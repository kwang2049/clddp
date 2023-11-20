from __future__ import annotations
import json
import logging
import os
import shutil
from typing import Dict, List, Optional
import wandb
import torch.distributed as dist
from transformers import Trainer
from torch.utils.data import Dataset
from clddp.dm import RetrievalDataset, RetrievedPassageIDList, Split
from clddp.evaluation import rerank_and_evaluate
from clddp.args.train_reranker import RerankingTrainingArguments
from clddp.reranker import RerankerConfig, RerankerInputExample, Reranker
from clddp.train import RetrievalTrainingData
from clddp.utils import is_device_zero, set_logger_format, parse_cli
from clddp.dataloader import load_dataset
from clddp.mine import MiningType, load_mined


class RerankingTrainer(Trainer):
    @property
    def reranker(self) -> Reranker:
        return self.model

    def _save(self, output_dir: Optional[str] = None) -> None:
        self.reranker.save(output_dir)

    def evaluate(
        self,
        eval_dataset: Optional[RetrievalDataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Optional[Dict[str, float]]:
        args: RerankingTrainingArguments = self.args
        if args.do_dev:
            dev_dataset: RetrievalDataset = self.eval_dataset
            assert dev_dataset is not None
            assert args.dev_retrieval_results
        else:
            return None

        retrieval_results = RetrievedPassageIDList.from_trec_csv(
            args.dev_retrieval_results
        )
        _, report = rerank_and_evaluate(
            retrieval_results=retrieval_results,
            reranker=self.reranker,
            eval_dataset=dev_dataset,
            split=args.dev_split,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            fp16=args.fp16,
            metric_key_prefix=metric_key_prefix,
        )
        self.control.should_evaluate = (
            False  # Otherwise it will evaluate it again for epoch ends
        )
        if is_device_zero():
            self.log(report)
            return report
        else:
            return None


class RerankingTrainingData(RetrievalTrainingData):
    def __getitem__(self, item: int):
        example = super().__getitem__(item)
        assert len(example.passages) > 1
        labels = [0 for _ in range(len(example.passages))]
        labels[0] = 1
        return RerankerInputExample(
            query=example.query, passages=example.passages, labels=labels
        )

    def __len__(self):
        return len(self.data)


def run(
    reranker: Reranker,
    args: RerankingTrainingArguments,
    training_data: Optional[Dataset] = None,
    dev_dataset: Optional[RetrievalDataset] = None,
    test_dataset: Optional[RetrievalDataset] = None,
) -> None:
    if is_device_zero():
        args.dump_arguments()

    if args.do_dev:
        assert dev_dataset is not None
    if args.do_train:
        assert training_data is not None
        # Begin training:
        trainer = RerankingTrainer(
            model=reranker,
            args=args,
            data_collator=lambda examples: {"examples": examples},
            train_dataset=training_data,
            eval_dataset=dev_dataset,
        )
        trainer.control.should_evaluate = True
        trainer._maybe_log_save_evaluate(
            tr_loss=None,
            model=reranker,
            trial=None,
            epoch=0,
            ignore_keys_for_eval=None,
        )
        trainer.train()
        logging.info("done training")

        # Saving:
        if is_device_zero():
            if trainer.state.best_model_checkpoint:
                shutil.copytree(
                    trainer.state.best_model_checkpoint,
                    os.path.dirname(
                        os.path.abspath(trainer.state.best_model_checkpoint)
                    ),
                    dirs_exist_ok=True,
                )
                logging.info(
                    f"Saved the best checkpoint from {trainer.state.best_model_checkpoint}"
                )
            else:
                trainer.save_model()
                logging.info("Saved the last checkpoint")

        # Make sure on the same page before evaluation:
        if dist.is_initialized():
            dist.barrier()

    # Final evaluation:
    if args.do_test:
        if os.path.exists(args.output_dir):
            reranker = Reranker.from_pretrained(args.output_dir)
            logging.info(f"Loaded from {args.output_dir} for evaluation")
        else:
            assert not args.do_train, "No checkpoints saved after training"
        retrieval_results = RetrievedPassageIDList.from_trec_csv(
            args.test_retrieval_results
        )
        retrieval_results, report = rerank_and_evaluate(
            retrieval_results=retrieval_results,
            reranker=reranker,
            eval_dataset=test_dataset,
            split=args.test_split,
            fp16=args.fp16,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
        )
        if is_device_zero():
            RetrievedPassageIDList.dump_trec_csv(
                retrieval_results=retrieval_results,
                fpath=os.path.join(args.output_dir, "retrieval_results.txt"),
            )
            freport = os.path.join(args.output_dir, "metrics.json")
            with open(freport, "w") as f:
                json.dump(report, f, indent=4)
            logging.info(f"Saved evaluation metrics to {freport}.")
            if wandb.run is None:
                wandb.init(
                    project=args.project,
                    name=args.run_name,
                    id=os.environ.get("WANDB_RUN_ID", None),
                )
            wandb_summary: dict = wandb.summary
            wandb_summary.update(report)
            wandb.finish()


def main(args: Optional[RerankingTrainingArguments] = None):
    set_logger_format(logging.INFO if is_device_zero() else logging.WARNING)
    if args is None:
        args = parse_cli(RerankingTrainingArguments)

    # Retriever building:
    config = RerankerConfig(
        model_name_or_path=args.model_name_or_path,
        sep=args.sep,
        max_length=args.max_length,
    )
    reranker = Reranker(config)

    # Data loading:
    train_dataset = load_dataset(
        enable=args.do_train,
        dataloader_name=args.train_dataloader,
        data_name_or_path=args.train_data,
    )
    if args.do_train:
        assert args.negatives_path
        assert os.path.exists(args.negatives_path)
        assert args.num_negatives
        load_mined(
            mined_path=args.negatives_path,
            mining_type=MiningType.negatives,
            dataset=train_dataset,
            split=Split.train,
            prograss_bar=is_device_zero(),
        )
    training_data = RerankingTrainingData(
        training_data=train_dataset.train_labeled_queries,
        num_negatives=args.num_negatives,
    )
    dev_dataset = train_dataset
    if args.dev_data != args.train_data:
        dev_dataset = load_dataset(
            enable=args.do_dev,
            dataloader_name=args.dev_dataloader,
            data_name_or_path=args.dev_data,
        )
    if args.quick_dev:
        assert args.do_dev
        save_pids_to_fpath = (
            os.path.join(args.output_dir, "quick_dev_pids.json")
            if is_device_zero()
            else None
        )
        dev_dataset = dev_dataset.to_quick_version(
            split=Split.dev,
            progress_bar=is_device_zero(),
            save_pids_to_fpath=save_pids_to_fpath,
        )
    test_dataset = train_dataset
    if args.test_data != args.train_data:
        test_dataset = load_dataset(
            enable=args.do_test,
            dataloader_name=args.test_dataloader,
            data_name_or_path=args.test_data,
        )
    if args.quick_test:
        assert args.do_test
        save_pids_to_fpath = (
            os.path.join(args.output_dir, "quick_test_pids.json")
            if is_device_zero()
            else None
        )
        test_dataset = test_dataset.to_quick_version(
            split=Split.test,
            progress_bar=is_device_zero(),
            save_pids_to_fpath=save_pids_to_fpath,
        )

    # Run training:
    run(
        reranker=reranker,
        args=args,
        training_data=training_data,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
    )
    logging.info("Done")


if __name__ == "__main__":
    # Example cli: torchrun --nproc_per_node=4 --master_port=29501 -m clddp.train
    main()
