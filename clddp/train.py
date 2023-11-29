from __future__ import annotations
import json
import logging
import os
import random
import shutil
from typing import Any, Dict, List, Optional
import wandb
import torch
import torch.distributed as dist
from transformers import Trainer
from torch.utils.data import Dataset
from clddp.dm import LabeledQuery, RetrievalDataset, RetrievedPassageIDList, Split
from clddp.evaluation import search_and_evaluate
from clddp.args.train import RetrievalTrainingArguments
from clddp.retriever import RetrievalTrainingExample, Retriever, RetrieverConfig
from clddp.utils import is_device_zero, set_logger_format, parse_cli
from clddp.dataloader import load_dataset
from clddp.mine import MiningType, load_mined


class RetrievalTrainer(Trainer):
    @property
    def retriever(self) -> Retriever:
        return self.model

    def _save(self, output_dir: Optional[str] = None) -> None:
        self.retriever.save(output_dir)

    def training_step(
        self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor | Any]
    ) -> torch.Tensor:
        loss = super().training_step(model, inputs)
        if dist.is_initialized():
            loss /= dist.get_world_size()  # Scale back the value for logging correctly
        return loss

    def evaluate(
        self,
        eval_dataset: Optional[RetrievalDataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Optional[Dict[str, float]]:
        args: RetrievalTrainingArguments = self.args
        if args.do_dev:
            dev_dataset = self.eval_dataset
            assert dev_dataset is not None
        else:
            return None
        _, report = search_and_evaluate(
            retriever=self.retriever,
            eval_dataset=dev_dataset,
            split=args.dev_split,
            topk=args.topk,
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


class RetrievalTrainingData(Dataset):
    def __init__(self, training_data: List[LabeledQuery], num_negatives: int):
        self.num_negatives = num_negatives
        with_positives = [lq for lq in training_data if len(lq.positives)]
        self.data = with_positives
        if len(with_positives) < len(training_data):
            logging.info(
                f"Kept {len(with_positives)} labeled queries which have positives out of {len(training_data)}."
            )
        if num_negatives:
            self.data = []
            for lq in with_positives:
                if len(lq.negatives) == 0:
                    logging.warn(
                        f"No negatives for query ID {lq.query.query_id} (num_negatives set to {num_negatives})"
                    )
                    continue
                self.data.append(lq)
            if len(with_positives) > len(self.data):
                logging.info(
                    f"Kept {len(self.data)} out of {len(training_data)}. The data are removed due to no negatives."
                )

    def __getitem__(self, item: int):
        lq = self.data[item]
        positives = list(lq.positives)
        negatives = list(lq.negatives) if lq.negatives else []
        random.shuffle(positives)
        random.shuffle(negatives)
        positive = positives[0]
        negatives = [negatives[i % len(negatives)] for i in range(self.num_negatives)]
        passages = [positive.passage] + [negative.passage for negative in negatives]
        return RetrievalTrainingExample(query=positive.query, passages=passages)

    def __len__(self):
        return len(self.data)


def run(
    retriever: Retriever,
    args: RetrievalTrainingArguments,
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
        trainer = RetrievalTrainer(
            model=retriever,
            args=args,
            data_collator=lambda examples: {"examples": examples},
            train_dataset=training_data,
            eval_dataset=dev_dataset,
        )
        trainer.control.should_evaluate = True
        trainer._maybe_log_save_evaluate(
            tr_loss=None,
            model=retriever,
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
            retriever = Retriever.from_pretrained(args.output_dir)
            logging.info(f"Loaded from {args.output_dir} for evaluation")
            if args.query_prompt is not None:
                retriever.set_query_prompt(args.query_prompt)
            if args.passage_prompt is not None:
                retriever.set_passage_prompt(args.passage_prompt)
        else:
            assert not args.do_train, "No checkpoints saved after training"
        retrieval_results, report = search_and_evaluate(
            retriever=retriever,
            eval_dataset=test_dataset,
            split=args.test_split,
            topk=args.topk,
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


def main(args: Optional[RetrievalTrainingArguments] = None) -> Retriever:
    set_logger_format(logging.INFO if is_device_zero() else logging.WARNING)
    if args is None:
        args = parse_cli(RetrievalTrainingArguments)

    # Retriever building:
    config = RetrieverConfig(
        query_model_name_or_path=args.query_model_name_or_path,
        passage_model_name_or_path=args.passage_model_name_or_path,
        shared_encoder=args.shared_encoder,
        sep=args.sep,
        pooling=args.pooling,
        similarity_function=args.similarity_function,
        query_max_length=args.query_max_length,
        passage_max_length=args.passage_max_length,
        sim_scale=args.sim_scale,
    )
    retriever = Retriever(config)
    if args.query_prompt is not None:
        retriever.set_query_prompt(args.query_prompt)
    if args.passage_prompt is not None:
        retriever.set_passage_prompt(args.passage_prompt)

    # Data loading:
    train_dataset = load_dataset(
        enable=args.do_train,
        dataloader_name=args.train_dataloader,
        data_name_or_path=args.train_data,
    )
    if args.negatives_path:
        assert args.do_train
        assert train_dataset is not None
        assert args.num_negatives
        load_mined(
            mined_path=args.negatives_path,
            mining_type=MiningType.negatives,
            dataset=train_dataset,
            split=Split.train,
            prograss_bar=is_device_zero(),
        )
    if args.positives_path:
        assert args.do_train
        assert train_dataset is not None
        load_mined(
            mined_path=args.positives_path,
            mining_type=MiningType.positives,
            dataset=train_dataset,
            split=Split.train,
            prograss_bar=is_device_zero(),
        )
    training_data = RetrievalTrainingData(
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
        retriever=retriever,
        args=args,
        training_data=training_data,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
    )
    logging.info("Done")
    return Retriever


if __name__ == "__main__":
    # Example cli: torchrun --nproc_per_node=4 --master_port=29501 -m clddp.train
    main()
