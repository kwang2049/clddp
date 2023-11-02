from __future__ import annotations
import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional
import wandb
import torch
import torch.distributed as dist
from transformers import Trainer
from torch.utils.data import Dataset
from clddp.dm import JudgedPassage, RetrievalDataset, RetrievedPassageIDList
from clddp.evaluation import search_and_evaluate
from clddp.args.train import RetrievalTrainingArguments
from clddp.retriever import RetrievalTrainingExample, Retriever, RetrieverConfig
from clddp.utils import is_device_zero, set_logger_format
from clddp.dataloader import load_dataset


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


class RetrievalTrainingDataset(Dataset):
    def __init__(self, train_data: List[JudgedPassage]):
        self.train_data = train_data

    def __getitem__(self, item: int):
        jp = self.train_data[item]
        return RetrievalTrainingExample(query=jp.query, passages=[jp.passage])

    def __len__(self):
        return len(self.train_data)


def run(
    args: RetrievalTrainingArguments,
    train_dataset: Optional[RetrievalTrainingDataset] = None,
    dev_dataset: Optional[RetrievalDataset] = None,
    test_dataset: Optional[RetrievalDataset] = None,
) -> None:
    if is_device_zero():
        args.dump_arguments()

    if args.do_dev:
        assert dev_dataset is not None
    if args.do_train:
        assert train_dataset is not None
        # Retriever building:
        config = RetrieverConfig(
            query_model_name_or_path=args.query_model_name_or_path,
            passage_model_name_or_path=args.passage_model_name_or_path,
            shared_encoder=args.shared_encoder,
            sep=args.sep,
            pooling=args.pooling,
            similarity_function=args.similarity_function,
            max_length=args.max_length,
            sim_scale=args.sim_scale,
        )
        retriever = Retriever(config)
        if args.query_prompt is not None:
            retriever.set_query_prompt(args.query_prompt)
        if args.passage_prompt is not None:
            retriever.set_passage_prompt(args.passage_prompt)

        # Begin training:
        trainer = RetrievalTrainer(
            model=retriever,
            args=args,
            data_collator=lambda examples: {"examples": examples},
            train_dataset=train_dataset,
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


if __name__ == "__main__":
    # Example cli: torchrun --nproc_per_node=4 --master_port=29501 -m clddp.train
    from clddp.utils import parse_cli

    set_logger_format(logging.INFO if is_device_zero() else logging.WARNING)
    args = parse_cli(RetrievalTrainingArguments)
    train_dataset = RetrievalTrainingDataset(
        load_dataset(
            enable=args.do_train,
            dataloader_name=args.train_dataloader,
            data_name_or_path=args.train_data,
        ).judged_passages_train
    )
    dev_dataset = load_dataset(
        enable=args.do_dev,
        dataloader_name=args.dev_dataloader,
        data_name_or_path=args.dev_data,
    )
    test_dataset = load_dataset(
        enable=args.do_test,
        dataloader_name=args.test_dataloader,
        data_name_or_path=args.test_data,
    )
    run(
        args=args,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
    )
    logging.info("Done")
