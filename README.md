# clddp: Contrastive Learning with Distributed Data Parallel

This Python package provides an implementation for constrastive learning with multiple GPUs (i.e. Distributed-Data-Parallel), with a special focus on neural retrieval.


## Installation
```bash
pip install -U clddp
```

If ColBERT is going to be used, please install its package additionally:
```bash
pip install git+https://github.com/stanford-futuredata/ColBERT.git@21b460a606bed606e8a7fa105ada36b18e8084ec
```


## Quick Start
Please have a look at the [examples](examples) for a quick start. For example, one can run the multi-GPU training with the following script:

### Training example
```bash
dataset="fiqa"
export DATA_DIR="data"
mkdir -p $DATA_DIR
export DATASET_PATH="$DATA_DIR/$dataset"
if [ ! -f "$DATASET_PATH.zip" ]; then
    wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$dataset.zip -P $DATA_DIR
fi
if [ ! -d "$DATASET_PATH" ]; then
    unzip $DATASET_PATH.zip -d $DATA_DIR
fi
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export WANDB_MODE="online"
export CHECKPOINT_DIR="checkpoints"
export CLI_ARGS="
--project="clddp_examples"
--checkpoint_dir=$CHECKPOINT_DIR
--query_model_name_or_path="sentence-transformers/msmarco-distilbert-base-tas-b"
--shared_encoder=True
--sep=blank
--pooling=cls
--similarity_function=dot_product
--query_max_length=350
--passage_max_length=350
--sim_scale=1.0
--fp16=True
--train_data=$DATASET_PATH
--train_dataloader=beir
--num_negatives=0
--dev_data=$DATASET_PATH
--dev_dataloader=beir
--do_dev=True
--quick_dev=False
--test_data=$DATASET_PATH
--test_dataloader=beir
--num_train_epochs=1
--eval_steps=0.4
--save_steps=0.4
"
export OUTPUT_DIR=$(python -m clddp.args.train $CLI_ARGS)
mkdir -p $OUTPUT_DIR
export LOG_PATH="$OUTPUT_DIR/logging.log"
echo "Logging file path: $LOG_PATH"
nohup torchrun --nproc_per_node=4 --master_port=29501 -m clddp.train $CLI_ARGS > $LOG_PATH &
```
### Search Example
This will run exact search with multiple GPUs and output the retrieval results in the TREC format (each row is `query-id` `Q0` `passage-id` `rank` `score` `exp`):
```bash
# Build a retriever checkpoint:
export CHECKPOINT_DIR="checkpoints/off-the-shell/tasb"
export CLI_ARGS="
--output_dir=$CHECKPOINT_DIR
--query_model_name_or_path="sentence-transformers/msmarco-distilbert-base-tas-b"
--shared_encoder=True
--sep=blank
--pooling=cls
--similarity_function=dot_product
--query_max_length=350
--passage_max_length=350
--sim_scale=1.0
"
python -m clddp.retriever $CLI_ARGS

# Run search:
dataset="fiqa"
export DATA_DIR="data"
export DATASET_PATH="$DATA_DIR/$dataset"
if [ ! -d $DATASET_PATH ]; then
    echo "Data path $DATASET_PATH does not exist"
    exit
fi
if [ ! -d $CHECKPOINT_DIR ]; then
    echo "Checkpoint path $CHECKPOINT_DIR does not exist"
    exit
fi
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CLI_ARGS="
--checkpoint_dir=$CHECKPOINT_DIR
--data_dir=$DATASET_PATH
--dataloader=beir
"
export OUTPUT_DIR=$(python -m clddp.args.search $CLI_ARGS)
mkdir -p $OUTPUT_DIR
export LOG_PATH="$OUTPUT_DIR/logging.log"
echo "Logging file path: $LOG_PATH"
nohup torchrun --nproc_per_node=4 --master_port=29501 -m clddp.search $CLI_ARGS > $LOG_PATH &
```
Evaluation is similar and the example can be found [here](examples/eval_fiqa.sh).

## Custom Data
For loading custom data, one needs to inherit the `clddp.dataloader.BaseDataLoader` and add it to the lookup map `clddp.DATA_LOADER_LOOKUP`:

```python
from clddp.train import main
from clddp.dataloader import BaseDataLoader, DATA_LOADER_LOOKUP
from clddp.dm import RetrievalDataset

class MyDataLoader(BaseDataLoader):
    def load_data(data_name_or_path: str, progress_bar: bool) -> RetrievalDataset:
        ...

DATA_LOADER_LOOKUP["my_dataloader"] = MyDataLoader

if __name__ == "__main__":
    main()  # Same for other entry points
```
Then, one can specify the `xxx_dataloader=my_dataloader` in the CLI arguments.