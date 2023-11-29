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
export CUDA_VISIBLE_DEVICES="0,1"
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
nohup torchrun --nproc_per_node=2 --master_port=29501 -m clddp.train $CLI_ARGS > $LOG_PATH &