# Build a retriever checkpoint:
export CHECKPOINT_DIR="checkpoints/off-the-shell/tasb"
export CLI_ARGS="
--output_dir=$CHECKPOINT_DIR
--query_model_name_or_path="sentence-transformers/msmarco-distilbert-base-tas-b"
--shared_encoder=True
--sep=blank
--pooling=cls
--similarity_function=dot_product
--max_length=350
--sim_scale=1.0
"
python -m clddp.retriever $CLI_ARGS

# Run mining positives and negatives:
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
export CUDA_VISIBLE_DEVICES="0,1"
export CLI_ARGS="
--checkpoint_dir=$CHECKPOINT_DIR
--data_dir=$DATASET_PATH
--dataloader=beir
"
export OUTPUT_DIR=$(python -m clddp.args.mine $CLI_ARGS)
mkdir -p $OUTPUT_DIR
export LOG_PATH="$OUTPUT_DIR/logging.log"
echo "Logging file path: $LOG_PATH"
torchrun --nproc_per_node=2 --master_port=29501 -m clddp.mine $CLI_ARGS

# Run training with the mined passages:
export NEGATIVES_PARTH="$OUTPUT_DIR/mined_with_filtering.txt"
export POSITIVES_PARTH="$OUTPUT_DIR/mined_positives.txt"
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
--max_length=350
--sim_scale=1.0
--fp16=True
--train_data=$DATASET_PATH
--train_dataloader=beir
--num_negatives=1
--negatives_path=$NEGATIVES_PARTH
--positives_path=$NEGATIVES_PARTH
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