# Build a retriever checkpoint:
export CHECKPOINT_DIR="checkpoints/off-the-shell/colbertv2"
export CLI_ARGS="
--output_dir=$CHECKPOINT_DIR
--query_model_name_or_path="colbert-ir/colbertv2.0"
--shared_encoder=True
--sep=blank
--pooling=no_pooling
--similarity_function=maxsim
--query_max_length=32
--passage_max_length=300
--sim_scale=1.0
"
python -m clddp.retriever $CLI_ARGS

# Run evaluation:
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
export OUTPUT_DIR=$(python -m clddp.args.evaluation $CLI_ARGS)
mkdir -p $OUTPUT_DIR
export LOG_PATH="$OUTPUT_DIR/logging.log"
echo "Logging file path: $LOG_PATH"
nohup torchrun --nproc_per_node=2 --master_port=29501 -m clddp.evaluation $CLI_ARGS > $LOG_PATH &