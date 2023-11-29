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

# Build a reranker checkpoint:
export RERANKER_CHECKPOINT_DIR="checkpoints/off-the-shell/minilm-l6-v2"
export CLI_ARGS="
--output_dir=$RERANKER_CHECKPOINT_DIR
--model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2"
--sep=blank
--max_length=512
"
python -m clddp.reranker $CLI_ARGS

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
export CUDA_VISIBLE_DEVICES="0,1"
export CLI_ARGS="
--checkpoint_dir=$CHECKPOINT_DIR
--reranker_checkpoint_dir=$RERANKER_CHECKPOINT_DIR
--data_dir=$DATASET_PATH
--dataloader=beir
--topk=100
"
export OUTPUT_DIR=$(python -m clddp.args.search $CLI_ARGS)
mkdir -p $OUTPUT_DIR
export LOG_PATH="$OUTPUT_DIR/logging.log"
echo "Logging file path: $LOG_PATH"
nohup torchrun --nproc_per_node=2 --master_port=29501 -m clddp.search $CLI_ARGS > $LOG_PATH &
