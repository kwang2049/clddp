# export CUDA_VISIBLE_DEVICES="0,1"
# export NGPUS=2
export CUDA_VISIBLE_DEVICES="5,6,7,8"
export NGPUS=4

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
# python -m clddp.retriever $CLI_ARGS

# Run hard-negative mining:
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
export CLI_ARGS="
--checkpoint_dir=$CHECKPOINT_DIR
--data_dir=$DATASET_PATH
--dataloader=beir
--negative_start_ranking=1
--negative_end_ranking=100
"
export MINING_OUTPUT_DIR=$(python -m clddp.args.mine $CLI_ARGS)
mkdir -p $MINING_OUTPUT_DIR
export LOG_PATH="$MINING_OUTPUT_DIR/logging.log"
echo "Logging file path: $LOG_PATH"
# torchrun --nproc_per_node=$NGPUS --master_port=29501 -m clddp.mine $CLI_ARGS

# Run search for dev:
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
export CLI_ARGS="
--checkpoint_dir=$CHECKPOINT_DIR
--data_dir=$DATASET_PATH
--dataloader=beir
--topk=100
--split=dev
"
export RETRIEVAL_DEV_OUTPUT_DIR=$(python -m clddp.args.search $CLI_ARGS)
mkdir -p $RETRIEVAL_DEV_OUTPUT_DIR
export LOG_PATH="$RETRIEVAL_DEV_OUTPUT_DIR/logging.log"
echo "Logging file path: $LOG_PATH"
# torchrun --nproc_per_node=$NGPUS --master_port=29501 -m clddp.search $CLI_ARGS

# Run search for test:
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
export CLI_ARGS="
--checkpoint_dir=$CHECKPOINT_DIR
--data_dir=$DATASET_PATH
--dataloader=beir
--topk=100
--split=test
"
export RETRIEVAL_TEST_OUTPUT_DIR=$(python -m clddp.args.search $CLI_ARGS)
mkdir -p $RETRIEVAL_TEST_OUTPUT_DIR
export LOG_PATH="$RETRIEVAL_TEST_OUTPUT_DIR/logging.log"
echo "Logging file path: $LOG_PATH"
# torchrun --nproc_per_node=$NGPUS --master_port=29501 -m clddp.search $CLI_ARGS

# Run training with the mined negatives:
export NEGATIVES_PARTH="$MINING_OUTPUT_DIR/mined_with_filtering.txt"
export DEV_RETRIEVAL_RESULTS="$RETRIEVAL_DEV_OUTPUT_DIR/ranking_results.txt"
export TEST_RETRIEVAL_RESULTS="$RETRIEVAL_TEST_OUTPUT_DIR/ranking_results.txt"
export WANDB_MODE="online"
export CHECKPOINT_DIR="checkpoints"
export CLI_ARGS="
--project="clddp_examples"
--checkpoint_dir=$CHECKPOINT_DIR
--model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2"
--sep=blank
--max_length=512
--fp16=True
--train_data=$DATASET_PATH
--train_dataloader=beir
--num_negatives=31
--negatives_path=$NEGATIVES_PARTH
--dev_data=$DATASET_PATH
--dev_dataloader=beir
--dev_retrieval_results=$DEV_RETRIEVAL_RESULTS
--do_dev=True
--quick_dev=False
--test_data=$DATASET_PATH
--test_dataloader=beir
--test_retrieval_results=$TEST_RETRIEVAL_RESULTS
--num_train_epochs=1
--eval_steps=0.4
--save_steps=0.4
"
export OUTPUT_DIR=$(python -m clddp.args.train_reranker $CLI_ARGS)
mkdir -p $OUTPUT_DIR
export LOG_PATH="$OUTPUT_DIR/logging.log"
echo "Logging file path: $LOG_PATH"
nohup torchrun --nproc_per_node=$NGPUS --master_port=29501 -m clddp.train_reranker $CLI_ARGS > $LOG_PATH &