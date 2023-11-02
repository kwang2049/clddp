export CLI_ARGS="
--output_dir="checkpoints/off-the-shell/tasb"
--query_model_name_or_path="sentence-transformers/msmarco-distilbert-base-tas-b"
--shared_encoder=True
--sep=blank
--pooling=cls
--similarity_function=dot_product
--max_length=350
--sim_scale=1.0
"
python -m clddp.retriever $CLI_ARGS