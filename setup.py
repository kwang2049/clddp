from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    readme = fh.read()

setup(
    name="clddp",
    version="0.0.6",
    author="Kexin Wang",
    author_email="kexin.wang.2049@gmail.com",
    description="A package for training and doing inference with contrastive learning with multiple GPUs (Pytorch-DDP).",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://https://github.com/kwang2049/ddp-cl",
    project_urls={
        "Bug Tracker": "https://github.com/kwang2049/ddp-cl/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "ujson",
        "wandb",
        "datasets",
        "more_itertools",
        "pytrec_eval",
        "sentence-transformers",
        "accelerate",
        "colbert @ git+https://github.com/stanford-futuredata/ColBERT.git@21b460a606bed606e8a7fa105ada36b18e8084ec#egg=colbert-0.2.0",
        "faiss-cpu",  # Not used. Just for removing some WARNINGs
    ],
)
