#!/bin/bash

# Ensure conda is initialized
source ~/miniconda3/etc/profile.d/conda.sh  # Adjust this path to your Conda installation

# Activate the conda environment
conda activate llm

# Run the Flask app and redirect output to out.txt
flask --app reactome_llm/ReactomeLLMRestAPI.py run >out.txt 2>&1 &


