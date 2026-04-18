# LoRA Fine-tuning   
Implement LoRA fine-tuning with PEFT.

## Python Requirements
This project requires Python 3.10 or later. The libraries needed are listed in the `requirements.txt` file.

## How to Access Data 
Data Source: https://huggingface.co/datasets/b-mc2/sql-create-context?row=0

The data is already imported in both `main.py` and `unit_test.py`

## How to Run `main.py`
1. Set up the GPU environment and pip install all required packages from `requirements.txt`.
2. Run `main.py` and execute the cells in order.

## How to Run `unit_test.py`
1. Set up the GPU environment and pip install all required packages from `requirements.txt`.
2. Run `unit_test.py` and execute the cells in order.

## Files Included 
- `main.py`: Jupyter Notebook with the assignment 3 implementation in the project description above.
- `unit_test.py`: Unit test script that tests fine-tuning function on a very small subset of data.
- `requirements.txt`: Python dependencies.
- `sample_output_hw3.txt`: Sample output showing example questions and GPT response.
- `my_model`: Folder containing the fine-tuned model parameters and its associated configuration files.
  - `README.md`: A markdown file providing documentation or instructions specific to the fine-tuned model.
  - `merges.txt`: A file containing merge operations for the tokenizer.
  - `tokenizer_config.json`: Configuration file for the tokenizer.
  - `adapter_config.json`: Configuration file for the adapter model.
  - `special_tokens_map.json`: A JSON file mapping special tokens.
  - `adapter_model.safetensors`: The fine-tuned adapter model weights saved in the safetensors format.
  - `tokenizer.json`: A JSON file containing the tokenizer's vocabulary and configuration.
  - `vocab.json`: A JSON file containing the vocabulary used by the tokenizer.

