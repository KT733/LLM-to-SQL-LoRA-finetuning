# Fine-Tuning SmolLM for Text-to-SQL Generation

This project fine-tunes **SmolLM2-1.7B-Instruct** using **LoRA** for the task of **text-to-SQL generation**. Given a natural language question and a database schema, the model generates an SQL query that answers the question. The project compares the base model against a LoRA fine-tuned version and evaluates improvements using SQL similarity scoring. :contentReference[oaicite:0]{index=0}

## Overview

Text-to-SQL is a practical NLP task where a model translates user questions into executable SQL queries. General-purpose LLMs can often produce plausible SQL, but they may struggle with schema grounding, joins, filtering conditions, and aggregation logic. This project improves performance by:

- using a domain-specific text-to-SQL dataset
- prompting the model with both the **question** and **schema**
- fine-tuning the base model with **parameter-efficient LoRA**
- evaluating outputs with normalized SQL similarity scoring :contentReference[oaicite:1]{index=1}

The workflow includes:

1. loading a text-to-SQL dataset
2. evaluating the base model
3. applying LoRA fine-tuning
4. re-evaluating the fine-tuned model
5. comparing qualitative and quantitative results :contentReference[oaicite:2]{index=2}

---

## Model and Dataset

### Base model
The project uses:

- **HuggingFaceTB/SmolLM2-1.7B-Instruct** :contentReference[oaicite:3]{index=3}

### Dataset
The training and testing data come from:

- **b-mc2/sql-create-context**
- loaded from `sql_create_context_v4.json` :contentReference[oaicite:4]{index=4}

Each example contains:
- a natural language **question**
- a database **schema/context**
- the ground-truth **SQL answer** :contentReference[oaicite:5]{index=5}

For this experiment:
- **500 samples** are used for training
- **50 samples** are used for testing :contentReference[oaicite:6]{index=6}

---

## Key Features

- Fine-tunes a causal language model for **schema-aware SQL generation**
- Uses **LoRA** for efficient adaptation instead of full model fine-tuning
- Supports **multi-GPU setup** with `DataParallel`
- Evaluates both **baseline** and **fine-tuned** model outputs
- Includes **SQL normalization** and **fuzzy similarity scoring**
- Provides both **quantitative** and **qualitative** performance comparison :contentReference[oaicite:7]{index=7}

---

## Project Workflow

## 1. Environment setup
The script configures CUDA devices and checks available GPUs before training. It is designed to run on GPU and includes memory-related configuration for PyTorch. :contentReference[oaicite:8]{index=8}

## 2. Data loading
The dataset is loaded into a pandas DataFrame, then split into:
- training subset
- testing subset :contentReference[oaicite:9]{index=9}

## 3. Base model inference
A prompt template is constructed in the following format:
- user question
- database schema
- instruction to output only SQL :contentReference[oaicite:10]{index=10}

The base model is queried with this prompt to generate SQL before fine-tuning.

## 4. SQL extraction and evaluation
Generated responses are parsed to isolate the SQL output. The script then:
- normalizes SQL formatting
- removes irrelevant formatting differences
- computes a similarity score against the ground-truth SQL using fuzzy matching :contentReference[oaicite:11]{index=11}

## 5. LoRA fine-tuning
The model is adapted using PEFT with LoRA. The script:
- freezes original model parameters
- enables gradient checkpointing
- applies LoRA to attention projection layers:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj` :contentReference[oaicite:12]{index=12}

Training examples are formatted as chat-style system, user, and assistant turns so the model learns to map schema-aware prompts to SQL outputs. :contentReference[oaicite:13]{index=13}

## 6. Post-training evaluation
After fine-tuning, the model is tested on the same held-out test questions, and the project compares:
- average similarity score before fine-tuning
- average similarity score after fine-tuning
- low-scoring examples for qualitative analysis :contentReference[oaicite:14]{index=14}

---

## LoRA Configuration

The project uses the following LoRA setup:

- `r = 4`
- `lora_alpha = 16`
- `lora_dropout = 0.05`
- `bias = "none"`
- `task_type = "CAUSAL_LM"` :contentReference[oaicite:15]{index=15}

This makes fine-tuning much more efficient than updating all model parameters.

---

## Training Configuration

The model is trained with Hugging Face `Trainer` using:

- batch size: `2`
- gradient accumulation steps: `8`
- warmup steps: `10`
- max steps: `100`
- learning rate: `1e-3`
- mixed precision: `fp16` :contentReference[oaicite:16]{index=16}

A fixed random seed is also used to improve reproducibility. :contentReference[oaicite:17]{index=17}

---

## Evaluation Method

The evaluation is based on SQL similarity rather than exact string match. This is important because valid SQL may differ in formatting while still expressing the same logic.

The script evaluates outputs by:
1. extracting the model’s SQL output
2. normalizing SQL syntax and formatting
3. comparing generated SQL with ground truth using fuzzy token similarity :contentReference[oaicite:18]{index=18}

This makes evaluation more tolerant to formatting differences while still penalizing incorrect joins, filters, and aggregations.

---

## Results

The fine-tuned model outperformed the baseline model on the evaluation examples. According to the analysis in the script:

- the baseline often produced structurally incorrect SQL
- common errors included:
  - missing joins
  - incorrect aggregation
  - unnecessary subqueries
  - incorrect filtering logic
- the fine-tuned model generated SQL that was much closer to ground truth
- improvement was especially visible in joins, conditions, and aggregation structure :contentReference[oaicite:19]{index=19}

The script notes that the fine-tuned model achieved roughly **5x–8x higher scores** on several of the lowest-performing examples, though some complex queries still showed room for improvement. :contentReference[oaicite:20]{index=20}
