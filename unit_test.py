# %%
#!pip install numpy torch transformers peft datasets sqlparse rapidfuzz
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, DataCollatorForLanguageModeling, TrainingArguments
import random
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import math
import os
import sqlparse
import difflib
from rapidfuzz import fuzz
import re
import pandas as pd

# %% [markdown]
# ## Environment Setup

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7"  
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# %%
torch.cuda.is_available()

# %% [markdown]
# ## Data Import

# %%
df = pd.read_json("hf://datasets/b-mc2/sql-create-context/sql_create_context_v4.json")
print(df.shape)
df.head()
train_df = df.iloc[1050:1100] 
train_df = train_df.reset_index()
test_df = df.iloc[1100:1110]
test_df = test_df.reset_index()

# %% [markdown]
# ## Load Model

# %%
checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
torch.cuda.empty_cache()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda:0"

# %%
tokenizer_small = AutoTokenizer.from_pretrained(checkpoint)
tokenizer_small.add_eos_token=True
tokenizer_small.pad_token_id=0
tokenizer_small.padding_side="left"
model_small = AutoModelForCausalLM.from_pretrained(checkpoint)
model_small = torch.nn.DataParallel(model_small)
model_small.to(device)

# %%
generator_params = {
    "max_new_tokens": 250, 
    "temperature": 0.1,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer_small.eos_token_id,
    "eos_token_id": tokenizer_small.eos_token_id
}

# %%
def query_lm(model, tokenizer, question, generator_params):
    messages = [{"role": "user", "content": question}]
    input_text=tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        if isinstance(model, torch.nn.DataParallel):
            outputs = model.module.generate(inputs, **generator_params)
        else:
            outputs = model.generate(inputs, **generator_params)
    return tokenizer_small.decode(outputs[0])

# %%
def extract_sql(response):
    '''Match the SQL query after assistant's answer, capturing everything between <|im_start|>assistant and <|im_end|>'''
    match = re.search(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", response, re.DOTALL)
    if match:
        return match.group(1).strip()  
    return " "  
    
def normalize_sql(sql):
    """Normalize SQL query by formatting it consistently and removing unnecessary punctuation."""
    formatted_sql = sqlparse.format(sql, reindent=True, keyword_case='upper')
    logical_operators = {">", "<", "=", "!=", ">=", "<="}
    tokens = re.split(r'(\W+)', formatted_sql) 
    cleaned_tokens = [
        token for token in tokens if token.strip() and (token.strip().isalnum() or token.strip() in logical_operators)
    ]
    return " ".join(cleaned_tokens)

def similarity_score(true_sql, gen_sql):
    """Compute similarity score between ground truth and genearted SQL."""
    true_sql = normalize_sql(true_sql)
    gen_sql = normalize_sql(gen_sql)
    
    # Exact match
    if true_sql == gen_sql:
        return 1.0
    
    # Token similarity 
    token_sim = fuzz.ratio(gen_sql, true_sql) / 100.0

    return token_sim

# %% [markdown]
# ### Base Model

# %%
question_template = """You will be asked to write an SQL query given a specific question and its database schema.

Question: {Question}
Schema: {Schema}

Provide only the SQL query in the response without additional text.
"""
questions = [question_template.format(Question=row.question, Schema=row.context) for _, row in test_df.iterrows()]
outputs=[query_lm(model_small, tokenizer_small, q, generator_params) for q in questions]
scores = [similarity_score(extract_sql(out), test_df['answer'][i]) for i, out in enumerate(outputs)]

# %%
base_results = [
    (i, questions[i], outputs[i], scores[i]) for i in range(len(scores))
]

base_results_sorted = sorted(base_results, key=lambda x: x[3])  

num = 25
for idx, q, out, score in base_results_sorted[:num]:
    print(f"Index: {idx}\nScore: {score}\nOutput: {extract_sql(out)}\n")

# %% [markdown]
# ### Fine-tuned Model

# %%
for param in model_small.parameters():
    param.requires_grad = False  
    if param.ndim == 1:
        param.data = param.data.to(torch.float32)

model_small.module.gradient_checkpointing_enable() 

class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)
model_small.module.lm_head = CastOutputToFloat(model_small.module.lm_head)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

for mod in model_small.named_modules():
    print(mod)

config = LoraConfig(
    r=2,
    lora_alpha=8,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
config.inference_mode = False

model = get_peft_model(model_small.module, config)
print_trainable_parameters(model)

ft_questions = [question_template.format(Question=row.question, Schema=row.context) for _, row in train_df.iterrows()]
ft_answers = [answer for answer in train_df['answer'] ]
qa_dict = {"question": ft_questions, "answer": ft_answers}
dataset = Dataset.from_dict(qa_dict)

def prepare_and_tokenize(example):
    q = example["question"]
    a = example["answer"]
    template = f'''<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
    <|im_start|>user\n
    {q} <|im_end|>
    <|im_start|>assistant\n {a}<|im_end|>'''
    return tokenizer_small(template, truncation=True, padding='max_length', max_length=2048)

mapped_train_dataset = dataset.map(prepare_and_tokenize, batched=False, remove_columns=['question', 'answer'])

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

trainer = Trainer(
    model=model,
    train_dataset=mapped_train_dataset,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        max_steps=8,
        learning_rate=1e-3,
        logging_steps=1,
        output_dir='test_outputs'
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer_small, mlm=False)
)
trainer.args._n_gpu = 1
model.config.use_cache = False

trainer.train()

# %%
model.config.use_cache = True
model.eval()
ft_outputs=[query_lm(model, tokenizer_small, q, generator_params) for q in questions]
ft_scores = [similarity_score(extract_sql(out), test_df['answer'].iloc[i]) for i, out in enumerate(ft_outputs)]

# %%
ft_results = [
    (i, ft_questions[i], ft_outputs[i], ft_scores[i]) for i in range(len(ft_scores))
]

ft_results_sorted = sorted(ft_results, key=lambda x: x[3])  

num = 25
for idx, q, out, score in ft_results_sorted[:num]:
    print(f"Index: {idx}\nScore: {score}\nOutput: {extract_sql(out)}\n")


