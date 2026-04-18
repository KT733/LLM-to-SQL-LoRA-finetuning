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
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"  
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # List the names of the available GPUs
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
else:
    print("No GPU available, using CPU instead.")

# %%
torch.cuda.is_available()

# %% [markdown]
# ## Data Import

# %%
df = pd.read_json("hf://datasets/b-mc2/sql-create-context/sql_create_context_v4.json")
print(df.shape)
df.head()

# %%
train_df = df.iloc[0:500] # subset 500 question-answer pairs for training 
test_df = df.iloc[1000:1050] # subset 50 question-answer pairs for testing 

# %% [markdown]
# ## Load Model

# %%
checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
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
    # Match the SQL query after assistant's answer, capturing everything between <|im_start|>assistant and <|im_end|>
    match = re.search(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", response, re.DOTALL)
    if match:
        return match.group(1).strip()  
    return " "  # Return None if no match is found
    
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

# Example Usage
gt = "SELECT COUNT(*) FROM head WHERE age > 56"
pred = "SELECT COUNT(*) FROM head WHERE age > 56"
print(similarity_score(gt, pred))  # Outputs similarity score


# %%
question_template = """You will be asked to write an SQL query given a specific question and its database schema.

Question: {Question}
Schema: {Schema}

Provide only the SQL query in the response without additional text.
"""
questions = [question_template.format(Question=row.question, Schema=row.context) for _, row in test_df.iterrows()]
outputs=[query_lm(model_small, tokenizer_small, q, generator_params) for q in questions]
scores = [similarity_score(extract_sql(out), test_df['answer'][i]) for i, out in enumerate(outputs)]

#example output
print("question: ", questions[0])
print("---------------------------------------------------------------------------")
print("output: ", outputs[0])
print("---------------------------------------------------------------------------")
print("score: ", scores[0])
print("---------------------------------------------------------------------------")

# %%
pd.set_option('display.max_colwidth', None)
print(test_df[test_df['question']=='Find the names and total checking and savings balances of accounts whose savings balance is higher than the average savings balance.'].answer)

# %%
print(f"Initial accuracy with base model: {np.sum(scores)/len(scores)}")

# %% [markdown]
# ## LoRA Fine-tuning

# %%
for param in model_small.parameters():
    # requires_grad will stop any gradients from being computed for that parameter and will stay frozen during backwards passes
    param.requires_grad = False  
    
    # changing 1d parameters such as biases to high precision may help with stability
    if param.ndim == 1:
        param.data = param.data.to(torch.float32)

# below helps with memory, but is not necessary
model_small.module.gradient_checkpointing_enable() 

# ensures that the final output logits of the model are full precision
class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)
model_small.module.lm_head = CastOutputToFloat(model_small.module.lm_head)

# %%
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

# %%
for mod in model_small.named_modules():
    print(mod)

# %%
config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
config.inference_mode = False

# %%
model = get_peft_model(model_small.module, config)
print_trainable_parameters(model)

# %%
ft_questions = [question_template.format(Question=row.question, Schema=row.context) for _, row in train_df.iterrows()]
ft_answers = [answer for answer in train_df['answer'] ]
qa_dict = {"question": ft_questions, "answer": ft_answers}
train_dataset = Dataset.from_dict(qa_dict)

# %%
def prepare_and_tokenize(example):
    q = example["question"]
    a = example["answer"]
    template = f'''<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
    <|im_start|>user\n
    {q} <|im_end|>
    <|im_start|>assistant\n {a}<|im_end|>'''
    return tokenizer_small(template, truncation=True, padding='max_length', max_length=2048)

# %%
mapped_train_dataset = train_dataset.map(prepare_and_tokenize, batched=False, remove_columns=['question', 'answer'])

# %%
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# %%
trainer = Trainer(
    model=model,
    train_dataset=mapped_train_dataset,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        max_steps=100,
        learning_rate=1e-3,
        fp16=True,
        logging_steps=1,
        output_dir='outputs'
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer_small, mlm=False)
)
trainer.args._n_gpu = 1
model.config.use_cache = False

# %%
torch.cuda.empty_cache()
trainer.train()

# %%
model.save_pretrained("./my_model")
tokenizer_small.save_pretrained("./my_model")

# %%
model.config.use_cache = True
model.eval()

# %% [markdown]
# ## Model Evaluation

# %% [markdown]
# ### Quantitative

# %%
ft_outputs=[query_lm(model, tokenizer_small, q, generator_params) for q in questions]
ft_scores = [similarity_score(extract_sql(out), test_df['answer'].iloc[i]) for i, out in enumerate(ft_outputs)]
print(f"Accuracy with fine-tuned model: {np.sum(ft_scores)/len(ft_scores)}")

# %% [markdown]
# ### Qualitative

# %%
# Print the base model results for the lowest-scoring indices
results = [
    (i, questions[i], outputs[i], scores[i]) for i in range(len(scores))
]

results_sorted = sorted(results, key=lambda x: x[3])  

print("Baseline results for the lowest-scoring indices:")
num_lowest = 3
for idx, q, out, score in results_sorted[:num_lowest]:
    print(f"Index: {idx}\nScore: {score}\nOutput: {extract_sql(out)}\n")

# %%
lowest_indices = [idx for idx, _, _, _ in results_sorted[:num_lowest]]

ft_lowest_results = [
    (idx, questions[idx], ft_outputs[idx], ft_scores[idx]) 
    for idx in lowest_indices
]

# Print the fine-tuned results for the lowest-scoring indices
print("Fine-tuned results for the lowest-scoring indices:")
for idx, q, out, score in ft_lowest_results:
    print(f"Index: {idx}\nScore: {score}\nOutput: {extract_sql(out)}\n")

# %%
# Print the ground truth results for the lowest-scoring indices to compare 
test_df_reindexed = test_df.reset_index(drop=True)
answers = [
    (idx, test_df_reindexed['answer'][idx]) 
    for idx in lowest_indices
]
for idx, answer in answers:
    print(f"Index: {idx}\nAnswer: {answer}")

# %% [markdown]
# The fine-tuned model significantly outperformed the baseline model across all three test cases. The baseline model produced SQL queries that were often structurally incorrect, missed key joins, or introduced unnecessary subqueries. The fine-tuned model generated outputs that were much closer to the ground truth, with improved accuracy in table joins, conditions, and aggregations. The baseline model struggled with generating correct joins, aggregations, and conditions, leading to very low scores. The fine-tuned model showed major improvements in SQL structure and accuracy, achieving 5-8x higher scores. There is still room for fine-tuning refinements, especially in cases requiring complex filtering and grouping logic (e.g., Index 40). Overall, fine-tuning significantly improved text-to-SQL conversion, making the model more effective for real-world database querying tasks.
# 
# Index: 40
# - Baseline Score: 0.093 (Very low)
# Issues: The output used nested subqueries instead of joins and failed to correctly filter based on the average balance.
# - Fine-tuned Score: 0.762 (High improvement)
# Improvements: The fine-tuned model correctly structured the join and filtering conditions but had a minor issue with grouping logic.
# 
# Index: 13
# - Baseline Score: 0.131 (Very low)
# Issues: The baseline model only selected the Gymnast_ID column, missing the necessary join with the people table.
# - Fine-tuned Score: 0.781 (High improvement)
# Improvements: The fine-tuned model correctly joined the people and gymnast tables and retrieved the correct Name column.
# 
# Index: 46
# - Baseline Score: 0.180 (Low)
# Issues: The baseline model misinterpreted table structures and used an unnecessary CASE statement instead of a direct sum operation.
# - Fine-tuned Score: 0.604 (Moderate improvement)
# Improvements: The fine-tuned model correctly summed balances across checking and savings but missed some structural details like explicit joins.

# %% [markdown]
# 


