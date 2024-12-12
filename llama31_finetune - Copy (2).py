from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType

# Load the dataset from CSV and select the top 2500 data points
csv_path = "./data/training.csv"
dataset = load_dataset('csv', data_files=csv_path, split='train')
dataset = dataset.select(range(min(len(dataset), 2500)))

# Load the tokenizer
model_name = "meta-llama/Llama-3.2-3B"  # Replace with your local path or Hugging Face repo name
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load the model with 4-bit quantization for QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,       # Enable 4-bit quantization
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="cpu")
model.config.pad_token_id = tokenizer.pad_token_id

# Configure LoRA/QLoRA
lora_config = LoraConfig(
    r=16,                            # Rank of the LoRA matrices
    lora_alpha=32,                   # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to these modules (common for LLaMA)
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM     # For causal language models
)

# Wrap the model with LoRA/QLoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenize the dataset and add labels
def tokenize_function(examples):
    tokens = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    tokens['labels'] = tokens['input_ids'].copy()
    return tokens

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split into train and test datasets
tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = tokenized_datasets['train']
test_dataset = tokenized_datasets['test']

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=4,   # Adjust batch size to fit into memory
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=10,
    gradient_accumulation_steps=4    # For effective batch size
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model("./finetuned_lora_model")

# Evaluate the model
eval_result = trainer.evaluate()
print(eval_result)
