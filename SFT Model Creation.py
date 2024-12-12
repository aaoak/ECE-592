from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Load the dataset from CSV
csv_path = "./data/training.csv"
dataset = load_dataset('csv', data_files=csv_path, split='train')
# Select the top 2500 data points
dataset = dataset.select(range(min(len(dataset), 10)))


# Load the tokenizer and model
model_name = "bigcode/starcoder2-3b"  # Replace with your local path or Hugging Face repo name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set pad_token to eos_token (LLaMA models don't have a pad_token by default)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

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
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=10
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

# Fine-tune the model
trainer.train()

# Save the trained model
trainer.save_model("./finetuned_model")

# Evaluate the model
eval_result = trainer.evaluate()
print(eval_result)
