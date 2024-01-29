from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


with open('your_filed.txt', 'r') as file:
    contentss = file.read()

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Tokenize the text
inputs = tokenizer(contentss, return_tensors='pt', truncation=True, max_length=512)

# Create a dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="your_filed.txt",
    block_size=128
)

# Create a data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# Load the model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
)

# Create the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Train the model
trainer.train()
