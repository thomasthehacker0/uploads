from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenized_dataset = tokenizer(dataset['input'], padding=True, truncation=True)
from torch.utils.data import Dataset
class ConversationDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)
from transformers import TextGenerationPipeline, GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained('gpt2')
pipeline = TextGenerationPipeline(model=model)
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16, # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                 # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset            # evaluation dataset
)

trainer.train()
