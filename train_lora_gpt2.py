#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import torch

# Load a sample dataset for fine-tuning (you can replace this with your own)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Load GPT-2 and the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Add a padding token (GPT-2 does not have a pad token by default)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("gpt2")

# Tokenize the dataset and add labels
def tokenize_function(examples):
    tokenized_output = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    # Set 'labels' to be the same as 'input_ids'
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()
    return tokenized_output

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Prepare for LoRa fine-tuning
lora_config = LoraConfig(
    r=8,  # Low-rank dimension
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"],  # GPT-2's attention layers to target for LoRa
)

# Apply LoRa to the GPT-2 model
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()  # Optional: to see which parameters are trainable

# Define a custom Trainer class with a compute_loss method
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").clone()  # Clone the labels
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Shift the logits and labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute the loss using cross-entropy
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize the custom trainer
trainer = CustomTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned LoRa model
peft_model.save_pretrained("./models/")
