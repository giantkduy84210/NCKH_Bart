import os
import numpy as np
import torch
import torch.nn as nn

from datasets import Dataset
from huggingface_hub import snapshot_download
from transformers import (
    BartModel,
    BartForConditionalGeneration,
    BartTokenizer,
    BartConfig,
    TrainingArguments,
    Trainer,
    GenerationConfig
)

from transformers.models.bart.modeling_bart import BartEncoder
# Load data from pose and text files

print("\n--- Loading Data From Files---")

def load_data_from_files(pose_path, text_path, num_samples=1000):
    data = []
    with open(pose_path, "r") as pose_file, open(text_path, "r") as text_file:
        pose_lines = pose_file.readlines()
        text_lines = text_file.readlines()

        assert len(pose_lines) == len(text_lines), "Mismatch in pose and text lines!"

        num_samples = min(num_samples, len(pose_lines))
        for i in range(num_samples):
            pose_array = np.array([float(x) for x in pose_lines[i].strip().split()])
            pose_array = pose_array.reshape(-1, 151)[:, :150]  # Only use 150 features
            text = text_lines[i].strip()
            data.append({"pose": pose_array, "text": text})

    return Dataset.from_list(data)

# Download dataset from Huggingface
data_folder = snapshot_download("ura-hcmut/how2sign", repo_type="dataset")

TRAIN_POSE = os.path.join(data_folder, "train.skels")
TRAIN_TEXT = os.path.join(data_folder, "train.text")
TEST_POSE = os.path.join(data_folder, "test.skels")
TEST_TEXT = os.path.join(data_folder, "test.text")

train_data = load_data_from_files(TRAIN_POSE, TRAIN_TEXT, num_samples=80)
test_data = load_data_from_files(TEST_POSE, TEST_TEXT, num_samples=20)

print("Sample input pose shape:", np.array(train_data[0]['pose']).shape)
print("Sample target text:", train_data[0]['text'])

class PoseBartEncoder(BartEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.pose_embedding = nn.Linear(150, config.d_model, bias=False)

    def forward(self, input_tensors=None, **kwargs):
        assert input_tensors is not None, "input_tensors not found in kwargs"

        inputs_embeds = self.pose_embedding(input_tensors.to(self.pose_embedding.weight.device))
        
        kwargs.pop("inputs_embeds", None)
        
        return super().forward(inputs_embeds=inputs_embeds, **kwargs)

class PoseBartModel(BartModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = PoseBartEncoder(config)

# Define PoseBART model
class PoseBART(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = PoseBartModel(config)

        # Lớp embedding riêng cho pose keypoints (150 -> d_model)
        self.pose_embedding = nn.Linear(150, config.d_model, bias=False)

        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_tensors=None, labels=None, attention_mask=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        assert input_tensors is not None, "Please pass pose matrix to `input_tensors`."
        result = super().forward(input_tensors=input_tensors, labels=labels, attention_mask=attention_mask, **kwargs)
        return result

# BART Configuration
config = BartConfig(
    encoder_layers=6,
    encoder_ffn_dim=4096,
    encoder_attention_heads=16,
    decoder_layers=6,
    decoder_ffn_dim=4096,
    decoder_attention_heads=16,
    d_model=1024
)

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PoseBART(config).to(device)

# Preprocessing function
def preprocess_function(examples):
    MAX_LENGTH_INPUT = 256
    MAX_LENGTH_OUTPUT = 32

    batch_size = len(examples["pose"])
    pose_tensors = torch.zeros(batch_size, MAX_LENGTH_INPUT, 150, dtype=torch.float32)
    attention_mask = torch.zeros(batch_size, MAX_LENGTH_INPUT, dtype=torch.long)
    
    for i, pose in enumerate(examples["pose"]):
        curr_len = min(len(pose), MAX_LENGTH_INPUT)
        pose_tensors[i, -curr_len:, :] = torch.tensor(pose[-curr_len:])
        attention_mask[i, -curr_len:] = 1  # Mark real frames
        
    targets = tokenizer(
        examples["text"],
        max_length=MAX_LENGTH_OUTPUT,
        truncation=True,
        padding='max_length',
        padding_side='right'
    )

    return {"input_tensors": pose_tensors, "attention_mask": attention_mask, "labels": targets["input_ids"]}

print("\n--- Preprocessing Data ---")
tokenized_train = train_data.map(preprocess_function, batched=True, remove_columns=["pose", "text"])
tokenized_test  = test_data.map(preprocess_function, batched=True, remove_columns=["pose", "text"])

training_args = TrainingArguments(
    output_dir="PoseBART_v1",
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=5, ##################################
    # weight_decay=0.01, # Usually used when data is small
    save_total_limit=2,
    logging_steps=1,
    logging_dir="logs",
    report_to='none'
)

def custom_collate_fn(batch):
    input_tensors = torch.stack([torch.tensor(item["input_tensors"], dtype=torch.float32) for item in batch])
    attention_masks = torch.stack([torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch])
    labels = torch.stack([torch.tensor(item["labels"], dtype=torch.long) for item in batch])
    return {"input_tensors": input_tensors, "attention_mask": attention_masks, "labels": labels}

# Train model
model.train()
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=custom_collate_fn,
)

    
print("\n--- Training ---")
trainer.train()


# Inference
model.eval()
generation_config = GenerationConfig(
	max_new_tokens = 128,
	stop_strings = ["</s>", "<pad>"],
)
def predict_pose_to_text(pose_array, attention_mask):
        pose_tensor = torch.tensor(pose_array, dtype=torch.float32).unsqueeze(0).to(device)
        #print("Pose tensor shape:", pose_tensor.shape)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
                generated_ids = model.generate(
                        input_tensors=pose_tensor,
                        attention_mask=attention_mask,
                        tokenizer=tokenizer,
                        generation_config=generation_config
                )

        #print("Generated token IDs:", generated_ids)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text if generated_text.strip() else "[EMPTY OUTPUT]"

# Test inference on a few examples
print("\n--- Testing Inference ---")
TEST_CASE_NUM = 10
for i in range(TEST_CASE_NUM):
        output_text = predict_pose_to_text(tokenized_train[i]["input_tensors"], tokenized_train[i]["attention_mask"])
        print(f"[Test case {i+1}]")
        print("Generated text:", output_text)
        print("Groundtruth:", train_data[i*7]["text"])
        print("\n")
