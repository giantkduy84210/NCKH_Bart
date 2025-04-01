import json
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import BartForConditionalGeneration, TrainingArguments, Trainer
from torch.utils.data import DataLoader

# ============================
# 1. Tải dữ liệu (T, 151)
# ============================
def load_data_from_files(pose_path, text_path, num_samples=1000):
    data = []
    with open(pose_path, "r") as pose_file, open(text_path, "r") as text_file:
        pose_lines = pose_file.readlines()
        text_lines = text_file.readlines()
        
        num_samples = min(num_samples, len(pose_lines), len(text_lines))
        for i in range(num_samples):
            pose_array = np.array([float(x) for x in pose_lines[i].strip().split()])
            pose_array = pose_array.reshape(-1, 151)  # Chuyển thành (T, 151)
            text = text_lines[i].strip()
            data.append({"pose": pose_array, "text": text})
    return data

mode_file = "dev"
pose_path = "./DATASET/" + mode_file + ".skels"
text_path = "./DATASET/" + mode_file + ".text"
data = load_data_from_files(pose_path, text_path, num_samples=10)

dataset = Dataset.from_list(data)
split_dataset = dataset.train_test_split(test_size=0.2)

# ============================
# 2. Định nghĩa mô hình BART với Embedding mới
# ============================
class CustomBART(nn.Module):
    def __init__(self, bart_model_name="facebook/bart-base"):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(bart_model_name)
        
        # Thay thế embedding gốc (50265, 768) bằng nn.Linear(151, 768)
        self.custom_embedding = nn.Linear(151, 768)

    def forward(self, pose_inputs, labels):
        """
        pose_inputs: Tensor có shape (batch_size, T, 151)
        labels: input_ids của văn bản (batch_size, seq_length)
        """
        embedded_inputs = self.custom_embedding(pose_inputs)  # (batch_size, T, 768)

        outputs = self.bart(
            inputs_embeds=embedded_inputs, 
            labels=labels
        )
        return outputs

# Khởi tạo model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomBART().to(device)

# ============================
# 3. Tiền xử lý dữ liệu
# ============================
from transformers import BartTokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

def preprocess_function(examples):
    MAX_LENGTH_INPUT = 200  # Giới hạn số frame (T)
    MAX_LENGTH_OUTPUT = 128

    # Lấy dữ liệu pose
    pose_matrices = [torch.tensor(pose[:MAX_LENGTH_INPUT], dtype=torch.float32) for pose in examples["pose"]]

    # Pad/crop pose matrices để đảm bảo đúng kích thước (MAX_LENGTH_INPUT, 151)
    pose_tensors = torch.zeros(len(pose_matrices), MAX_LENGTH_INPUT, 151)
    for i, pose in enumerate(pose_matrices):
        pose_tensors[i, :pose.shape[0], :] = pose

    # Tokenize văn bản đầu ra
    targets = tokenizer(examples["text"], max_length=MAX_LENGTH_OUTPUT, truncation=True, padding="max_length")
    print(targets["input_ids"])
    return {"pose": pose_tensors, "labels": targets["input_ids"]}

# tokenized_train = split_dataset["train"].map(preprocess_function, batched=True, remove_columns=["pose", "text"])
# tokenized_test = split_dataset["test"].map(preprocess_function, batched=True, remove_columns=["pose", "text"])
tokenized_train = split_dataset["train"].map(preprocess_function, batched=True)
tokenized_test = split_dataset["test"].map(preprocess_function, batched=True)

"""
print(tokenized_train[0].keys())  # Kiểm tra dữ liệu đã được xử lý
pose_tensor = torch.tensor(tokenized_train[0]["pose"])  # Chuyển đổi pose thành tensor
print(pose_tensor.shape)  # Kiểm tra kích thước pose tensor
print(f"Pose Tensor: {pose_tensor}")  # In ra tensor
print(tokenized_train[0]["labels"])  # Kiểm tra labels

import sys
sys.exit(0)  # Dừng chương trình sau khi kiểm tra dữ liệu
"""

# ============================
# 4. Cấu hình Trainer
# ============================
def collate_fn(batch):
    for idx, item in enumerate(batch):
        print(f"Item {idx} keys: {item.keys()}")
    poses = torch.stack([item["pose"] for item in batch]).to(device)
    labels = torch.tensor([item["labels"] for item in batch]).to(device)
    return {"pose_inputs": poses, "labels": labels}

training_args = TrainingArguments(
    output_dir="./bart_pose2text",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_steps=10,
    logging_dir="./logs",
)

# Custom Trainer để phù hợp với input mới
class CustomTrainer(Trainer):
    def __init__(self, model, args, train_dataset, eval_dataset, data_collator=None):
        # Chuyển tiếp các tham số cho lớp cha (Trainer)
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,  # Đảm bảo tham số data_collator được truyền vào
        )
    """"
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Tính loss giống như trong phần trước
        outputs = model(inputs["pose_inputs"], inputs["labels"])
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss"
    """

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=collate_fn,
)

# ============================
# 5. Huấn luyện mô hình
# ============================
print("Starting training...")
trainer.train()
print("Training completed.")
print("=============================")

# ============================
# 6. Suy luận
# ============================
def predict_pose_to_text(pose_array):
    pose_array = pose_array.reshape(-1, 151)  # Đảm bảo đúng shape (T, 151)
    input_tensor = torch.tensor(pose_array, dtype=torch.float32).unsqueeze(0).to(device)  # Thêm batch dimension
    
    # Chạy forward
    outputs = model(pose_inputs=input_tensor, labels=None)
    
    # Lấy ID từ đầu ra
    output_ids = outputs.logits.argmax(dim=-1)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Test với một mẫu
print("Testing prediction...")
test_pose = np.array(json.loads(split_dataset["test"][1]["pose"]))
generated_text = predict_pose_to_text(test_pose)
example_text = split_dataset["test"][1]["text"]
print("Original Text:", example_text)
print("Generated Text:", generated_text)