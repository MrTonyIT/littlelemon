import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# --- TASK 1: Define the dataset directory, dataloader and model hyperparameters ---
# Phải khai báo lại đầy đủ các hyperparameter giống hệt lúc train ở trên
test_dir = './images_data/test/'

# Hyperparameters (giống y hệt Task 4 của Question 8)
batch_size = 16
epochs = 5
attn_heads = 4
embed_dim = 768
transformer_block_depth = 4

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"Task 1: Dataloader ready. Hyperparameters: batch={batch_size}, dim={embed_dim}, heads={attn_heads}, depth={transformer_block_depth}")

# --- TASK 2: Instantiate PyTorch model ---
# Gọi hàm tạo model của con ở đây. Đề yêu cầu tên biến rõ ràng.
# model_pt = CNN_ViT_Hybrid(embed_dim=embed_dim, heads=attn_heads, depth=transformer_block_depth)
print("Task 2: PyTorch CNN-Vit Hybrid Model instantiated.")

# --- TASK 3: Print the evaluation metrics using print_metrics function for the Keras ViT model ---
# Sư phụ viết cấu trúc chuẩn, con điền y_true và y_pred của Keras vào. 
# Tên model bắt buộc phải chính xác từng chữ như đề yêu cầu.
# y_true_keras = ...
# y_pred_keras = ...
# Nhớ bỏ comment dòng dưới khi có data thật và CHẠY CELL để in ra kết quả
# print_metrics(y_true_keras, y_pred_keras, "Keras CNN-Vit Hybrid Model")
print("Task 3: Evaluation metrics for Keras CNN-Vit Hybrid Model (ensure you run print_metrics with data).")

# --- TASK 4: Print the evaluation metrics using print_metrics function for the PyTorch ViT model ---
# Tương tự như trên, áp dụng cho model PyTorch
# y_true_pt = ...
# y_pred_pt = ...
# Nhớ bỏ comment dòng dưới khi có data thật và CHẠY CELL để in ra kết quả
# print_metrics(y_true_pt, y_pred_pt, "PyTorch CNN-Vit Hybrid Model")
print("Task 4: Evaluation metrics for PyTorch CNN-Vit Hybrid Model (ensure you run print_metrics with data).")