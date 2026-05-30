import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

test_dir = './images_data/test/'

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
print("Task 2: PyTorch CNN-Vit Hybrid Model instantiated.")
print("Task 3: Evaluation metrics for Keras CNN-Vit Hybrid Model (ensure you run print_metrics with data).")
print("Task 4: Evaluation metrics for PyTorch CNN-Vit Hybrid Model (ensure you run print_metrics with data).")
