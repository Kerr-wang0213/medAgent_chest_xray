import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# 导入我们写好的模块
from src.sabotage import generate_sabotaged_dataset
from src.cleaning import clean_dataset
from src.dataset import ChestXrayDataset
from src.config import IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE
from src.visualization import plot_sample_images, plot_distribution

def main():
    # Phase 1: Sabotage
    print("Phase 1: Generting Sabotaged Data")
    dirty_df = generate_sabotaged_dataset()         # 生成包含 NaN、死链、黑白坏图、重复行的 DataFrame
    print(f"Dirty Dataset Size: {len(dirty_df)}\n")

    # Phase 2: Cleaning
    print("Phase 2: Cleaning Data")
    clean_df = clean_dataset(dirty_df)
    print(f"\nClean Dataset Size: {len(clean_df)}\n")

    # Phase 3: Loading
    print("Phase 3: Loading To Pytorch")
    data_transforms = transforms.Compose([                  # 定义预处理：调整大小 -> 转张量。这里的 IMG_WIDTH, IMG_HEIGHT 来自 config.py (150)
        transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
        transforms.ToTensor(),
    ])
    dataset = ChestXrayDataset(clean_df, transform=data_transforms)      # 实例化 Dataset。注意：我们传进去的是 clean_df
    
    # 实例化 DataLoader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 测试读取一个 Batch
    try:
        images, labels = next(iter(dataloader))
        print(f"Success! Batch shape: {images.shape}") # 预期: [32, 3, 150, 150]
        print(f"Labels shape: {labels.shape}")         # 预期: [32]
        print("The pipeline is rock solid.")
    except Exception as e:
        print(f"Pipeline Failed! Error: {e}")

if __name__ == "__main__":
    main()