from pathlib import Path  # 导入 Path 用于管理输出文件夹
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

# 导入我们写好的模块
from src.sabotage import generate_sabotaged_dataset
from src.cleaning import clean_dataset
from src.dataset import ChestXrayDataset
from src.config import IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, NORM_MEAN, NORM_STD, RESULTS_DIR
from src.visualization import plot_sample_images, plot_distribution
from src.model import ChestXRayResNet18
from src.train import train_model

def main():
    #  Setup: 准备输出目录
    VIS_DIR = Path(RESULTS_DIR)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Sabotage (数据破坏)
    print("Phase 1: Generating Sabotaged Data")
    dirty_df = generate_sabotaged_dataset()
    print(f"Dirty Dataset Size: {len(dirty_df)}\n")

    # Visualization Step 1
    print("   Visualizing Dirty Data...")
    plot_distribution(
        dirty_df, 
        title="Class Distribution (Dirty)", 
        save_filename=VIS_DIR / "01_distribution_dirty.png"
    )

    corrupted_samples = dirty_df[dirty_df['filepath'].str.contains('corrupt_', na=False)]
    if not corrupted_samples.empty:
        plot_sample_images(
            corrupted_samples,
            title="Sabotaged Images (Simulated Errors)", 
            save_filename=VIS_DIR / "02_samples_corrupted.png"
        )
    print(">> Saved dirty visualizations to folder.\n")


    # Phase 2: Cleaning (数据清洗)
    print("Phase 2: Cleaning Data")
    clean_df = clean_dataset(dirty_df)
    print(f"\nClean Dataset Size: {len(clean_df)}\n")

    # Visualization Step 2
    print(">> Visualizing Clean Data...")
    plot_distribution(
        clean_df, 
        title="Class Distribution (Cleaned)", 
        save_filename=VIS_DIR / "03_distribution_clean.png"
    )
    plot_sample_images(
        clean_df, 
        title="Valid Training Samples", 
        save_filename=VIS_DIR / "04_samples_clean.png"
    )
    print(">> Saved clean visualizations to folder.\n")


    # Phase 3: Loading (加载到 PyTorch)
    print("Phase 3: Loading To Pytorch")
    
    # 定义预处理流水线
    data_transforms = transforms.Compose([
        transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    
    # 实例化 Dataset
    dataset = ChestXrayDataset(clean_df, transform=data_transforms)

    # 打印数据描述
    dataset.preprocessed_data_describe(raw_total=len(dirty_df), batch_size=BATCH_SIZE)


    # [核心修改] Phase 4: Splitting & Training
    print("\n>>> PHASE 4: DATA SPLITTING & TRAINING <<<")
    
    # 1. 确定切分比例 (80% 训练, 10% 验证, 10% 测试)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size  # 剩下的都给测试，防止除不尽
    
    print(f"Total images: {total_size}")
    print(f"Splitting into -> Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # 2. 执行随机切分
    # generator=torch.Generator().manual_seed(42) 保证每次切分结果一样，方便复现实验
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) 
    )
    
    # 3. 把切好的数据塞进 DataLoader
    # num_workers=0 为了兼容性，防止部分系统报错
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    # Test DataLoader 暂时留着备用，或者给 test.py 使用
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False) 

    print("DataLoaders created successfully.")

    # 4. 初始化模型
    print("Initializing Model...")
    model = ChestXRayResNet18(num_classes=2)

    # 5. 开始训练
    trained_model = train_model(model, train_loader, val_loader)

    print("\n>> Training Complete. Back to Main.")
    
    # 如果你想在这里顺便把 test 也跑了，可以解开下面这行的注释，需要先导入 test_model
    # from src.test import test_model
    # test_model(trained_model, test_loader)

if __name__ == "__main__":
    main()