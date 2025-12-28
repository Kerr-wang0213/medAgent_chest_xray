from pathlib import Path  # 导入 Path 用于管理输出文件夹
from torchvision import transforms
from torch.utils.data import DataLoader

# 导入我们写好的模块
from src.sabotage import generate_sabotaged_dataset
from src.cleaning import clean_dataset
from src.dataset import ChestXrayDataset
from src.config import IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, NORM_MEAN, NORM_STD
from src.visualization import plot_sample_images, plot_distribution

def main():
    #  Setup: 准备输出目录
    # 定义保存可视化图片的文件夹。如果不写：图片会散落在项目根目录，很乱。
    VIS_DIR = Path("visualizations_output")
    VIS_DIR.mkdir(parents=True, exist_ok=True) # 如果文件夹不存在就创建

    # Phase 1: Sabotage
    print("Phase 1: Generating Sabotaged Data")
    dirty_df = generate_sabotaged_dataset()
    print(f"Dirty Dataset Size: {len(dirty_df)}\n")

    # Visualization Step 1: 
    print("   Visualizing Dirty Data...")
    
    # 1. 画脏数据的分布图 (可能包含不平衡)
    plot_distribution(
        dirty_df, 
        title="Class Distribution (Dirty)", 
        save_filename=VIS_DIR / "01_distribution_dirty.png"
    )

    # 2. 专门把我们要展示的“坏图”挑出来画一下
    # 逻辑：文件名里包含 'corrupt_' 的就是我们在 sabotage 阶段生成的坏图
    corrupted_samples = dirty_df[dirty_df['filepath'].str.contains('corrupt_', na=False)]
    
    if not corrupted_samples.empty:        # 只有当确实存在坏图时才画，防止报错
        plot_sample_images(
            corrupted_samples, 
            title="Sabotaged Images (Simulated Errors)", 
            save_filename=VIS_DIR / "02_samples_corrupted.png"
        )
    print(">> Saved dirty visualizations to folder.\n")

    # Phase 2: Cleaning
    print("Phase 2: Cleaning Data")
    clean_df = clean_dataset(dirty_df)
    print(f"\nClean Dataset Size: {len(clean_df)}\n")

    # Visualization Step 2:
    print(">> Visualizing Clean Data...")

    # 1. 画清洗后的分布图 (确认数据量变化)
    plot_distribution(
        clean_df, 
        title="Class Distribution (Cleaned)", 
        save_filename=VIS_DIR / "03_distribution_clean.png"
    )
    # 2. 随机画几张正常的图
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
        # 1. 统一尺寸
        transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
        
        # 2. 转为 Tensor (0-255 -> 0.0-1.0)
        transforms.ToTensor(),
        
        # 3. 归一化 (关键步骤！)
        # 将数据分布拉动到以 0 为中心，标准差为 1 的正态分布形态
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    
    # 实例化 Dataset
    dataset = ChestXrayDataset(clean_df, transform=data_transforms)

    # raw_total 参数传 len(dirty_df)，这是你在 Phase 1 生成的原始数据量
    # batch_size 参数传 BATCH_SIZE，来自 config
    dataset.preprocessed_data_describe(raw_total=len(dirty_df), batch_size=BATCH_SIZE)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 测试读取一个 Batch
    try:
        images, labels = next(iter(dataloader))
        print(f"Success! Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print("The pipeline is rock solid.")
    except Exception as e:
        print(f"Pipeline Failed! Error: {e}")

if __name__ == "__main__":
    main()