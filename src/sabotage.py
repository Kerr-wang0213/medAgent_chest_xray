import shutil
from pathlib import Path # 确保导入了 Path，虽然下面可能通过 config 导入了对象，显式导入更安全
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm  # 导入进度条库。如果不写：处理大量图片时不知道进度，用户体验差。
# 从配置文件导入常量。如果不写：路径和种子需要硬编码，难以维护。
from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR, RANDOM_SEED

def generate_sabotaged_dataset():
    """
    Generate a dataset with artificial errors (NaNs and corrupted images).
    1. Duplicate Rows
    2. NaN Labels
    3. Broken File Paths
    4. Corrupted Images (White/Black)
    """
    print("[Sabotage] Indexing raw data...")
    
    processed_dir = Path(PROCESSED_DATA_DIR)
    if processed_dir.exists():
        shutil.rmtree(processed_dir)   # 递归删除文件夹及其内容
    
    # 重新创建空文件夹
    processed_dir.mkdir(parents=True, exist_ok=True)
   
    splits = ['train', 'test', 'val']
    filepaths = []  # 初始化用于存储路径的列表。如果不写：无法收集数据。
    labels = []   # 初始化用于存储标签的列表。如果不写：无法对应图片的类别。
    dataset_splits = []   #用来记录这一行数据属于 train, test 还是 val


    for split in splits:     # 遍历每一个划分集合（训练/测试/验证）。如果不写：只能读取部分数据。
        split_dir = RAW_DATA_DIR / split   # 拼接当前划分的路径。如果不写：无法定位文件夹。
        
        for label in ['NORMAL', 'PNEUMONIA']:    # 遍历每一个类别（正常/肺炎）。如果不写：无法区分正负样本。
            class_dir = split_dir / label    # 拼接类别文件夹路径。如果不写：无法进入具体类别的文件夹。
            
            if class_dir.exists():      # 检查文件夹是否存在。如果不写：如果路径不对程序会报错崩溃。
                # iterdir() 直接产生 Path 对象，比 os.listdir 只是给个文件名要好用
                for f_path in class_dir.iterdir():
                    if f_path.suffix.lower() in ['.jpeg', '.jpg', '.png']:
                        filepaths.append(str(f_path)) # 将对应的标签加入列表, Path对象转str存入列表，如果不写：丢失类别信息。
                        labels.append(label)  # 将对应的标签加入列表。如果不写：丢失类别信息。
                        dataset_splits.append(split)    #记录当前循环是在哪个 split 里


    df = pd.DataFrame({'filepath': filepaths, 'label': labels, 'dataset_split': dataset_splits})

    rng = np.random.RandomState(RANDOM_SEED)

    # --- part 1: 模拟重复数据 (Duplicate Rows) ---
    duplicates = df.sample(n=30, random_state=rng)
    
    # 将重复行拼接到原 DataFrame 底部。
    # ignore_index=True 会重置索引，模拟“由于多次录入导致的重复记录”。
    df = pd.concat([df, duplicates], axis=0, ignore_index=True)
    print(f"[Sabotage] Added 30 duplicate rows. Final count: {len(df)}")

    # --- Part 2: Simulate Missing Labels (NaN) ---
    nan_indices = rng.choice(df.index, 50, replace=False)
    df.loc[nan_indices, 'label'] = np.nan
    print(f"[Sabotage] Created 50 NaN labels.")

    # --- Part 3: Broken Paths ---
    broken_indices = rng.choice(df.index, 20, replace=False)
    df.loc[broken_indices, 'filepath'] = "broken/path/to/nowhere/error_image.jpg"
    print(f"[Sabotage] Created 20 broken file paths.")

    # --- Part 4: Simulate Image Corruption --- 随机选择15个索引用于破坏
    processed_dir = Path(PROCESSED_DATA_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)

    corruption_indices = rng.choice(df.index, 15, replace=False)
    overexposed_indices = corruption_indices[:10]   # 前10个用于制造过曝（全白）
    underexposed_indices = corruption_indices[10:]  # 后5个用于制造欠曝（全黑）

    print(f"[Sabotage] Generating corrupted images in {PROCESSED_DATA_DIR}...")  # 打印日志。如果不写：用户不知道正在进行文件IO操作。


    def save_corrupt_image(idx, mode):
        original_path_str = df.loc[idx, 'filepath']
        
        if "broken/path" in original_path_str:
            return

        original_path = Path(original_path_str)
        
        try:
            with Image.open(original_path) as img:
                size = img.size
            
            if mode == 'white':
                new_img = Image.new('RGB', size, (255, 255, 255))
                prefix = 'corrupt_white_'
            else:
                new_img = Image.new('RGB', size, (0, 0, 0))
                prefix = 'corrupt_black_'
            
            new_filename = f"{prefix}{original_path.name}"
            save_path = processed_dir / new_filename
            
            new_img.save(save_path)
            
            df.loc[idx, 'filepath'] = str(save_path)
            
        except Exception as e:
            print(f"Error processing {original_path}: {e}")

    # 循环生成全白图片。如果不写：不会生成过曝样本。
    for idx in tqdm(overexposed_indices, desc="Generating White Images"):
        save_corrupt_image(idx, 'white')
        
    # 循环生成全黑图片。如果不写：不会生成欠曝样本。
    for idx in tqdm(underexposed_indices, desc="Generating Black Images"):
        save_corrupt_image(idx, 'black')


    return df    # 返回包含脏数据的DataFrame。如果不写：主程序拿不到处理后的数据表。