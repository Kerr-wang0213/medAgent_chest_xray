import os  # 导入系统接口。如果不写：无法使用 os.listdir 遍历文件。
import cv2  # 导入OpenCV库。如果不写：无法读取和修改图像像素。
import numpy as np
import pandas as pd
from tqdm import tqdm  # 导入进度条库。如果不写：处理大量图片时不知道进度，用户体验差。
# 从配置文件导入常量。如果不写：路径和种子需要硬编码，难以维护。
from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR, RANDOM_SEED

def generate_sabotaged_dataset():
    """
    Generate a dataset with artificial errors (NaNs and corrupted images).
    """
    print("[Sabotage] Indexing raw data...")
    
   
    splits = ['train', 'test', 'val']
    filepaths = []  # 初始化用于存储路径的列表。如果不写：无法收集数据。
    labels = []   # 初始化用于存储标签的列表。如果不写：无法对应图片的类别。

    # 遍历每一个划分集合（训练/测试/验证）。如果不写：只能读取部分数据。
    for split in splits:
        # 拼接当前划分的路径。如果不写：无法定位文件夹。
        split_dir = RAW_DATA_DIR / split
        
        # 遍历每一个类别（正常/肺炎）。如果不写：无法区分正负样本。
        for label in ['NORMAL', 'PNEUMONIA']:
            # 拼接类别文件夹路径。如果不写：无法进入具体类别的文件夹。
            class_dir = split_dir / label
            
            # 检查文件夹是否存在。如果不写：如果路径不对程序会报错崩溃。
            if class_dir.exists():
                # 列出文件夹下所有文件。如果不写：无法获取文件名。
                files = os.listdir(class_dir)
                
                # 遍历每一个文件。如果不写：无法逐个处理图片。
                for f in files:
                    # 过滤非图片文件。如果不写：可能会读入系统隐藏文件导致报错。
                    if f.lower().endswith(('.jpeg', '.jpg', '.png')):
                        # 将完整路径加入列表。如果不写：丢失文件位置信息。
                        filepaths.append(str(class_dir / f))
                        # 将对应的标签加入列表。如果不写：丢失类别信息。
                        labels.append(label)

    # 创建Pandas DataFrame。如果不写：无法进行高效的数据清洗和索引操作。
    df = pd.DataFrame({'filepath': filepaths, 'label': labels})
    
    # 设置随机种子。如果不写：每次运行生成的坏数据不一样，实验不可复现。
    np.random.seed(RANDOM_SEED)

    # --- Part 1: Simulate Missing Labels (NaN) ---
    # 随机选择50个索引。如果不写：无法确定哪些数据要变成 NaN。
    nan_indices = np.random.choice(df.index, 50, replace=False)
    
    # 将选中的标签设为NaN。如果不写：数据集中没有缺失值，无法演示清洗缺失值的技术。
    df.loc[nan_indices, 'label'] = np.nan
    
    print(f"[Sabotage] Created 50 NaN labels.")  # 打印日志。如果不写：不知道操作是否成功。

    # --- Part 2: Simulate Image Corruption ---
    # 随机选择15个索引用于破坏。如果不写：无法确定哪些图片要被损坏。
    corruption_indices = np.random.choice(df.index, 15, replace=False)
    
    # 前10个用于制造过曝（全白）。如果不写：无法区分破坏类型。
    overexposed_indices = corruption_indices[:10]
    
    # 后5个用于制造欠曝（全黑）。如果不写：同上。
    underexposed_indices = corruption_indices[10:]

    print(f"[Sabotage] Generating corrupted images in {PROCESSED_DATA_DIR}...")  # 打印日志。如果不写：用户不知道正在进行文件IO操作。

    # 定义内部函数用于保存坏图。如果不写：代码会有大量重复，可读性差。
    def save_corrupt_image(idx, mode):
        # 获取原始图片路径。如果不写：不知道要读哪张图。
        original_path = df.loc[idx, 'filepath']
        
        # 读取原始图片。如果不写：无法获取原图尺寸，无法生成同样大小的坏图。
        img = cv2.imread(original_path)
        
        # 如果读取失败（文件损坏）则跳过。如果不写：程序可能会崩溃。
        if img is None: return

        # 根据模式生成全白或全黑图片。如果不写：无法生成指定的坏数据。
        if mode == 'white':
            new_img = np.ones_like(img) * 255  # 全白像素值是255。
            prefix = 'corrupt_white_'
        else:
            new_img = np.zeros_like(img)       # 全黑像素值是0。
            prefix = 'corrupt_black_'

        # 获取原始文件名。如果不写：无法构造新文件名。
        filename = os.path.basename(original_path)
        
        # 构造新文件名。如果不写：文件名冲突或无法识别坏图。
        new_filename = f"{prefix}{filename}"
        
        # 构造完整保存路径。如果不写：不知道文件存哪儿。
        save_path = PROCESSED_DATA_DIR / new_filename
        
        # 将坏图写入硬盘。如果不写：物理文件没有生成，只有逻辑指向，会导致后续读取报错。
        cv2.imwrite(str(save_path), new_img)
        
        # 更新DataFrame中的路径指向新生成的坏图。如果不写：表格里存的还是原图路径，破坏无效。
        df.loc[idx, 'filepath'] = str(save_path)

    # 循环生成全白图片。如果不写：不会生成过曝样本。
    for idx in tqdm(overexposed_indices, desc="Generating White Images"):
        save_corrupt_image(idx, 'white')
        
    # 循环生成全黑图片。如果不写：不会生成欠曝样本。
    for idx in tqdm(underexposed_indices, desc="Generating Black Images"):
        save_corrupt_image(idx, 'black')

    # 返回包含脏数据的DataFrame。如果不写：主程序拿不到处理后的数据表。
    return df