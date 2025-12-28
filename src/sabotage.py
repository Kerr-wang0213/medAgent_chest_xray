from pathlib import Path # 确保导入了 Path，虽然下面可能通过 config 导入了对象，显式导入更安全
import cv2  # 导入OpenCV库。如果不写：无法读取和修改图像像素。
import numpy as np  # 导入NumPy库。如果不写：无法进行矩阵运算和随机数生成。
import pandas as pd  # 导入Pandas库。如果不写：无法创建和操作数据表格。
from tqdm import tqdm  # 导入进度条库。如果不写：处理大量图片时不知道进度，用户体验差。
# 从配置文件导入常量。如果不写：路径和种子需要硬编码，难以维护。
from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR, RANDOM_SEED

def generate_sabotaged_dataset():
    """
    Generate a dataset with artificial errors (NaNs and corrupted images).
    """
    print("[Sabotage] Indexing raw data...")  # 打印状态。如果不写：用户不知道程序卡住了还是在运行。

    splits = ['train', 'test', 'val']     # 定义数据集划分列表。如果不写：循环无法进行。
    filepaths = []  # 初始化用于存储路径的列表。如果不写：无法收集数据。
    labels = []   # 初始化用于存储标签的列表。如果不写：无法对应图片的类别。


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


    df = pd.DataFrame({'filepath': filepaths, 'label': labels})
    np.random.seed(RANDOM_SEED)

    # --- Part 1: Simulate Missing Labels (NaN) ---
    nan_indices = np.random.choice(df.index, 50, replace=False)
    df.loc[nan_indices, 'label'] = np.nan
    print(f"[Sabotage] Created 50 NaN labels.")

    # --- Part 2: Simulate Image Corruption --- 随机选择15个索引用于破坏
    corruption_indices = np.random.choice(df.index, 15, replace=False)
    overexposed_indices = corruption_indices[:10]   # 前10个用于制造过曝（全白）
    underexposed_indices = corruption_indices[10:]  # 后5个用于制造欠曝（全黑）

    print(f"[Sabotage] Generating corrupted images in {PROCESSED_DATA_DIR}...")  # 打印日志。如果不写：用户不知道正在进行文件IO操作。


    def save_corrupt_image(idx, mode):       # 定义内部函数用于保存坏图。如果不写：代码会有大量重复，可读性差。
        original_path = df.loc[idx, 'filepath']
        # 这里的 original_path 是字符串，为了用 Path 的功能，我们把它包一下
        # 虽然 cv2.imread 接受字符串，但我们需要 Path 来处理文件名
        p_original = Path(original_path) 
        
        img = cv2.imread(original_path)   # 读取原始图片，OpenCV 读取还是用字符串路径最稳。如果不写：无法获取原图尺寸，无法生成同样大小的坏图。
        
        if img is None: return    # 如果读取失败（文件损坏）则跳过。如果不写：程序可能会崩溃。

        if mode == 'white':       # 根据模式生成全白或全黑图片。如果不写：无法生成指定的坏数据。
            new_img = np.ones_like(img) * 255  # 全白像素值是255。
            prefix = 'corrupt_white_'
        else:
            new_img = np.zeros_like(img)       # 全黑像素值是0。
            prefix = 'corrupt_black_'
        
        filename = p_original.name    # 获取原始文件名。如果不写：无法构造新文件名。

        new_filename = f"{prefix}{filename}"      # 构造新文件名。如果不写：文件名冲突或无法识别坏图。
        save_path = PROCESSED_DATA_DIR / new_filename
        
        cv2.imwrite(str(save_path), new_img)     # 将坏图写入硬盘。如果不写：物理文件没有生成，只有逻辑指向，会导致后续读取报错。

        df.loc[idx, 'filepath'] = str(save_path)       # 更新DataFrame中的路径指向新生成的坏图。如果不写：表格里存的还是原图路径，破坏无效。

    # 循环生成全白图片。如果不写：不会生成过曝样本。
    for idx in tqdm(overexposed_indices, desc="Generating White Images"):
        save_corrupt_image(idx, 'white')
        
    # 循环生成全黑图片。如果不写：不会生成欠曝样本。
    for idx in tqdm(underexposed_indices, desc="Generating Black Images"):
        save_corrupt_image(idx, 'black')


    return df    # 返回包含脏数据的DataFrame。如果不写：主程序拿不到处理后的数据表。