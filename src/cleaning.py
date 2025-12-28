import cv2  # 导入OpenCV。如果不写：无法读取图像进行质量检测。
import numpy as np  # 导入NumPy。如果不写：无法计算平均像素值。
import pandas as pd  # 导入Pandas。如果不写：无法操作DataFrame。
from tqdm import tqdm  # 导入进度条。如果不写：扫描几千张图片时用户会以为死机了。

def check_image_quality(filepath):
    """
    Check if image is too dark or too bright.
    Returns: True (Good), False (Bad)
    """
    try:
        # 以灰度模式读取图片。如果不写：无法进行亮度计算，或者彩色图计算逻辑更复杂。
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        # 检查图片是否读取成功。如果不写：遇到坏路径会报错。
        if img is None:
            return False
        
        # 计算图像的平均像素亮度。如果不写：没有依据判断图片是否异常。
        mean_brightness = np.mean(img)
        
        # 判断是否过曝（大于240视作全白）。如果不写：无法过滤全白坏图。
        if mean_brightness > 240: 
            return False
        # 判断是否欠曝（小于5视作全黑）。如果不写：无法过滤全黑坏图。
        elif mean_brightness < 5: 
            return False
        # 正常范围。如果不写：正常图片也会被当成未定义状态。
        else:
            return True
    except:
        # 捕获任何异常。如果不写：单个文件错误会导致整个程序中断。
        return False

def clean_dataset(df):
    """
    Execute cleaning pipeline: Drop NaN -> Pixel Scan.
    """
    print("\n[Cleaning] Starting cleaning pipeline...")  # 打印日志。如果不写：流程状态不明确。
    
    # --- Step 1: Handle NaN ---
    # 记录初始行数。如果不写：无法计算清洗掉多少数据。
    initial_count = len(df)
    
    # 计算缺失标签的数量。如果不写：报告中无法展示发现了多少错误。
    nan_count = df['label'].isnull().sum()
    print(f"[Check 1] Missing labels detected: {nan_count}")  # 打印发现的错误。如果不写：缺乏信息反馈。
    
    # 删除标签为空的行。如果不写：模型训练时遇到 NaN 标签会直接报错。
    df_step1 = df.dropna(subset=['label']).copy()
    print(f"[Fix 1] Dropped rows with missing labels. Remaining: {len(df_step1)}")  # 打印结果。如果不写：不知道第一步清洗效果。
    
    # --- Step 2: Handle Image Quality ---
    print("[Check 2] Scanning image quality (Pixel-Level)...")  # 打印日志。如果不写：用户不知道正在进行耗时操作。
    
    # 初始化有效掩码列表。如果不写：无法记录哪些图片是好的。
    valid_mask = []
    
    # 计数坏图数量。如果不写：最后无法统计剔除数量。
    bad_count = 0
    
    # 遍历第一次清洗后的所有图片路径。如果不写：无法逐张检查。
    for filepath in tqdm(df_step1['filepath']):
        # 调用质量检查函数。如果不写：无法判断图片好坏。
        is_good = check_image_quality(filepath)
        
        # 将结果存入掩码。如果不写：无法筛选DataFrame。
        valid_mask.append(is_good)
        
        # 统计坏图。如果不写：统计数据缺失。
        if not is_good:
            bad_count += 1
            
    print(f"[Check 2] Corrupted images detected: {bad_count}")  # 打印日志。如果不写：缺乏反馈。
    
    # 应用掩码过滤数据。如果不写：坏图依然存在于数据集中。
    df_final = df_step1[valid_mask].copy()
    
    print(f"[Fix 2] Removed corrupted images.")  # 打印日志。如果不写：不知道操作完成。
    print(f"Cleaning Complete. Data reduced from {initial_count} to {len(df_final)}.")  # 总结。如果不写：不知道最终还能用多少数据。
    
    # 返回清洗后的DataFrame。如果不写：后续步骤拿不到干净数据。
    return df_final