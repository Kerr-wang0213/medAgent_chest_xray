from pathlib import Path  # 导入Path对象。如果不写：无法检查文件是否存在。
import numpy as np  # 导入NumPy。如果不写：无法计算平均像素值。
import pandas as pd  # 导入Pandas。如果不写：无法操作DataFrame。
from tqdm import tqdm  # 导入进度条。如果不写：扫描几千张图片时用户会以为死机了。
from PIL import Image, UnidentifiedImageError # 导入PIL。如果不写：无法读取图像，且无法保证RGB格式兼容。

def check_image_quality(filepath):
    """
    Check if image is too dark or too bright using PIL.
    Returns: True (Good), False (Bad)
    """
    try:
        # 使用 PIL 打开图片。with 语句确保文件使用后自动关闭。
        # 如果不写：可能导致文件句柄泄露，且不像 cv2 那样容易发生颜色错误。
        with Image.open(filepath) as img:
            # 转换为灰度图 ('L' mode)。
            # 如果不写：RGB图片有三个通道，计算平均亮度会比较麻烦。
            gray_img = img.convert('L')
            
            # 将图片转为 numpy 数组并计算平均值。
            # 如果不写：没有量化依据来判断图片是黑是白。
            mean_brightness = np.array(gray_img).mean()
            
            # 判断是否过曝（平均像素 > 240，接近纯白）。
            # 如果不写：无法过滤之前制造的“全白坏图”。
            if mean_brightness > 240: 
                return False
            
            # 判断是否欠曝（平均像素 < 5，接近纯黑）。
            # 如果不写：无法过滤之前制造的“全黑坏图”。
            elif mean_brightness < 5: 
                return False
            
            # 正常范围。
            else:
                return True
                
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        # 捕获文件找不到或图片损坏的异常。
        # 如果不写：遇到坏文件程序会直接崩溃，无法继续清洗后续图片。
        return False

def clean_dataset(df):
    """
    Execute full cleaning pipeline: 
    Duplicates -> NaN -> Broken Paths -> Pixel Scan.
    """
    print("\n[Cleaning] Starting cleaning pipeline...")
    initial_count = len(df)
    
    # --- Step 1: Handle Duplicates (处理重复数据) ---
    # 这是应对之前 "Sabotage" 中制造的重复行。
    # drop_duplicates() 默认保留第一条。
    df_step1 = df.drop_duplicates().copy()
    
    dup_count = initial_count - len(df_step1)
    print(f"[Check 1] Duplicate rows removed: {dup_count}") # 如果不写：不知道删了多少重复项。
    
    # --- Step 2: Handle NaN (处理缺失标签) ---
    nan_count = df_step1['label'].isnull().sum()
    print(f"[Check 2] Missing labels detected: {nan_count}")
    
    # 删除标签为空的行。
    # 如果不写：PyTorch 的 Loss Function 无法处理 NaN 标签，训练必报错。
    df_step2 = df_step1.dropna(subset=['label']).copy()
    print(f"[Fix 2] Dropped rows with missing labels. Remaining: {len(df_step2)}")
    
    # --- Step 3 & 4: Path Validation & Image Quality (路径与质量检查) ---
    print("[Check 3 & 4] Scanning file paths and image quality...") 
    
    valid_mask = []
    broken_path_count = 0
    bad_image_count = 0
    
    # 遍历当前 DataFrame。
    # tqdm 用于显示进度条。如果不写：用户不知道还要等多久。
    for filepath in tqdm(df_step2['filepath']):
        path_obj = Path(filepath)
        
        # [Step 3] 检查物理文件是否存在 (Path.exists)。
        # 这是应对之前 "Sabotage" 中制造的 "broken/path/to/nowhere"。
        if not path_obj.exists():
            valid_mask.append(False)
            broken_path_count += 1
            continue # 跳过后续检查，直接看下一张
        
        # [Step 4] 检查图片内容质量 (Pixel Check)。
        is_good = check_image_quality(filepath)
        
        valid_mask.append(is_good)
        
        if not is_good:
            bad_image_count += 1
            
    print(f"[Check 3] Broken file paths detected: {broken_path_count}") # 如果不写：不知道有多少路径是错的。
    print(f"[Check 4] Corrupted images (black/white) detected: {bad_image_count}") # 如果不写：不知道有多少图片是坏的。
    
    # 应用掩码过滤数据。
    # 如果不写：坏数据依然保留，会导致模型学习到错误的特征。
    df_final = df_step2[valid_mask].copy()
    
    # 重置索引。
    # 如果不写：索引会断断续续（如 0, 1, 5, 8...），可能导致 Dataset __getitem__ 报错。
    df_final = df_final.reset_index(drop=True)
    
    print(f"[Fix 3 & 4] Removed invalid files.")
    print(f"Cleaning Complete. Data reduced from {initial_count} to {len(df_final)}.") 
    
    return df_final