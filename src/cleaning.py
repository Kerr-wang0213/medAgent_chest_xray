from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError

def check_image_quality(filepath):
    """
    Check if image is too dark or too bright.
    Returns: True (Good), False (Bad)
    """
    try:
        with Image.open(filepath) as img:
            gray_img = img.convert('L')
            
            mean_brightness = np.array(gray_img).mean()

            if mean_brightness > 240: 
                return False
            
            elif mean_brightness < 5: 
                return False

            else:
                return True
                
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return False

def clean_dataset(df):
    """
    Execute full cleaning pipeline: 
    Duplicates -> NaN -> Broken Paths -> Pixel Scan.
    """
    print("\n[CLEANING] Starting cleaning pipeline...")
    initial_count = len(df)
    
    # --- 1: Handle Duplicates --
    df_step1 = df.drop_duplicates().copy()
    dup_count = initial_count - len(df_step1)
    print(f"[Fix 1] Duplicate rows removed: {dup_count}")
    
    # --- 2: Handle NaN  ---
    nan_count = df_step1['label'].isnull().sum()
    df_step2 = df_step1.dropna(subset=['label']).copy()
    print(f"[Fix 2] Missing-label rows removed: {nan_count}")
    
    # --- 3 & 4: Path Validation & Image Quality ---  
    valid_mask = []
    broken_path_count = 0
    bad_image_count = 0
    
    for filepath in tqdm(df_step2['filepath']):
        path_obj = Path(filepath)
        
        # 3: Path.exists
        if not path_obj.exists():
            valid_mask.append(False)
            broken_path_count += 1
            continue
        
        # 4: Pixel Check
        is_good = check_image_quality(filepath)
        
        valid_mask.append(is_good)
        
        if not is_good:
            bad_image_count += 1
            
    print(f"[Fix 3] Broken-path files removed: {broken_path_count}")
    print(f"[Fix 4] Corrupted images (black/white) removed: {bad_image_count}")
    
    df_final = df_step2[valid_mask].copy()
    
    df_final = df_final.reset_index(drop=True)

    print(f"Cleaning Complete. Data reduced from {initial_count} to {len(df_final)}.") 
    
    return df_final