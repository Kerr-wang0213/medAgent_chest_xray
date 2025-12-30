import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR, RANDOM_SEED

def generate_sabotaged_dataset():
    """
    Generate a dataset with artificial errors (NaNs and corrupted images).
    1. Duplicate Rows
    2. NaN Labels
    3. Broken File Paths
    4. Corrupted Images (White/Black)
    """  
    processed_dir = Path(PROCESSED_DATA_DIR)
    if processed_dir.exists():
        shutil.rmtree(processed_dir)   # Delete folder and its contents
    
    processed_dir.mkdir(parents=True, exist_ok=True)
   
    splits = ['train', 'test', 'val']
    filepaths = []  # List storage paths
    labels = []   # List storage labels
    dataset_splits = []   # Record data belonging to: training set, test set, or validation set


    for split in splits:
        split_dir = RAW_DATA_DIR / split
        
        for label in ['NORMAL', 'PNEUMONIA']:
            class_dir = split_dir / label
            
            if class_dir.exists():
                for f_path in class_dir.iterdir():
                    if f_path.suffix.lower() in ['.jpeg', '.jpg', '.png']:
                        filepaths.append(str(f_path))
                        labels.append(label)
                        dataset_splits.append(split)


    df = pd.DataFrame({'filepath': filepaths, 'label': labels, 'dataset_split': dataset_splits})

    rng = np.random.RandomState(RANDOM_SEED)

    # 1: Duplicate Rows
    duplicates = df.sample(n=30, random_state=rng)
    
    df = pd.concat([df, duplicates], axis=0, ignore_index=True)
    print(f"[SABOTAGED] Added 30 duplicate rows. Final count: {len(df)}")

    # 2: Simulate Missing Labels (NaN)
    nan_indices = rng.choice(df.index, 50, replace=False)
    df.loc[nan_indices, 'label'] = np.nan
    print(f"[SABOTAGED] Created 50 NaN labels.")

    # 3: Broken Paths
    broken_indices = rng.choice(df.index, 20, replace=False)
    df.loc[broken_indices, 'filepath'] = "broken/path/to/nowhere/error_image.jpg"
    print(f"[SABOTAGED] Created 20 broken file paths.")

    # 4: Simulate Image Corruption
    processed_dir = Path(PROCESSED_DATA_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)

    corruption_indices = rng.choice(df.index, 15, replace=False)
    overexposed_indices = corruption_indices[:10]
    underexposed_indices = corruption_indices[10:]

    print(f"[SABOTAGED] Generating corrupted images in {PROCESSED_DATA_DIR}:")


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

    for idx in tqdm(overexposed_indices, desc="Generating White Images"):
        save_corrupt_image(idx, 'white')
        
    for idx in tqdm(underexposed_indices, desc="Generating Black Images"):
        save_corrupt_image(idx, 'black')


    return df