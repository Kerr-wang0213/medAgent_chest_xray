import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class ChestXrayDataset(Dataset):
    """
    Standard PyTorch Dataset.
    Assumes the input DataFrame is already CLEANED.
    """
    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): Cleaned df 
            transform (callable, optional): Image preprocessing function
        """
        self.df = df
        self.transform = transform
        self.label_map = {'NORMAL': 0, 'PNEUMONIA': 1}   # Mapping dictionary for labels (label name: number)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['filepath']
        label_str = row['label']
        
        image = Image.open(img_path).convert('RGB')    # Load images, processing grayscale images as RGB three-channel images, to accommodate subsequent model training.

        if self.transform:                  # Apply Transformations (Resize, ToTensor, Normalize)
            image = self.transform(image)

        label = self.label_map[label_str]   # Map labels from strings to integers (0 or 1)    
        return image, torch.tensor(label, dtype=torch.long)     # Return image tensor and label tensor 
    


    def preprocessed_data_describe(self, original_total=None, batch_size=32):   # 预处理后的数据描述报告，包括：留存率计算、类别分布统计、完整性检查
        """       
        Args:
            original_total (int, optional): Volume of dirty data
            batch_size (int, optional): Batch Size for DataLoader Smoke Test
        """
        clean_total = len(self)
        print("[PREPROCESSED DATA DESCRIBE]")
 
        # 1. Data Retention
        print(f"1. Data Volume:")
        print(f"   Current Size: {clean_total}")
        if original_total is not None:
            diff = original_total - clean_total
            survival_rate = (clean_total / original_total) * 100 if original_total > 0 else 0

            print(f"   Original Size: {original_total}")
            print(f"   Removed Rows : {diff}")
            print(f"   Survival Rate: {survival_rate:.2f}%\n")
            if survival_rate < 80:
                print("   [WARNING] High data loss detected! Check cleaning logic.\n")
        else:
            print("   (Raw total not provided, skipping retention rate)")


        # 2. Class Distribution
        print(f"2. Class Distribution:")
        label_counts = self.df['label'].value_counts().to_dict()
        for label, count in label_counts.items():
            ratio = (count / clean_total) * 100
            print(f"   {label:<10}: {count} ({ratio:.1f}%)")


        # 3. Integrity Check
        print(f"3. Integrity Check:")
        suspicious_count = 0
        check_limit = min(clean_total, 50) # Check the first 50 only to prevent lag.
        
        for i in range(check_limit):
            img, _ = self[i] 
            if isinstance(img, torch.Tensor):    # Convert to NumPy format
                img_arr = img.numpy()
            else:
                img_arr = np.array(img)

            if img_arr.min() == img_arr.max():   # Determine if the image is pure black or pure white
                suspicious_count += 1

        if suspicious_count == 0:
            print("   Sample check passed (No pure black/white images).")
        else:
            print(f"   ! Found {suspicious_count} suspicious images in sample!")


        # 4. DataLoader Smoke Test
        print(f"4. DataLoader Test (Batch Size={batch_size}):")
        try:
            test_loader = DataLoader(self, batch_size=batch_size, shuffle=True)    # 临时创建一个 loader 跑一次
            images, labels = next(iter(test_loader))
            print(f"   Batch loaded successfully! Image: {images.shape}, Label: {labels.shape}.\n")
        except Exception as e:
            print(f"   ! DataLoader Error: {e}\n")