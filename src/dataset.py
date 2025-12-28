import torch
from torch.utils.data import Dataset
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
            df (pd.DataFrame): 这里的 df 必须是清洗过后的。
            transform (callable, optional): 图像增强/预处理函数。
        """
        self.df = df
        self.transform = transform
        
        self.label_map = {'NORMAL': 0, 'PNEUMONIA': 1}   # 定义标签映射字典。如果不写：模型无法识别 'NORMAL' 是什么，必须转为数字。

    def __len__(self):                # 返回数据集大小。如果不写：DataLoader 无法知道有多少数据，无法划分 batch。
        return len(self.df)

    def __getitem__(self, idx):       # 获取当前行数据。如果不写：无法获取对应的文件路径和标签。
        row = self.df.iloc[idx]
        img_path = row['filepath']
        label_str = row['label']
        
        # 1. 加载图像 (使用 PIL)
        # .convert('RGB') 极其重要，防止单通道图片报错
        image = Image.open(img_path).convert('RGB')

        # 2. 应用变换 (Resize, ToTensor, Normalize)，如果不写：图片尺寸不一，且不是 Tensor 格式，无法喂给神经网络。
        if self.transform:
            image = self.transform(image)

        # 3. 处理标签
        label = self.label_map[label_str]   # 将字符串映射为整数 (0 或 1)。
        
        # 返回图片张量和标签张量
        return image, torch.tensor(label, dtype=torch.long)    # dtype=torch.long 是分类任务标签的标准格式
    

    def preprocessed_data_describe(self, raw_total=None, batch_size=32):   # 数据描述报告,集成：留存率计算、类别分布统计、完整性检查。
        """       
        Args:
            raw_total (int, optional): 原始（脏）数据量。如果不传则不计算留存率。
            batch_size (int, optional): 测试DataLoader时的批次大小。
        """
        clean_total = len(self)
        print(" PREPROCESSED DATA DESCRIBE (数据描述)")
 
        # --- 1. Data Retention (留存率) ---
        print(f"1. Data Volume (数据量):")
        print(f"   Current Size: {clean_total}")
        
        if raw_total is not None:
            diff = raw_total - clean_total
            survival_rate = (clean_total / raw_total) * 100 if raw_total > 0 else 0
            print(f"   Original Size: {raw_total}")
            print(f"   Removed Rows : {diff}")
            print(f"   Survival Rate: {survival_rate:.2f}%")
            
            if survival_rate < 80:
                print("   [WARNING] High data loss detected! Check cleaning logic.")
        else:
            print("   (Raw total not provided, skipping retention rate)")

        # --- 2. Class Balance (类别分布) ---
        print(f"\n2. Class Distribution (类别分布):")
        label_counts = self.df['label'].value_counts().to_dict()
        for label, count in label_counts.items():
            ratio = (count / clean_total) * 100
            print(f"   - {label:<10}: {count} ({ratio:.1f}%)")
            
        # --- 3. Integrity Check (完整性检查) ---
        print(f"\n3. Integrity Check (完整性检查):")
        suspicious_count = 0
        check_limit = min(clean_total, 50) # 只抽查前50个，防止卡顿
        
        for i in range(check_limit):
            img, _ = self[i] 
            
            if isinstance(img, torch.Tensor):    # 统一转 numpy
                img_arr = img.numpy()
            else:
                img_arr = np.array(img)

            if img_arr.min() == img_arr.max():          # 判断是否是纯黑或纯白图
                suspicious_count += 1
                
        if suspicious_count == 0:
            print("   [PASS] Sample check passed (No pure black/white images).")
        else:
            print(f"   [FAIL] Found {suspicious_count} suspicious images in sample!")