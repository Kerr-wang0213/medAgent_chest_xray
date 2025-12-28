import torch
from torch.utils.data import Dataset
from PIL import Image

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