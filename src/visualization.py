import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from pathlib import Path

def plot_distribution(df, title, save_filename):
    """
    绘制并保存类别分布图。
    对应 main.py 中的 EDA 分布绘制。
    """
    plt.figure(figsize=(10, 6))
    
    # 使用 seaborn 绘制计数图
    # x='dataset_split' 表示横轴是 train/val/test
    # hue='label' 表示按颜色区分 NORMAL/PNEUMONIA
    sns.countplot(x='dataset_split', hue='label', data=df, palette='viridis')
    
    plt.title(title)
    plt.xlabel('Dataset Split')
    plt.ylabel('Count')
    plt.legend(title='Diagnosis')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 保存图片到 results 文件夹（或者当前目录）
    plt.tight_layout()
    plt.savefig(save_filename)
    print(f"[Visualization] Saved distribution plot to {save_filename}")
    plt.close() # 关闭画布释放内存

def plot_sample_images(df, title, save_filename, num_samples=5):
    """
    随机抽取样本并拼图展示。
    对应 main.py 中的 坏图展示 和 清洗后样本展示。
    """
    # 如果数据不足 num_samples，就取全部
    n = min(len(df), num_samples)
    if n == 0:
        print("[Visualization] No samples to plot.")
        return

    sample_df = df.sample(n=n, random_state=42)
    
    fig, axes = plt.subplots(1, n, figsize=(15, 3))
    if n == 1: axes = [axes] # 处理只有一张图的情况
    
    fig.suptitle(title, fontsize=16)
    
    for ax, (_, row) in zip(axes, sample_df.iterrows()):
        img_path = row['filepath']
        label = row['label']
        
        try:
            # 使用 PIL 读取
            with Image.open(img_path) as img:
                ax.imshow(img, cmap='gray')
                
            # 获取文件名显示在标题里，方便确认是哪张图
            fname = Path(img_path).name
            # 如果文件名太长，截断一下
            display_name = (fname[:10] + '..') if len(fname) > 10 else fname
            
            ax.set_title(f"{label}\n{display_name}", fontsize=10)
            ax.axis('off')
            
        except Exception as e:
            ax.set_title("Error loading")
            ax.axis('off')
            print(f"Error plotting {img_path}: {e}")
            
    plt.tight_layout()
    plt.savefig(save_filename)
    print(f"[Visualization] Saved sample images to {save_filename}")
    plt.close()