import matplotlib.pyplot as plt  # 导入Matplotlib绘图库。如果不写：无法生成图表。
import seaborn as sns  # 导入Seaborn库。如果不写：无法生成美观的统计图。
import cv2  # 导入OpenCV。如果不写：无法读取要展示的图片。
import pandas as pd  # 导入Pandas。如果不写：无法处理数据标签。
from .config import RESULTS_DIR  # 导入结果保存路径。如果不写：不知道图存哪儿。

def plot_sample_images(df, title, filename, num_samples=5):
    """
    Plot random sample images and save to results folder.
    """
    # 如果DataFrame为空则直接返回。如果不写：空数据会导致报错。
    if df.empty: return

    # 随机采样指定数量的图片。如果不写：展示全部图片会内存溢出或太乱。
    samples = df.sample(min(len(df), num_samples), random_state=42)
    
    # 创建画布。如果不写：图表没有载体。
    plt.figure(figsize=(15, 5))
    
    # 遍历采样数据。如果不写：无法画出每一张子图。
    for i, (idx, row) in enumerate(samples.iterrows()):
        # 创建子图位置。如果不写：所有图片会重叠在一起。
        plt.subplot(1, num_samples, i+1)
        
        # 读取图片。如果不写：无法获取图像内容。
        img = cv2.imread(row['filepath'])
        
        # 检查图片是否读取成功。如果不写：坏路径会导致报错。
        if img is not None:
            # 将BGR转为RGB颜色空间。如果不写：图片颜色会显示成奇怪的蓝色调（OpenCV默认BGR）。
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)  # 显示图片。如果不写：画布上是空的。
        else:
            # 如果读不出来，显示文字提示。如果不写：用户不知道这里原本有图。
            plt.text(0.5, 0.5, "Error", ha='center', color='red')
            
        # 处理标签显示（NaN处理）。如果不写：NaN标签可能会导致报错或显示难看。
        label_text = str(row['label']) if pd.notna(row['label']) else "NaN"
        
        # 设置子图标题。如果不写：不知道这张图是什么类别。
        plt.title(f"Label: {label_text}")
        
        # 关闭坐标轴。如果不写：会有丑陋的刻度线。
        plt.axis('off')
    
    # 设置总标题。如果不写：图表缺乏整体描述。
    plt.suptitle(title, fontsize=16)
    
    # 自动调整布局。如果不写：子图之间可能会重叠遮挡。
    plt.tight_layout()
    
    # 保存图片到结果目录。如果不写：图表只能看一眼，无法放入报告。
    plt.savefig(RESULTS_DIR / filename)
    
    # 关闭画布释放内存。如果不写：循环画图时内存会飙升。
    plt.close()
    
    print(f"[Viz] Saved figure: {filename}")  # 打印日志。如果不写：不知道图片保存成功没。

def plot_distribution(df, title, filename):
    """
    Plot class distribution bar chart.
    """
    # 创建画布。如果不写：无法画图。
    plt.figure(figsize=(8, 6))
    
    # 统计类别数量，并把 NaN 填充为字符串以便显示。如果不写：NaN会被忽略，看不到缺失情况。
    counts = df['label'].fillna('Missing (NaN)').value_counts()
    
    # 绘制柱状图。如果不写：无法直观展示分布。
    sns.barplot(x=counts.index, y=counts.values, palette='viridis')
    
    # 设置标题。如果不写：不知道图表含义。
    plt.title(title)
    
    # 设置Y轴标签。如果不写：不知道Y轴数字代表什么。
    plt.ylabel('Count')
    
    # 在柱子上方标注具体数值。如果不写：很难看出精确数量。
    for i, v in enumerate(counts.values):
        plt.text(i, v + 50, str(v), ha='center')
        
    # 保存图表。如果不写：图表无法持久化。
    plt.savefig(RESULTS_DIR / filename)
    
    # 关闭画布。如果不写：占用内存。
    plt.close()
    
    print(f"[Viz] Saved figure: {filename}")  # 打印日志。如果不写：缺乏反馈。