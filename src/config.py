from pathlib import Path  # 导入Path对象用于路径操作。如果不写：无法使用面向对象的路径管理，需要回退到繁琐的os.path字符串拼接。

# ==========================================
# 1. Path Configuration
# ==========================================

# 获取当前文件的父目录的父目录，即项目根目录。如果不写：程序不知道项目在哪里，导致无法找到数据文件。
BASE_DIR = Path(__file__).resolve().parent.parent

# 定义数据总目录。如果不写：后续代码无法引用 data 文件夹。
DATA_DIR = BASE_DIR / 'data'

# 定义处理后的数据保存目录。如果不写：生成的坏数据和中间文件无处存放。
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# 定义结果图表保存目录。如果不写：可视化生成的图片无法保存。
RESULTS_DIR = BASE_DIR / 'results'

# 定义模型文件保存目录。如果不写：训练好的模型无法持久化保存。
MODEL_SAVE_DIR = BASE_DIR / 'models'

# 定义原始数据存放目录。如果不写：程序找不到 Kaggle 下载的数据源。
RAW_DATA_DIR = DATA_DIR / 'raw' / 'chest_xray'

# 定义训练集具体路径。如果不写：后续需要手动拼接路径，降低代码复用性。
TRAIN_DIR = RAW_DATA_DIR / 'train'

# 定义测试集具体路径。如果不写：同上，不方便引用测试集。
TEST_DIR = RAW_DATA_DIR / 'test'

# 定义验证集具体路径。如果不写：同上。
VAL_DIR = RAW_DATA_DIR / 'val'

# 自动创建处理数据文件夹（如果不存在）。如果不写：当程序试图写入文件时会报 FileNotFoundError。
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 自动创建结果文件夹。如果不写：保存图表时会报错。
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 自动创建模型文件夹。如果不写：保存模型时会报错。
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Hyperparameters
# ==========================================

# 定义批次大小。如果不写：各模块间批次大小可能不一致，导致训练出错。
BATCH_SIZE = 32

# 定义学习率。如果不写：优化器可能使用默认值，不一定适合当前任务。
LEARNING_RATE = 0.001

# 定义训练轮数。如果不写：无法全局控制训练时长。
EPOCHS = 10

# 定义图片输入宽度。如果不写：预处理时无法统一图片尺寸，导致矩阵维度不匹配。
IMG_WIDTH = 150

# 定义图片输入高度。如果不写：同上。
IMG_HEIGHT = 150

# 定义输入数据的形状元组。如果不写：构建神经网络第一层时缺少 input_shape 参数。
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)

# 定义随机种子。如果不写：每次运行的数据划分和破坏结果都不一样，无法复现实验。
RANDOM_SEED = 42

# 调试代码，仅当直接运行此脚本时执行。如果不写：不影响功能，但无法快速检查路径配置是否正确。
if __name__ == "__main__":
    print(f"Project Root: {BASE_DIR}")  # 打印根目录路径。如果不写：无法直观确认路径。
    if TRAIN_DIR.exists():  # 检查训练集是否存在。如果不写：无法提前发现路径配置错误。
        print("Status: Train dir found.")
    else:
        print("Status: Train dir NOT found.")
