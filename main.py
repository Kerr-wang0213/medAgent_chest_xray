# 导入自定义模块。如果不写：无法调用各个功能模块的代码。
from src.sabotage import generate_sabotaged_dataset
from src.cleaning import clean_dataset
from src.visualization import plot_sample_images, plot_distribution

def main():
    """
    Main execution flow for Phase 1: Data Preparation.
    """
    print("=== CA6000 Assignment: Chest X-Ray AI Diagnosis ===")  # 打印项目头。如果不写：仪式感缺失。
    
    # ---------------------------------------------------------
    # 1. Data Acquisition & Fault Injection
    # ---------------------------------------------------------
    # 调用故障注入模块获取脏数据。如果不写：没有数据可以处理。
    df_dirty = generate_sabotaged_dataset()
    
    # ---------------------------------------------------------
    # 2. EDA (Before Cleaning)
    # ---------------------------------------------------------
    print("\n--- Visualizing Dirty Data ---")  # 打印阶段提示。如果不写：日志不清晰。
    
    # 绘制清洗前的分布图。如果不写：无法在报告中展示原始数据的混乱（含NaN）。
    plot_distribution(df_dirty, 
                      "Class Distribution (Raw with Errors)", 
                      "01_distribution_dirty.png")
    
    # 筛选出被故意损坏的图片路径。如果不写：无法专门展示坏图样本。
    bad_files = df_dirty[df_dirty['filepath'].str.contains('corrupt_')]
    
    # 如果找到了坏文件，就画出来。如果不写：当没生成坏文件时会报错。
    if not bad_files.empty:
        plot_sample_images(bad_files, 
                           "Artificially Corrupted Samples", 
                           "02_samples_corrupted.png")

    # ---------------------------------------------------------
    # 3. Data Cleaning
    # ---------------------------------------------------------
    # 调用清洗模块执行双层清洗。如果不写：数据质量差，模型训练效果会很烂。
    df_clean = clean_dataset(df_dirty)
    
    # ---------------------------------------------------------
    # 4. Post-Cleaning Analysis
    # ---------------------------------------------------------
    print("\n--- Visualizing Cleaned Data ---")  # 打印阶段提示。如果不写：日志不清晰。
    
    # 绘制清洗后的分布图。如果不写：无法证明清洗工作有效（NaN消失）。
    plot_distribution(df_clean, 
                      "Class Distribution (Cleaned)", 
                      "03_distribution_clean.png")
    
    # 展示清洗后的正常样本。如果不写：无法确认最终进入模型的是什么图。
    plot_sample_images(df_clean, 
                       "Cleaned Training Samples", 
                       "04_samples_clean.png")

    print("\n=== Phase 1 Complete. Ready for Model Training. ===")  # 结束语。如果不写：不知道程序运行完了。

# 程序的入口判断。如果不写：被import时会意外执行main函数。
if __name__ == "__main__":
    main()