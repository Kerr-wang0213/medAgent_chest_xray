import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.metrics import confusion_matrix

def plot_distribution(df, title, save_filename):
    """
    Draw a bar chart of the data distribution across dataset splits and labels.
    """
    plt.figure(figsize=(10, 6))
    
    # x='dataset_split' - train/val/test
    # hue='label' - NORMAL/PNEUMONIA
    sns.countplot(x='dataset_split', hue='label', data=df, palette='viridis')
    
    plt.title(title)
    plt.xlabel('Dataset Split')
    plt.ylabel('Count')
    plt.legend(title='Diagnosis')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_filename)
    print(f"[VISUALIZATION] Saved distribution plot to {save_filename}")
    plt.close()


def plot_sample_images(df, title, save_filename, num_samples=5):
    """
    Randomly select samples and display them in a jigsaw puzzle format.
    """
    n = min(len(df), num_samples)
    if n == 0:
        print("[VISUALIZATION] No samples to plot.")
        return

    sample_df = df.sample(n=n, random_state=42)
    
    fig, axes = plt.subplots(1, n, figsize=(15, 3))
    if n == 1: axes = [axes]
    
    fig.suptitle(title, fontsize=16)
    
    for ax, (_, row) in zip(axes, sample_df.iterrows()):
        img_path = row['filepath']
        label = row['label']
        
        try:
            with Image.open(img_path) as img:
                ax.imshow(img, cmap='gray')

            fname = Path(img_path).name
            display_name = (fname[:10] + '..') if len(fname) > 10 else fname
            
            ax.set_title(f"{label}\n{display_name}", fontsize=10)
            ax.axis('off')
            
        except Exception as e:
            ax.set_title("Error loading")
            ax.axis('off')
            print(f"Error plotting {img_path}: {e}")
            
    plt.tight_layout()
    plt.savefig(save_filename)
    print(f"[VISUALIZATION] Saved sample images to {save_filename}")
    plt.close()


def plot_training_history(history, save_filename):
    """
    Plot the Loss and Accuracy curves during the training process
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 1. Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r--', label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 2. Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r--', label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_filename)
    print(f"[VISUALIZATION] Training history plot saved to {save_filename}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_filename):
    """
    Draw a confusion matrix heat map
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(save_filename)
    print(f"[VISUALIZATION] Confusion Matrix saved to {save_filename}")
    plt.close()