import torch
from sklearn.metrics import classification_report, accuracy_score
from .config import RESULTS_DIR
from .visualization import plot_confusion_matrix

def evaluate_model(model, test_loader):
    """
    Evaluate the trained model on the test dataset and print a detailed report.
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            
    class_names = ['NORMAL', 'PNEUMONIA']
    
    acc = accuracy_score(all_labels, all_preds)
    print(f"Overall Accuracy: {acc*100:.2f}%\n")
    
    print("Detailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    save_path = RESULTS_DIR / "confusion_matrix.png"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    plot_confusion_matrix(all_labels, all_preds, class_names, save_path)
    
    print("="*50 + "\n")