import torch
import torch.nn as nn
import torch.optim as optim
import time


from .config import EPOCHS, LEARNING_RATE, MODEL_SAVE_DIR, RESULTS_DIR
from .visualization import plot_training_history

def train_model(model, train_loader, val_loader):
    """
    Train and validate via loops
    Save the best model
    Call visualization.py to plot
    """
    print(f"[TRAINING] Start training for {EPOCHS} epochs:")
    start_time = time.time()
 
    # Phase A: Training
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        train_loss = 0.0
        train_correct = 0
        total = 0
            
        model.train()        # set the model to training mode
        for images, labels in train_loader:
            output = model(images)
            loss = loss_fn(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item()
            _, pred = torch.max(output, 1)
            total += labels.size(0)
            train_correct += (pred == labels).sum().item()
            
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100.0 * train_correct / total
        

        # Phase B: Validation
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                
                val_loss_sum += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss_sum / len(val_loader)
        epoch_val_acc = 100.0 * val_correct / val_total
        

        # Phase C: Log & Save
        print(f"[TRAINING] Epoch {epoch+1}/{EPOCHS} | "
              f"Train: {epoch_train_acc:.2f}% (accuracy), {epoch_train_loss:.4f} (loss) | "
              f"Val: {epoch_val_acc:.2f}% (accuracy), {epoch_val_loss:.4f} (loss)")
        
        history['train_loss'].append(epoch_train_loss)   # Update History
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            save_path = MODEL_SAVE_DIR / "best_model.pth"
            torch.save(model.state_dict(), save_path)          # Save the best model
            
    total_time = time.time() - start_time
    print(f"[TRAINING] Training Complete in {total_time:.0f}s. Best Val Acc: {best_acc:.2f}%\n")
    
    plot_training_history(history, save_filename=RESULTS_DIR / "training_history.png")    # Call visualization.py to plot
    print("Saved Loss and Accuracy curves to folder.\n")
    return model