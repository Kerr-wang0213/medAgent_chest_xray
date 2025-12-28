import torch
import torch.nn as nn
import torch.optim as optim
import time
from pathlib import Path

# 1. 从 config 导入必要的常量 (注意：没有 DEVICE，也没有绘图库)
from .config import EPOCHS, LEARNING_RATE, MODEL_SAVE_DIR, RESULTS_DIR

# 2. 导入 visualization 里的函数，把画图的工作“外包”出去
from .visualization import plot_training_history

def train_model(model, train_loader, val_loader):
    """
    核心训练逻辑：
    - 自动检测设备 (CPU/GPU)
    - 训练与验证循环
    - 保存最佳模型
    - 调用 visualization 生成报表
    """
    
    print(f"[Training] Start training for {EPOCHS} epochs...")
    start_time = time.time()
    
    # 1. 准备模型与优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 准备记录数据的字典
    history = {'train_loss': [], 'epoch_train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    
    # 2. 开始 Epoch 循环
    for epoch in range(EPOCHS):
        train_loss = 0.0
        train_correct = 0
        total = 0
            
        model.train()        # set the model to training mode
        for images, labels in train_loader:

            output = model(images)      # forward propagation
            loss = loss_fn(output, labels)   # calculate the loss
            optimizer.zero_grad()    # set the previous computed gradients to zero
            loss.backward()          # backward propagation
            optimizer.step()         # update parameters
    
            train_loss += loss.item()
            _, pred = torch.max(output, 1)
            total += labels.size(0)
            train_correct += (pred == labels).sum().item()
            

        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100.0 * train_correct / total
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        
        # ==========================
        # Phase B: Validation (验证)
        # ==========================
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad(): # 验证时不计算梯度
            for images, labels in val_loader:
                
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                
                val_loss_sum += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss_sum / len(val_loader)
        epoch_val_acc = 100.0 * val_correct / val_total
        
        # ==========================
        # Phase C: Log & Save (记录)
        # ==========================
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train: {epoch_train_acc:.2f}% (Loss: {epoch_train_loss:.4f}) | "
              f"Val: {epoch_val_acc:.2f}% (Loss: {epoch_val_loss:.4f})")
        
        # 更新历史记录
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # 保存最佳模型
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            save_path = MODEL_SAVE_DIR / "best_model.pth"
            torch.save(model.state_dict(), save_path)
            
    # 3. 训练结束
    total_time = time.time() - start_time
    print(f"\n[Training] Done in {total_time:.0f}s. Best Val Acc: {best_acc:.2f}%")
    
    # 4. 调用 visualization.py 里的函数画图 (train.py 自己不画)
    plot_training_history(history, save_filename=RESULTS_DIR / "training_history.png")
    
    return model