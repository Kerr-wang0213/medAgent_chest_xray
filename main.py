from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

from src.config import IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, NORM_MEAN, NORM_STD, RESULTS_DIR, MODEL_SAVE_DIR
from src.sabotage import generate_sabotaged_dataset
from src.cleaning import clean_dataset
from src.dataset import ChestXrayDataset
from src.visualization import plot_sample_images, plot_distribution
from src.model import ChestXRayResNet18
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    VIS_DIR = Path(RESULTS_DIR)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Generating Sabotaged Data
    print("\nPhase 1: Generating Sabotaged Data:")
    dirty_df = generate_sabotaged_dataset()
    print(f"Dirty Dataset Size: {len(dirty_df)}\n")
    
    # Visualize Dirty Data
    plot_distribution(
        dirty_df, 
        title="Class Distribution (Dirty)", 
        save_filename=VIS_DIR / "01_distribution_dirty.png"
    )
    corrupted_samples = dirty_df[dirty_df['filepath'].str.contains('corrupt_', na=False)]
    if not corrupted_samples.empty:
        plot_sample_images(
            corrupted_samples,
            title="Sabotaged Images (Simulated Errors)", 
            save_filename=VIS_DIR / "02_samples_corrupted.png"
        )
    print("Saved dirty data visualizations to folder.\n")


    # Phase 2: Cleaning Data
    print("Phase 2: Cleaning Data:")
    clean_df = clean_dataset(dirty_df)
    print(f"Clean Dataset Size: {len(clean_df)}\n")

    # Visualize Clean Data
    plot_distribution(
        clean_df, 
        title="Class Distribution (Cleaned)", 
        save_filename=VIS_DIR / "03_distribution_clean.png"
    )
    plot_sample_images(
        clean_df, 
        title="Valid Training Samples", 
        save_filename=VIS_DIR / "04_samples_clean.png"
    )
    print("Saved clean data visualizations to folder.\n")


    # Phase 3: Loading To Pytorch
    print("Phase 3: Loading To Pytorch:")
    data_transforms = transforms.Compose([                 # Define the preprocessing pipeline
        transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    dataset = ChestXrayDataset(clean_df, transform=data_transforms)     # Initialize Dataset.
    dataset.preprocessed_data_describe(original_total=len(dirty_df), batch_size=BATCH_SIZE)   # Print the description of preprocessed data.


    # Phase 4: Data Splitting & Training (80% training set, 10% validation set, 10% test set)
    print("Phase 4: Data Splitting & Training:")
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"Total images: {total_size}")
    print(f"Splitting into: Train: {train_size}, Val: {val_size}, Test: {test_size}")

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) 
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False) 
    print("DataLoaders created successfully.")

    model = ChestXRayResNet18(num_classes=2)
    train_model(model, train_loader, val_loader)


    # Phase 5: Evaluation on Test Set
    print("Phase 5: Evaluation on Test Set:")
    best_model_path = MODEL_SAVE_DIR / "best_model.pth"
    if best_model_path.exists():
        test_model = ChestXRayResNet18(num_classes=2)
        checkpoint = torch.load(best_model_path, map_location='cpu')
        test_model.load_state_dict(checkpoint)
        evaluate_model(test_model, test_loader)
    else:
        print("! best_model.pth not found!\n")

    # Completed
    print("Congratulation! All Phases Completed!\n")

if __name__ == "__main__":
    main()