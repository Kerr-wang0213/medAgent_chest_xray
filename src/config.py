from pathlib import Path

# Path Configuration
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw' / 'chest_xray'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

MODEL_SAVE_DIR = BASE_DIR / 'models'
TRAIN_DIR = RAW_DATA_DIR / 'train'
RESULTS_DIR = BASE_DIR / 'results'


# Automatically create folders
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)


# Hyperparameters
RANDOM_SEED = 42
IMG_WIDTH = 224
IMG_HEIGHT = 224

INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 1e-4

NORM_MEAN = [0.485, 0.456, 0.406]     # Normalization Constants (ImageNet Standards)
NORM_STD = [0.229, 0.224, 0.225]


if __name__ == "__main__":
    print(f"Project Root: {BASE_DIR}")
    if TRAIN_DIR.exists():
        print("Status: Train dir found.")
    else:
        print("Status: Train dir NOT found.")
