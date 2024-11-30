import torch

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 784  # 28x28
HIDDEN_SIZES = [256, 128,128,128]
NUM_CLASSES = 10
BATCH_SIZE = 32

LEARNING_RATE = 0.098050
DROPOUT_RATE = 0.1
NUM_LAYERS = 3

EPOCHS = 50