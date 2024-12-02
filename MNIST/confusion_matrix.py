import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import matplotlib.pyplot as plt
from MNIST.model import prepare_data, MNISTClassifier
from configuration.init_config import get_optimizer_config

file_name = 'mnist_logs/best_model_Adam.pth'
name = os.path.basename(file_name).split('_')[2].split('.')[0]
optimizer_config = get_optimizer_config(name)
LEARNING_RATE = optimizer_config["lr"]
EPOCHS = 50
BATCH_SIZE = optimizer_config["batch_size"]
DROPOUT_RATE = optimizer_config["dropout_rate"]
INPUT_SIZE = optimizer_config["input_size"]
HIDDEN_SIZES = optimizer_config["hidden_sizes"]
NUM_CLASSES = optimizer_config["num_classes"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load best model
def load_best_model():
    model = MNISTClassifier(INPUT_SIZE, HIDDEN_SIZES, NUM_CLASSES, dropout_rate=DROPOUT_RATE).to(DEVICE)
    model.load_state_dict(torch.load(f'mnist_logs/best_model_{name}.pth', weights_only=True))
    return model

def evaluate_with_confusion_matrix(model, test_loader):
    # Model eval mode
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            # Get model predictions
            output = model(data)
            preds = output.argmax(dim=1, keepdim=True)

            # Get predictions and targets
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Compute the conf. matrix
    cm = confusion_matrix(all_targets, all_preds)
    return cm

def plot_confusion_matrix(cm):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    _, test_loader = prepare_data(BATCH_SIZE)
    model = load_best_model()
    cm = evaluate_with_confusion_matrix(model, test_loader)
    plot_confusion_matrix(cm)