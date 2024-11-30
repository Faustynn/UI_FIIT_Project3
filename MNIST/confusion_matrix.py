import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from MNIST.model import prepare_data, MNISTClassifier
from configuration.init_config import DEVICE, INPUT_SIZE, HIDDEN_SIZES, NUM_CLASSES, DROPOUT_RATE, BATCH_SIZE

def load_best_model():
    model = MNISTClassifier(INPUT_SIZE, HIDDEN_SIZES, NUM_CLASSES, dropout_rate=DROPOUT_RATE).to(DEVICE)
    model.load_state_dict(torch.load('mnist_logs/best_model.pth'))
    return model

def evaluate_with_confusion_matrix(model, test_loader):
    #model eval mode
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            # get model predictions
            output = model(data)
            preds = output.argmax(dim=1, keepdim=True)

            # get predictions and targets
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Compute the cm
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