from MNIST.model import train_model, evaluate_model, prepare_data, MNISTClassifier
from configuration.init_config import DEVICE, INPUT_SIZE, HIDDEN_SIZES, NUM_CLASSES, LEARNING_RATE, EPOCHS, BATCH_SIZE, DROPOUT_RATE
import torch.optim as optim
import torch.nn as nn
import torch
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up loging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', handlers=[
    logging.FileHandler("mnist_logs/mnist.log"),
    logging.StreamHandler()
])

def main():
    train_loader, test_loader = prepare_data(BATCH_SIZE)

    # Optimization algorithms
    optimizers = {
        "SGD": optim.SGD,
        "SGD_Momentum": lambda params, lr: optim.SGD(params, lr=lr, momentum=0.9),
        "Adam": optim.Adam
    }

    criterion = nn.CrossEntropyLoss()

    results = {}

    for opt_name, opt_class in optimizers.items():
        # Initialize model for evert optimizers
        model = MNISTClassifier(INPUT_SIZE, HIDDEN_SIZES, NUM_CLASSES, dropout_rate=DROPOUT_RATE).to(DEVICE)
        optimizer = opt_class(model.parameters(), lr=LEARNING_RATE)

        train_losses, test_losses = [], []

        for epoch in range(EPOCHS):
            train_loss = train_model(model, train_loader, optimizer, criterion)
            test_loss, accuracy = evaluate_model(model, test_loader, criterion)

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            print(f"{opt_name} - Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Accuracy = {accuracy:.2f}%")
            logging.info(f"{opt_name} - Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Accuracy = {accuracy:.2f}%")

        results[opt_name] = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "final_accuracy": accuracy
        }
    # Save the best model
    model_filename = f"mnist_logs/best_model_{opt_name}_{accuracy:.2f}.pth"
    torch.save(model.state_dict(), model_filename)

    # Print results
    for opt_name, result in results.items():
        print(f"\n{opt_name} Results:")
        print(f"Final Test Accuracy: {result['final_accuracy']:.2f}%")

        # Add into log file
        logging.info(f"\n{opt_name} Results:")
        logging.info(f"Final Test Accuracy: {result['final_accuracy']:.2f}%")

if __name__ == "__main__":
    main()