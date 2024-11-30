import optuna
import torch
import logging
from MNIST.model import train_model, evaluate_model, prepare_data, MNISTClassifier
from configuration.init_config import DEVICE, EPOCHS

# Set up loging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', handlers=[
    logging.FileHandler("search_logs/model_search.log"),
    logging.StreamHandler()
])

# Dict. to store the best trials
best_trials = {
    'SGD': [],
    'SGD_Momentum': [],
    'Adam': []
}

def objective(trial):
    optimizer_name = trial.suggest_categorical('optimizer', ['SGD', 'SGD_Momentum', 'Adam'])

    if optimizer_name == 'SGD':
        lr = trial.suggest_float('sgd_lr', 1e-5, 1e-1, log=True)
        momentum = 0.0
    elif optimizer_name == 'SGD_Momentum':
        lr = trial.suggest_float('sgd_momentum_lr', 1e-5, 1e-1, log=True)
        momentum = trial.suggest_float('momentum', 0.5, 0.9)
    elif optimizer_name == 'Adam':
        lr = trial.suggest_float('adam_lr', 1e-5, 1e-1, log=True)
        momentum = 0.0

    # Common hyperpar.
    input_size = trial.suggest_categorical('input_size', [784])
    hidden_sizes = [trial.suggest_categorical('hidden_size_{}'.format(i), [256, 128]) for i in
                    range(trial.suggest_int('num_layers', 2, 5))]
    num_classes = trial.suggest_categorical('num_classes', [10])
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)

    train_loader, test_loader = prepare_data(batch_size)
    model = MNISTClassifier(input_size, hidden_sizes, num_classes, dropout_rate).to(DEVICE)

    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD_Momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        train_model(model, train_loader, optimizer, criterion)

    # eval model
    test_loss, accuracy = evaluate_model(model, test_loader, criterion)
    return test_loss

def logging_callback(study, trial):
    optimizer_name = trial.params['optimizer']
    best_trials[optimizer_name].append(trial)
    best_trials[optimizer_name] = sorted(best_trials[optimizer_name], key=lambda t: t.value)[:3]
    logging.info(f"{optimizer_name}, Trial {trial.number} finished with value: {trial.value}")

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    logging.info(f"A new study created in memory with name: {study.study_name}")

    study.optimize(objective, n_trials=100, callbacks=[logging_callback])
    logging.info(f"Best parameters: {study.best_params} with optimizer: {study.best_params['optimizer']}")

    for optimizer_name, trials in best_trials.items():
        logging.info(f"\nTop 3 trials for {optimizer_name}:")
        for trial in trials:
            logging.info(f"Trial {trial.number} with value: {trial.value} and params: {trial.params}")