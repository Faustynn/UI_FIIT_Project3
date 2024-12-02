optimizer_configs = {
    'SGD': {
        'optimizer': 'SGD',
        'lr': 0.011667030131725665,
        'input_size': 784,
        'num_layers': 3,
        'hidden_sizes': [256, 256, 256],
        'num_classes': 10,
        'batch_size': 64,
        'dropout_rate': 0.1
    },
    'SGD_Momentum': {
        'optimizer': 'SGD_Momentum',
        'lr': 0.008005276745626849,
        'momentum': 0.8987060343120898,
        'input_size': 784,
        'num_layers': 2,
        'hidden_sizes': [256, 256],
        'num_classes': 10,
        'batch_size': 256,
        'dropout_rate': 0.2
    },
    'Adam': {
        'optimizer': 'Adam',
        'lr': 7.85101358317508e-05,
        'input_size': 784,
        'num_layers': 3,
        'hidden_sizes': [256, 128, 128],
        'num_classes': 10,
        'batch_size': 64,
        'dropout_rate': 0.2
    }
}

def get_optimizer_config(optimizer_name):
    return optimizer_configs.get(optimizer_name)