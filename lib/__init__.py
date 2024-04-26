import os

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

config = {
    'logs': 'logs/'
}
datasets_path = {
    'air': 'datasets/air_quality',
    'la': 'datasets/metr_la',
    'bay': 'datasets/pems_bay',
    'synthetic': 'datasets/synthetic',
    'exchange_rate':'datasets/exchange_rate',
    'electricity': 'datasets/electricity',
    'solar': 'datasets/solar',
    'P19': 'datasets/P19',
    'P12': 'datasets/P12',
    'PAM': 'datasets/PAM',
}
epsilon = 1e-8

for k, v in config.items():
    config[k] = os.path.join(base_dir, v)
for k, v in datasets_path.items():
    datasets_path[k] = os.path.join(base_dir, v)
