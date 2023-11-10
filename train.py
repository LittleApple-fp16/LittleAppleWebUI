from ditk import logging
from cyberharem.train import train_plora


def run_train_plora(dataset_name, charname, min_step, bs, epoc):
    logging.try_init_root(logging.INFO)
    params = {
        'source': 'dataset/' + dataset_name,
        'name': charname,
        'batch_size': bs,
        'workdir': 'runs/' + dataset_name,
        'epochs': int(epoc),
    }
    if min_step:
        params['min_steps'] = int(min_step)
    train_plora(**params)


if __name__ == '__main__':
    print('run webui.py instead')
