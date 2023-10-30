from ditk import logging
from cyberharem.train import train_plora


def run_train_plora(dataset_name, charname, min_step):
    logging.try_init_root(logging.INFO)
    if min_step:
        train_plora(
            source='dataset/'+dataset_name,
            name=charname,
            min_steps=int(min_step),
            workdir='runs/'+dataset_name,
        )
    else:
        train_plora(
            source='dataset/' + dataset_name,
            name=charname,
            workdir='runs/' + dataset_name,
        )


if __name__ == '__main__':
    run_train_plora('./dataset/chen', 'chen', 1000)