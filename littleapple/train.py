from ditk import logging
from cyberharem.train import train_plora


def run_train_plora(dataset_name, charname, min_step, bs, epoc, is_pipeline=False):
    logging.try_init_root(logging.INFO)
    # charname_e = re.sub(r'[^\w\s()]', '', ''.join([word if not (u'\u4e00' <= word <= u'\u9fff') else lazy_pinyin(charname)[i] for i, word in enumerate(charname)]))
    if is_pipeline:
        params = {
            'source': 'pipeline\\dataset\\' + dataset_name,
            'name': charname,
            'batch_size': bs,
            'workdir': 'pipeline\\runs\\' + dataset_name,
            'epochs': int(epoc),
        }
    else:
        params = {
            'source': 'dataset\\' + dataset_name,
            'name': charname,
            'batch_size': bs,
            'workdir': 'runs\\' + dataset_name,
            'epochs': int(epoc),
        }
    if min_step:
        params['min_steps'] = int(min_step)
    train_plora(**params)


if __name__ == '__main__':
    print('run webui.py instead')