from ditk import logging
from cyberharem.train import train_plora
from littleapple.exceptions import DatasetTypeError


def run_train_plora(dataset_name, min_step, bs, epoc, is_batch=False, batch_list=None, is_pipeline=False):
    logging.try_init_root(logging.INFO)
    # charname_e = re.sub(r'[^\w\s()]', '', ''.join([word if not (u'\u4e00' <= word <= u'\u9fff') else lazy_pinyin(charname)[i] for i, word in enumerate(charname)]))
    if not is_batch:
        dataset_list = [dataset_name]
    else:
        dataset_list = batch_list
    for dataset in dataset_list:
        if is_pipeline:
            params = {
                'source': 'pipeline\\dataset\\' + dataset,
                'name': dataset,
                'batch_size': bs,
                'workdir': 'pipeline\\runs\\' + dataset,
                'epochs': int(epoc),
            }
        else:
            if dataset.endswith(' (kohya)'):  # from dataset_dropdown
                raise DatasetTypeError(dataset, "正在尝试加载hcp数据集")
            params = {
                'source': 'dataset\\' + dataset,
                'name': dataset,
                'batch_size': bs,
                'workdir': 'runs\\hcpdiff\\' + dataset,
                'epochs': int(epoc),
            }
        if min_step:
            params['min_steps'] = int(min_step)
        else:
            params['no_min_steps'] = True
        train_plora(**params)


if __name__ == '__main__':
    print('run webui.py instead')
