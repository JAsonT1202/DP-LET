from data.data_loader import Dataset_MILAN, Dataset_Pred
from torch.utils.data import DataLoader
"""
Choose the mode and dataset
mode: ['train', 'val', 'test', 'pred']
dataset: ['MILAN', 'TRENTINO']

"""

data_dict = {
    'MILAN': Dataset_MILAN,
    # 'TRENTINO': Dataset_TRENTINO,
}


def data_provider(mode, **configs):
    """
    mode choose from ['train', 'val', 'test', 'pred']

    """
    data_configs = configs['data']
    model_configs = configs['model']

    Data = data_dict[data_configs['dataset_name']]
    # timeenc default: 1
    timeenc = 0 if model_configs['embed_type'] != 'timeF' else 1

    if mode == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = data_configs['batch_size']
        freq = data_configs['freq']
    elif mode == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = data_configs['freq']
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = data_configs['batch_size']
        freq = data_configs['freq']

    if configs['model']['dim'] == 'time_feat':
        data_set = Data(dataset_filename=data_configs['dataset_filename'],
                        mode=mode,
                        in_label_out_lens=[model_configs['in_lens'], model_configs['label_lens'], model_configs['out_lens']],
                        scale=True,
                        timeenc=timeenc,
                        freq=freq,
                        load_multi_feat=True,
                        )

    else:
        data_set = Data(dataset_filename=data_configs['dataset_filename'],
                        mode=mode,
                        in_label_out_lens=[model_configs['in_lens'], model_configs['label_lens'], model_configs['out_lens']],
                        scale=True,
                        timeenc=timeenc,
                        freq=freq
                        )

    print(mode, len(data_set))
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=data_configs['num_workers'],
        drop_last=drop_last
    )

    return data_set, data_loader

