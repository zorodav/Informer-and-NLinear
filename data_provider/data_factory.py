from data_provider.data_loader import ODE_Lorenz, PDE_KS, Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Lorenz_Official, KS_Official
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'ODE_Lorenz': ODE_Lorenz,
    'PDE_KS': PDE_KS,
    'KS_Official': KS_Official,
    'Lorenz_Official': Lorenz_Official,
}


def data_provider(args, flag):
    Data = data_dict[args['dataset']['name']]
    print('Data:', Data)
    timeenc = 0 if args['embed'] != 'timeF' else 1
    train_only = args['training']['train_only']

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args['batch_size']
        freq = args['freq']
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args['freq']
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args['batch_size']
        freq = args['freq']
    if flag == 'pred':
        data_set = Data(
            root_path=args['root_path'],
            flag=flag,
            size=[args['seq_len'], args['label_len'], args['pred_len']],
            features=args['features'],
            timeenc=timeenc,
            freq=freq,
            train_only=train_only,
        )
    else:
        data_set = Data(
            root_path=args['root_path'],
            flag=flag,
            size=[args['seq_len'], args['label_len'], args['pred_len']],
            features=args['features'],
            timeenc=timeenc,
            freq=freq,
            train_only=train_only,
            pair_id=args['dataset']['pair_id'],
        )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args['num_workers'],
        drop_last=drop_last)
    return data_set, data_loader
