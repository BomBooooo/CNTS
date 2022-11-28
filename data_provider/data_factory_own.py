from data_provider.data_loader_own import CLoadData, ALoadData, SLoadData
from torch.utils.data import DataLoader


def data_provider(args, flag, task):
    if args.model == 'Combined':
        Data = CLoadData
    else:
        Data = ALoadData

    
    if flag == 'train':
        batch_size = args.batch_size
        shuffle_flag = True
    else:
        batch_size = 1
        shuffle_flag = False
    drop_last = True

    data_set = Data(
        data_name=args.data,
        root_path=args.root_path,
        flag=flag,
        task=task,
        idx=args.idx, 
        seq_len=args.seq_len,
        features=args.features,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
