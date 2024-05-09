import torch
from torch import nn
from dataset.librispeech import LibriDataset
import os
from model.cpc import CPC
from torch.utils.data import DataLoader


def load_audio_model(args):
    strides = [5, 4, 2, 2, 2]
    filter_sizes = [10, 8, 4, 4, 4]
    padding = [2, 2, 2, 2, 1]
    genc_hidden = 512
    gar_hidden = 256

    model = CPC(
        args,
        strides=strides,
        filter_sizes=filter_sizes,
        padding=padding,
        genc_input=1,
        genc_hidden=genc_hidden,
        gar_hidden=gar_hidden,
    )
    return model

def librispeech_loader(opt, num_workers=16):
    if opt.validate:
        print("Using Train / Val split")
        train_dataset = LibriDataset(opt, 
                                     root=os.path.join(opt.data_input_dir, "LibriSpeech/train-clean-100"),
                                     flist=os.path.join(opt.data_input_dir, "LibriSpeech100_labels_split/train_val_train.txt")
                                     )
        test_dataset = LibriDataset(opt=opt,
                                    root=os.path.join(opt.data_input_dir, "LibriSpeech/train-clean-100"),
                                    flist=os.path.join(opt.data_input_dir, "LibriSpeech100_labels_split/train_val_val.txt")
                                    )
    else:
        print("Using Train+val / Test dataset")
        train_dataset = LibriDataset(opt=opt,
                                     root=os.path.join(opt.data_input_dir, "LibriSpeech/train-clean-100"),
                                     flist=os.path.join(opt.data_input_dir, "LibriSpeech100_labels_split/train_split.txt")
                                   )
        test_dataset = LibriDataset(opt=opt, 
                                    root=os.path.join(opt.data_input_dir, "LibriSpeech/train-clean-100"), 
                                    flist=os.path.join(opt.data_input_dir, "LibriSpeech100_labels_split/test_split.txt")
                                    )
    
    train_loader = DataLoader(train_dataset,
                              batch_size=opt.batch_size,
                              shuffle=False,
                              drop_last=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_dataset,
                             batch_size=opt.batch_size,
                             shuffle=False,
                             drop_last=True,
                             num_workers=num_workers)
    return train_loader, train_dataset, test_loader, test_dataset