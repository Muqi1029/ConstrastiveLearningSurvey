import argparse
import sys
import time
import os
import torch
from torch import optim
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
from torch.utils.data import DataLoader

import models
from dataset.loader import load_dataset
from dataset.transform import load_transform
from lemniscate.linear_average import LinearAverage
from lemniscate.nce_average import NCEAverage
from loss.NCECriterion import NCECriterion
from utils import AverageMeter, ProgressMeter
from metrics.knn import NN, kNN


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--nce-k', default=4096, type=int,
                    metavar='K', help='number of negative samples for NCE')
parser.add_argument('--nce-t', default=0.07, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--nce-m', default=0.5, type=float,
                    help='momentum for non-parametric updates')
parser.add_argument('--iter_size', default=1, type=int,
                    help='caffe style iter size')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    # get dataloader
    transform_train, transform_test = load_transform()
    train_dataloader, test_dataloader, classes = load_dataset(
        "cifar10",
        transform=transform_train,
        test_transform=transform_test,
    )

    # load model
    print("==> Building model...")
    net = models.__dict__['ResNet18'](low_dim=args.low_dim)
    ndata = len(train_dataloader.dataset)

    # define leminiscate
    if args.nce_k > 0:
        lemniscate = NCEAverage(args.low_dim, ndata,
                                args.nce_k, args.nce_t, args.nce_m)
    else:
        lemniscate = LinearAverage(args.low_dim, ndata, args.nce_t, args.nce_m)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    start_epoch = 0

    if args.resume:
        # load the checkpoint
        print("==> Resuming from checkpoint...")
        if not os.path.exists("./checkpoint"):
            os.makedirs("./checkpoint")
        checkpoint_path = os.path.join("./checkpoint", args.resume)
        assert os.path.exists(checkpoint_path)

        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        lemniscate = checkpoint['lemniscate']
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['start_epoch']

    # define loss function
    if hasattr(lemniscate, 'K'):
        criterion = NCECriterion(ndata)
    else:
        criterion = nn.CrossEntropyLoss()

    net.to(device)
    lemniscate.to(device)
    criterion.to(device)

    if args.test_only:
        acc = kNN(0,
                  net,
                  lemniscate,
                  train_dataloader,
                  test_dataloader,
                  200, args.nce_t, 1)
        sys.exit(0)

    # define optimizer
    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(args.start_epoch, args.epochs + args.start_epoch):
        train(train_dataloader=train_dataloader,
              model=net, lemniscate=lemniscate, criterion=criterion,
              optimizer=optimizer, epoch=epoch, device=device)


def train(train_dataloader: DataLoader, 
          model: nn.Module, 
          lemniscate, 
          criterion, 
          optimizer, 
          epoch: int,
          device: str):

    batch_time = AverageMeter("batch_time", fmt=":.3f")
    data_time = AverageMeter("data_time", fmt=":.3f")
    losses = AverageMeter("losses", fmt=":.3f")
    progress = ProgressMeter(len(train_dataloader), 
                             meters=[batch_time, data_time, losses],
                             prefix=f"Epoch {epoch}")


    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()

    for i, (inputs, targets, indices) in enumerate(train_dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets, indices = inputs.to(device), targets.to(device), indices.to(device)

        # compute output
        features = model(inputs)
        output = lemniscate(features, indices)
        loss = criterion(output, indices) 

        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        progress.display(i + 1)

if __name__ == '__main__':
    main()
