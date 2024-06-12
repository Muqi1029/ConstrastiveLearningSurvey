"""run script"""
import argparse
from torchvision import models
from dataset.contrastive_learning_dataset import ContrastiveLearningDataset, load_dataloader
from models.loader import load_model
import torch.backends.cudnn as cudnn
import torch
from trainer import Trainer


parser = argparse.ArgumentParser("constrastive learning arugmentor")
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--data', metavar='DIR', default='data',
                    help='path to dataset')
parser.add_argument('--dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument("--model-name", default="simclr")
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')

parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')


def main():
    # 0. get args
    args = parser.parse_args()
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
    
    # 1. get dataset and dataloader
    dataset_loader = ContrastiveLearningDataset(args.data)
    train_dataset = dataset_loader.get_dataset(args.dataset_name, args.n_views)
    dataloader = load_dataloader(dataset=train_dataset, batch_size=args.batch_size, 
                                  num_workers=args.workers)

    # 2. load model
    model = load_model(model_name=args.model_name, base_model=args.arch, out_dim=args.out_dim).to(args.device)

    # 3. optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=len(dataloader),
                                                           eta_min=0,
                                                           last_epoch=-1)
    
    # 4. train
    with torch.cuda.device(args.gpu_index):
        trainer = Trainer(model=model,
                          dataloader=dataloader, 
                            optimizer=optimizer,
                            scheduler=scheduler, 
                            args=args)
        trainer.train()




if __name__ == "__main__":
    main()