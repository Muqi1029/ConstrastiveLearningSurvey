import argparse
import os
import warnings
from moco.builder import Moco
from moco import loader
import torch
from torch import nn
from torchvision import models
from utils import *
from torchvision import transforms, datasets
from torch.utils.data import DataLoader



model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="Pytorch ImageNet Training")


parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "--seed",
    default=None,
    type=int
)


#  ------------------ Moco --------------------
parser.add_argument(
    "--moco-dim", default=128, type=int, help="feature dimension (default: 128)"
)
parser.add_argument(
    "--moco-k",
    default=65536,
    type=int,
    help="queue size; number of negative keys (default: 65536)",
)
parser.add_argument(
    "--moco-m",
    default=0.999,
    type=float,
    help="moco momentum of updating key encoder (default: 0.999)",
)
parser.add_argument(
    "--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)"
)


# options for moco v2
parser.add_argument("--mlp", action="store_true", help="use mlp head")
parser.add_argument(
    "--aug-plus", action="store_true", help="use moco v2 data augmentation"
)
parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")


# -------------------- Training args ------------------------------
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.03,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--schedule",
    default=[120, 160],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by 10x)",
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)

# --------------------- Print Log ---------------------
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)

# ---------------------- Persistence -------------------------------
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)

# ----------------------- Distribution -------------------------
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://224.66.41.62:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)

# --------------------------- GPU ----------------------------------
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")


def main():
    args = parser.parse_args()
    if args.seed is not None:
        seed_everything(args.seed)
        
    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )
    main_worker(args.gpu, 1, args)
    

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    
    print("=> creating model '{}'".format(args.arch))
    model = Moco(base_encoder=models.__dict__[args.arch], 
                 dim=args.moco_dim,
                 K=args.moco_k,
                 m=args.moco_m,
                 t=args.moco_t,
                 mlp=args.mlp)
    model.eval()

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    
    init_lr = args.lr * args.batch_size / 256
    
    if args.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()  

    optimizer = torch.optim.SGD(
        optim_params,
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )        

    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ]
    
    # Data loading code
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data, "train"),
        loader.TwoCropsTransform(transforms.Compose(augmentation)))
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=(train_sampler is None), num_workers=args.workers,
                              pin_memory=True, sampler=train_sampler, drop_last=True)
    
    print("start to train".center(50, "-"))

    for epoch in range(args.start_epoch, args.epochs):
        
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, init_lr, epoch, args)

        train(train_loader, model, criterion, optimizer, epoch, args)
        
        if not args.multiprocess_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            
            save_checkpoint({
                'epoch': epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))


if __name__ == "__main__":
    main()
