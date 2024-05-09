import torch
from torch import nn
from torch.utils.data import DataLoader
import time


class AverageMeter:
    """Computes and stores the average and current value
    """
    def __init__(self, name, fmt=":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0
    
    def update(self, val, n :int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def KNN(epoch: int,
        net: nn.Module, 
        trainloader: DataLoader, 
        testloader: DataLoader, 
        K: int, 
        sigma: float, 
        ndata: int, 
        low_dim: int = 128):
    """_summary_

    Args:
        epoch (int): _description_
        net (nn.Module): _description_
        trainloader (DataLoader): _description_
        testloader (DataLoader): _description_
        K (int): _description_
        sigma (float): _description_
        ndata (int): the number of data
        low_dim (int, optional): the feature dim of each img. Defaults to 128.
    """
    net.eval()
    net_time = AverageMeter("Inference time", ":.3f")
    cls_time = AverageMeter("Classification time", ":.3f")
    top1_acc = AverageMeter("TOP1 ACC", ":.2f")
    top5_acc = AverageMeter("TOP5 ACC", ":.2f")
    progress = ProgressMeter(num_batches=len(testloader), meters=[net_time, cls_time, top1_acc, top5_acc])

    total = 0
    correct_t = 0
    test_size = len(testloader.dataset)

    if hasattr(trainloader.dataset, "imgs"):
       train_labels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
    else:
       train_labels = torch.LongTensor(trainloader.dataset.targets).cuda()
    
    train_features = torch.zeros((low_dim, ndata)).cuda()
    C = int(train_labels.max()) + 1

    with torch.no_grad():
        train_transform = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = DataLoader(trainloader.dataset, batch_size=100, shuffle=False, num_workers=4)
        for batch_idx, (inputs, _, targets, indexes) in enumerate(temploader):
            batch_size = len(inputs)
            inputs = inputs.cuda()
            features = net(inputs)
            train_features[:, batch_idx*batch_size: min((batch_idx+1)*batch_size, ndata)] = features.data.T
    trainloader.dataset.transform = train_transform
    
    top1 = 0
    top5 = 0
    end = time.time()
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            batch_size = len(inputs)
            end = time.time()
            targets = targets.cuda()
            inputs = inputs.cuda()

            features = net(inputs)
            total += batch_size
            
            net_time.update(time.time() - end)
            end = time.time()
            
            dist = features @ train_features
            yd, yi = torch.topk(dist, K=K, dim=1, largest=True, sorted=True)
            
            candidates = train_labels.reshape(1, -1).expand(batch_size, -1)
            retrieval = torch.gather(candidates, dim=1, index=yi)
            
            retrieval_one_hot.resize_(batch_size * K, C).zero_()
            retrieval_one_hot.scatter_(dim=1, index=retrieval, src=1)
            
            yd_transform = torch.exp(yd.clone() / sigma)
            probs = torch.sum(retrieval_one_hot.view(batch_size, -1, C) * yd_transform.view(batch_size, -1, 1), dim=1)
            _, predictions = probs.sort(dim=1, descending=True)
            
            correct = predictions == targets.view(-1, 1)
            cls_time.update(time.time() - end)

            top1 += correct[:, :1].sum().item()
            top5 += correct[:, :5].sum().item()
            top1_acc.update(top1, n=batch_size)
            top5_acc.update(top5, n=batch_size)
            progress.display(batch_idx)
    print(top1*100./total)
    return top1*100./total 
    
            
            
            
        
            
            
        