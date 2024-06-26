import torch
import shutil


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(initial_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    lr = initial_lr
    if epoch < 120:
        lr = initial_lr
    elif epoch >= 120 and epoch < 160:
        lr = initial_lr * 0.1
    else:
        lr = initial_lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxK = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxK, 1, True, True)
    pred = pred.T
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


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

    def display(self, cur_batch):
        entries = [self.prefix + self.batch_fmtstr.format(cur_batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        # {prefix} {[cur_batch/num_batches]} {meter1} {meter2}

    def _get_batch_fmtstr(self, num_batches):
        # [ cur_batch / num_batches]
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
