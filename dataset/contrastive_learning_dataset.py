"""Create dataset"""
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from data_aug.gaussian_blur import GaussianBlur


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


class ContrastiveLearningDataset(Dataset):
    def __init__(self, root_folder: str):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations 
        as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose(
            [transforms.RandomResizedCrop(size=size),
             transforms.RandomHorizontalFlip(),
             transforms.RandomApply(
                [color_jitter], p=0.8),
             transforms.RandomGrayscale(
                p=0.2),
             GaussianBlur(
                kernel_size=int(0.1 * size)),
             transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {
            'cifar10': lambda: datasets.CIFAR10(
                self.root_folder,
                train=True,
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(32), n_views),
                download=True),

            'stl10': lambda: datasets.STL10(
                self.root_folder, split='unlabeled',
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(
                        96),
                    n_views),
                download=True)}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            print("invalid dataset")
        else:
            return dataset_fn()


def load_dataloader(dataset, batch_size: int, shuffle=True,
        num_workers=1, pin_memory=True, drop_last=True):
    return DataLoader(dataset, batch_size=batch_size, 
                      shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory, drop_last=drop_last)
    