
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from .distribute_dataset import distribute_dataset


def load_cifar100(root, transforms=None, image_size=32,  
                 train_batch_size=64, valid_batch_size=64,
                 distribute=False, split=1.0, rank=0, seed=666):

    train_transforms = tfs.Compose([
        tfs.Resize((image_size,image_size)),
        tfs.RandomHorizontalFlip(),  
        tfs.ToTensor(),
        tfs.Normalize([0.5073715, 0.4867007, 0.441096], [0.26750046, 0.25658613, 0.27630225])
    ])

    valid_transforms = tfs.Compose([
        tfs.Resize((image_size,image_size)),
        tfs.ToTensor(),
        tfs.Normalize([0.5073715, 0.4867007, 0.441096], [0.26750046, 0.25658613, 0.27630225])
    ])

    if train_batch_size is None:
        train_batch_size = 1
    if split is None:
        split = [1.0]
    train_set = CIFAR100(root, True, train_transforms, download=True)
    valid_set = CIFAR100(root, False, valid_transforms, download=True)
    if distribute:
        train_set = distribute_dataset(train_set, split, rank, seed=seed)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=valid_batch_size, drop_last=True)
    return train_loader, valid_loader, (3, image_size, image_size), 100
