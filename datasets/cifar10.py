
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from .distribute_dataset import distribute_dataset


def load_cifar10(root, transforms=None, image_size=32,  
                 train_batch_size=64, valid_batch_size=64,
                 distribute=False, split=1.0, rank=0, seed=666):
    # 对训练集进行数据增强
    transform_train = tfs.Compose([
        tfs.Resize((image_size,image_size)),
        # 随机水平翻转
        tfs.RandomHorizontalFlip(),
        # 以0.5的概率随机垂直翻转
        tfs.RandomVerticalFlip(p=0.5),
        # 将图片转换为tensor格式
        tfs.ToTensor(),
        # 对RGB三个通道的像素值进行归一化，mean和std分别为CIFAR10的均值和标准差
        tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # 对测试集进行数据处理，不进行数据增强
    transform_test = tfs.Compose([
        tfs.Resize((image_size,image_size)),
        tfs.ToTensor(),
        tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    if train_batch_size is None:
        train_batch_size = 1
    if split is None:
        split = [1.0]
    train_set = CIFAR10(root, True, transform_train, download=True)
    valid_set = CIFAR10(root, False, transform_test, download=True)
    if distribute:
        train_set = distribute_dataset(train_set, split, rank, seed=seed)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=valid_batch_size, drop_last=True)
    return train_loader, valid_loader, (3, image_size, image_size), 10
