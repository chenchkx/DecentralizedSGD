

"""
A simple example to calculate the top eigenvectors for the hessian of
ResNet18 network for CIFAR-10
"""
import os
import argparse
import torch
# import skeletor
from datasets import load_dataset
from networks import load_model

from hessian_eigenthings import compute_hessian_eigenthings
nfs_dataset_path1 = '/mnt/nfs4-p1/ckx/datasets/'
nfs_dataset_path2 = '/nfs4-p1/ckx/datasets/'


def main(args):

    # check nfs dataset path
    if os.path.exists(nfs_dataset_path1):
        args.dataset_path = nfs_dataset_path1
    elif os.path.exists(nfs_dataset_path2):
        args.dataset_path = nfs_dataset_path2

    trainloader, testloader,  _, classes = load_dataset(root=args.dataset_path, name=args.dataset_name, image_size=args.image_size,
                                                                    train_batch_size=256, valid_batch_size=64)
    if args.fname:
        print("Loading model from %s" % args.fname)
        model = torch.load(args.fname, map_location="cpu").cuda()
    else:
        model = load_model('ResNet18', classes, pretrained=True)

    'Mar15_01:56:19_vipa-110_CIFAR10s56-512-csgd-fixed-16-ResNet18_M-1-0.8-0.0-0.1-0.0-60-6000-6000-666-False'
    'Mar15_01:56:31_vipa-110_CIFAR10s56-512-ring-fixed-16-ResNet18_M-1-0.8-0.0-0.1-0.0-60-6000-6000-666-False'
    'Mar15_09:56:57_vipa-111_CIFAR10s56-64-csgd-fixed-16-ResNet18_M-1-0.1-0.0-0.1-0.0-60-6000-6000-666-False'
    'Mar15_09:57:08_vipa-111_CIFAR10s56-64-ring-fixed-16-ResNet18_M-1-0.1-0.0-0.1-0.0-60-6000-6000-666-False'
    
    if args.pretrain_type == 1:
        model.load_state_dict(torch.load('logs_perf/CIFAR10/dict/Mar15_01:56:19_vipa-110_CIFAR10s56-512-csgd-fixed-16-ResNet18_M-1-0.8-0.0-0.1-0.0-60-6000-6000-666-False.t7')['state_dict'])
    elif args.pretrain_type == 2:
        model.load_state_dict(torch.load('logs_perf/CIFAR10/dict/Mar15_01:56:31_vipa-110_CIFAR10s56-512-ring-fixed-16-ResNet18_M-1-0.8-0.0-0.1-0.0-60-6000-6000-666-False.t7')['state_dict'])
    elif args.pretrain_type == 3:
        model.load_state_dict(torch.load('logs_perf/CIFAR10/dict/Mar15_09:56:57_vipa-111_CIFAR10s56-64-csgd-fixed-16-ResNet18_M-1-0.1-0.0-0.1-0.0-60-6000-6000-666-False.t7')['state_dict'])
    elif args.pretrain_type == 4:
        model.load_state_dict(torch.load('logs_perf/CIFAR10/dict/Mar15_09:57:08_vipa-111_CIFAR10s56-64-ring-fixed-16-ResNet18_M-1-0.1-0.0-0.1-0.0-60-6000-6000-666-False.t7')['state_dict'])
        
    criterion = torch.nn.CrossEntropyLoss()
    eigenvals, eigenvecs = compute_hessian_eigenthings(
        model,
        testloader,
        criterion,
        args.num_eigenthings,
        mode=args.mode,
        # power_iter_steps=args.num_steps,
        max_possible_gpu_samples=args.max_samples,
        # momentum=args.momentum,
        full_dataset=args.full_dataset,
        use_gpu=args.cuda,
    )
    print("Eigenvecs:")
    print(eigenvecs)
    print("Eigenvals:")
    print(eigenvals)
    # track.metric(iteration=0, eigenvals=eigenvals)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_eigenthings", default=5, type=int, help="number of eigenvals/vecs to compute")
    parser.add_argument("--dataset_path", type=str, default='datasets')
    parser.add_argument("--dataset_name", type=str, default='CIFAR10',
                                            choices=['CIFAR10','CIFAR100','TinyImageNet'])
    parser.add_argument("--image_size", type=int, default=56, help='input image size')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", default=16, type=int, help="test set batch size")
    parser.add_argument("--momentum", default=0.0, type=float, help="power iteration momentum term")
    parser.add_argument("--num_steps", default=50, type=int, help="number of power iter steps")
    parser.add_argument("--max_samples", default=2048, type=int)
    parser.add_argument("--cuda", action="store_true", help="if true, use CUDA/GPUs")
    parser.add_argument("--full_dataset", action="store_true", help="if true, loop over all batches in set for each gradient step",
    )
    parser.add_argument("--fname", default="", type=str)
    parser.add_argument("--mode", default='power_iter', type=str, choices=["power_iter", "lanczos"])
    parser.add_argument("--pretrain_type", default=1, type=int)
    args = parser.parse_args()

# python hessian_main.py --pretrain_type 1
    main(args)
    # skeletor.supply_args(extra_args)
    # skeletor.execute(main)