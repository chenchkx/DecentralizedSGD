
# TinyImageNet training 100000
# 50000/(512*16) = 12.20
# 50000/(64*16)  = 96.16

## AlexNet
python main.py --dataset_name "TinyImageNet" --image_size 64 --batch_size 64 --mode "ring" --size 16 --lr 0.1 --model "AlexNet_M" --warmup_step 0 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
python main.py --dataset_name "TinyImageNet" --image_size 64 --batch_size 512 --mode "ring" --size 16 --lr 0.8 --model "AlexNet_M" --warmup_step 0 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0

# training with amp（automatic mixed precision）
python main.py --dataset_name "TinyImageNet" --image_size 64 --batch_size 64 --mode "ring" --size 16 --lr 0.1 --model "AlexNet_M" --warmup_step 0 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0 --amp
python main.py --dataset_name "TinyImageNet" --image_size 64 --batch_size 512 --mode "ring" --size 16 --lr 0.8 --model "AlexNet_M" --warmup_step 0 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0 --amp


## ResNet18
python main.py --dataset_name "TinyImageNet" --image_size 56 --batch_size 64 --mode "ring" --size 16 --lr 0.1 --model "ResNet18_M" --warmup_step 120 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0
python main.py --dataset_name "TinyImageNet" --image_size 56 --batch_size 512 --mode "ring" --size 16 --lr 0.8 --model "ResNet18_M" --warmup_step 120 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0

# training with amp（automatic mixed precision）
python main.py --dataset_name "TinyImageNet" --image_size 56 --batch_size 64 --mode "ring" --size 16 --lr 0.1 --model "ResNet18_M" --warmup_step 120 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0 --amp
python main.py --dataset_name "TinyImageNet" --image_size 56 --batch_size 512 --mode "ring" --size 16 --lr 0.8 --model "ResNet18_M" --warmup_step 120 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0 --amp

python main.py --dataset_name "TinyImageNet" --image_size 56 --batch_size 64 --mode "ring" --shuffle 'random' --size 16 --lr 0.1 --model "ResNet18_M" --warmup_step 120 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0 --amp
python main.py --dataset_name "TinyImageNet" --image_size 56 --batch_size 512 --mode "ring" --shuffle 'random' --size 16 --lr 0.8 --model "ResNet18_M" --warmup_step 120 --milestones 2400 4800 --early_stop 6000 --epoch 6000 --seed 666 --pretrained 1 --device 0 --amp