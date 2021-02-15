python amc_search_vgg.py \
    --job=train \
    --suffix=amc_vgg16_30 \
    --model=vgg16 \
    --dataset=cifar10 \
    --preserve_ratio=0.3 \
    --lbound=0.2 \
    --rbound=1 \
    --reward=acc_reward \
    --data_root=C:\\Users\\lenovo\\dataset\\cifar \
    --ckpt_path=C:\\Users\\lenovo\\Desktop\\cacp\\amc_vgg\\checkpoints\\vgg16_cifar10.pt \
    --seed=2020 \
    --rmsize=200 \
    --bsize=64 

