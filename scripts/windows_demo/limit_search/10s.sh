python amc_search_vgg.py \
    --job=train \
    --suffix=amc_vgg16_10 \
    --model=vgg16 \
    --dataset=cifar10 \
    --preserve_ratio=0.1 \
    --lbound=0.1 \
    --rbound=1 \
    --reward=acc_reward \
    --data_root=C:\\Users\\lenovo\\dataset\\cifar \
    --ckpt_path=C:\\Users\\lenovo\\Desktop\\cacp\\amc_vgg\\checkpoints\\vgg16_cifar10.pt \
    --seed=2020 \
    --warmup=50\
    --rmsize=50 \
    --bsize=64 \
    --train_episode=150 

