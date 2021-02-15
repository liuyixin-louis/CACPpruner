python amc_search_vgg.py \
    --job=train \
    --suffix=amc_vgg16_search_10 \
    --model=vgg16 \
    --dataset=cifar10 \
    --preserve_ratio=0.1 \
    --lbound=0.2 \
    --rbound=1 \
    --reward=acc_reward \
    --data_root=C:\\Users\\lenovo\\dataset\\cifar \
    --ckpt_path=C:\\Users\\lenovo\\Desktop\\cacp\\amc_vgg\\checkpoints\\vgg16_cifar10.pt \
    --seed=2020 \
    --warmup=300 \
    --rmsize=200  \
    --bsize=64 \
    --train_episode=800 

