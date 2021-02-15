python ft_vgg.py \
--action 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1 \
--model=vgg16  \
--dataset=cifar10 \
--lr=0.01 \
--n_gpu=1 \
--batch_size=256 \
--n_worker=2 \
--lr_type=cos \
--n_epoch=200 \
--wd=4e-5 \
--seed=2020 
 