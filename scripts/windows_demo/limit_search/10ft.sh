python ft_vgg.py \
--action 0.25,0.5,0.5625,0.1875,0.9375,0.25,0.15625,0.109375,0.109375,0.28125,0.109375,0.109375,0.703125 \
--model=vgg16  \
--dataset=cifar10 \
--lr=0.01 \
--n_gpu=1 \
--batch_size=256 \
--n_worker=0 \
--lr_type=cos \
--n_epoch=150 \
--wd=4e-5 \
--seed=2020 
 