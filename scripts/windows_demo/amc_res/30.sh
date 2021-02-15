python ft_vgg.py \
--action 0.75,0.625,1.0,0.25,0.90625,0.34375,0.90625,0.421875,0.625,0.203125,0.765625,0.203125,0.40625 \
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
 