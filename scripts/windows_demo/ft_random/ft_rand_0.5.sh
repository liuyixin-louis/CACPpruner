python ft_vgg.py \
--action 0.75,0.375,0.8125,1.0,0.8125,0.53125,0.71875,0.875,0.296875,0.59375,0.921875,0.265625,0.203125 \
--model=vgg16  \
--dataset=cifar10 \
--lr=0.01 \
--n_gpu=1 \
--batch_size=256 \
--n_worker=0 \
--lr_type=cos \
--n_epoch=200 \
--wd=4e-5 \
--seed=2020 
 