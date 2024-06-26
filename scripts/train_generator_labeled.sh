gpu=0
epochs=500
seed=1
ratio=0.01
sample=100
save=100
bs=64
lr=0.0002
sens=Male
target=Attractive
name=S\=$sens\&Y\=$target/$seed@$ratio-epochs$epochs-bs$bs-lr$lr

python train_labeled.py --sens $sens --target $target --batch_size $bs --save_interval $save --sample_interval $sample --seed $seed --ratio $ratio --img_size 128 --shortcut_layers 1 --inject_layers 1 --experiment_name $name --gpu $gpu --epochs $epochs
