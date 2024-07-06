epochs=100
ratio=0.01
sample=100
save=100
bs=64
lr=0.0002
sens=Male
target=Attractive
threshold=0.9
seed=$1
gpu=$2
name=semi/S\=$sens\&Y\=$target/$seed@$ratio-epochs$epochs-bs$bs-lr$lr

labeled_epochs=500
path=output/labeled/S\=$sens\&Y\=$target/$seed@$ratio-epochs$labeled_epochs-bs$bs-lr$lr/checkpoint/generator.$labeled_epochs.pth

python train_semi.py --threshold $threshold --model_path $path --sens $sens --target $target --batch_size $bs --save_interval $save --sample_interval $sample --seed $seed --ratio $ratio --img_size 128 --shortcut_layers 1 --inject_layers 1 --experiment_name $name --gpu $gpu --epochs $epochs
