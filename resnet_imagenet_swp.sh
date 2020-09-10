PYTHON="/home/ubuntu/anaconda3/envs/neurosim_test/bin/python"

############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi

model=resnet18_imagenet_quant
dataset=imagenet
data_path="~/imagenet/"
epochs=60
batch_size=256
group_ch=16
wbit=4
abit=4
mode=mean
lambda=0.00015
k=4
ratio=0.3
wd=0.0001
lr=0.01

save_path="./save/resnet18_imagenet/${model}_w${wbit}_a${abit}_mode_${mode}_k${k}_lambda${ub}_wd${wd}_lambda${lambda}_ratio${ratio}_swpTrue/"
log_file="${model}_w${wbit}_a${abit}_mode${mode}_k${k}_lambda${ub}_wd${wd}_lambda${lambda}_ratio${ratio}_swpTrue.log"
pretrained_model="./save/resnet18_imagenet/resnet18_imagenet_quant_w4_a4_mode_mean_k4_lambda_wd0.0001_swpFalse/model_best.pth.tar"

export CUDA_VISIBLE_DEVICES=4,5,6,7

$PYTHON -W ignore train.py --dataset ${dataset} \
    --data_path ${data_path} \
    --model ${model} \
    --save_path ${save_path} \
    --epochs ${epochs} \
    --log_file ${log_file} \
    --lr  ${lr} \
    --schedule 30 40 45 \
    --gammas 0.1 0.1 0.1 \
    --batch_size ${batch_size} \
    --q_mode ${mode} \
    --ngpu 4 \
    --wd ${wd} \
    --lamda ${lambda} \
    --ratio ${ratio} \
    --k ${k} \
    --clp \
    --swp \
    --a_lambda 0.01 \
    --group_ch ${group_ch} \
    --wbit ${wbit} \
    --abit ${abit} \
    --resume ${pretrained_model} \
    --fine_tune \
