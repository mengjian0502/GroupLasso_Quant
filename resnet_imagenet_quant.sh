PYTHON="/home/li/.conda/envs/pytorch/bin/python"

############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi

model=resnet18_imagenet_quant
dataset=imagenet
data_path="/opt/imagenet/imagenet_compressed"
epochs=110
batch_size=256
group_ch=16
wbit=4
abit=4
mode=mean
k=4
ratio=0.7
wd=0.0001
lr=0.1

save_path="./save/resnet18_imagenet/${model}_w${wbit}_a${abit}_mode_${mode}_k${k}_lambda${ub}_wd${wd}_swpFalse_tmp/"
log_file="${model}_w${wbit}_a${abit}_mode${mode}_k${k}_lambda${ub}_wd${wd}_swpFalse_tmp.log"
# pretrained_model="./save/resnet20/w4_a4_quant_baseline_gdrq/decay0.0005_w4_a4_baseline/model_best.pth.tar"

$PYTHON -W ignore train.py --dataset ${dataset} \
    --data_path ${data_path} \
    --model ${model} \
    --save_path ${save_path} \
    --epochs ${epochs} \
    --log_file ${log_file} \
    --lr  ${lr} \
    --schedule 30 60 85 95 \
    --gammas 0.1 0.1 0.1 0.1 \
    --batch_size ${batch_size} \
    --ngpu 4 \
    --wd ${wd} \
    --k ${k} \
    --clp \
    --a_lambda 0.01 \
    --group_ch ${group_ch} \
    --wbit ${wbit} \
    --abit ${abit} \

    
