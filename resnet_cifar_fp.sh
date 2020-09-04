PYTHON="/home/mengjian/anaconda3/envs/neurosim_test/bin/python3"

############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi

model=resnet20_FP
dataset=cifar10
epochs=200
batch_size=128
optimizer=SGD

wbit=32
abit=32
mode=mean
k=2
ratio=0.7
wd=0.0005
lr=0.1

save_path="./save/${model}/${model}_w${wbit}_a${abit}_wd${wd}_swpFalse_GN/"
log_file="${model}_w${wbit}_a${abit}_wd${wd}_swpFalse_GN.log"
# pretrained_model="./save/resnet20/w4_a4_quant_baseline_gdrq/decay0.0005_w4_a4_baseline/model_best.pth.tar"

$PYTHON -W ignore train.py --dataset ${dataset} \
    --data_path ./dataset/ \
    --model ${model} \
    --save_path ${save_path} \
    --epochs ${epochs} \
    --log_file ${log_file} \
    --lr  ${lr} \
    --schedule 60 120 \
    --gammas 0.1 0.1 \
    --batch_size ${batch_size} \
    --ngpu 1 \
    --wd ${wd} \
    --wbit ${wbit} \
    --abit ${abit} \

    
