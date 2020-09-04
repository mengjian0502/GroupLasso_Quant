PYTHON="/home/mengjian/anaconda3/envs/neurosim_test/bin/python3"

############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi

model=resnet18_quant
dataset=cifar10
epochs=200
batch_size=128
optimizer=SGD
group_ch=16
wbit=2
abit=2
mode=sawb
k=2
ratio=0.7
wd=0.0005
lr=0.1

save_path="./save/resnet18/${model}_w${wbit}_a${abit}_mode_${mode}_k${k}_lambda${ub}_wd${wd}_swpFalse_tmp/"
log_file="${model}_w${wbit}_a${abit}_mode${mode}_k${k}_lambda${ub}_wd${wd}_swpFalse_tmp.log"
eval_model="./save/resnet18/resnet18_quant_w2_a2_mode_sawb_k2_lambda_wd0.0005_swpFalse/model_best.pth.tar"
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
    --k ${k} \
    --clp \
    --a_lambda 0.01 \
    --group_ch ${group_ch} \
    --wbit ${wbit} \
    --abit ${abit} \
    --resume ${eval_model} \
    --fine_tune \
    --evaluate \

    
