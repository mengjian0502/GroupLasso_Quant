PYTHON="/home/mengjian/anaconda3/envs/neurosim_test/bin/python3"

############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi

model=resnet20_W2_quant
dataset=cifar10
epochs=200
batch_size=128
optimizer=SGD
group_ch=16
wbit=2
abit=2
mode=mean
k=2
ratio=0.7
wd=0.0005
lr=0.01

ub=0.001
lb=0.001
diff=0.001
gamma=0.1

save_path="./save/${model}/${model}_w${wbit}_a${abit}_mode${mode}_k${k}_lambda${ub}_ratio${ratio}_wd${wd}_lr${lr}_swpFalse_groupch${group_ch}_push${push}_gamma${gamma}/"
log_file="${model}_w${wbit}_a${abit}_mode${mode}_k${k}_lambda${ub}_ratio${ratio}_wd${wd}_lr${lr}_swpFalse_groupch${group_ch}_push${push}_${gamma}.log"
# pretrained_model="./save/resnet20_W2_quant/resnet20_W2_quant_w2_a2_mode_mean_k2_lambda_wd0.0005_swpFalse/resnet20_W2_quant_w2_a2_modemean_k2_lambda_wd0.0005_swpFalse.log"
pretrained_model="./save/resnet20_FP/resnet20_FP_w32_a32_wd0.0005_swpFalse/model_best.pth.tar"

$PYTHON -W ignore train.py --dataset ${dataset} \
    --data_path ./dataset/ \
    --model ${model} \
    --save_path ${save_path} \
    --lr  ${lr} \
    --schedule 80 120 160 \
    --gammas 0.1 0.1 0.5 \
    --epochs ${epochs} \
    --batch_size ${batch_size} \
    --q_mode ${mode} \
    --log_file ${log_file} \
    --ngpu 1 \
    --wd ${wd} \
    --lamda ${ub} \
    --ratio ${ratio} \
    --clp \
    --a_lambda 0.01 \
    --k ${k} \
    --swp \
    --group_ch ${group_ch} \
    --resume ${pretrained_model} \
    --fine_tune \
    --wbit ${wbit} \
    --abit ${abit} \
    --gamma ${gamma} \
    # --evaluate \
    
