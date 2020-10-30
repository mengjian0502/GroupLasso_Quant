PYTHON="/home/mengjian/anaconda3/bin/python"

############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi

model=resnet20_quant
dataset=cifar10
epochs=200
batch_size=128
optimizer=SGD
col_size=16
group_ch=8
targeted_group=8
wbit=4
abit=4
mode=mean
k=2
ratio=0.5
wd=0.0005
lr=0.005

ub=0.002
lb=0.002
diff=0.0005
push=False

pretrained_model="./save/resnet20_quant_w4_a4_modemean_k_lambda_wd0.0005_swpFalse/model_best.pth.tar"

for lambda in $(seq ${lb} ${diff} ${ub})
do
save_path="./save/sparsity_analysis/${model}_grp${group_ch}/${model}_w${wbit}_a${abit}_mode${mode}_k${k}_lambda${lambda}_ratio${ratio}_wd${wd}_lr${lr}_swpFalse_groupch${group_ch}_push${push}_lr${lr}_global_mp_percentile${ratio}_optim_update1.0/"
log_file="${model}_w${wbit}_a${abit}_mode${mode}_k${k}_lambda${lambda}_ratio${ratio}_wd${wd}_lr${lr}_swpFalse_groupch${group_ch}_push${push}_lr${lr}_global_mp_percentile${ratio}_optim_update1.0.log"

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
    --lamda ${lambda} \
    --ratio ${ratio} \
    --clp \
    --a_lambda 0.01 \
    --k ${k} \
    --swp \
    --group_ch ${group_ch} \
    --targeted_group ${targeted_group} \
    --g_multi 2 2 1 \
    --resume ${pretrained_model} \
    --fine_tune \
    --wbit ${wbit} \
    --abit ${abit} \
    --col_size ${col_size} \
    --lr_scale;
done
