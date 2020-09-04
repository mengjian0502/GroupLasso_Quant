PYTHON="/home/mengjian/anaconda3/envs/neurosim_test/bin/python3"

############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi

model=vgg7_quant
dataset=cifar10
epochs=200
batch_size=128
optimizer=SGD
col_size=16
group_ch=4
wbit=4
abit=4
mode=mean
k=2
ratio=0.7
wd=0.0005
lr=0.01

ub=0.001
lb=0.001
diff=0.00025
lamda_lin=0.001
push=False

pretrained_model="./save/vgg7_quant/vgg7_quant_w4_a4_mode_mean_k2_lambda_wd0.0002_swpFalse/model_best.pth.tar"
# pretrained_model="./save/resnet18/resnet18_quant_w4_a4_mode_mean_k2_lambda_wd0.0005_swpFalse/model_best.pth.tar"
# eval_model="./save/resnet18_quant_grp4/resnet18_quant_w4_a4_modemean_k2_lambda0.0025_ratio0.7_wd0.0005_lr0.01_swpFalse_groupch4_pushTrue_iter4000_g01/model_best.pth.tar"
# eval_model="./save/resnet18/resnet18_quant_w4_a4_modemean_k3_lambda0.002_ratio0.7_wd0.0005_lr0.01_swpTrue/model_best.pth.tar"


for lambda in $(seq ${lb} ${diff} ${ub})
do

save_path="./save/${model}_grp${group_ch}/${model}_w${wbit}_a${abit}_mode${mode}_k${k}_lambda${lambda}_lamda_lin${lamda_lin}_ratio${ratio}_wd${wd}_lr${lr}_swpFalse_groupch${group_ch}_push${push}_iter4000_g01_eval/"
log_file="${model}_w${wbit}_a${abit}_mode${mode}_k${k}_lambda${lambda}_lamda_lin${lamda_lin}_ratio${ratio}_wd${wd}_lr${lr}_swpFalse_groupch${group_ch}_push${push}_iter4000_g01_eval.log"

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
    --lamda_lin ${lamda_lin} \
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
    --col_size ${col_size}
done