PYTHON="/home/mengjian/anaconda3/envs/neurosim_test/bin/python3"

############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi

model=resnet20_quant_eval
dataset=cifar10
epochs=200
batch_size=128
optimizer=SGD
group_ch=16
wbit=4
abit=4
mode=mean
k=2
ratio=0.7
wd=0.0005
lr=0.01

ub=0.001
lb=0.001
diff=0.001

col_size=16
cellBit=2
adc_prec=6


save_path="./save/${model}/resnet20_quant_w4_a4_modemean_k2_lambda0.0020_ratio0.7_wd0.0005_lr0.01_swpFalse_groupch8_pushFalse_iter4000_g01_eval/"
log_file="${model}_w${wbit}_a${abit}_mode${mode}_k${k}_groupch${group_ch}_colsize${col_size}_cellBit${cellBit}_adc_prec${adc_prec}.log"
pretrained_model="./save/resnet20/resnet20_quant_w4_a4_modemean_k_lambda_wd0.0005_swpFalse/model_best.pth.tar"
# pretrained_model="./save/resnet18/resnet18_quant_w4_a4_mode_mean_k2_lambda_wd0.0005_swpFalse/model_best.pth.tar"
# eval_model="./save/resnet18_quant_grp16/resnet18_quant_w4_a4_modemean_k3_lambda0.002_ratio0.7_wd0.0005_lr0.01_swpTrue/model_best.pth.tar"

$PYTHON -W ignore inference.py --dataset ${dataset} \
    --data_path ./dataset/ \
    --model ${model} \
    --save_path ${save_path} \
    --batch_size ${batch_size} \
    --q_mode ${mode} \
    --log_file ${log_file} \
    --ngpu 1 \
    --k ${k} \
    --group_ch ${group_ch} \
    --resume ${pretrained_model} \
    --fine_tune \
    --wbit ${wbit} \
    --abit ${abit} \
    --col_size ${col_size} \
    --cellBit ${cellBit} \
    --adc_prec ${adc_prec} \
    --evaluate \
    
