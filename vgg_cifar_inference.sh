PYTHON="/home/mengjian/anaconda3/envs/neurosim_test/bin/python3"

############ directory to save result #############

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./dataset
fi

model=vgg7_quant_eval
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

cellBit=2
adc_prec=6

pretrained_model="./save/vgg7_quant/vgg7_quant_w4_a4_mode_mean_k2_lambda_wd0.0002_swpFalse/model_best.pth.tar"
# eval_model="./save/vgg7_quant_grp4/vgg7_quant_w4_a4_modemean_k2_lambda0.00050_ratio0.7_wd0.0005_lr0.01_swpFalse_groupch4_pushFalse_iter4000_g01_eval/checkpoint.pth.tar"

save_path="./save/vgg7_quant/vgg7_quant_w4_a4_mode_mean_k2_lambda_wd0.0002_swpFalse_eval/"
log_file="vgg7_quant_w4_a4_mode_mean_k2_lambda_wd0.0002_swpFalse_eval.log"

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