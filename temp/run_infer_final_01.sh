export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$(pwd):$PYTHONPATH
export TOKENIZERS_PARALLELISM=false


: " For final infer
"


python main.py \
    --base "./config/vaflow_sda_dit_small.yaml" "./config/vaflow_sda_dit_infer.yaml" \
    -f "_small_jt_e49_5dopri5_final_infer_on_test_x1" \
    -t False \
    -i True \
    --devices 4, \
    model.params.guidance_scale=5.0 \
    model.params.sample_method=dopri5 \
    model.params.vaflow_ckpt_path="./log/2025_02_16-00_32_00-vaflow_sda_dit_small_joint_tune_vae/ckpt/epoch=0049-step=1.43e+05.ckpt" 



