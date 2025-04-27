export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$(pwd):$PYTHONPATH
export TOKENIZERS_PARALLELISM=false


: " For final infer
"

# Uncond ag

# python main.py \
#     --base "./config/vaflow_sda_dit_uncond.yaml" "./config/vaflow_sda_dit_uncond_ag_infer.yaml" \
#     -f "_uncond_e59_e6ag_2dopri5_final_infer_on_test_x1" \
#     -t False \
#     -i True \
#     --devices 4,5,6,7 \
#     model.params.guidance_scale=2.0 \
#     model.params.sample_method=dopri5 \
#     model.params.vaflow_ckpt_path="./log/2025_02_18-13_18_10-vaflow_sda_dit_uncond_joint_tune_vae/ckpt/epoch=0059-step=1.71e+05.ckpt" \
#     model.params.vaflow_ckpt_path_ag="./log/2025_03_02-20_23_17-vaflow_sda_dit_uncond_joint_tune_vae_repeat/ckpt/epoch=0006-step=2.00e+04.ckpt"


# Noise

# python main.py \
#     --base "./config/vaflow_sda_dit_noise.yaml" "./config/vaflow_sda_dit_infer.yaml" \
#     -f "_noise_e94_5dopri5_final_infer_on_test_x1" \
#     -t False \
#     -i True \
#     --devices 4,5,6,7 \
#     model.params.guidance_scale=5.0 \
#     model.params.sample_method=dopri5 \
#     model.params.vaflow_ckpt_path="./log/2025_02_18-23_57_33-vaflow_sda_dit_noise/ckpt/epoch=0094-step=2.17e+05.ckpt" 


# VidTok

# python main.py \
#     --base "./config/vaflow_sda_dit_vidtok.yaml" "./config/vaflow_sda_dit_infer_ablation_vidtok.yaml" \
#     -f "_vidtok_e44_5dopri5_final_infer_on_test_x1" \
#     -t False \
#     -i True \
#     --devices 4,5,6,7 \
#     model.params.guidance_scale=5.0 \
#     model.params.sample_method=dopri5 \
#     model.params.vaflow_ckpt_path="./log/2025_02_25-00_17_17-vaflow_sda_dit_vidtok_joint_tune_vae/ckpt/epoch=0044-step=1.28e+05.ckpt" 




