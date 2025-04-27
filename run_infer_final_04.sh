export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$(pwd):$PYTHONPATH
export TOKENIZERS_PARALLELISM=false


: " For final infer
"


# Base

# python main.py \
#     --base "./config/vaflow_sda_dit_base.yaml" "./config/vaflow_sda_dit_infer.yaml" \
#     -f "_base_jt_e49_5dopri5_final_infer_on_test_x1" \
#     -t False \
#     -i True \
#     --devices 5,6,7 \
#     model.params.guidance_scale=5.0 \
#     model.params.sample_method=dopri5 \
#     model.params.vaflow_ckpt_path="./log/2025_02_16-00_28_09-vaflow_sda_dit_base_joint_tune_vae/ckpt/epoch=0049-step=1.43e+05.ckpt" 



# CAVP

# python main.py \
#     --base "./config/vaflow_sda_dit_cavp.yaml" "./config/vaflow_sda_dit_infer_ablation_cavp.yaml" \
#     -f "_cavp_e49_5dopri5_final_infer_on_test_x1" \
#     -t False \
#     -i True \
#     --devices 4,5,6,7 \
#     model.params.guidance_scale=5.0 \
#     model.params.sample_method=dopri5 \
#     model.params.vaflow_ckpt_path="./log/2025_02_21-10_27_28-vaflow_sda_dit_cavp_joint_tune_vae/ckpt/epoch=0049-step=1.43e+05.ckpt" 


# # VidTok

# python main.py \
#     --base "./config/vaflow_sda_dit_vidtok.yaml" "./config/vaflow_sda_dit_infer_ablation_vidtok.yaml" \
#     -f "_vidtok_e44_5dopri5_final_infer_on_test_x1" \
#     -t False \
#     -i True \
#     --devices 4,5,6,7 \
#     model.params.guidance_scale=5.0 \
#     model.params.sample_method=dopri5 \
#     model.params.vaflow_ckpt_path="./log/2025_02_25-00_17_17-vaflow_sda_dit_vidtok_joint_tune_vae/ckpt/epoch=0044-step=1.28e+05.ckpt" 


# # Uncond ag

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


# Full 0/1 guidance

# python main.py \
#     --base "./config/vaflow_sda_dit_full.yaml" "./config/vaflow_sda_dit_infer.yaml" \
#     -f "_fullraw_jt_e49_0dopri5_final_infer_on_test_x1" \
#     -t False \
#     -i True \
#     --devices 4,5,6,7 \
#     model.params.guidance_scale=0.0 \
#     model.params.sample_method=dopri5 \
#     model.params.vaflow_ckpt_path="./log/2025_02_08-11_17_40-vaflow_sda_dit_joint_tune_vae/ckpt/epoch=0049-step=1.21e+05.ckpt" 


# python main.py \
#     --base "./config/vaflow_sda_dit_full.yaml" "./config/vaflow_sda_dit_infer.yaml" \
#     -f "_fullraw_jt_e49_1dopri5_final_infer_on_test_x1" \
#     -t False \
#     -i True \
#     --devices 4,5,6,7 \
#     model.params.guidance_scale=1.0 \
#     model.params.sample_method=dopri5 \
#     model.params.vaflow_ckpt_path="./log/2025_02_08-11_17_40-vaflow_sda_dit_joint_tune_vae/ckpt/epoch=0049-step=1.21e+05.ckpt" 
