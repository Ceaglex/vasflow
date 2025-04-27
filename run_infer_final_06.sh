export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$(pwd):$PYTHONPATH
export TOKENIZERS_PARALLELISM=false


: " For vae training
"


# python main.py \
#     --base "./config/vae_beta.yaml" "./config/clip_infer.yaml" \
#     -f "_clip_infer" \
#     -t False \
#     -i True \
#     --devices 0,1,2,3


# python main.py \
#     --base "./config/vaflow_beta.yaml" "./config/vaflow_beta_infer.yaml" \
#     -f "_infer_on_val" \
#     -t False \
#     -i True \
#     --devices 0,


# python main.py \
#     --base "./config/vaflow_sda_dit.yaml" "./config/vaflow_sda_dit_infer.yaml" \
#     -f "_joint_tuned_e89_ablation" \
#     -t False \
#     -i True \
#     --devices 4,5,6,7 \
    # model.params.guidance_scale=3.0 \
    # model.params.sample_steps=50 \
    # model.params.sample_method="euler". # Other commonly used solvers are "dopri5", "midpoint" and "heun3"


# python main.py \
#     --base "./config/vaflow_sda_dit_cavp.yaml" "./config/vaflow_sda_dit_infer_ablation_cavp.yaml" \
#     -f "_cavp_joint_tuned_infer_on_test" \
#     -t False \
#     -i True \
#     --devices 1,2,3,4,5,6,7 \
#     model.params.vaflow_ckpt_path="./log/2025_02_21-10_27_28-vaflow_sda_dit_cavp_joint_tune_vae/ckpt/epoch=0049-step=1.43e+05.ckpt" 

# python main.py \
#     --base "./config/vaflow_sda_dit_noise.yaml" "./config/vaflow_sda_dit_infer_ablation.yaml" \
#     -f "_noise_joint_tuned_infer_on_test" \
#     -t False \
#     -i True \
#     --devices 1,2,3,4,5,6,7 \
#     model.params.vaflow_ckpt_path="./log/2025_02_18-23_57_33-vaflow_sda_dit_noise/ckpt/epoch=0074-step=1.71e+05.ckpt" 


# python main.py \
#     --base "./config/vaflow_sda_dit.yaml" "./config/vaflow_sda_dit_infer_ablation.yaml" \
#     -f "_full_joint_tuned_infer_on_test" \
#     -t False \
#     -i True \
#     --devices 1,2,3,4,5,6,7 \
#     model.params.vaflow_ckpt_path="./log/2025_02_08-11_17_40-vaflow_sda_dit_joint_tune_vae/ckpt/epoch=0059-step=1.45e+05.ckpt" 


# python main.py \
#     --base "./config/vaflow_sda_dit_uncond.yaml" "./config/vaflow_sda_dit_uncond_ag_infer.yaml" \
#     -f "_uncond_ag_infer_on_ablation_joint_tuned_e9" \
#     -t False \
#     -i True \
#     --devices 1,

# python main.py \
#     --base "./config/vaflow_sda_dit_custom_ddpm.yaml" "./config/vaflow_sda_dit_custom_ddpm_infer.yaml" \
#     -f "_ddpm_e79_dpm_test1" \
#     -t False \
#     -i True \
#     --devices 1,2,3


# python main.py \
#     --base "./config/vaflow_sda_dit_uncond.yaml" "./config/vaflow_sda_dit_uncond_infer.yaml" \
#     -f "_uncond_infer_on_test" \
#     -t False \
#     -i True \
#     --devices 1,

# python main.py \
#     --base "./config/vaflow_sda_dit_uncond.yaml" "./config/vaflow_sda_dit_uncond_ag_infer.yaml" \
#     -f "_uncond_infer_on_test_ag_w2_e1" \
#     -t False \
#     -i True \
#     --devices 1, \
#     model.params.vaflow_ckpt_path_ag="./log/2025_03_02-20_23_17-vaflow_sda_dit_uncond_joint_tune_vae_repeat/ckpt/epoch=0000-step=2.85e+03.ckpt"


# python main.py \
#     --base "./config/vaflow_sda_dit_uncond.yaml" "./config/vaflow_sda_dit_uncond_ag_infer.yaml" \
#     -f "_uncond_infer_on_test_ag_w2_e2" \
#     -t False \
#     -i True \
#     --devices 1, \
#     model.params.vaflow_ckpt_path_ag="./log/2025_03_02-20_23_17-vaflow_sda_dit_uncond_joint_tune_vae_repeat/ckpt/epoch=0001-step=5.71e+03.ckpt"

# python main.py \
#     --base "./config/vaflow_sda_dit_uncond.yaml" "./config/vaflow_sda_dit_uncond_ag_infer.yaml" \
#     -f "_uncond_infer_on_test_ag_w2_e3" \
#     -t False \
#     -i True \
#     --devices 1, \
#     model.params.vaflow_ckpt_path_ag="./log/2025_03_02-20_23_17-vaflow_sda_dit_uncond_joint_tune_vae_repeat/ckpt/epoch=0002-step=8.56e+03.ckpt"


# flow_ckpt_path_2="/root/workspace/vaflow/log/2025_03_02-20_23_17-vaflow_sda_dit_uncond_joint_tune_vae_repeat/ckpt"
# es=(7 8 9) 
# for epoch in ${es[@]}; do
#     # 查找对应的检查点
#     # ckpt=$(find ${flow_ckpt_path_2} -name "epoch=00[0-9]*${adjusted_epoch}*-step=*.ckpt" | sort | head -1)
#     ckpt=$(find ${flow_ckpt_path_2} -name "epoch=000${epoch}*-step=*.ckpt")
    
#     if [ -n "$ckpt" ]; then
#         echo "Running with checkpoint: $ckpt (epoch=$epoch)"
        
#         python main.py \
#             --base "./config/vaflow_sda_dit_uncond.yaml" "./config/vaflow_sda_dit_uncond_ag_infer.yaml" \
#             -f "_uncond_infer_on_test_ag_w2_e${epoch}" \
#             -t False \
#             -i True \
#             --devices 1,2,3 \
#             model.params.vaflow_ckpt_path_ag="$ckpt"
#     else
#         echo "Could not find checkpoint for adjusted epoch ${adjusted_epoch} in ${flow_ckpt_path_2}"
#     fi
# done


flow_ckpt_path_2="/root/workspace/vaflow/log/2025_02_13-00_52_05-vaflow_sda_dit_uncond/ckpt"
es=(14 24 34 44 54 64 74) 
for epoch in ${es[@]}; do
    # 查找对应的检查点
    # ckpt=$(find ${flow_ckpt_path_2} -name "epoch=00[0-9]*${adjusted_epoch}*-step=*.ckpt" | sort | head -1)
    ckpt=$(find ${flow_ckpt_path_2} -name "epoch=*${epoch}*-step=*.ckpt")
    
    if [ -n "$ckpt" ]; then
        echo "Running with checkpoint: $ckpt (epoch=$epoch)"
        
        python main.py \
            --base "./config/vaflow_sda_dit_uncond.yaml" "./config/vaflow_sda_dit_uncond_ag_infer.yaml" \
            -f "_uncond_infer_on_test_ag_w2_e${epoch}" \
            -t False \
            -i True \
            --devices 1,2,3 \
            model.params.vaflow_ckpt_path_ag="$ckpt"
    else
        echo "Could not find checkpoint for epoch ${epoch} in ${flow_ckpt_path_2}"
    fi
done

