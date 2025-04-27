export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$(pwd):$PYTHONPATH
export TOKENIZERS_PARALLELISM=false



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

