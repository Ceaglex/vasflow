export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$(pwd):$PYTHONPATH
export TOKENIZERS_PARALLELISM=false


: " For vae training
"



python\
    main.py \
    --base "./config/vaflow_sda_dit_noise_text_mel.yaml" \
    -f "_noise_sr16000" \
    -t True \
    -i False \
    --devices "0,1,2,3"