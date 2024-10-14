lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-1B \
    --tasks mmlu \
    --num_fewshot 5 \
    --device cuda:0 \
    --batch_size auto \
    --output_path results \


lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-1B \
    --tasks drop \
    --num_fewshot 3 \
    --device cuda:0 \
    --batch_size auto \
    --output_path results \