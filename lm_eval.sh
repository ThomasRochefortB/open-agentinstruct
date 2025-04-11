# lm_eval --model hf \
#     --model_args pretrained=HuggingFaceTB/SmolLM2-135M-Instruct \
#     --tasks mmlu \
#     --device cuda:0 \
#     --batch_size 16

# lm_eval --model hf \
#     --model_args pretrained=HuggingFaceTB/SmolLM2-135M-Instruct \
#     --tasks mmlu \
#     --device cuda:0 \
#     --batch_size 16

lm_eval --model hf \
    --model_args pretrained=./finetuned_models \
    --tasks bbh \
    --device cuda:0 \
    --batch_size auto