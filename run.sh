model="llama2-7b-chat-hf"
method="low_rank"
type="unstructured"
suffix="weightonly"
save_dir="out/$model/$type/${method}_${suffix}/align/"

python main_low_rank.py \
    --model $model \
    --prune_method $method \
    --prune_data align \
    --rank 10 \
    --top_remove \
    --save $save_dir \
    --eval_zero_shot \
    --eval_attack \
    --save_attack_res 