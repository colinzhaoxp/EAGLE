# training script
# accelerate launch -m --mixed_precision=bf16 --num_processes=1 --dynamo_backend=inductor --dynamo_mode=reduce-overhead eagle.train.main --epoch 1 --basepath pretrain/vicuna-7b-v1.3 --tmpdir ./generated_data --cpdir ./output_model --configpath ./eagle/train/vicuna_7B_config.json 2&>1 | tee train.log

# accelerate launch -m --mixed_precision=bf16 eagle.train.main --epoch 1 --basepath pretrain/vicuna-7b-v1.3 --tmpdir ./generated_data --cpdir ./output_model --configpath ./eagle/train/vicuna_7B_config.json


# evaluation script
python -m eagle.evaluation.gen_ea_answer_vicuna --ea-model-path pretrain/EAGLE-Vicuna-7B-v1.3 --base-model-path pretrain/vicuna-7b-v1.3 --model-id vicuna-7b

# python -m eagle.evaluation.gen_baseline_answer_vicuna --ea-model-path output_model/state_20 --base-model-path pretrain/vicuna-7b-v1.3

# python -m eagle.evaluation.speed --model-name pretrain/vicuna-7b-v1.3 --jsonl-file mt_bench/ess-vicuna-70b-fp16-temperature-1.0.jsonl --jsonl-file-base mt_bench/ess-vicuna-70b-fp16-baseline-temperature-1.0.jsonl