python run_summarization.py \
    --model_name_or_path facebook/mbart-large-50 \
    --do_train \
    --do_eval \
    --train_file de.train.csv \
    --validation_file de.test.csv \
    --output_dir ./models/mbart-large-50-spelling-de/ \
    --overwrite_output_dir \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --gradient_accumulation_steps=4 \
    --learning_rate="3e-4" \
    --warmup_ratio 0.1 \
    --predict_with_generate \
	--logging_steps="10" \
    --save_total_limit="2" \
    --max_target_length=1024 \
    --max_source_length=1024 \
    --lang="de"

# facebook/mbart-large-50