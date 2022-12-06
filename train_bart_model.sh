python run_summarization.py \
    --model_name_or_path facebook/bart-base \
    --do_train \
    --do_eval \
    --evaluation_strategy="steps" \
    --eval_steps=500 \
    --train_file en.train.csv \
    --validation_file en.test.csv \
    --output_dir ./models/bart-base-en-mix/ \
    --overwrite_output_dir \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=32 \
    --gradient_accumulation_steps=32 \
    --learning_rate="4e-4" \
    --num_train_epochs=3 \
    --predict_with_generate \
	--logging_steps="10" \
    --save_total_limit="2" \
    --max_target_length=1024 \
    --max_source_length=1024 \
    --fp16

exit

python run_summarization.py \
    --model_name_or_path facebook/bart-base \
    --do_train \
    --do_eval \
    --train_file en.train.csv \
    --validation_file en.test.csv \
    --output_dir ./models/bart-base-spelling-en-repro/ \
    --overwrite_output_dir \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=4 \
    --gradient_accumulation_steps=8 \
    --learning_rate="3e-4" \
    --num_train_epochs="1" \
    --predict_with_generate \
	--logging_steps="10" \
    --save_total_limit="2" \
    --max_target_length=1024 \
    --max_source_length=1024

exit()


python run_summarization.py \
    --model_name_or_path facebook/bart-base \
    --do_train \
    --do_eval \
    --train_file en.all.train.csv \
    --validation_file en.all.test.csv\
    --output_dir ./models/bart-base-all-spelling-en/ \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=16 \
    --gradient_accumulation_steps=8 \
    --learning_rate="3e-4" \
    --num_train_epochs=2 \
    --predict_with_generate \
	--logging_steps="10" \
    --save_total_limit="2" \
    --max_target_length=1024 \
    --max_source_length=1024