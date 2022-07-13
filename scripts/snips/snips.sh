#cd inverse
source ~/init_conda.sh
conda activate py36
python train.py  --do_test  --log_output_path ./log/log.txt --test_path ./prompt_data/snips/prompt_snips_shot_5/ --test_file sinps-test-1-shot-5 --gen_batch_size=50 --num_finetune_epochs 2 --dev_path ./prompt_data/snips/prompt_snips_shot_5/ --dev_file sinps-dev-1-shot-5 --train_path ./prompt_data/snips/prompt_snips_shot_5/ --train_file sinps-train-1-shot-5 --dataset snips --pred_path ./pred/ --model_selection_path ./model_selection/ --ft_model_output_dir ./ft_model/ --model_output_dir ./model/
