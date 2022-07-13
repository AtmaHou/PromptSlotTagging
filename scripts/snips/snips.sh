#cd inverse
source ~/init_conda.sh
conda activate py36
python train.py  --do_train --do_dev --do_test  --log_output_path ./log/log.txt --test_path ./prompt_data/snips/prompt_snips/ --test_file snips_test_1 --gen_batch_size=50 --num_finetune_epochs 2 --dev_path ./prompt_data/snips/prompt_snips/ --dev_file snips_dev_1 --train_path ./prompt_data/snips/prompt_snips/ --train_file snips_train_1 --dataset snips --pred_path ./pred/ --model_selection_path ./model_selection/ --ft_model_output_dir ./ft_model/ --model_output_dir ./model/
