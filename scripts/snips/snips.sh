#cd inverse
source ~/init_conda.sh
conda activate py36
python train.py  --do_test  --log_output_path ./prompt_data/MIT_M/prompt_MIT_M/mit_m.10_shot/log.txt --test_path ./prompt_data/snips/prompt_snips/ --test_file snips_test_1 --gen_batch_size=50 --num_finetune_epochs 10 --dev_path ./prompt_data/snips/prompt_snips/ --dev_file snips_dev_1 --train_path ./prompt_data/snips/prompt_snips/ --train_file snips_train_1 --dataset snips --pred_path ./pred/ --model_selection_path ./model_selection/