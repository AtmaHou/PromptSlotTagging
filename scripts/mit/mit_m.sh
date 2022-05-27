cd inverse
source ~/init_conda.sh
conda activate py36
python train.py --do_test --log_output_path ./pred/mit_m.10_shot/log.txt --test_path ./prompt_data/MIT_M/prompt_MIT_M/ --test_file mit_m.10_shot --gen_batch_size=50 --num_finetune_epochs 2 --dataset mit --pred_path ./pred/ --model_selection_path ./model_selection/
