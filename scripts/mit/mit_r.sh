#cd inverse
source ~/init_conda.sh
conda activate py36
python train.py --do_test --log_output_path ./prompt_data/MIT_R/prompt_MIT_R/mit_r.10_shot/log.txt --test_path ./prompt_data/MIT_R/prompt_MIT_R/ --test_file mit_r.10_shot --gen_batch_size=50 --num_finetune_epochs 4 --dataset mit --pred_path ./pred/ --model_selection_path ./model_selection/
