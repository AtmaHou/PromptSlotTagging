#cd inverse
source ~/init_conda.sh
conda activate py36
python train.py --do_test --log_output_path ./prompt_data/MIT_MM/prompt_MIT_MM/mir_mm.10_shot/log.txt --test_path ./prompt_data/MIT_MM/prompt_MIT_MM/ --test_file mit_mm.10_shot --gen_batch_size=50 --num_finetune_epochs 4 --dataset mit --pred_path ./pred/ --model_selection_path ./model_selection/
