import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train',action='store_true',help='Whether to train')
    parser.add_argument('--do_test',action='store_true',help='Whether to test')
    parser.add_argument('--do_dev',action='store_true',help='Whether to dev')
    parser.add_argument("--log_output_path", default='/users5/chencheng/prompt_for_mit/mit_data/MIT_R/prompt_MIT_R/mit_r.20_shot/log.txt', type=str,
                        help="The output directory where the log files will be written.")
    parser.add_argument("--model_output_dir", default='/users5/chencheng/prompt_for_mit/model/', type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--ft_model_output_dir", default='/users5/chencheng/prompt_for_mit/ftmodel/', type=str,
                        help="The output directory where the ft_model checkpoints will be written.")
    parser.add_argument('--train_path',default='/users5/chencheng/prompt_for_mit/mit_data/MIT_R/prompt_MIT_R/')
    parser.add_argument('--dev_path',default='')
    parser.add_argument('--model_selection_path',default='/users5/chencheng/prompt_for_mit/model_selection/')
    parser.add_argument('--test_path',default='/users5/chencheng/prompt_for_mit/mit_data/MIT_R/prompt_MIT_R/')
    parser.add_argument('--pred_path',default='')
    parser.add_argument('--test_file',default='mit_r.20_shot')
    parser.add_argument('--train_file',default='')
    parser.add_argument('--dev_file',default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--num_finetune_epochs', type=int, default=2)
    parser.add_argument('--train_batch_size', type=int, default=16,
                        help="the number of the cluster per training batch")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--max_finetune_steps", default=-1, type=int,
                        help="If > 0: set total number of finetune steps to perform. Override num_finetune_epochs.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument("--warmup_rate", default=0.1, type=float,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--fp16', action='store_true', help='fp16')
    parser.add_argument('--n_save_per_epoch', type=int, default=1)
    parser.add_argument('--n_save_epochs', type=int, default=1)
    parser.add_argument('--n_valid', type=int, default=374)
    parser.add_argument('--n_gpu',type=int, default=1)
    parser.add_argument('--gen_length', type=int, default=40,
                        help="max length of the generation utterances")
    parser.add_argument('--gen_batch_size',type=int,default=16)
    parser.add_argument('--gen_mode', type=str, default='sample', choices=["sample", "greed"],
                        help="generate each token in greed or sample mode")
    parser.add_argument("--wo_pretrained", action='store_true',
                        help='training with the random initialized model')
    parser.add_argument('--wo_pretrained_layer', type=int, default=2)
    parser.add_argument('--load_model_dir',type=str,default='')
    parser.add_argument('--dataset', default='mit', choices=['mit', 'snips'],
                        help='Specify which of the dataset is used, snips for the cross-domain and mit for the in-domain.')

    args = parser.parse_args()
    return args