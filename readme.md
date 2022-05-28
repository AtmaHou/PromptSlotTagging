# Prompt Slot Tagging

This is the code of the Findings of ACL 2022 paper: [Inverse is Better! Fast and Accurate Prompt for Few-shot Slot Tagging](https://arxiv.org/pdf/2204.00885.pdf).

## Get Started

### Requirement
```
python >= 3.6.13
torch >= 1.10.2
transformers >= 4.10.2
```

### Step1: Prepare data and scripts
- Download prompted few-shot data at [download mit data](https://github.com/AtmaHou/PromptSlotTagging/releases/download/prompt_data/prompt_data.zip).
- You can also generate data using `original_data` and `utils`:
    - train data for snips using `preprocessor.py`,other data \(dev and test data\) using `rechecker_preprocessor.py`
    - change the path in `__main__`
- few-shot original data example：
```json
{
  "domain_name": [
    {  // episode
      "support": {  // support set
        "seq_ins": [["we", "are", "friends", "."], ["how", "are", "you", "?"]],  // input sequence
        "seq_outs": [["O", "O", "O", "O"], ["O", "O", "O", "O"]]  // output sequence in sequence labeling task
      },
      "query": {  // query set
        "seq_ins": [["we", "are", "friends", "."], ["how", "are", "you", "?"]],
        "seq_outs": [["O", "O", "O", "O"], ["O", "O", "O", "O"]]
      }
    },
    ...
  ],
  ...
}
```
- few-shot prompted data example：
```json
{
  "domain_name": [
    {  // episode
      "domain": "domain_name"
      "support": {  // support set
        "original_seq_in": [["Jack", "is", "my", "friend", "."], ["how", "are", "you", "?"]],  // input sequence
        "original_seq_out": [["B-name", "O", "O", "O", "O"], ["O", "O", "O", "O"]],  // output sequence in sequence labeling task
	"prompt_seq_in": [["'", "Jack", "is", "my", "friend", ".", "'", "name", "refers", "to"]...],
	"prompt_seq_out": [["'", "Jack", "is", "my", "friend", ".", "'", "name", "refers", "to", "Jack"]...],
	"checker_prompt_in": [["'", "Jack", "is", "my", "friend", ".", "'", "name", "refers", "to", "Jack", ".", "loaction", "refers", "to"]...],
	"checker_prompt_out": [["'", "Jack", "is", "my", "friend", ".", "'", "name", "refers", "to", "Jack", ".", "loaction", "refers", "to", "none"]...]
      },
      "query": {  // query set
      	"domain": "domain_name",
        "original_seq_in": [["Jack", "is", "my", "friend", "."], ["how", "are", "you", "?"]],  // input sequence
        "original_seq_out": [["O", "O", "O", "O"], ["O", "O", "O", "O"]],  // output sequence in sequence labeling task
	"prompt_seq_in": [["'", "Jack", "is", "my", "friend", ".", "'", "name", "refers", "to"]...],
	"prompt_seq_out": [["'", "Jack", "is", "my", "friend", ".", "'", "name", "refers", "to", "Jack"]...]
      }
    },
    ...
  ],
  ...
}
```
- For MIT data(In-domain): 
For example, `./prompt_data/MIT_M/prompt_MIT_M/mit_m.10_shot.json` is the path for prompted 10-shot MIT_movie data. 
    - Then you need to:
        - set `test_path` in `./scripts/mit/mit_m.sh` as `./prompt_data/MIT_M/prompt_MIT_M/` (the `/` at the end is needed).
        - set `test_file` in `./scripts/mit/mit_m.sh` as `mit_m.10_shot`.
        - set `data_set` in `./scripts/mit/mit_m.sh` as `mit`
        - mkdir pred
        - mkdir model_selection

- For SNIPS data(Meta-Source Transfer): 
For example, `./prompt_data/snips/prompt_snips/snips_train_2.json`, `./prompt_data/snips/prompt_snips/snips_dev_2.json` and `./prompt_data/snips/prompt_snips/snips_test_2.json` is the train, dev and test path for the 2nd of 7 domain of prompted 1-shot SNIPS data. (the last number of the file name is the domain number).
    - Then you need to:
        - set `train_path`, `dev_path` and `test_path` in `./scripts/snips/snips.sh` as `./prompt_data/snips/prompt_snips/` (the `/` at the end is needed).
        - set `train_file` in `./scripts/snips/snips.sh` as `snips_train_2`.
        - set `dev_file` in `./scripts/snips/snips.sh` as `snips_dev_2`.
        - set `test_file` in `./scripts/snips/snips.sh` as `snips_test_2`.
        - set `data_set` in `./scripts/snips/snips.sh` as `snips`
        - mkdir pred
        - mkdir model_selection

### Step2: Train and test the main model

##### Example for MIT:
```bash
source ./scripts/mit/mit_m.sh
```

```bash
source ./scripts/mit/mit_mm.sh
```

```bash
source ./scripts/mit/mit_r.sh
```

##### Example for SNIPS:
```bash
source ./scripts/snips/snips.sh
```  
Note: Each time re-run a setting you need to clear the corresponding pred path manaully.
## Project Architecture


- `inverse`:
    - `original_data`: original data to construct prompt data 
    - `scripts`: running scripts.
        - `mit`: running scripts for mit (In-domain).
            - `mit_m.sh`: running script for MIT-movie
            - `mit_mm.sh`: running script for MIT-movie-hard
            - `mit_r.sh`: running script for MIT-restaurant
        - `snips`: running scripts for SNIPS (Meta-Source Transfer).
            -  `snips.sh`: running script for SNIPS
    - `utils`: tools to construct prompt data 
    - `train.py`: the entry file of the whole project.
    - `opt.py`: the definition and default settings of the arguments.
    - `model.py`: modified GPT2 modeling for our generation.
    - `dataloader.py`: the definition of dataloader, which transfrom raw prompted data into model inputs.
    - `eval.py`: the eval function based on the outputs of the model.
    - `conlleval.pl`: the conlleval scripts from https://www.clips.uantwerpen.be/conll2000/chunking/conlleval.txt.
   

## License for code and data
Apache License 2.0

