# Exploring Task Selection for Intermediate-Task Transfer Learning

| [Installation](#installation) | [Prompt Transfer](#prompttransfer) | [Task Selection](#taskselection) | [Datasets](#datasets) |

This repository contains the implementation of [Exploring Task Selection for Intermediate-Task Transfer Learning](https://drive.google.com/file/d/1xzOWiR1Psu62UOTaPGpvB6r7S9eiaP3k/view?usp=sharing) with prompt tuning and prompt transfer.

In this repository, we explore the effectiveness of task selection approaches based on prompt transfer. We first train soft prompts with a frozen T5 model, then continually train the prompt weight on resource-constrained tasks. We investiage task selection methods including data size, text embedding, prompt-based task embedding, and further extend task embedding on prompt weight  to max-pairwise similarity (MAX).

Our code is run by submitting job via HTCondor. 
Please specify the following scrips in your submit file. Otherwise, you can comment out the lines for job submission setup in those bash script. 
For instruction of HTConder, please read the [documents](https://repos.lsv.uni-saarland.de/mmosbach/htcondor-test/-/tree/master). 


## Installation

### Python Version

* Python >= 3.8

### Environment

Create an environment from file and activate the environment.

```
conda create -n intermediate-task-selection
conda activate intermediate-task-selection
```

Install denpendencies

```bash
. install_dependencies.sh
```

We suggest to install editable mode. This works for most of case. 

```bash
pip install -e .
```

## Datasets

We following 23 tasks, paired with their respective evaluation metrics across NLI, 
paraphrase detection, semantic similarity, question answering, reading comprehension, Grammatical acceptability, Word sense disambiguation, Sentiment analysis, coreference resolution.
We split as the datasets with less than 1K training samples as target task and 10 datasets as target tasks.
To run the following task, please use the provided names below:

### Source Tasks

We consider the dataset with richer annotation number as source tasks.
We train 13 source task, then initalized these pre-trained prompt weight for contually training on target tasks. We apply learning rate of 5e-1 for all source task.


| Dataset            | Metrics                        |
| ------------------ | ------------------------------ |
| mnli               | accuracy                       |
| mnli_mismatched    | accuracy                       |
| mnli_matched       | accuracy                       |
| qqp                | accuracy, f1                   |
| qnli               | accuracy                       |
| superglue-record   | f1, em                         |
| cxc                | pearson, spearmanr             |
| squad              | f1, em                         |
| drop               | f1, em                         |
| sst2               | accuracy                       |
| winogrande         | accuracy                       |
| hellaswag          | accuracy                       |
| superglue-multirc  | f1, em                         |
| cosmosqa           | accuracy                       |
| race               | accuracy                       |


### Target Tasks

We prompt tune 10 target task as baseline and transfer 13 source prompt to each target task. We apply learning rate of 2.


| Dataset            | Metrics                        |
| ------------------ | ------------------------------ |
| boolq              | accuracy                       |
| cola               | matthews_correlation           |
| stsb               | pearson, spearmanr             |
| superglue-wic      | accuracy                       |
| cr                 | accuracy                       |
| mrpc               | accuracy, f1                   |
| rte                | accuracy                       |
| superglue-wsc      | accuracy                       |
| superglue-copa     | accuracy                       |
| cb                 | f1_multiclass, accuracy        |




## Prompt Transfer

This section performs intermediate task transfer with prompt tuning. 
This involes (1) prompt tuning initialized from the sampled embedding's token and prompt transfer initialzied from pretrained prompt on source task.

To reproduce the results of Table 5.2 (Effect of prompt transfer), you need to execute both scripts for prompt tuning and the prompt transfer one.

We applied same configuration either for prompt tuning or for prompt transfer.
Example scripts for running prompt tuning, prompt transfer, and task selection are provided in ``seq2seq/scripts`. 

We provide the example scripts to run prompt tuning, prompt transfer and task selection using thee config files in `seq2seq/configs`. To run the models, please first do

```bash
cd compacter/seq2seq
```

### Configuration File and Arguments

When training model with prompt tuning, all code for training requires a configration file defined in `configs/prompt_tuning_tokens_config/t5-base` folder.
Our implemetation manages files according the fine-tuning methods and model type.
Feel free to create your own directory.

Note that to enable flexible confugrate the hyperparameters, we provide partial arguments in the main python script (`run_seq2seq.py`) for allowing sweep over the testing values. 
These arguments will overwrite the arguments in config files.


### Run Prompt Tuning

To perform prompt tuning on a single task, execute the following command. This script trains prompt tuning with initialization from the language model's vocabulary method. For 10 tasks, we use a learning rate of 2, and for the source tasks, a learning rate of 5e-1.

```bash
. script/prompt_tuning_tokens_init.sh
```


To get the average performance, please run:

```bash
python dev/get_prompt_scores.py
```


### Run Prompt Transfer

The commands train prompt tuning with initialization from pretrained prompt weights. Please specify the checkpoints to CKPTS.

```bash
. script/tf_prompt_tuning.py
```

To get the relative performance, please run:

```bash
python dev/get_prompt_transfer.py
```


### Creating Ranking Files

After training a set of models on a target tasks, one can create a ranking of emprical prompt transfer as ground-truth reference for evaluting the predction of task embedding.
For evalauting all 13 source tasks transfering to RTE, please do:

```bash
python dev/get_transfer_ranking.py \
	--tgt_task=rte \
	--output_dir=PATH_TO_YOUR_DIR/spot_eval/transfer_ranking
```

We save the result file in the `--output_dir` and require it for evalaution ranking.
Each file is named as `eval_tgt-TASKNAME_seed-VALUE` which contains prompt tuning's and prompt transfer's performances and ranking of intermediate tasks sorted by prompt transfer. 


As previsous part, you can specify all the tasks once you done all the training jobs. 

```bash
. scripts/run_transfer_ranking.sh
```



## Task Selection

In order to evalaute the transferability of an sorce task to given target task, one would need run prompt tuning on all tasks.


- vocab similarity

```bash
. scripts/run_vocab_sim.sh
```

- task embeddings

With our training scripts, we save the prompt weight along in the file `prefix_shared.bin`.
Through all task embedding expereiemnts, the weights are calculated for the task embeddigns.

Except for prompt similarity (`feature_mean`), we provdes additional constructions for task embeddings.
The following values are supports for `--task_embedding_type ` arugments : `feature_mean`, `flatten`, `unigram`, `bigram`,`max_pairwise`.

```bash
. scripts/get_prompt_similarity.sh
```

We save the predicted ranking file `eval_tgt-TASKNAME_seed-VALUE.json` in the `--output_dir`.


### Evalaution on Task Selection

To evalaute the task selection methods (random, size, text embedding or task embeddings), one reuqires both prediction file and reference file.

In our work, we evalaute with `ndcg`, `regret_at_k `. This is support to `--method`.

```bash
python dev/get_ndcg.py \
	--pred_dir=PATH_TO_YOUR_DIR \
	--pred_dir=PATH_TO_YOUR_DIR \
	--output_dir=PATH_TO_YOUR_DIR \
	--method=ndcg
```

You can replace the ndcg with  `regret_at_k` and `top_k_performance`. 


For evalauton prediction of *data size* and *random* methods. you can directly pass boolean argument as follow:

```bash
python dev/get_ndcg.py 	--ranking_by_random
```

For data size method, please do:  

```bash
python dev/get_ndcg.py 	--ranking_by_size
```

## Fine-tuning methods and adding tasks

This repo is developed based on [COMPACTER](https://github.com/rabeehk/compacter) and contains the implementation of recent parameter-efficient finetuning methods. Thus, the repo also provides the example scripts to run T5 model with full model tuning and others parameter efficeint fine-tuning. For full fine-tuning please run:

```bash
. scripts/baseline.sh
```  

Other parameter fine-tuning methods, please check the scripts in `scripts` folds.
(Adapter, AdapterDrop, Low-Rank, BitFit, Compacter, Compacter++, PHM-Adapters, Intrinsic-SAID)

If you wish to add a new task, you will need to create a new dataset class in `/data/{tasks,postprocessors}.py` and its corresponding configuration file. For example, when running a task, a configuration file is required, such as prompt_tuning_tokens_config/t5-baseprompt_tuning_tokens_init_boolq.json.


## Bibliography

If you find this repo useful, please cite our work:

```bash
@inproceedings{lin2024taskselection,
  title={Exploring Task Selection For Intermediate Task Transfer},
  author={Pin-Jie Lin},
  year={2024}
}
```


## Contact Information

For help or issues using our code, please submit a issue or contact to pjlin@lsv.uni-saarland.de
