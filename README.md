# HiFlaky: Hierarchy-Aware Flakiness Classification

This is the replication package associated with the paper: 'HiFlaky: Hierarchy-Aware Flakiness Classification'

### Requirements

This is a list of all required python packages:

- Python == 3.12.3
- imbalanced-learn==0.12.3
- pandas==2.2.3
- scikit-learn==1.5.2
- transformers==4.44.2
- torch==2.3.0




### Dataset

The data folder contains the following :

* **data/flaky_token_train_v3.json**, **data/flaky_token_val_v3.json**, **data/flaky_token_test_v3.json**: contains data obtained after an augmentation of the original data. 
* **data/flaky_label_v2.taxonomy**: predefines hierarchical structure. 
* **data/flaky_prob_v3.json**: calculates the prior probability between parent-child pair in train dataset.

### Train

```bash
python train.py config/flaky_config_v3.json
```

+ optimizer -> train.set_optimizer: default torch.optim.Adam
+ learning rate decay schedule callback -> train_modules.trainer.update_lr
+ earlystop callback -> train.py 
+ Hyper-parameters are set in config.train
