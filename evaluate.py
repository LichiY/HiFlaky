# --coding:utf-8--

import helper.logger as logger
from models.model import HiAGM
import torch
import sys
from helper.configure import Configure
import os
from data_modules.data_loader import data_loaders
from data_modules.vocab import Vocab
from train_modules.criterions import ClassificationLoss
from train_modules.trainer import Trainer
from helper.utils import load_checkpoint

#�����Ż���
def set_optimizer(config, model):

    params = model.optimize_params_dict()
    if config.train.optimizer.type == 'Adam':
        return torch.optim.Adam(lr=config.train.optimizer.learning_rate,
                                params=params)
    else:
        raise TypeError("Recommend the Adam optimizer")


def evaluate(config):

    corpus_vocab = Vocab(config,
                         min_freq=5,
                         max_size=52000)


    _, _, test_loader = data_loaders(config, corpus_vocab)


    hiagm = HiAGM(config, corpus_vocab, model_type=config.model.type, model_mode='TRAIN')
    hiagm.to(config.train.device_setting.device)


    criterion = ClassificationLoss(os.path.join(config.data.data_dir, config.data.hierarchy),
                                   corpus_vocab.v2i['label'],
                                   recursive_penalty=config.train.loss.recursive_regularization.penalty,
                                   recursive_constraint=config.train.loss.recursive_regularization.flag)
    optimize = set_optimizer(config, hiagm)


    model_checkpoint = config.train.checkpoint.dir
    dir_list = os.listdir(model_checkpoint)
    assert len(dir_list), "No model file in checkpoint directory!!"
    assert os.path.isfile(os.path.join(model_checkpoint, config.test.best_checkpoint)), \
        "The predefined checkpoint file does not exist."
    model_file = os.path.join(model_checkpoint, config.test.best_checkpoint)
    logger.info('Loading Previous Checkpoint...')
    logger.info('Loading from {}'.format(model_file))
    _, config = load_checkpoint(model_file=model_file,
                                model=hiagm,
                                config=config)

    trainer = Trainer(model=hiagm,
                      criterion=criterion,
                      optimizer=optimize,
                      vocab=corpus_vocab,
                      config=config)
    hiagm.eval()

    trainer.eval(test_loader, -1, 'TEST')
    return


if __name__ == "__main__":
    configs = Configure(config_json_file=sys.argv[1])


    if configs.train.device_setting.device == 'cuda':
        os.system('CUDA_VISIBLE_DEVICES=' + str(configs.train.device_setting.visible_device_list))
    else:
        os.system("CUDA_VISIBLE_DEVICES=''")
    torch.manual_seed(2019)
    torch.cuda.manual_seed(2019)
    logger.Logger(configs)


    if not os.path.isdir(configs.train.checkpoint.dir):
        os.mkdir(configs.train.checkpoint.dir)

    evaluate(configs)
