"""
# ==================================
# AUTHOR : Yan Li, Qiong Wang
# CREATE DATE : 03.13.2020
# Contact : liyanxian19@gmail.com
# ==================================
# Change History: None
# ==================================
"""
from __future__ import print_function

########## Set seed ##########
from numpy.random import seed
from tensorflow import set_random_seed
seed(1)
set_random_seed(2)



########## import python libs ##########
import argparse
import math
import os
from distutils.version import LooseVersion

########## Import third-party libs ##########
from tensorflow.python import keras
assert LooseVersion(keras.__version__) > LooseVersion('2.0.8')
from tensorflow.python.keras import backend as K

########## Import our libs ##########
from dataset import *
from model import llfnet
from utils import *



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--data_root', type=str, default='./Data') # YOUR_DATA_ROOT
parser.add_argument('--dataset', type=str, default='Hand-designed_test') # YOUR_DATA_SET.txt in YOUR_DATA_ROOT
parser.add_argument('--data_size', type=str, default='Image') # Changed to 'Patch' if there exists CPU/GPU memory issues

parser.add_argument('--iterations', type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument('--model_path', type=str, default='./Model') # YOUR_MODEL_PATH
parser.add_argument('--move_path', type=str, default='LT') # LT: Left-right, Top-bottom
parser.add_argument('--model_infovis', type=bool, default=True)
parser.add_argument("--finetune", action='store_true')
config = parser.parse_args()



def train(train_generator, model_train, iter_n, max_iter,
          val_generator=None, val_steps=None, logger=None):

    init_lr = config.learning_rate
    # Update learning rate
    schedule = {"step": {"80": 5e-5, "160": 1e-5, "200": 1e-6}}
    update_lr = update_learning_rate(schedule, init_lr, iter_n, max_iter)
    if update_lr != init_lr:
        init_lr = update_lr
        K.set_value(model_train.optimizer.lr, init_lr)
        logger.info('=> updated learning rate: {}'.format(K.get_value(model_train.optimizer.lr)))

    steps_per_epoch = config.iterations
    if val_generator is None:
        model_train.fit_generator(train_generator, steps_per_epoch=steps_per_epoch,
                                  epochs=iter_n+1, initial_epoch=iter_n, verbose=2, workers=config.workers)
    else:
        model_train.fit_generator(train_generator, steps_per_epoch=steps_per_epoch,
                                  epochs=iter_n+1, initial_epoch=iter_n,
                                  validation_data=val_generator, validation_steps=val_steps,
                                  verbose=2, workers=config.workers)
    model_weights = model_train.get_weights()
    return model_weights

def run(hp, logger=None):
    ########## prepare dataset ##########
    train_generator = prepare_train_generator(hp, logger=logger)

    input_chns = int(math.ceil(len(hp["network"]["train_crop_seqs"]) / hp["network"]["sample"]))
    # padding
    if "train_pad" in hp["network"].keys():
        pad_n_h = hp["network"]["train_pad"][0]
        pad_n_w = hp["network"]["train_pad"][1]
        input_shape = (hp["network"]["train_size"][0]+2*pad_n_h,
                       hp["network"]["train_size"][1]+2*pad_n_w, input_chns)
    else:
        input_shape = (hp["network"]["train_size"][0],
                       hp["network"]["train_size"][1], input_chns)

    ########## prepare model ##########
    input_layer_names = hp["network"]["input_layer_candidates"]
    llfnet_model = llfnet(input_layer_names, input_shape, config=config, hp=hp, phase="train", logger=logger)

    # finetune the model?
    if config.finetune:
        logger.info("=> finetune from {}".format(config.model_path))
        llfnet_model, iter_st = load_weight(llfnet_model, config.model_path)
    else:
        iter_st = 0

    max_iter = hp["max_n_iter"]
    ########## start to train (patch-wise training) ##########
    for iter_n in range(iter_st, max_iter):
        logger.info("------------------ iteration {}----------------------".format(iter_n))
        model_weights = train(train_generator,
                              llfnet_model,
                              iter_n,
                              max_iter)
        save_path_file_new = '%s_iter%04d' % ('wlf_train', iter_n)
        if not os.path.exists(os.path.join(config.model_path, save_path_file_new + '.hdf5')):
            llfnet_model.save(os.path.join(config.model_path, save_path_file_new + '.hdf5'))



if __name__ == '__main__':
    ########## call logger ##########
    logger = get_logger()

    ########## setup GPU ##########
    if config.gpus > 1:
        pass
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable printing tensorflow gpu info etc.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # load hyper-parameters
    baseline = 1
    density = 9
    dataset_repr = "ca" # ma: micro-lens array (narrow), ca: camera array (wide)
    json_file_name = "hyper-parameters_llf-net_b{}d{}_{}.json".format(baseline,
                                                                      density,
                                                                      dataset_repr)

    hp = get_hp("manual", json_file_name)
    run(hp=hp, logger=logger)
