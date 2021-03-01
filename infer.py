"""
# ==================================
# AUTHOR : Yan Li, Qiong Wang
# CREATE DATE : 03.13.2020
# Contact : liyanxian19@gmail.com
# ==================================
# Change History: None
# ==================================
"""
########## Import python libs ##########
from __future__ import print_function
import argparse
import os
import time
from collections import OrderedDict

########## Import third-party libs ##########
import matplotlib.pyplot as plt

########## Import our libs ##########
from dataset import get_preds_data
from model import llfnet
from utils import *



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--data_root', type=str, default='./Data') # YOUR_DATA_ROOT
parser.add_argument('--dataset', type=str, default='Hand-designed_test') # YOUR_DATA_SET.txt in YOUR_DATA_ROOT
parser.add_argument('--data_size', type=str, default='Image') # Changed to 'Patch' if there exists CPU/GPU memory issues
parser.add_argument('--model_path', type=str, default='./Model') # YOUR_MODEL_PATH
parser.add_argument('--move_path', type=str, default='LT') # LT: Left-right, Top-bottom
parser.add_argument('--model_infovis', type=bool, default=True)
config = parser.parse_args()



def infer(model_weights_medium, data_for_predictions, model_pred=None, logger=None, show_time=False):

    def get_weights(model_weights_medium):
        # load model weights
        if isinstance(model_weights_medium, str):
            model_pred.load_weights(model_weights_medium)

    for data_key, data_value in data_for_predictions.items():
        imgs = data_value[0]
        get_weights(model_weights_medium)
        start = time.time()
        ########## predict ##########
        outputs = model_pred.predict(imgs, batch_size=config.batch_size)
        end = time.time()
        if show_time:
            logger.info("=> elapsed time: {}s".format(end-start))
        logger.info("=> output_tmp: {}".format(outputs.shape))

    return outputs

def run():
    ########## prepare dataset ##########
    lf_shape = (1080, 1920, 9, 9, 3) # WLF, using CPU
    #lf_shape = (560, 976,9, 9, 3) # WLF, using GPU 'gtx1080 ti'
    config.lf_shape = lf_shape

    input_chns = 9 # input channels to LLF-Net
    input_img_shape = [lf_shape[0], lf_shape[1]] # input image shape to MANet
    # padding
    if input_img_shape[0] % 8 == 0 and input_img_shape[0] % 8 == 0:
        config.pad = None
        input_shape = (input_img_shape[0], input_img_shape[1], input_chns)
    else:
        pad_n_hl, pad_n_hr, pad_n_wl, pad_n_wr = 0, 0, 0, 0
        if input_img_shape[0] % 8 != 0:
            pad_n_hl = int(8 - input_img_shape[0] % 8)/2
            pad_n_hr = (8 - input_img_shape[0] % 8) - pad_n_hl
        if input_img_shape[1] % 8 != 0:
            pad_n_wl = int(8 - input_img_shape[1] % 8)/2
            pad_n_wr = (8 - input_img_shape[1] % 8) - pad_n_wl
        config.pad = [pad_n_hl, pad_n_hr, pad_n_wl, pad_n_wr]
        input_shape = (input_img_shape[0]+(pad_n_hl+pad_n_hr), input_img_shape[1]+(pad_n_wl+pad_n_wr), input_chns)
    config.input_shape = input_shape

    preds_x = get_preds_data(config, logger=logger)
    data_for_predictions = OrderedDict()
    data_for_predictions = {config.dataset: [preds_x]}

    ########## prepare model ##########
    input_layer_names = ["x90d", "x0d", "x45d", "xm45d"]
    llfnet_model = llfnet(input_layer_names, input_shape, config=config, phase="test", logger=logger)

    ########## start to infer ##########
    for model_id, model_weights_file in enumerate(os.listdir(config.model_path)):
        if '.hdf5' in model_weights_file:
            logger.info("load model weights {}".format(model_weights_file))
            """
            if model_id == 0:
                # dry run
                x = np.zeros((1, lf_shape[0], lf_shape[1], lf_shape[2]), dtype=np.float32)
                infer(os.path.join(config.model_path, model_weights_file),
                      {"dry_run": [[x, x, x, x]]},
                      model_pred=llfnet_model,
                      logger=logger)
            """
            # infer
            outputs = infer(os.path.join(config.model_path, model_weights_file),
                            data_for_predictions,
                            model_pred=llfnet_model,
                            logger=logger,
                            show_time=True)
            logger.info(outputs.shape)
            plt.imsave('./Results/example.png', outputs[0, ..., 0])



if __name__ == '__main__':
    ########## call logger ##########
    logger = get_logger()

    ########## choose CPU or GPU ##########
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # "-1": cpu, "0": 'gtx1080 ti' (if you have)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable printing tensorflow gpu info etc.

    run()
