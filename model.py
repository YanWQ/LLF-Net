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
import math
from distutils.version import LooseVersion

########## Import third-party libs ##########
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Lambda
from tensorflow.python.keras.layers import UpSampling3D
from tensorflow.python.keras.layers import Average, concatenate
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import RMSprop
if LooseVersion(keras.__version__) > LooseVersion('2.1.2'):
    from tensorflow.python.keras.utils import multi_gpu_model

########## Import our libs ##########
from submodule import *


def smoothL1(y_true, y_pred):
    HUBER_DELTA = 0.3 # 1.0
    x = K.abs(y_true - y_pred)
    x = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return K.sum(x)

def get_input(input_layer_names, input_shape):
    input_layers = []
    for _, input_layer_name in enumerate(input_layer_names):
        # input
        x = Input(shape=input_shape, name=input_layer_name)
        input_layers.append(x)
    return input_layers

def get_3x3(x, f_n, logger=None):
    if len(x.shape) == 4:
        B, H, W, C = K.int_shape(x)
    elif len(x.shape) == 5:
        B, D, H, W, C = K.int_shape(x)
    nb_3x3s = int((C/f_n - 1)/2)
    if logger is not None:
        logger.debug('=> C and f_n: {} {}'.format(C, f_n))
    x_3x3s = []
    for interval in range(1, nb_3x3s+1):
        x_3x3 = Lambda(slicing, arguments={'index': (int((C/f_n - 1)/2)-interval),
                                           'index_end': (int((C/f_n - 1)/2)+interval+1),
                                           'interval': interval,
                                           'split': int(C/f_n)})(x)
        x_3x3s.append(x_3x3)
    return x_3x3s

def llfnet(input_layer_names, input_shape, config=None, hp=None, phase=None, logger=None):
    output_layers = []

    # paras of layers
    conv_type, ks, activ = "conv", 2, "relu"

    # paras of cost volume (cv)
    min_disp, max_disp, num_disp_labels = 0, 50, 128

    # 0. Input
    input_layers = get_input(input_layer_names, input_shape)

    # 1. Feature extraction
    nb_filt1 = 16
    fe_filt = [nb_filt1*2, nb_filt1*2, nb_filt1*4, nb_filt1*4]
    feature_s_paras = {'ks': ks, 'stride': [2, 1], 'padding': "zero", 'filter': fe_filt*1,
                       'activation': activ, 'conv_type': conv_type,
                       'pyr': True, 'layer_nums': 2, "ret_feat_levels": 1}
    # feature share module
    feature_s_m = feature_extraction_m((input_shape[0], input_shape[1], 1), feat_paras=feature_s_paras)

    feature_streams = []
    fs_ts_ids = []
    for stream_id, x in enumerate(input_layers):
        if stream_id > 1:
            logger.info("skip stream{}".format(stream_id))
            continue
        feature_stream = []
        # iterate views of a stream
        for x_sid in range(0, input_shape[2]):
            x_sub = Lambda(slicing, arguments={'index': x_sid,
                                               'index_end': x_sid+1})(x)
            x_sub = feature_s_m(x_sub)
            feature_stream.append(x_sub)

        # | stream (vertical)
        if stream_id == 0:
            t_ids = list(range(input_shape[2]))
            s_ids = [int((input_shape[2]-1)/2)]*input_shape[2]
        # - stream (horizontal)
        elif stream_id == 1:
            t_ids = [int((input_shape[2]-1)/2)]*input_shape[2]
            s_ids = list(range(input_shape[2]))
        fs_ts_ids.append((t_ids, s_ids))
        feature_streams.append(feature_stream)

    # 2. Cost volume
    # iterate pyramid levels reversely
    pyr_cost_volume = [] # pyramid cost volume
    pyr_levels_l = list(range(2))
    skip_pyr_levels = 1
    for pyr_level in pyr_levels_l[::-1]:
        if pyr_level < skip_pyr_levels:
            logger.info("=> skip pyramid level: {}".format(pyr_level))
            continue
        else:
            logger.info("=> pyramid level: {}".format(pyr_level))
        cv_streams = []
        scale_factor = math.pow(2, pyr_level+1)

        # Cost volume per pyramid level
        # 2.1 build (shift+cost) cost volume per stream
        for fs_id, feature_stream in enumerate(feature_streams):
            # pyramid feature stream
            pyr_fs = [fs_ep for fs_ep in feature_stream]
            cost_volume = Lambda(compute_cost_volume,
                                 arguments={"t_s_ids": fs_ts_ids[fs_id],
                                            "min_disp": min_disp/scale_factor,
                                            "max_disp": max_disp/scale_factor,
                                            "labels": int(num_disp_labels / scale_factor),
                                            "move_path": "LT"})(pyr_fs)
            cv_streams.append(cost_volume)

        # 2.2 fuse multiple streams (across views + across streams or intra + inter)
        if len(cv_streams) > 1:
            if input_shape[2] > 3:
                cv_streams_3x3s = []
                # divide
                for cv_stream in cv_streams:
                    cv_stream_3x3s = get_3x3(cv_stream,  nb_filt1*2)
                    cv_streams_3x3s.append(cv_stream_3x3s)
                cv_streams_3x3s = list(map(list, zip(*cv_streams_3x3s))) # transpose
                # concat
                concat_cost_volumes = []
                for cv_streams_3x3 in cv_streams_3x3s:
                    concat_cost_volume = concatenate(cv_streams_3x3)
                    concat_cost_volumes.append(concat_cost_volume)
                # sum over divided
                cost_volume = Average()(concat_cost_volumes)
            else:
                cost_volume = concatenate(cv_streams)

        pyr_cost_volume.append(cost_volume)

    # 3/4.Cost aggregation + Regression
    for idx in range(len(pyr_cost_volume)):
        # 3.Cost aggregation
        ca_paras = {'conv_type': conv_type, 'ks': 3, 'stride': 2, 'padding': "same", 'filter': nb_filt1*4,
                    'activation': activ, 'n_dc': 1}
        # view
        pcv_tmp_l = []
        for i in range(2):
            ind_st = i*nb_filt1*2*3
            ind_end = (i+1)*nb_filt1*2*3
            pcv_sub = Lambda(slicing, arguments={'index': ind_st, 'index_end': ind_end})(pyr_cost_volume[idx])
            pcv_sub_ca = channel_attention_m(pcv_sub, residual=True)
            pcv_tmp_l.append(pcv_sub_ca)
        pcv_tmp = concatenate(pcv_tmp_l)
        # stream
        pcv_tmp = channel_attention_m(pcv_tmp, residual=True, stream=True)
        output = cost_aggregation(pcv_tmp, ca_paras=ca_paras)

        # upsampling
        up_scale = int(input_shape[0]/K.int_shape(output)[2])
        x_shape = output.get_shape().as_list()
        output = Lambda(upsample_ops,
                        arguments={'width': int(up_scale * x_shape[2]),
                                   'height': int(up_scale * x_shape[3]),
                                   'axis': 1,
                                   'scale': 1,
                                   'interp': "bilinear"})(output)
        output = Lambda(upsample_ops,
                        arguments={'width': int(2 * x_shape[1]),
                                   'height': int(2 * x_shape[3]),
                                   'axis': 2,
                                   'scale': 1,
                                   'interp': "bilinear"})(output)

        # 4.Regression
        output = Lambda(lambda op: tf.nn.softmax(op, axis=1))(output)
        output = Lambda(soft_min_reg,
                        arguments={"axis": 1,
                                   "min_disp": min_disp,
                                   "max_disp": max_disp,
                                   "labels": num_disp_labels},
                        name="sm_disp{}".format(pyr_level))(output)
    output_layers.append(output)

    # Set optimizer, and compile
    if phase == 'train':
        # Set optimizer with learning rate
        learning_rate = hp["network"]['learning_rate']
        opt = RMSprop(lr=learning_rate)

        llfnet_model = Model(inputs=input_layers,
                            outputs=output_layers)
        if config.gpus > 1:
            llfnet_model = multi_gpu_model(llfnet_model, gpus=config.gpus)
        llfnet_model.compile(optimizer=opt, loss=smoothL1)

    else:
        llfnet_model = Model(inputs=input_layers,
                             outputs=output_layers)
        if config.gpus > 1:
            llfnet_model = multi_gpu_model(llfnet_model, gpus=config.gpus)

    if config.model_infovis:
        llfnet_model.summary()
    return llfnet_model
