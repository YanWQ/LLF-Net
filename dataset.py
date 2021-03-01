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
import os
import math

########## Import third-party libs ##########
import numpy as np
import cv2
import h5py

########## Import our libs ##########
from dataset_augmenter import *



########## light field camera/micro-lens array IDs ##########
def get_lf_ca(config=None):
    _, _, l_t, l_s, _ = config.lf_shape
    dataset_view_nums = l_t * l_s
    ca = np.arange(dataset_view_nums)
    move_path = config.move_path
    if move_path == "LT":
        ca = np.reshape(ca, newshape=(1, dataset_view_nums))
    elif move_path == "RT":
        ca = np.reshape(np.fliplr(np.reshape(ca, newshape=(l_t, l_s))), newshape=(1, dataset_view_nums))
    elif move_path == "LD":
        ca = np.reshape(np.flipud(np.reshape(ca, newshape=(l_t, l_s))), newshape=(1, dataset_view_nums))
    return ca

########## light field scene path list ##########
def read_lf_scene_path_list(data_root='', dataset_name='', logger=None):
    lf_dir = os.path.abspath(os.getcwd())
    lf_list = ''
    with open('{}/{}.txt'.format(data_root, dataset_name)) as f:
        logger.info("Loading data from {}.txt".format(dataset_name))
        lines = f.read().splitlines()
        for line_cnt, line in enumerate(lines):
            if line != '':
                if (line_cnt + 1) == len(lines):
                    lf_list += os.path.join(lf_dir, line)
                else:
                    lf_list += os.path.join(lf_dir, line) + ' '
            logger.info('Scene: {}'.format(line))
    return lf_list.split(' ')

########## load light field images ##########
def load_lf_images(frame_paths, ca, color_space, dataset_img_shape):

    _, _, l_t, l_s, _ = dataset_img_shape

    lf_img = np.zeros(((len(frame_paths),) + dataset_img_shape[:-1]), np.uint8)

    dataset_view_nums = l_t * l_s
    scene_id = 0

    # a frame means a scene
    for frame_path in frame_paths:
        # load images
        # cam_id is a coordinate in LT (origin) system
        for cam_id in range(dataset_view_nums):
            # cam_map_id: camera mapping id (used for capturing paths)
            cam_map_id = ca[0, cam_id]
            if color_space == "gray":
                try:
                    tmp = np.float32(cv2.imread(os.path.join(frame_path, 'input_Cam0%.2d.png' % cam_map_id), 0))
                except:
                    print(os.path.join(frame_path, 'input_Cam0%.2d.png..does not exist' % cam_map_id))
                lf_img[scene_id, :, :, cam_id // l_s, cam_id - l_t * (cam_id // l_s)] = tmp
            del tmp

        scene_id = scene_id + 1
    return lf_img

########## load light field data ##########
def load_lf_data(config, color_space=None, frame_paths=None, logger=None):
    if frame_paths is None:
        frame_paths = read_lf_scene_path_list(data_root=config.data_root,
                                              dataset_name=config.dataset,
                                              logger=logger)
    # light field camera/micro-lens array IDs/NOs
    ca = get_lf_ca(config)
    # load light field images
    infer_imgs = load_lf_images(frame_paths, ca, color_space, config.lf_shape)

    return infer_imgs

def load_lf_fromh5(lf_h5_path, lf_h5_dataset, logger=None):
    f = h5py.File(lf_h5_path, 'r')
    if "imgs" in lf_h5_dataset:
        imgs_data = f["imgs"]
        logger.info("=> loading light field images from h5")
    if "disps" in lf_h5_dataset:
        labels_data = f["disps"]
        logger.info("=> loading light field disps from h5")
        return imgs_data, labels_data
    else:
        return imgs_data



########################################################################
# Prepare data samples for inference
########################################################################
########## prepare preds data ##########
def prepare_preds_data(lf_imgs_data, config=None, logger=None):
    B, H, W, T, S = lf_imgs_data.shape
    assert T == S

    preds_crop_seqs = [i for i in range(0, config.input_shape[-1])]
    crop_seqs = np.array(preds_crop_seqs) # np

    scene_nums = B # number of scenes
    # spatial coordinate of central view
    stride_v = H
    stride_u = W
    # angular coordinate of central view
    l_t = crop_seqs[int((len(crop_seqs)-1)/2)]
    l_s = crop_seqs[int((len(crop_seqs)-1)/2)]
    if logger is not None:
        logger.info("Central view {},{}".format(l_t, l_s))

    if config.data_size == "Image":
        x_shape = (scene_nums, stride_v, stride_u, config.input_shape[-1])
        x90d = np.zeros(x_shape, dtype=np.float32)
        x0d = np.zeros(x_shape, dtype=np.float32)
        x45d = np.zeros(x_shape, dtype=np.float32)
        xm45d = np.zeros(x_shape, dtype=np.float32)
    elif config.data_size == 'Patch':
        scale_val = 32
        patch_size = [int((H/2)/16)*16+16, int((W/2)/16)*16+16]
        if H % scale_val != 0:
            patch_size[0] += 16
        if W % scale_val != 0:
            patch_size[1] += 16

        sip_height = patch_size[0]
        sip_width = patch_size[1]
        stride_h = patch_size[0] - 32
        stride_w = patch_size[1] - 32

        sip_rows = int(math.ceil((lf_imgs_data.shape[1] - sip_height) / stride_h)) + 1
        sip_cols = int(math.ceil((lf_imgs_data.shape[2] - sip_width) / stride_w)) + 1

        patch_nums = scene_nums * sip_rows * sip_cols

        img_v_ids = list(range(0, lf_imgs_data.shape[1] - sip_height, stride_h))
        img_u_ids = list(range(0, lf_imgs_data.shape[2] - sip_width, stride_w))
        v_append = lf_imgs_data.shape[1] - sip_height
        h_append = lf_imgs_data.shape[2] - sip_width
        img_v_ids.append(v_append)
        img_u_ids.append(h_append)

        patch_num_l = []
        for scene_id in range(scene_nums):
            for v_id in img_v_ids:
                for u_id in img_u_ids:
                    patch_num_l.append([scene_id, v_id, u_id])
        patch_num_l.sort()

        x_shape = (patch_nums, sip_height, sip_width, len(crop_seqs))
        x90d = np.zeros(x_shape, dtype=np.float32)
        x0d = np.zeros(x_shape, dtype=np.float32)
        x45d = np.zeros(x_shape, dtype=np.float32)
        xm45d = np.zeros(x_shape, dtype=np.float32)

    start1 = crop_seqs[0]
    end1 = crop_seqs[-1]
    x90d_t = preds_crop_seqs
    x0d_s = preds_crop_seqs
    if config.data_size == "Image":
        for scene_id in range(scene_nums):
            for v in range(0, 1):
                for u in range(0, 1):
                    x90d[scene_id, v:v + stride_v, u:u + stride_u, :] = \
                        np.moveaxis(lf_imgs_data[scene_id, v:v + stride_v, u:u + stride_u, x90d_t, l_s], 0, -1).astype('float32')
                    x0d[scene_id, v:v + stride_v, u:u + stride_u, :] = \
                        np.moveaxis(lf_imgs_data[scene_id, v:v + stride_v, u:u + stride_u, l_t, x0d_s], 0, -1).astype('float32')
                    for kkk in range(start1, end1 + 1):
                        x45d[scene_id, v:v + stride_v, u:u + stride_u, int((kkk - start1))] = lf_imgs_data[scene_id,
                                                                                       v:v + stride_v,
                                                                                       u:u + stride_u,
                                                                                       end1 + start1 - kkk,
                                                                                       kkk].astype('float32')
                        xm45d[scene_id, v:v + stride_v, u:u + stride_u, int((kkk - start1))] = lf_imgs_data[scene_id,
                                                                                        v:v + stride_v,
                                                                                        u:u + stride_u,
                                                                                        kkk, kkk].astype('float32')
    elif config.data_size == 'Patch':
        for patch_id in range(patch_nums):
            scene_id = patch_num_l[patch_id][0]
            v = patch_num_l[patch_id][1]
            u = patch_num_l[patch_id][2]
            x0d[patch_id, ...] = \
                np.moveaxis(lf_imgs_data[scene_id, v: v + sip_height, u: u + sip_width, l_t, crop_seqs.tolist()], 0, -1).astype('float32')
            x90d[patch_id, ...] = \
                np.moveaxis(lf_imgs_data[scene_id, v: v + sip_height, u: u + sip_width, x90d_t, l_s], 0, -1).astype('float32')
            for kkk in range(start1, end1 + 1):
                x45d[patch_id, ..., kkk - start1] = lf_imgs_data[scene_id, v: v + sip_height,
                                                    u: u + sip_width, (T-1) - kkk, kkk].astype('float32')

                xm45d[patch_id, ..., kkk - start1] = lf_imgs_data[scene_id, v: v + sip_height,
                                                     u: u + sip_width, kkk, kkk].astype('float32')

    if config.pad is not None:
        pad_n_hl, pad_n_hr = config.pad[:2]
        pad_n_wl, pad_n_wr = config.pad[2:]
        x90d = np.pad(x90d, ((0, 0), (pad_n_hl, pad_n_hr), (pad_n_wl, pad_n_wr), (0, 0)), mode='reflect')
        x0d = np.pad(x0d, ((0, 0), (pad_n_hl, pad_n_hr), (pad_n_wl, pad_n_wr), (0, 0)), mode='reflect')
        x45d = np.pad(x45d, ((0, 0), (pad_n_hl, pad_n_hr), (pad_n_wl, pad_n_wr), (0, 0)), mode='reflect')
        xm45d = np.pad(xm45d, ((0, 0), (pad_n_hl, pad_n_hr), (pad_n_wl, pad_n_wr), (0, 0)), mode='reflect')

    x90d = np.float32((1 / 255) * x90d)
    x0d = np.float32((1 / 255) * x0d)
    x45d = np.float32((1 / 255) * x45d)
    xm45d = np.float32((1 / 255) * xm45d)
    return [x90d, x0d, x45d, xm45d]

########## get prediction data ##########
def get_preds_data(config, logger=None):
    preds_imgs_data = load_lf_data(config,
                                   color_space="gray",
                                   logger=logger)

    preds_x = prepare_preds_data(preds_imgs_data,
                                 config=config,
                                 logger=logger)
    return preds_x



########################################################################
# Prepare data samples for training
########################################################################
########## get a batch of training patches ##########
def get_train_patches_data(lf_img_data=None, lf_label_data=None, hp=None, bool_masks=None):
    B, H, W, T, S, C = lf_img_data.shape

    batch_size = hp["network"]["train_batch_size"]
    patch_height, patch_width = hp["network"]["train_size"]
    label_height, label_width = hp["network"]["label_size"]
    crop_seqs = np.array(hp["network"]["train_crop_seqs"])

    # decrease
    rem_edge = hp["network"]["augmentation"]["rem_edge"]
    # increase
    interval = hp["network"]["augmentation"]["crop_interval"]
    scales = hp["network"]["augmentation"]["scales"]
    # sample
    ca_sample = hp["network"]["sample"] # sample camera array

    ### initialize x and y shape
    x_shape = (batch_size, patch_height, patch_width, int(len(crop_seqs)/ca_sample))
    y_shape = (batch_size, label_height, label_width)

    ### initialize x and y data
    x90d = np.zeros(x_shape, dtype="float32")
    x0d = np.zeros(x_shape, dtype="float32")
    x45d = np.zeros(x_shape, dtype="float32")
    xm45d = np.zeros(x_shape, dtype="float32")
    y = np.zeros(y_shape, dtype="float32")

    ### get first index and last index of NxN camera array that you set in "train_crop_seqs"
    if ca_sample == 1:
        start1 = crop_seqs[0]
        end1 = crop_seqs[-1]
    else:
        start1 = hp["network"]["sample_seqs"][0]
        end1 = hp["network"]["sample_seqs"][-1]
    crop_half1 = int(0.5 * (patch_height - label_height))

    # generate batch of image stacks and labels
    for batch_id in range(batch_size):
        sum_diff = 0
        valid = 0
        while (sum_diff < rem_edge * patch_height * patch_width) or (valid == 0):
            # shift augmentation for 7x7, 5x5, ... light fields
            ct_os, cs_os = get_aug_cts(crop_seqs)

            if not hp["dataset"]["dataset_filterout"]:
                # randomly choose a scale
                scale = get_aug_scale(scales)
                # randomly crop with intervals
                v_st, u_st = get_aug_vu(scale, [H, W], [patch_height, patch_width], interval)

                valid = 1

            scene_id = get_aug_sid(scene_nums=lf_img_data.shape[0])

            if hp["dataset"]["dataset_filterout"]:
                # randomly choose a scale
                scale = get_aug_scale(scales)
                if np.amax(lf_label_data[scene_id, :, :, 4 + ct_os, 4 + cs_os]) >= 50:
                    scale = 2
                # deal with patches (in Hand-designed_train dataset) with disparities larger than 50 pixels
                elif np.amax(lf_label_data[scene_id, :, :, 4 + ct_os, 4 + cs_os]) >= 100:
                    scale = 3

                if np.amax(lf_label_data[scene_id, :, :, 4 + ct_os, 4 + cs_os]) >= 150:
                    valid = 0
                else:
                    # randomly crop with intervals
                    v_st, u_st = get_aug_vu(scale, [H, W], [patch_height, patch_width], interval)
                    valid = 1

            if valid:
                if ca_sample == 1:
                    seqs = crop_seqs + ct_os
                    if hp["dataset_baseline"] == "wide":
                        seqs_v = crop_seqs + cs_os
                    elif hp["dataset_baseline"] == "narrow":
                        seqs_v = crop_seqs[::-1] + cs_os
                else:
                    seqs = np.array(hp["network"]["sample_seqs"]) + ct_os
                    if hp["dataset_baseline"] == "wide":
                        seqs_v = np.array(hp["network"]["sample_seqs"]) + cs_os
                    elif hp["dataset_baseline"] == "narrow":
                        seqs_inv = np.array(hp["network"]["sample_seqs"][::-1]) + cs_os

                # randomly convert rgb to gray
                R, G, B = get_aug_rgb()
                sum_diff = get_sum_diff(lf_img_data[scene_id, v_st: v_st + scale * patch_height:scale,
                                        u_st: u_st + scale * patch_width:scale, 4 + ct_os, 4 + cs_os, :],
                                        (R, G, B),
                                        patch_height, patch_width)

                # gray-scaled, cropped and scaled image stacks
                x0d[batch_id, :, :, :] = np.squeeze(
                    R * lf_img_data[scene_id:scene_id + 1, v_st: v_st + scale * patch_height:scale,
                        u_st: u_st + scale * patch_width:scale, 4 + ct_os, seqs.tolist(), 0].astype(
                        'float32') +
                    G * lf_img_data[scene_id:scene_id + 1, v_st: v_st + scale * patch_height:scale,
                        u_st: u_st + scale * patch_width:scale, 4 + ct_os, seqs.tolist(), 1].astype(
                        'float32') +
                    B * lf_img_data[scene_id:scene_id + 1, v_st: v_st + scale * patch_height:scale,
                        u_st: u_st + scale * patch_width:scale, 4 + ct_os, seqs.tolist(), 2].astype(
                        'float32'))

                x90d[batch_id, :, :, :] = np.squeeze(
                    R * lf_img_data[scene_id:scene_id + 1, v_st: v_st + scale * patch_height:scale,
                        u_st: u_st + scale * patch_width:scale, seqs_v.tolist(), 4 + cs_os, 0].astype(
                        'float32') +
                    G * lf_img_data[scene_id:scene_id + 1, v_st: v_st + scale * patch_height:scale,
                        u_st: u_st + scale * patch_width:scale, seqs_v.tolist(), 4 + cs_os, 1].astype(
                        'float32') +
                    B * lf_img_data[scene_id:scene_id + 1, v_st: v_st + scale * patch_height:scale,
                        u_st: u_st + scale * patch_width:scale, seqs_v.tolist(), 4 + cs_os, 2].astype(
                        'float32'))
                for kkk in range(start1, end1 + 1, ca_sample):
                    x45d[batch_id, :, :, int((kkk - start1)/ca_sample)] = np.squeeze(
                        R * lf_img_data[scene_id:scene_id + 1, v_st: v_st + scale * patch_height:scale,
                            u_st: u_st + scale * patch_width:scale, 8 - kkk + ct_os, kkk + cs_os, 0].astype(
                            'float32') +
                        G * lf_img_data[scene_id:scene_id + 1, v_st: v_st + scale * patch_height:scale,
                            u_st: u_st + scale * patch_width:scale, 8 - kkk + ct_os, kkk + cs_os, 1].astype(
                            'float32') +
                        B * lf_img_data[scene_id:scene_id + 1, v_st: v_st + scale * patch_height:scale,
                            u_st: u_st + scale * patch_width:scale, 8 - kkk + ct_os, kkk + cs_os, 2].astype(
                            'float32'))

                    xm45d[batch_id, :, :, int((kkk - start1)/ca_sample)] = np.squeeze(
                        R * lf_img_data[scene_id:scene_id + 1, v_st: v_st + scale * patch_height:scale,
                            u_st: u_st + scale * patch_width:scale, kkk + ct_os, kkk + cs_os, 0].astype('float32') +
                        G * lf_img_data[scene_id:scene_id + 1, v_st: v_st + scale * patch_height:scale,
                            u_st: u_st + scale * patch_width:scale, kkk + ct_os, kkk + cs_os, 1].astype('float32') +
                        B * lf_img_data[scene_id:scene_id + 1, v_st: v_st + scale * patch_height:scale,
                            u_st: u_st + scale * patch_width:scale, kkk + ct_os, kkk + cs_os, 2].astype('float32'))

                # y scale_factor*lf_label_data[random_index, scaled_label_size, scaled_label_size]
                if len(lf_label_data.shape) == 5:
                    y[batch_id, :, :] = ca_sample * (1.0 / scale) * lf_label_data[scene_id,
                                                        v_st + scale * crop_half1: v_st + scale * crop_half1 + scale * label_height:scale,
                                                        u_st + scale * crop_half1: u_st + scale * crop_half1 + scale * label_width:scale,
                                                        4 + ct_os, 4 + cs_os]
                else:
                    y[batch_id, :, :] = ca_sample * (1.0 / scale) * lf_label_data[scene_id,
                                                        v_st + scale * crop_half1: v_st + scale * crop_half1 + scale * label_height:scale,
                                                        u_st + scale * crop_half1: u_st + scale * crop_half1 + scale * label_width:scale]

                valid = check_disp_range(y[batch_id, :, :])

    # normalize
    x90d = np.float32((1.0 / 255.0) * x90d)
    x0d = np.float32((1.0 / 255.0) * x0d)
    x45d = np.float32((1.0 / 255.0) * x45d)
    xm45d = np.float32((1.0 / 255.0) * xm45d)

    return x90d, x0d, x45d, xm45d, y

########## get train generator ##########
def get_train_generator(hp, data_format="h5", logger=None):
    # get mask data
    bool_masks = None

    h5_is = []
    if "Hand-designed_train" in hp["dataset"]["train_dataset"]:
        dataset_rel_path = os.path.join(os.getcwd(), 'Data\Hand-designed_train')
        dataset_frame_s = 16 - 8
        h5_is.append([dataset_rel_path, dataset_frame_s])
        if logger is not None:
            logger.info("{}".format(dataset_rel_path))
    if "Flying-objects_train" in hp["dataset"]["train_dataset"]:
        dataset_rel_path = os.path.join(os.getcwd(), 'Data\FlyingObjects')
        dataset_frame_s = 345 - 8
        h5_is.append([dataset_rel_path, dataset_frame_s])
        logger.info("{}".format(dataset_rel_path))

    imgs_labels_data = []
    for h5_i in h5_is:
        h5_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)),
                               h5_i[0])

        if data_format == "h5":
            lf_h5_path = os.path.join(h5_path, "train_.h5")
            logger.debug("-> light field h5 path: {}".format(lf_h5_path))
            imgs_data, labels_data = load_lf_fromh5(lf_h5_path, ["imgs", "disps"], logger=logger)
            imgs_labels_data.append([imgs_data, labels_data])
            logger.debug("-> shape of loaded light field images: {}".format(imgs_data.shape))

    repeat_cnt = 0
    repeat_total = hp["network"]["repeat"]
    logger.info("=> the total number of repeat: {}".format(repeat_total))
    batch_imgs_data = None
    batch_labele_data = None
    while True:
        if data_format == "h5":
            if repeat_cnt % repeat_total == 0:
                repeat_cnt = 0
                # get dataset id
                dataset_id = np.random.randint(0, high=len(h5_is))
                # get frame id
                frame_id = np.random.randint(0, high=h5_is[dataset_id][1]) # dataset_frame_s is exclusive
                logger.debug("frame_id {}".format(frame_id))
                batch_imgs_data = imgs_labels_data[dataset_id][0][frame_id:frame_id+hp["network"]["train_batch_size"], ...]
                batch_labele_data = imgs_labels_data[dataset_id][1][frame_id:frame_id+hp["network"]["train_batch_size"], ...]
                # need B, H, W, T, S, C
                batch_imgs_data = np.moveaxis(batch_imgs_data, (1, 2), (3, 4)) # scenes, T, S, height, width, channel => scenes, height, width, T, S, channel
                batch_labele_data = np.moveaxis(batch_labele_data, (1, 2), (3, 4))
            repeat_cnt += 1

        x90d, x0d, x45d, xm45d, y = get_train_patches_data(batch_imgs_data, batch_labele_data,
                                                           hp=hp,
                                                           bool_masks=bool_masks)
        # augment patches data
        x90d, x0d, x45d, xm45d, y = data_augmentation([x90d, x0d, x45d, xm45d, y], hp)

        if "train_pad" in hp["network"].keys():
            pad_n_h = hp["network"]["pad"][0]
            pad_n_w = hp["network"]["pad"][1]
            x90d = np.pad(x90d, ((0, 0), (pad_n_h, pad_n_h), (pad_n_w, pad_n_w), (0, 0)), mode='reflect')
            x0d = np.pad(x0d, ((0, 0), (pad_n_h, pad_n_h), (pad_n_w, pad_n_w), (0, 0)), mode='reflect')
            x45d = np.pad(x45d, ((0, 0), (pad_n_h, pad_n_h), (pad_n_w, pad_n_w), (0, 0)), mode='reflect')
            xm45d = np.pad(xm45d, ((0, 0), (pad_n_h, pad_n_h), (pad_n_w, pad_n_w), (0, 0)), mode='reflect')

        y = y[:, :, :, np.newaxis]
        yield ([x90d, x0d, x45d, xm45d], y)

########## prepare train generator ##########
def prepare_train_generator(hp, logger=None):
    train_generator = get_train_generator(hp, logger=logger)
    return train_generator
