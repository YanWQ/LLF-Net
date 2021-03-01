"""
# ==================================
# AUTHOR : Yan Li, Qiong Wang
# CREATE DATE : 03.13.2020
# Contact : liyanxian19@gmail.com
# ==================================
# Change History: None
# ==================================
"""
########## Import third-party libs (numpy, tensorflow) ##########
import numpy as np

########## remove training patches ##########
def check_disp_range(gt_disp):
    c_flag = 1
    if np.amax(gt_disp) > 2 and np.amax(gt_disp) < 50:
        c_flag = 1
    else:
        c_flag = 0
    return c_flag


########## augment training patches -> data augmentation ##########
def get_aug_rgb():
    rand_3color = 0.05 + np.random.rand(3)
    rand_3color = rand_3color / np.sum(rand_3color)
    R = rand_3color[0]
    G = rand_3color[1]
    B = rand_3color[2]
    return R, G, B

def get_aug_sid(rem=None, scene_nums=16):
    if rem is None:
        scene_ids = np.array(list(range(scene_nums)))

    scene_id = np.random.choice(scene_ids)
    return scene_id

def get_aug_cts(crop_seqs):
    # ct_os: central t view_offset, cs_os: central s view_offset
    if len(crop_seqs) == 9:
        ct_os = 0
        cs_os = 0
    elif len(crop_seqs) == 7:
        ct_os = np.random.randint(0, 3) - 1
        cs_os = np.random.randint(0, 3) - 1
    elif len(crop_seqs) == 5:
        ct_os = np.random.randint(0, 5) - 2
        cs_os = np.random.randint(0, 5) - 2
    elif len(crop_seqs) == 3:
        ct_os = np.random.randint(0, 7) - 3
        cs_os = np.random.randint(0, 7) - 3
    return ct_os, cs_os

def get_aug_scale(scales):
    kk = np.random.randint(scales)
    if kk < 8:
        scale = 1
    elif kk < 14:
        scale = 2
    elif kk < 17:
        scale = 3
    return scale

def get_aug_vu(scale, image_size, patch_size, interval=1):
    v_st = np.random.choice(np.arange(0, image_size[0] - scale * patch_size[0], interval))
    u_st = np.random.choice(np.arange(0, image_size[1] - scale * patch_size[1], interval))
    return v_st, u_st

def get_cv_offsets(crop_seqs):
    # ct_os: central t view_offset, cs_os: central s view_offset
    if len(crop_seqs) == 9:
        ct_os = 0
        cs_os = 0
    elif len(crop_seqs) == 7:
        ct_os = np.random.randint(0, 3) - 1
        cs_os = np.random.randint(0, 3) - 1
    elif len(crop_seqs) == 5:
        ct_os = np.random.randint(0, 5) - 2
        cs_os = np.random.randint(0, 5) - 2
    return ct_os, cs_os

def get_sum_diff(central_img, color_coefs, patch_height, patch_width):
    R, G, B = color_coefs
    image_center = (1.0 / 255.0) * np.squeeze(
        R * central_img[..., 0].astype('float32') +
        G * central_img[..., 1].astype('float32') +
        B * central_img[..., 2].astype('float32'))
    sum_diff = np.sum(np.abs(image_center - np.squeeze(image_center[int(0.5 * patch_height), int(0.5 * patch_width)])))
    return sum_diff


def data_augmentation(img_label_data, hp, aug_num=None, arrangement=None):
    aug_nums = hp["network"]["augmentation"]["rot_flip"]
    # mi checked 2019.1.17 12.06pm with orig code; 2020.01.04 rechecked
    x90d, x0d, x45d, xm45d, y = img_label_data
    # reverse indices
    indices = np.flip(np.arange(x90d.shape[-1]), axis=0).tolist()
    for batch_id in range(y.shape[0]):
        if hp["network"]["augmentation"]["gamma"] != []:
            gray_rand = (hp["network"]["augmentation"]["gamma"][1] - hp["network"]["augmentation"]["gamma"][0]) * np.random.rand() \
                        + hp["network"]["augmentation"]["gamma"][0]
            x90d[batch_id, :, :, :] = pow(x90d[batch_id, :, :, :], gray_rand)
            x0d[batch_id, :, :, :] = pow(x0d[batch_id, :, :, :], gray_rand)
            x45d[batch_id, :, :, :] = pow(x45d[batch_id, :, :, :], gray_rand)
            xm45d[batch_id, :, :, :] = pow(xm45d[batch_id, :, :, :], gray_rand)

        if aug_num is None:
            aug_num = np.random.randint(0, aug_nums)
        if aug_num == 1:  # 90 degree
            x90d_tmp3 = np.copy(np.rot90(x90d[batch_id, :, :, :], 1, (0, 1)))
            x0d_tmp3 = np.copy(np.rot90(x0d[batch_id, :, :, :], 1, (0, 1)))
            x45d_tmp3 = np.copy(np.rot90(x45d[batch_id, :, :, :], 1, (0, 1)))
            xm45d_tmp3 = np.copy(np.rot90(xm45d[batch_id, :, :, :], 1, (0, 1)))

            if hp["dataset_baseline"] == "wide":
                x90d[batch_id, :, :, :] = x0d_tmp3[:, :, indices]
                x45d[batch_id, :, :, :] = xm45d_tmp3
                x0d[batch_id, :, :, :] = x90d_tmp3
                xm45d[batch_id, :, :, :] = x45d_tmp3[:, :, indices]
            elif hp["dataset_baseline"] == "narrow":
                x90d[batch_id, :, :, :] = x0d_tmp3
                x45d[batch_id, :, :, :] = xm45d_tmp3
                x0d[batch_id, :, :, :] = x90d_tmp3[:, :, indices]
                xm45d[batch_id, :, :, :] = x45d_tmp3[:, :, indices]

            y[batch_id, ...] = np.copy(np.rot90(y[batch_id, ...], 1, (0, 1)))
        elif aug_num == 2:  # 180 degree
            x90d_tmp4 = np.copy(np.rot90(x90d[batch_id, :, :, :], 2, (0, 1)))
            x0d_tmp4 = np.copy(np.rot90(x0d[batch_id, :, :, :], 2, (0, 1)))
            x45d_tmp4 = np.copy(np.rot90(x45d[batch_id, :, :, :], 2, (0, 1)))
            xm45d_tmp4 = np.copy(np.rot90(xm45d[batch_id, :, :, :], 2, (0, 1)))

            x90d[batch_id, :, :, :] = x90d_tmp4[:, :, indices]
            x0d[batch_id, :, :, :] = x0d_tmp4[:, :, indices]
            x45d[batch_id, :, :, :] = x45d_tmp4[:, :, indices]
            xm45d[batch_id, :, :, :] = xm45d_tmp4[:, :, indices]

            y[batch_id, ...] = np.copy(np.rot90(y[batch_id, ...], 2, (0, 1)))
        elif aug_num == 3:  # 270 degree
            x90d_tmp5 = np.copy(np.rot90(x90d[batch_id, :, :, :], 3, (0, 1)))
            x0d_tmp5 = np.copy(np.rot90(x0d[batch_id, :, :, :], 3, (0, 1)))
            x45d_tmp5 = np.copy(np.rot90(x45d[batch_id, :, :, :], 3, (0, 1)))
            xm45d_tmp5 = np.copy(np.rot90(xm45d[batch_id, :, :, :], 3, (0, 1)))

            if hp["dataset_baseline"] == "wide":
                x90d[batch_id, :, :, :] = x0d_tmp5
                x0d[batch_id, :, :, :] = x90d_tmp5[:, :, indices]
                x45d[batch_id, :, :, :] = xm45d_tmp5[:, :, indices]
                xm45d[batch_id, :, :, :] = x45d_tmp5
            elif hp["dataset_baseline"] == "narrow":
                x90d[batch_id, :, :, :] = x0d_tmp5[:, :, indices]
                x0d[batch_id, :, :, :] = x90d_tmp5
                x45d[batch_id, :, :, :] = xm45d_tmp5[:, :, indices]
                xm45d[batch_id, :, :, :] = x45d_tmp5

            y[batch_id, ...] = np.copy(np.rot90(y[batch_id, ...], 3, (0, 1)))
        elif aug_num == 4:  # transpose
            if hp["dataset_baseline"] == "narrow":
                x90d_tmp6 = np.copy(np.transpose(np.squeeze(x90d[batch_id, :, :, :]), (1, 0, 2)))
                x0d_tmp6 = np.copy(np.transpose(np.squeeze(x0d[batch_id, :, :, :]), (1, 0, 2)))
                x45d_tmp6 = np.copy(np.transpose(np.squeeze(x45d[batch_id, :, :, :]), (1, 0, 2)))
                xm45d_tmp6 = np.copy(np.transpose(np.squeeze(xm45d[batch_id, :, :, :]), (1, 0, 2)))

                x0d[batch_id, :, :, :] = np.copy(x90d_tmp6[:, :, indices])
                x90d[batch_id, :, :, :] = np.copy(x0d_tmp6[:, :, indices])
                x45d[batch_id, :, :, :] = np.copy(x45d_tmp6[:, :, indices])
                xm45d[batch_id, :, :, :] = np.copy(xm45d_tmp6)  # [:,:,::-1])
        elif aug_num == 5:  # horizontal flipping
            x90d[batch_id, :, :, :] = np.copy(np.flip(x90d[batch_id, :, :, :], axis=1))
            x0d[batch_id, :, :, :] = np.copy(np.flip(np.flip(x0d[batch_id, :, :, :], axis=1), axis=2))
            x45d[batch_id, :, :, :] = np.copy(np.flip(np.flip(x45d[batch_id, :, :, :], axis=1), axis=2))
            xm45d[batch_id, :, :, :] = np.copy(np.flip(np.flip(xm45d[batch_id, :, :, :], axis=1), axis=2))
            y[batch_id, ...] = np.copy(np.flip(y[batch_id, ...], axis=1))
        elif aug_num == 6:  # vertical flipping
            x90d[batch_id, :, :, :] = np.copy(np.flip(np.flip(x90d[batch_id, :, :, :], axis=0), axis=2))
            x0d[batch_id, :, :, :] = np.copy(np.flip(x0d[batch_id, :, :, :], axis=0))
            x45d[batch_id, :, :, :] = np.copy(np.flip(np.flip(x45d[batch_id, :, :, :], axis=0), axis=2))
            xm45d[batch_id, :, :, :] = np.copy(np.flip(np.flip(xm45d[batch_id, :, :, :], axis=0), axis=2))
            y[batch_id, ...] = np.copy(np.flip(y[batch_id, ...], axis=0))

    return x90d, x0d, x45d, xm45d, y
