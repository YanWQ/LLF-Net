import logging
import os
import json

def get_logger():
    logger_name = "run_logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def get_hp(nn_work_mode, json_file_name):
    # load hyper-parameters from json
    with open(os.path.join("Paras", json_file_name)) as json_file:
        json_data = json.load(json_file)
        hp = json_data[nn_work_mode]
    return hp

def load_weight(model, output_model_path):
    list_name = os.listdir(output_model_path)
    list_name.sort()
    ckp_name = list_name[-1]
    idx = ckp_name.find("iter")
    iter_n = int(ckp_name[idx + 4:idx + 8]) + 1
    model.load_weights(os.path.join(output_model_path, ckp_name))
    print("Weights will be loaded from the previous checkpoint \n(%s)" % ckp_name)
    return model, iter_n

def update_learning_rate(policy, learning_rate, iter, max_iter):
    for steps, lr in policy["step"].items():
        if iter == int(steps):
            learning_rate = lr
    return learning_rate


