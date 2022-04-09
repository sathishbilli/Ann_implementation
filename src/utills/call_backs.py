import tensorflow as tf
import numpy as np
import time
import os

def get_timestamp(name):
    tiemstamp=time.asctime().replace(" ","_").replace(":","_")
    unique_name=f"{name}_at_{tiemstamp}"

    return unique_name

def get_callbacks(config,x_train):
    logs=config["logs"]
    unique_dir_name=get_timestamp("tb_logs")
    
    TENSORBOARD_ROOT_LOG_DIR=os.path.join(logs["logs_dir"],logs["TENSORBOARD_ROOT_LOG_DIR"],unique_dir_name)
    os.makedirs(TENSORBOARD_ROOT_LOG_DIR,exist_ok=True)
    tensorboard_callbacks=tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_ROOT_LOG_DIR)
    file_writer=tf.summary.create_file_writer(logdir=TENSORBOARD_ROOT_LOG_DIR)
    with file_writer.as_default():
        img=np.reshape(x_train[10:30],(-1,28,28,1))# 20 28x28 1
        tf.summary.image("20 handwriten smaple",img,max_outputs=25,step=0)
    params=config["params"]
    early_stopping=tf.keras.callbacks.EarlyStopping(patience=params["patience"],restore_best_weights=params["restore_best_weights"],monitor='val_accuracy')
    
    ckpt_path=os.path.join(config["artifacts"]["artifacts_dir"],config["artifacts"]["CHECKPOINT_DIR"])
    os.makedirs(ckpt_path,exist_ok=True)
    check_point_path=os.path.join(ckpt_path,"model_check_point.h5")

    check_point=tf.keras.callbacks.ModelCheckpoint(check_point_path,save_best_only=True)
    callback_list=[tensorboard_callbacks,early_stopping,check_point]

    return callback_list
