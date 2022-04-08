import tensorflow as tf
import time
import os
def get_log_path(log_dir="logs/fit"):
    uniquename=time.strftime("logs_%Y_%m_%d_%H_%H_%S")
    log_path=os.path.join(log_dir,uniquename)
    print(f"saving logs at {log_path}")
    return log_path
log_dir=get_log_path()
def call_back_fun(log_dir):
    tensorboard_callbacks=tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    early_stopping=tf.keras.callbacks.EarlyStopping(patience=1,restore_best_weights=True,monitor='val_accuracy')

    check_point_path="model_check_point.h5"
    check_point=tf.keras.callbacks.ModelCheckpoint(check_point_path,save_best_only=True)
    callback_list=[tensorboard_callbacks,early_stopping,check_point]
    print(callback_list)
    return callback_list
