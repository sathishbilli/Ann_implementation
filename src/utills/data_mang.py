import tensorflow as tf

def get_data(validation_datasize):
    mnist=tf.keras.datasets.mnist
    (x_tr_full,y_tr_full),(x_te_full,y_test)=mnist.load_data()
    x_valid,x_tr=x_tr_full[:validation_datasize]/255.,x_tr_full[validation_datasize:]/255.
    y_valid,y_tr=y_tr_full[:validation_datasize],y_tr_full[validation_datasize:]

    x_test=x_te_full/255
    return (x_tr,y_tr),(x_valid,y_valid),(x_test,y_test)


