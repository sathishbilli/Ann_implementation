import tensorflow as tf
import time
import os
def create_model(loss_fun,Optimizer,Metrics,no_classes):
    layers=[
      tf.keras.layers.Flatten(input_shape=[28,28],name="inputLayer"),
      tf.keras.layers.Dense(300,activation="relu",name="hiddenLayer"),
      tf.keras.layers.Dense(100,activation="relu",name="hiddenLayer2"),
      tf.keras.layers.Dense(no_classes,activation="softmax",name="outputLayer")
    ]
    model_clf=tf.keras.models.Sequential(layers)
    model_clf.summary()
    
    model_clf.compile(optimizer=Optimizer,loss=loss_fun,metrics=Metrics)
    return model_clf# untrained model

def get_unique_filename(filename):
  unique_filename=time.strftime(f"%Y%m%d_%H%H%s_{filename}")
  return unique_filename


def save_model(model,model_name,model_dir):
  unique_filename=get_unique_filename(model_name)
  path_to_model=os.path.join(model_dir,unique_filename)
  model.save(path_to_model)
