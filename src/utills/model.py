import tensorflow as tf
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