import os
from src.utills.common import read_config
from src.utills.data_mang import get_data
from src.utills.model import create_model, save_model
import matplotlib.pyplot as plt
import pandas as pd



import argparse

def training(config_path):
    config = read_config(config_path)
    
    validation_datasize = config["params"]["validation_datasize"]
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)

    LOSS_FUNCTION = config["params"]["loss_function"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    NUM_CLASSES = config["params"]["num_classes"]

    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)

    EPOCHS = config["params"]["epochs"]
    VALIDATION_SET = (X_valid, y_valid)

    # create callbacks
    #CALLBACK_LIST = get_callbacks(config, X_train)

    histroy = model.fit(X_train, y_train, epochs=EPOCHS,
                    validation_data=VALIDATION_SET)
    

    artifacts_dir = config["artifacts"]["artifacts_dir"]
    model_dir = config["artifacts"]["model_dir"]
    
    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    
    model_name = config["artifacts"]["model_name"]
    
    

    save_model(model, model_name, model_dir_path)
    save_img(histroy,filename="plot.png", plot_dir="plots")

def save_img(histroy,filename,plot_dir):
    pd.DataFrame(histroy.history).plot(figsize=(10,7))
    plt.grid(True)
    plt.show()
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, filename)
    plt.savefig(plot_path)




if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config.yaml")

    parsed_args = args.parse_args()

    training(config_path=parsed_args.config)