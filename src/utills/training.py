from genericpath import exists
from src.utills.common import read_config
from src.utills.data_mang import get_data
from src.utills.model import create_model
from src.utills.model import save_model
import os
import argparse
def training(config_path):
    config=read_config(config_path)
    loss_fun=config["params"]["loss_function"]
    Optimizer=config["params"]["optimizer"]
    Metrics=config["params"]["metrics"]
    no_classes=config["params"]["no_classes"]

    validation_datasize=config["params"]["validation_datasize"]
    (x_tr,y_tr),(x_valid,y_valid),(x_test,y_test)=get_data(validation_datasize)
    model=create_model(loss_fun,Optimizer,Metrics,no_classes)
    
    Epochs=config["params"]["epochs"]
    Validation=(x_valid,y_valid)
    histroy=model.fit(x_tr,y_tr,epochs=Epochs,validation_data=Validation)
    model_name=config["artifacts"]["model_name"]
    artifacts_dir=config["artifacts"]["artifacts_dir"]
    
    model_dir=config["artifacts"]["model_dir"]
    model_dir_path=os.path.join(artifacts_dir,model_dir)
    os.mkdir(model_dir_path,exist_ok=True)

    save_model(model,model_name,model_dir_path)




    if __name__ == '__main__':
        args = argparse.parser.parse_args()

        args.add_argument('--config','-c',default='config.yaml')

        parsed_args = args.parse_args()
        training(config_path=parsed_args.config)