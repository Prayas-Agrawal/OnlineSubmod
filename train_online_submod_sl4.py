from cords.utils.config_utils import load_config_data

#config_file = './cords/configs/SL/config_gradmatchpb_mnist.py'
config_file = './cords/configs/SL/config_onlinesubmod_mnist.py'
cfg = load_config_data(config_file)
print(cfg)
from train_sl4 import TrainClassifier
clf = TrainClassifier(cfg)
clf.train()