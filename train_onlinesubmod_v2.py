from cords.utils.config_utils import load_config_data

#config_file = './cords/configs/SL/config_gradmatchpb_mnist.py'
config_file = './cords/configs/SL/config_onlinesubmodPB_cifar10.py'
cfg = load_config_data(config_file)
print(cfg)
from train_sl_onlinesubmod import TrainClassifier
clf = TrainClassifier(cfg)
clf.train()