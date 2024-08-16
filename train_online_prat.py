from cords.utils.config_utils import load_config_data

#config_file = './cords/configs/SL/config_gradmatchpb_mnist.py'
config_file = './cords/configs/SL/custom_configs_prateek_glister/config_onlinesubmod_cifar10_0.1.py'
cfg = load_config_data(config_file)
print(cfg)
from train_sl_prat import TrainClassifier
clf = TrainClassifier(cfg)
clf.train()