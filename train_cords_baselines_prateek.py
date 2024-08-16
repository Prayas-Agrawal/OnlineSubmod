
import sys
from cords.utils.config_utils import load_config_data

config_file_path = (sys.argv)


print(config_file_path[1])

config_file = './cords/configs/SL/custom_configs_craig_pb/config_craig_cifar_frac_0.1.py'

config_file = config_file_path[1]
cfg = load_config_data(config_file)
print(cfg)
from train_sl import TrainClassifier
clf = TrainClassifier(cfg)
clf.train()