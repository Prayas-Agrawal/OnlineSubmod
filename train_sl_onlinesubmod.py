import logging
import wandb
import os
import os.path as osp
import sys
import time
import torch
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim
from ray import tune
from cords.selectionstrategies.helpers.ssl_lib.param_scheduler import scheduler as step_scheduler
from cords.utils.data.data_utils import WeightedSubset
from cords.utils.data.dataloader.SL.adaptive import GLISTERDataLoader, AdaptiveRandomDataLoader, StochasticGreedyDataLoader,\
    CRAIGDataLoader, GradMatchDataLoader, OnlineSubmodDataLoader, RandomDataLoader, WeightedRandomDataLoader, MILODataLoader, SELCONDataLoader
from cords.utils.data.dataloader.SL.nonadaptive import FacLocDataLoader, MILOFixedDataLoader
from cords.utils.data.datasets.SL import gen_dataset
from cords.utils.models import *
from cords.utils.data.data_utils.collate import *
import pickle
from datetime import datetime
import submod
from submodlib import FacilityLocationFunction, GraphCutFunction, \
    DisparityMinFunction, DisparitySumFunction, LogDeterminantFunction, SetCoverFunction, ProbabilisticSetCoverFunction
from generate_order import compute_dino_image_embeddings, load_feature_model
import torchvision
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision import transforms
from torch.utils.data import DataLoader

args = dict(
    select_every=dict(count=5, mode="step", use_prev_best_arm_for_next_batch= True),
    # warm_start = 40,
    sampling_mode="Uniform",
    submod_budget=0.1,
    batch_size=128,
    num_val_points=32,
    lamb=0.1,
    pi=1.1,
    epochs = 300,
    eta_n = 0.5,
    model_name = "resnet18",
)
device = "cuda"
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, z):
        super(MyDataset, self).__init__()
        self.x = x
        self.y = y
        self.z = z

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        z = self.z[index]
        return (x, y), z
def extract_features(dataloader, load_from=None):
    feature_model, feature_extractor = load_feature_model(device)
    if(load_from is not None and os.path.exists(load_from)):
        print(f"***Features exist, Loading presaved from {load_from}***")
        features, labels, images = None, None, None
        with open(load_from, 'rb') as f:
            features, labels, images = pickle.load(f)
        return features, labels, images
    print(f"***Features dont exist, Creating new features for {load_from}***")
    features_list = []
    labels_list = []
    images_list = []
    t = 0
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        # Convert the input tensor to FloatTensor if it's of type ByteTensor
        if images.dtype == torch.uint8:
            images = images.float()
        features_batch = compute_dino_image_embeddings(images,device, model=feature_model, feature_extractor=feature_extractor, return_tensor=True)
        features_list.append(features_batch)
        labels_list.extend(labels)
        images_list.append(images)

    # Concatenate all feature batches into a single tensor
    features = torch.cat(features_list, dim=0)
    labels = torch.tensor(labels_list)
    images = torch.cat(images_list, dim=0)
    print(f"***Saving features and labels***")
    with open(load_from, 'wb') as f:
        pickle.dump((features, labels, images), f)
    return features, labels, images
data_dir = "/home/ganesh/AAI_prayas/AAAI/data"
features_dir = f"{data_dir}/features"
score_dir = f"{data_dir}/score_pkl"
subset_dir = f"{data_dir}/subset_pkl"
features_train = f"{data_dir}/features/train.pkl"
features_test = f"{data_dir}/features/test.pkl"
seed_value = 42 
torch.manual_seed(seed_value)

# Define data transforms
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

process_train_dataloader = DataLoader(trainset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
process_test_dataloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
# Extract features from train and test dataloaders
print(f"train loader:{len(process_train_dataloader)}, test loader: {len(process_test_dataloader)}")
x_train_features, x_train_labels, x_train_images = extract_features(process_train_dataloader, load_from=features_train)
x_test_features, x_test_labels, x_test_images = extract_features(process_test_dataloader, load_from=features_test)

# Create new dataloaders for features with the batch size of 64
train_features_dataset = MyDataset(x_train_features, x_train_images, x_train_labels )
test_features_dataset = MyDataset(x_test_features, x_test_images, x_test_labels, )

class TrainClassifier:
    def __init__(self, config_file_data):
        self.cfg = config_file_data
        results_dir = osp.abspath(osp.expanduser(self.cfg.train_args.results_dir))
        
        if self.cfg.dss_args.type in ['StochasticGreedyExploration', 'WeightedRandomExploration', 'SGE', 'WRE']:
            subset_selection_name = self.cfg.dss_args.type + "_" + self.cfg.dss_args.submod_function + "_" + str(self.cfg.dss_args.kw)
        elif self.cfg.dss_args.type in ['MILO']:
            subset_selection_name = self.cfg.dss_args.type + "_" + self.cfg.dss_args.submod_function + "_" + str(self.cfg.dss_args.gc_ratio) + "_" + str(self.cfg.dss_args.kw)
        else:
            subset_selection_name = self.cfg.dss_args.type
            
        all_logs_dir = os.path.join(results_dir, 
                                    self.cfg.setting,
                                    self.cfg.dataset.name,
                                    subset_selection_name,
                                    self.cfg.model.architecture,
                                    str(self.cfg.dss_args.fraction),
                                    str(self.cfg.dss_args.select_every),
                                    str(self.cfg.train_args.run))

        os.makedirs(all_logs_dir, exist_ok=True)
        # setup logger
        plain_formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                                            datefmt="%m/%d %H:%M:%S")
        now = datetime.now()
        current_time = now.strftime("%y/%m/%d %H:%M:%S")
        self.logger = logging.getLogger(__name__+"  " + current_time)
        self.logger.setLevel(logging.INFO)
        s_handler = logging.StreamHandler(stream=sys.stdout)
        s_handler.setFormatter(plain_formatter)
        s_handler.setLevel(logging.INFO)
        self.logger.addHandler(s_handler)
        f_handler = logging.FileHandler(os.path.join(all_logs_dir, self.cfg.dataset.name + "_" +
                                                     self.cfg.dss_args.type + ".log"), mode='w')
        f_handler.setFormatter(plain_formatter)
        f_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(f_handler)
        self.logger.propagate = False

    
    """
    ############################## Loss Evaluation ##############################
    """

    def model_eval_loss(self, data_loader, model, criterion):
        total_loss = 0
        with torch.no_grad():
            for _, inputs, targets in data_loader:
                inputs, targets = inputs.to(self.cfg.train_args.device), \
                                  targets.to(self.cfg.train_args.device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss

    """
    ############################## Model Creation ##############################
    """

    def create_model(self):
        if self.cfg.model.architecture == 'RegressionNet':
            model = RegressionNet(self.cfg.model.input_dim)
        elif self.cfg.model.architecture == 'ResNet18':
            model = ResNet18(self.cfg.model.numclasses)
            if self.cfg.dataset.name in ['cifar10', 'cifar100', 'tinyimagenet']:
                model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                model.maxpool = nn.Identity()
        elif self.cfg.model.architecture == 'ResNet101':
            model = ResNet101(self.cfg.model.numclasses)
            if self.cfg.dataset.name in ['cifar10', 'cifar100', 'tinyimagenet']:
                model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                model.maxpool = nn.Identity()
        elif self.cfg.model.architecture == 'MnistNet':
            model = MnistNet()
        elif self.cfg.model.architecture == 'ResNet164':
            model = ResNet164(self.cfg.model.numclasses)
        elif self.cfg.model.architecture == 'MobileNet':
            model = MobileNet(self.cfg.model.numclasses)
        elif self.cfg.model.architecture == 'MobileNetV2':
            model = MobileNetV2(self.cfg.model.numclasses)
        elif self.cfg.model.architecture == 'MobileNet2':
            model = MobileNet2(output_size=self.cfg.model.numclasses)
        elif self.cfg.model.architecture == 'HyperParamNet':
            model = HyperParamNet(self.cfg.model.l1, self.cfg.model.l2)
        elif self.cfg.model.architecture == 'ThreeLayerNet':
            model = ThreeLayerNet(self.cfg.model.input_dim, self.cfg.model.num_classes, self.cfg.model.h1, self.cfg.model.h2)
        elif self.cfg.model.architecture == 'LSTM':
            model = LSTMClassifier(self.cfg.model.numclasses, self.cfg.model.wordvec_dim, \
                 self.cfg.model.weight_path, self.cfg.model.num_layers, self.cfg.model.hidden_size)
        else:
            raise(NotImplementedError)
        model = model.to(self.cfg.train_args.device)
        return model

    """
    ############################## Loss Type, Optimizer and Learning Rate Scheduler ##############################
    """

    def loss_function(self):
        if self.cfg.loss.type == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
            criterion_nored = nn.CrossEntropyLoss(reduction='none')
        elif self.cfg.loss.type == "MeanSquaredLoss":
            criterion = nn.MSELoss()
            criterion_nored = nn.MSELoss(reduction='none')
        return criterion, criterion_nored

    def optimizer_with_scheduler(self, model):
        if self.cfg.optimizer.type == 'sgd':
            if ('ResNet' in self.cfg.model.architecture) and ('lr1' in self.cfg.optimizer.keys()) and ('lr2' in self.cfg.optimizer.keys()) and ('lr3' in self.cfg.optimizer.keys()):
                optimizer = optim.SGD( [{"params": model.linear.parameters(), "lr": self.cfg.optimizer.lr1},
                                        {"params": model.layer4.parameters(), "lr": self.cfg.optimizer.lr2},
                                        {"params": model.layer3.parameters(), "lr": self.cfg.optimizer.lr2},
                                        {"params": model.layer2.parameters(), "lr": self.cfg.optimizer.lr2},
                                        {"params": model.layer1.parameters(), "lr": self.cfg.optimizer.lr2},
                                        {"params": model.conv1.parameters(), "lr": self.cfg.optimizer.lr3}],
                                    lr=self.cfg.optimizer.lr,
                                    momentum=self.cfg.optimizer.momentum,
                                    weight_decay=self.cfg.optimizer.weight_decay,
                                    nesterov=self.cfg.optimizer.nesterov)
            else:
                optimizer = optim.SGD(model.parameters(),
                                    lr=self.cfg.optimizer.lr,
                                    momentum=self.cfg.optimizer.momentum,
                                    weight_decay=self.cfg.optimizer.weight_decay,
                                    nesterov=self.cfg.optimizer.nesterov)
        elif self.cfg.optimizer.type == "adam":
            optimizer = optim.Adam(model.parameters(), lr=self.cfg.optimizer.lr, weight_decay=self.cfg.optimizer.weight_decay)
        elif self.cfg.optimizer.type == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(), lr=self.cfg.optimizer.lr)

        if self.cfg.scheduler.type == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=self.cfg.scheduler.T_max)
        elif self.cfg.scheduler.type == 'cosine_annealing_WS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                   T_0=self.cfg.scheduler.T_0,
                                                                   T_mult=self.cfg.scheduler.T_mult)
        elif self.cfg.scheduler.type == 'linear_decay':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                        step_size=self.cfg.scheduler.stepsize, 
                                                        gamma=self.cfg.scheduler.gamma)
        elif self.cfg.scheduler.type == 'multistep':    
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg.scheduler.milestones,
                                                             gamma=self.cfg.scheduler.gamma)
        elif self.cfg.scheduler.type == 'cosine_annealing_step':
            scheduler = step_scheduler.CosineAnnealingLR(optimizer, max_iteration=self.cfg.scheduler.max_steps)
        else:
            scheduler = None
        return optimizer, scheduler

    @staticmethod
    def generate_cumulative_timing(mod_timing):
        tmp = 0
        mod_cum_timing = np.zeros(len(mod_timing))
        for i in range(len(mod_timing)):
            tmp += mod_timing[i]
            mod_cum_timing[i] = tmp
        return mod_cum_timing

    @staticmethod
    def save_ckpt(state, ckpt_path):
        torch.save(state, ckpt_path)

    @staticmethod
    def load_ckpt(ckpt_path, model, optimizer):
        checkpoint = torch.load(ckpt_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss = checkpoint['loss']
        metrics = checkpoint['metrics']
        return start_epoch, model, optimizer, loss, metrics

    def count_pkl(self, path):
        if not osp.exists(path):
            return -1
        return_val = 0
        file = open(path, 'rb')
        while(True):
            try:
                _ = pickle.load(file)
                return_val += 1
            except EOFError:
                break
        file.close()
        return return_val
    def do_dss(self, args, step, epoch):
        select_every_count = args["select_every"]["count"]
        select_every_mode = args["select_every"]["mode"]
        num = epoch if select_every_mode == "epoch" else step
        return select_every_count > 0 and num%select_every_count == 0 and num > 0
    
    # def do_dss(self, args, step, epoch, warm_start):
    #     if(warm_start >= epoch+1): return False
    #     select_every_count = args["select_every"]["count"]
    #     select_every_mode = args["select_every"]["mode"]
    #     num = epoch if select_every_mode == "epoch" else step
    #     return select_every_count > 0 and num%select_every_count == 0 and num > 0
        

    def train(self, **kwargs):
        """
        ############################## General Training Loop with Data Selection Strategies ##############################
        """
        # Loading the Dataset
        logger = self.logger
        # if ('trainset' in kwargs) and ('validset' in kwargs) and ('testset' in kwargs) and ('num_cls' in kwargs):
        #     trainset, validset, testset, num_cls = kwargs['trainset'], kwargs['validset'], kwargs['testset'], kwargs['num_cls']
        # else:
        #     #logger.info(self.cfg)
        #     if self.cfg.dataset.feature == 'classimb':
        #         trainset, validset, testset, num_cls = gen_dataset(self.cfg.dataset.datadir,
        #                                                         self.cfg.dataset.name,
        #                                                         self.cfg.dataset.feature,
        #                                                         classimb_ratio=self.cfg.dataset.classimb_ratio, dataset=self.cfg.dataset)
        #     else:
        #         trainset, validset, testset, num_cls = gen_dataset(self.cfg.dataset.datadir,
        #                                                         self.cfg.dataset.name,
        #                                                         self.cfg.dataset.feature, dataset=self.cfg.dataset)

        trn_batch_size = self.cfg.dataloader.batch_size
        val_batch_size = self.cfg.dataloader.batch_size
        tst_batch_size = self.cfg.dataloader.batch_size

        batch_sampler = lambda _, __ : None
        drop_last = False
        
        # if self.cfg.dataset.name == "sst2_facloc" and self.count_pkl(self.cfg.dataset.ss_path) == 1 and self.cfg.dss_args.type == 'FacLoc':
        #     self.cfg.dss_args.type = 'Full'
        #     file_ss = open(self.cfg.dataset.ss_path, 'rb')
        #     ss_indices = pickle.load(file_ss)
        #     file_ss.close()
        #     trainset = torch.utils.data.Subset(trainset, ss_indices)

        if 'collate_fn' not in self.cfg.dataloader.keys():
            collate_fn = None
        else:
            collate_fn = self.cfg.dataloader.collate_fn


        # Creating the Data Loaders
        trainloader = torch.utils.data.DataLoader(train_features_dataset, batch_size=trn_batch_size, sampler=batch_sampler(train_features_dataset, trn_batch_size),
                                                  shuffle=False, pin_memory=False, collate_fn = collate_fn, drop_last=drop_last)

        valloader = torch.utils.data.DataLoader(test_features_dataset, batch_size=val_batch_size, sampler=batch_sampler(test_features_dataset, val_batch_size),
                                                shuffle=False, pin_memory=False, collate_fn = collate_fn, drop_last=drop_last)

        testloader = torch.utils.data.DataLoader(test_features_dataset, batch_size=tst_batch_size, sampler=batch_sampler(test_features_dataset, tst_batch_size),
                                                 shuffle=False, pin_memory=False, collate_fn = collate_fn, drop_last=drop_last)
	
        train_eval_loader = torch.utils.data.DataLoader(train_features_dataset, batch_size=trn_batch_size * 20, sampler=batch_sampler(train_features_dataset, trn_batch_size),
                                                  shuffle=False, pin_memory=False, collate_fn = collate_fn, drop_last=drop_last)

        val_eval_loader = torch.utils.data.DataLoader(test_features_dataset, batch_size=val_batch_size * 20, sampler=batch_sampler(test_features_dataset, val_batch_size),
                                                shuffle=False, pin_memory=False, collate_fn = collate_fn, drop_last=drop_last)

        test_eval_loader = torch.utils.data.DataLoader(test_features_dataset, batch_size=tst_batch_size * 20, sampler=batch_sampler(test_features_dataset, tst_batch_size),
                                                 shuffle=False, pin_memory=False, collate_fn = collate_fn, drop_last=drop_last)
						 
        substrn_losses = list()  # np.zeros(cfg['train_args']['num_epochs'])
        trn_losses = list()
        val_losses = list()  # np.zeros(cfg['train_args']['num_epochs'])
        tst_losses = list()
        subtrn_losses = list()
        timing = []
        trn_acc = list()
        val_acc = list()  # np.zeros(cfg['train_args']['num_epochs'])
        tst_acc = list()  # np.zeros(cfg['train_args']['num_epochs'])
        best_acc = list()
        curr_best_acc = 0
        subtrn_acc = list()  # np.zeros(cfg['train_args']['num_epochs'])

        # Checkpoint file
        checkpoint_dir = osp.abspath(osp.expanduser(self.cfg.ckpt.dir))
        
        if self.cfg.dss_args.type in ['StochasticGreedyExploration', 'WeightedRandomExploration', 'SGE', 'WRE']:
            subset_selection_name = self.cfg.dss_args.type + "_" + self.cfg.dss_args.submod_function + "_" + str(self.cfg.dss_args.kw)
        elif self.cfg.dss_args.type in ['MILO']:
            subset_selection_name = self.cfg.dss_args.type + "_" + self.cfg.dss_args.submod_function + "_" + str(self.cfg.dss_args.gc_ratio) + "_" + str(self.cfg.dss_args.kw)
        else:
            subset_selection_name = self.cfg.dss_args.type
        
        ckpt_dir = os.path.join(checkpoint_dir, 
                                self.cfg.setting,
                                self.cfg.dataset.name,
                                subset_selection_name,
                                self.cfg.model.architecture,
                                str(self.cfg.dss_args.fraction),
                                str(self.cfg.dss_args.select_every),
                                str(self.cfg.train_args.run))
                                
        checkpoint_path = os.path.join(ckpt_dir, 'model.pt')
        os.makedirs(ckpt_dir, exist_ok=True)

        # Model Creation
        model = self.create_model()
        if self.cfg.train_args.wandb:
            wandb.watch(model)

        # model1 = self.create_model()

        #Initial Checkpoint Directory
        init_ckpt_dir = os.path.abspath(os.path.expanduser("checkpoints"))
        os.makedirs(init_ckpt_dir, exist_ok=True)
        
        model_name = ""
        for key in self.cfg.model.keys():
            if r"/" not in str(self.cfg.model[key]):
                model_name += (str(self.cfg.model[key]) + "_")

        if model_name[-1] == "_":
            model_name = model_name[:-1]
            
        if not os.path.exists(os.path.join(init_ckpt_dir, model_name + ".pt")):
            ckpt_state = {'state_dict': model.state_dict()}
            # save checkpoint
            self.save_ckpt(ckpt_state, os.path.join(init_ckpt_dir, model_name + ".pt"))
        else:
            checkpoint = torch.load(os.path.join(init_ckpt_dir, model_name + ".pt"))
            model.load_state_dict(checkpoint['state_dict'])

        # Loss Functions
        criterion, criterion_nored = self.loss_function()

        
        if self.cfg.scheduler.type == "cosine_annealing_step":
            if self.cfg.dss_args.type == "Full":
                self.cfg.scheduler.max_steps = math.ceil(len(list(dataloader.batch_sampler)) * self.cfg.train_args.num_epochs)
            else:
                self.cfg.scheduler.max_steps = math.ceil(len(list(dataloader.subset_loader.batch_sampler)) * self.cfg.train_args.num_epochs)
                 # * self.cfg.dss_args.fraction)

        # Getting the optimizer and scheduler
        optimizer, scheduler = self.optimizer_with_scheduler(model)

        """
        ############################## Custom Dataloader Creation ##############################
        """

        if 'collate_fn' not in self.cfg.dss_args:
                self.cfg.dss_args.collate_fn = None

        if self.cfg.dss_args.type in ['OnlineSubmod', 'OnlineSubmodPB', 'OnlineSubmod-Warm', 'OnlineSubmodPB-Warm']:
            """
            ############################## OnlineSubmod Dataloader Additional Arguments ##############################
            """
            self.cfg.dss_args.model = model
            self.cfg.dss_args.loss = criterion_nored
            self.cfg.dss_args.eta = self.cfg.optimizer.lr
            self.cfg.dss_args.num_classes = self.cfg.model.numclasses
            self.cfg.dss_args.num_epochs = self.cfg.train_args.num_epochs
            self.cfg.dss_args.device = self.cfg.train_args.device
            dataloader = OnlineSubmodDataLoader(trainloader, valloader, self.cfg.dss_args, logger, args["num_val_points"],
                                batch_size=self.cfg.dataloader.batch_size,
                                shuffle=False,
                                pin_memory=False, # PIN MEMORY
                                )

        else:
            raise NotImplementedError

        """
        ################################################# Checkpoint Loading #################################################
        """

        if self.cfg.ckpt.is_load:
            start_epoch, model, optimizer, ckpt_loss, load_metrics = self.load_ckpt(checkpoint_path, model, optimizer)
            logger.info("Loading saved checkpoint model at epoch: {0:d}".format(start_epoch))
            for arg in load_metrics.keys():
                if arg == "val_loss":
                    val_losses = load_metrics['val_loss']
                if arg == "val_acc":
                    val_acc = load_metrics['val_acc']
                if arg == "tst_loss":
                    tst_losses = load_metrics['tst_loss']
                if arg == "tst_acc":
                    tst_acc = load_metrics['tst_acc']
                    best_acc = load_metrics['best_acc']
                if arg == "trn_loss":
                    trn_losses = load_metrics['trn_loss']
                if arg == "trn_acc":
                    trn_acc = load_metrics['trn_acc']
                if arg == "subtrn_loss":
                    subtrn_losses = load_metrics['subtrn_loss']
                if arg == "subtrn_acc":
                    subtrn_acc = load_metrics['subtrn_acc']
                if arg == "time":
                    timing = load_metrics['time']
        else:
            start_epoch = 0

        """
        ################################################# Training Loop #################################################
        """
        opt_subset = {
            "images": None,
            "features": None,
            "labels": None,
        }
        
        
        train_time = 0
        for epoch in range(start_epoch, self.cfg.train_args.num_epochs+1):
            """
            ################################################# Evaluation Loop #################################################
            """
            
            print_args = self.cfg.train_args.print_args
            if (epoch % self.cfg.train_args.print_every == 0) or (epoch == self.cfg.train_args.num_epochs) or (epoch == 0):
                trn_loss = 0
                trn_correct = 0
                trn_total = 0
                val_loss = 0
                val_correct = 0
                val_total = 0
                tst_correct = 0
                tst_total = 0
                tst_loss = 0
                model.eval()
                logger_dict = {}
                if ("trn_loss" in print_args) or ("trn_acc" in print_args):
                    samples=0
		            
                    with torch.no_grad():
                        for (_, inputs), targets in train_eval_loader:

                            inputs, targets = inputs.to(self.cfg.train_args.device), \
                                              targets.to(self.cfg.train_args.device, non_blocking=True)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            trn_loss += (loss.item() * train_eval_loader.batch_size)
                            samples += targets.shape[0]
                            if "trn_acc" in print_args:
                                _, predicted = outputs.max(1)
                                trn_total += targets.size(0)
                                trn_correct += predicted.eq(targets).sum().item()
                        trn_loss = trn_loss/samples
                        trn_losses.append(trn_loss)
                        logger_dict['trn_loss'] = trn_loss
                    if "trn_acc" in print_args:
                        trn_acc.append(trn_correct / trn_total)
                        logger_dict['trn_acc'] = trn_correct / trn_total

                if ("val_loss" in print_args) or ("val_acc" in print_args):
                    samples =0
                    with torch.no_grad():
                        for (_, inputs), targets in val_eval_loader:

                            inputs, targets = inputs.to(self.cfg.train_args.device), \
                                              targets.to(self.cfg.train_args.device, non_blocking=True)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            val_loss += (loss.item() * val_eval_loader.batch_size)
                            samples += targets.shape[0]
                            if "val_acc" in print_args:
                                _, predicted = outputs.max(1)
                                val_total += targets.size(0)
                                val_correct += predicted.eq(targets).sum().item()
                        val_loss = val_loss/samples
                        val_losses.append(val_loss)
                        logger_dict['val_loss'] = val_loss

                    if "val_acc" in print_args:
                        val_acc.append(val_correct / val_total)
                        logger_dict['val_acc'] = val_correct / val_total

                if ("tst_loss" in print_args) or ("tst_acc" in print_args):
                    samples =0
                    with torch.no_grad():
                        for (_, inputs), targets in test_eval_loader:

                            inputs, targets = inputs.to(self.cfg.train_args.device), \
                                              targets.to(self.cfg.train_args.device, non_blocking=True)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            tst_loss += (loss.item() * test_eval_loader.batch_size)
                            samples += targets.shape[0]
                            if "tst_acc" in print_args:
                                _, predicted = outputs.max(1)
                                tst_total += targets.size(0)
                                tst_correct += predicted.eq(targets).sum().item()
                        tst_loss = tst_loss/samples
                        tst_losses.append(tst_loss)
                        logger_dict['tst_loss'] = tst_loss

                    if (tst_correct/tst_total) > curr_best_acc:
                        curr_best_acc = (tst_correct/tst_total)

                    if "tst_acc" in print_args:
                        tst_acc.append(tst_correct / tst_total)
                        best_acc.append(curr_best_acc)
                        logger_dict['tst_acc'] = tst_correct / tst_total
                        logger_dict['best_acc'] = curr_best_acc

                if "subtrn_acc" in print_args:
                    if epoch == 0:
                        subtrn_acc.append(0)
                        logger_dict['subtrn_acc'] = 0
                    else:    
                        subtrn_acc.append(subtrn_correct / subtrn_total)
                        logger_dict['subtrn_acc'] = subtrn_correct / subtrn_total

                if "subtrn_losses" in print_args:
                    if epoch == 0:
                        subtrn_losses.append(0)
                        logger_dict['subtrn_loss'] = 0
                    else: 
                        subtrn_losses.append(subtrn_loss)
                        logger_dict['subtrn_loss'] = subtrn_loss

                print_str = "Epoch: " + str(epoch)
                logger_dict['Epoch'] = epoch
                logger_dict['Time'] = train_time
                timing.append(train_time)
                
                if self.cfg.train_args.wandb:
                    wandb.log(logger_dict)

                """
                ################################################# Results Printing #################################################
                """

                for arg in print_args:
                    if arg == "val_loss":
                        print_str += " , " + "Validation Loss: " + str(val_losses[-1])

                    if arg == "val_acc":
                        print_str += " , " + "Validation Accuracy: " + str(val_acc[-1])

                    if arg == "tst_loss":
                        print_str += " , " + "Test Loss: " + str(tst_losses[-1])

                    if arg == "tst_acc":
                        print_str += " , " + "Test Accuracy: " + str(tst_acc[-1])
                        print_str += " , " + "Best Accuracy: " + str(best_acc[-1])

                    if arg == "trn_loss":
                        print_str += " , " + "Training Loss: " + str(trn_losses[-1])

                    if arg == "trn_acc":
                        print_str += " , " + "Training Accuracy: " + str(trn_acc[-1])

                    if arg == "subtrn_loss":
                        print_str += " , " + "Subset Loss: " + str(subtrn_losses[-1])

                    if arg == "subtrn_acc":
                        print_str += " , " + "Subset Accuracy: " + str(subtrn_acc[-1])

                    if arg == "time":
                        print_str += " , " + "Timing: " + str(timing[-1])

                # report metric to ray for hyperparameter optimization
                if 'report_tune' in self.cfg and self.cfg.report_tune and len(dataloader) and epoch > 0:
                    tune.report(mean_accuracy=np.array(val_acc).max())

                logger.info(print_str)

            subtrn_loss = 0
            subtrn_correct = 0
            subtrn_total = 0
            model.train()
            start_time = time.time()
            step = 0
            # moment_sum = 0
            # best_arm = None
            # use_prev = args["select_every"]["use_prev_best_arm_for_next_batch"]
            for (_, images), targets, weights in tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.cfg.train_args.num_epochs}"):
                images = images.to(self.cfg.train_args.device, non_blocking=True)
                targets = targets.to(self.cfg.train_args.device, non_blocking=True)
                optimizer.zero_grad()
                final_images =  images
                final_labels = targets
                with torch.autocast(device_type=device):
                    outputs = model(final_images)
                    loss = criterion(outputs, final_labels)
                # loss = torch.dot(losses, weights / (weights.sum()))
                loss.backward()
                subtrn_loss += loss.item()
                optimizer.step()
                if self.cfg.scheduler.type == "cosine_annealing_step":
                    scheduler.step()
                if not self.cfg.is_reg:
                    _, predicted = outputs.max(1)
                    subtrn_total += final_labels.size(0)
                    subtrn_correct += predicted.eq(final_labels).sum().item()
                step += 1
            epoch_time = time.time() - start_time
            if (scheduler is not None) and (self.cfg.scheduler.type != "cosine_annealing_step"):
                scheduler.step()
            # timing.append(epoch_time)
            train_time += epoch_time
            

            """
            ################################################# Checkpoint Saving #################################################
            """

            if ((epoch + 1) % self.cfg.ckpt.save_every == 0) and self.cfg.ckpt.is_save:

                metric_dict = {}

                for arg in print_args:
                    if arg == "val_loss":
                        metric_dict['val_loss'] = val_losses
                    if arg == "val_acc":
                        metric_dict['val_acc'] = val_acc
                    if arg == "tst_loss":
                        metric_dict['tst_loss'] = tst_losses
                    if arg == "tst_acc":
                        metric_dict['tst_acc'] = tst_acc
                        metric_dict['best_acc'] = best_acc
                    if arg == "trn_loss":
                        metric_dict['trn_loss'] = trn_losses
                    if arg == "trn_acc":
                        metric_dict['trn_acc'] = trn_acc
                    if arg == "subtrn_loss":
                        metric_dict['subtrn_loss'] = subtrn_losses
                    if arg == "subtrn_acc":
                        metric_dict['subtrn_acc'] = subtrn_acc
                    if arg == "time":
                        metric_dict['time'] = timing

                ckpt_state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': self.loss_function(),
                    'metrics': metric_dict
                }

                # save checkpoint
                self.save_ckpt(ckpt_state, checkpoint_path)
                logger.info("Model checkpoint saved at epoch: {0:d}".format(epoch + 1))

        """
        ################################################# Results Summary #################################################
        """
        original_idxs = set([x for x in range(len(train_features_dataset))])
        encountered_idxs = []
        if self.cfg.dss_args.type != 'Full':
            for key in dataloader.selected_idxs.keys():
                encountered_idxs.extend(dataloader.selected_idxs[key])
            encountered_idxs = set(encountered_idxs)
            rem_idxs = original_idxs.difference(encountered_idxs)
            encountered_percentage = len(encountered_idxs)/len(original_idxs)

            # logger.info("Selected Indices: ") 
            # logger.info(dataloader.selected_idxs)
            logger.info("Percentages of data samples encountered during training: %.2f", encountered_percentage)
            # logger.info("Not Selected Indices: ")
            # logger.info(rem_idxs)

            if self.cfg.train_args.wandb:
                wandb.log({
                           "Data Samples Encountered(in %)": encountered_percentage
                           })
                           
        logger.info(self.cfg.dss_args.type + " Selection Run---------------------------------")
        logger.info("Final SubsetTrn: {0:f}".format(subtrn_loss))
        if "val_loss" in print_args:
            if "val_acc" in print_args:
                logger.info("Validation Loss: %.2f , Validation Accuracy: %.2f", val_loss, val_acc[-1])
            else:
                logger.info("Validation Loss: %.2f", val_loss)

        if "tst_loss" in print_args:
            if "tst_acc" in print_args:
                logger.info("Test Loss: %.2f, Test Accuracy: %.2f, Best Accuracy: %.2f", tst_loss, tst_acc[-1], best_acc[-1])
            else:
                logger.info("Test Data Loss: %f", tst_loss)
        logger.info('---------------------------------------------------------------------')
        logger.info(self.cfg.dss_args.type)
        logger.info('---------------------------------------------------------------------')

        """
        ################################################# Final Results Logging #################################################
        """

        if "val_acc" in print_args:
            val_str = "Validation Accuracy: "
            for val in val_acc:
                if val_str == "Validation Accuracy: ":
                    val_str = val_str + str(val)
                else:
                    val_str = val_str + " , " + str(val)
            logger.info(val_str)

        if "tst_acc" in print_args:
            tst_str = "Test Accuracy: "
            for tst in tst_acc:
                if tst_str == "Test Accuracy: ":
                    tst_str = tst_str + str(tst)
                else:
                    tst_str = tst_str + " , " + str(tst)
            logger.info(tst_str)

            tst_str = "Best Accuracy: "
            for tst in best_acc:
                if tst_str == "Best Accuracy: ":
                    tst_str = tst_str + str(tst)
                else:
                    tst_str = tst_str + " , " + str(tst)
            logger.info(tst_str)

        if "time" in print_args:
            time_str = "Time: "
            for t in timing:
                if time_str == "Time: ":
                    time_str = time_str + str(t)
                else:
                    time_str = time_str + " , " + str(t)
            logger.info(time_str)

        omp_timing = np.array(timing)
        # omp_cum_timing = list(self.generate_cumulative_timing(omp_timing))
        logger.info("Total time taken by %s = %.4f ", self.cfg.dss_args.type, omp_timing[-1])
        return trn_acc, val_acc, tst_acc, best_acc, omp_timing
