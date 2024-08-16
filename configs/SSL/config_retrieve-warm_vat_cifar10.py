# Learning setting
config = dict(setting="SSL",
              dataset=dict(name="cifar10",
                           root="../data",
                           feature="dss",
                           type="pre-defined",
                           num_labels=4000,
                           val_ratio=0.1,
                           ood_ratio=0.5,
                           random_split=False,
                           whiten=False,
                           zca=True,
                           labeled_aug='WA',
                           unlabeled_aug='WA',
                           wa='t.t.f',
                           strong_aug=False),

              dataloader=dict(shuffle=True,
                              pin_memory=True,
                              num_workers=8,
                              l_batch_size=50,
                              ul_batch_size=50),

              model=dict(architecture='wrn',
                         type='pre-defined',
                         numclasses=10),

              ckpt=dict(is_load=False,
                        is_save=True,
                        checkpoint_model='model.ckpt',
                        checkpoint_optimizer='optimizer.ckpt',
                        start_iter=None,
                        checkpoint=10000),

              loss=dict(type='CrossEntropyLoss',
                        use_sigmoid=False),

              optimizer=dict(type="sgd",
                             momentum=0.9,
                             lr=0.03,
                             weight_decay=0,
                             nesterov=True,
                             tsa=False,
                             tsa_schedule='linear'),

              scheduler=dict(lr_decay="cos",
                             warmup_iter=0),

              ssl_args=dict(alg='vat',
                            coef=0.3,
                            ema_teacher=False,
                            ema_teacher_warmup=False,
                            ema_teacher_factor=0.999,
                            ema_apply_wd=False,
                            em=0,
                            threshold=None,
                            sharpen=None,
                            temp_softmax=None,
                            consis='ce',
                            eps=6,
                            xi=1e-6,
                            vat_iter=1
                            ),

              ssl_eval_args=dict(weight_average=False,
                                 wa_ema_factor=0.999,
                                 wa_apply_wd=False),

              dss_args=dict(type="RETRIEVE-Warm",
                            fraction=0.1,
                            select_every=20,
                            kappa=0.5,
                            linear_layer=False,
                            selection_type='Supervised',
                            greedy='Stochastic',
                            valid=True),

              train_args=dict(iteration=500000,
                              max_iter=-1,
                              device="cuda",
                              results_dir='results/',
                              disp=256,
                              seed=96)
              )
