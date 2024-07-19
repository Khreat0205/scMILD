import sys, os
sys.path.append(r"src")
import train_scMILD
import torch 
torch.set_num_threads(16)
train_scMILD.train_scMILD(data_dir='data/Lupus',
                          dataset="Lupus",
                          device_num=4,
                          data_dim=3000,
                          mil_latent_dim=16, 
                          student_batch_size=1024,
                          teacher_learning_rate=1e-4, 
                          student_learning_rate=1e-3,
                          encoder_learning_rate=1e-3,
                          scMILD_epoch=100,
                          scMILD_neg_weight=0.1, 
                          scMILD_stuOpt=1,
                          scMILD_patience=10,
                          val_combined_metric=False,
                          add_suffix="_0604_lupus_test_433_op_gmm_device4_only_using_loss_switch",
                          n_exp=8, exp=None, gmm=True, use_loss = True)