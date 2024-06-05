import sys, os
sys.path.append(r"src")
import train_scMILD
train_scMILD.train_scMILD(data_dir='data/Lupus',
                          dataset="Lupus",
                          device_num=1,
                          data_dim=3000,
                          mil_latent_dim=16, 
                          student_batch_size=256,
                          teacher_learning_rate=1e-4, 
                          student_learning_rate=1e-4,
                          encoder_learning_rate=1e-4,
                          scMILD_epoch=500,
                          scMILD_neg_weight=0.1, 
                          scMILD_stuOpt=1,
                          scMILD_patience=15,
                          val_combined_metric=True,
                          add_suffix="_0604_lupus_test_444_op_gmm",
                          n_exp=8, exp=None, gmm=True)