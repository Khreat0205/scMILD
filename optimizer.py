import os 
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from termcolor import colored
import numpy as np
import copy
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, f1_score
# from torch.utils.tensorboard import SummaryWriter


class Optimizer:
    def __init__(self, exp, model_teacher, model_student, model_encoder, optimizer_teacher, optimizer_student,
                optimizer_encoder,bag_train_dl, bag_val_dl, bag_test_dl, instance_train_dl, instance_val_dl, instance_test_dl,
                n_epochs, device, val_combined_metric=True, stuOptPeriod=3, stu_loss_weight_neg = 0.3, writer=None, patience=15, csv='tmp.csv', saved_path=None, epoch_warmup=0):
        self.model_teacher = model_teacher
        self.model_student = model_student
        self.model_encoder = model_encoder
        
        self.optimizer_teacher = optimizer_teacher
        self.optimizer_student = optimizer_student
        self.optimizer_encoder = optimizer_encoder
        
        self.bag_train_dl = bag_train_dl
        self.bag_val_dl = bag_val_dl
        self.bag_test_dl = bag_test_dl
        
        self.instance_train_dl = instance_train_dl
        self.instance_val_dl = instance_val_dl
        self.instance_test_dl = instance_test_dl
        
        self.n_epochs = n_epochs
        self.device = device
        self.val_combined_metric = val_combined_metric
        self.stu_loss_weight_neg = stu_loss_weight_neg
        self.stuOptPeriod = stuOptPeriod
        
        self.writer = writer
        
        self.patience = patience
        self.exp = exp
        self.csv = csv
        
        self.saved_path = saved_path
        
        self.best_threshold_withAttnScore = 0.5
        self.best_threshold_withStuPred = 0.5
        self.epoch_warmup = epoch_warmup
    def optimize(self):
        best_combined_metric = float('inf')
        best_auc = 0
        best_model_wts_teacher = None
        best_model_wts_encoder = None
        best_model_wts_student = None
        no_improvement = 0
        for epoch in tqdm(range(self.n_epochs),desc='Training'):
            loss_training, bag_auc_training = self.optimize_teacher(epoch)
            loss_val, bag_auc_ByTeacher_withAttnScore, bag_f1macro_ByTeacher_withAttnScore, bag_accuracy_ByTeacher_withAttnScore = self.evaluate_teacher(epoch, test=False)
            self.writer.add_scalar('Train/Loss', loss_training, epoch)
            self.writer.add_scalar('Train/AUC', bag_auc_training, epoch)
            
            self.writer.add_scalar('Val/Loss', loss_val, epoch)
            self.writer.add_scalar('Val/AUC', bag_auc_ByTeacher_withAttnScore, epoch)
            self.writer.add_scalar('Val/F1-Macro', bag_f1macro_ByTeacher_withAttnScore, epoch)
            self.writer.add_scalar('Val/Accuracy', bag_accuracy_ByTeacher_withAttnScore, epoch)
            
            if epoch % self.stuOptPeriod == 0:
                self.optimize_student(epoch)
            
            if self.val_combined_metric:
                combined_metric = (1-bag_auc_ByTeacher_withAttnScore) + loss_val
            else:
                combined_metric = (1-bag_auc_ByTeacher_withAttnScore)
            
            if epoch > self.epoch_warmup:
                if combined_metric < best_combined_metric:
                    best_combined_metric = combined_metric
                    best_model_wts_teacher = copy.deepcopy(self.model_teacher.state_dict())
                    best_model_wts_student = copy.deepcopy(self.model_student.state_dict())
                    best_model_wts_encoder = copy.deepcopy(self.model_encoder.state_dict())
                    loss_test, test_auc, test_f1macro_withAttn, test_accuracy = self.evaluate_teacher(epoch, test=True)
                    self.writer.add_scalar('Test/Loss', loss_test, epoch)
                    self.writer.add_scalar('Test/AUC', test_auc, epoch)
                    self.writer.add_scalar('Test/F1-Macro', test_f1macro_withAttn, epoch)
                    self.writer.add_scalar('Test/Accuracy', test_accuracy, epoch)
                    
                    
                    no_improvement = 0
                else:
                    no_improvement += 1
                    if no_improvement >= self.patience:
                        print(colored(f'Early stopping at epoch {epoch}',"red"))
                        break
        self.model_teacher.load_state_dict(best_model_wts_teacher)
        self.model_encoder.load_state_dict(best_model_wts_encoder)
        self.model_student.load_state_dict(best_model_wts_student)
        
        if self.saved_path is not None:
            if not os.path.exists(self.saved_path):
                os.makedirs(self.saved_path)
            torch.save(self.model_teacher, f"{self.saved_path}/model_teacher_exp{self.exp}.pt")
            torch.save(self.model_encoder, f"{self.saved_path}/model_encoder_exp{self.exp}.pt")
            torch.save(self.model_student, f"{self.saved_path}/model_student_exp{self.exp}.pt")
        
        self.evaluate_teacher(epoch, test=False)
        
        loss_test, test_auc, test_f1macro_withAttn, test_accuracy= self.evaluate_teacher(epoch, test=True)
        result_df = pd.DataFrame({'exp': [self.exp], 'loss':[loss_test], 'AUC':[test_auc], 'F1-macro':[test_f1macro_withAttn], 'Accuracy':[test_accuracy]})
        if not os.path.exists(self.csv):
            result_df.to_csv(self.csv, index=False)
        else:
            result_df.to_csv(self.csv, mode='a', index=False, header=False)
        
        return 0
    
    def optimize_teacher(self, epoch):
        self.model_encoder.train()
        self.model_teacher.train()
        self.model_student.eval()
        # 1. Bag-level training
        loader = self.bag_train_dl
        # 2. Optimize
        instance_label_gt = []
        instance_label_pred = []
        bag_label_gt = []
        bag_label_pred = []
        total_loss = 0.0
        total_samples = 0
        for i, (t_data, t_bagids, t_labels) in enumerate(loader):
            t_data, t_labels, t_bagids = t_data.to(self.device), t_labels.to(self.device), t_bagids.to(self.device)
            
            feat = self.model_encoder(t_data)[:, :self.model_teacher.input_dims]
            
            inner_ids = t_bagids[len(t_bagids)-1]
            unique, inverse, counts = torch.unique(inner_ids, sorted=True, return_inverse=True, return_counts=True)
            bag_idx = torch.cat([(inverse == x).nonzero()[0] for x in range(len(unique))]).sort()[1]
            bags = unique[bag_idx]
            counts = counts [bag_idx]
            
            batch_instance_label_gt = []
            batch_instance_label_pred = []
            batch_bag_label_pred = torch.empty((len(bags),2), dtype=torch.float, device=self.device)
            
            for b, bag in enumerate(bags):
                bag_instances = feat[inner_ids == bag]
                bag_pred = self.model_teacher(bag_instances)
                instance_attn_score = self.model_teacher.attention_module(bag_instances)
                batch_bag_label_pred[b] = bag_pred
                batch_instance_label_pred.append(instance_attn_score.squeeze(0))
                batch_instance_label_gt.append(t_labels[b]*torch.ones(len(instance_attn_score.squeeze(0)), dtype=torch.long, device=self.device))
            batch_instance_label_pred = torch.cat(batch_instance_label_pred, dim=0)
            batch_instance_label_gt = torch.cat(batch_instance_label_gt, dim=0)
            bag_prediction = torch.softmax(batch_bag_label_pred, dim=1)
            loss_teacher = -1. * (t_labels * torch.log(bag_prediction[:, 1]+1e-5) + (1. - t_labels) * torch.log(1. - bag_prediction[:, 1]+1e-5))
            total_loss += torch.sum(loss_teacher).item()
            total_samples += loss_teacher.size(0)
            loss_teacher = loss_teacher.mean()
            self.optimizer_teacher.zero_grad()
            self.optimizer_encoder.zero_grad()
            
            loss_teacher.backward()
            self.optimizer_teacher.step()
            self.optimizer_encoder.step()
            
            instance_label_gt.append(batch_instance_label_gt)
            instance_label_pred.append(batch_instance_label_pred)
            bag_label_gt.append(t_labels)
            bag_label_pred.append(bag_prediction)
        
        avg_loss = total_loss / total_samples
        instance_label_gt = torch.cat(instance_label_gt, dim=0)
        instance_label_pred = torch.cat(instance_label_pred, dim=0)
        bag_label_gt = torch.cat(bag_label_gt, dim=0)
        bag_label_pred = torch.cat(bag_label_pred, dim=0)
        self.estimated_attn_score_norm_param_min = instance_label_pred.min()
        self.estimated_attn_score_norm_param_max = instance_label_pred.max()
        # instance_label_pred_normed = self.norm_AttnScore2Prob(instance_label_pred)
        # instance_auc_ByTeacher = roc_auc_score(instance_label_gt.cpu().detach().numpy(), instance_label_pred_normed.cpu().detach().numpy())
        bag_auc_ByTeacher = roc_auc_score(bag_label_gt.cpu().detach().numpy(), bag_label_pred.cpu().detach().numpy()[:,1])
        
        return avg_loss, bag_auc_ByTeacher
    def norm_AttnScore2Prob(self, attn_score):
        return (attn_score - self.estimated_attn_score_norm_param_min) / (self.estimated_attn_score_norm_param_max - self.estimated_attn_score_norm_param_min)
    
    def optimize_student(self, epoch):
        self.model_teacher.train()
        self.model_encoder.train()
        self.model_student.train()
        loader = self.instance_train_dl
        instance_label_gt = torch.zeros([loader.dataset.__len__()], dtype=torch.long, device=self.device)
        instance_label_pred = torch.zeros([loader.dataset.__len__()], dtype=torch.float, device=self.device)
        instance_corresponding_bag_idx = torch.zeros([loader.dataset.__len__()], dtype=torch.long, device=self.device)
        for iter, (t_data, t_bag_ids, _, t_instance_labels) in enumerate(loader): 
            t_data, t_bag_ids, t_instance_labels = t_data.to(self.device), t_bag_ids.to(self.device), t_instance_labels.to(self.device)
            feat = self.model_encoder(t_data)[:, :self.model_student.input_dims]
            with torch.no_grad():
                instance_attn_score = self.model_teacher.attention_module(feat)
                pseudo_instance_label = self.norm_AttnScore2Prob(instance_attn_score).clamp(min=1e-5, max=1-1e-5).squeeze(0)
                pseudo_instance_label[t_instance_labels == 0] = 0
            instance_prediction = self.model_student(feat)
            instance_prediction = torch.softmax(instance_prediction, dim=1)
            loss_student = -1. * torch.mean(self.stu_loss_weight_neg * (1-pseudo_instance_label) * torch.log(instance_prediction[:, 0] + 1e-5) + (1-self.stu_loss_weight_neg) * pseudo_instance_label * torch.log(instance_prediction[:, 1] + 1e-5))
            
            self.optimizer_encoder.zero_grad()
            self.optimizer_student.zero_grad()
            loss_student.backward()
            self.optimizer_encoder.step()
            self.optimizer_student.step()
            
            instance_label_gt[iter*t_data.shape[0]:(iter+1)*t_data.shape[0]] = t_instance_labels
            instance_label_pred[iter*t_data.shape[0]:(iter+1)*t_data.shape[0]] = instance_prediction[:, 1]
            instance_corresponding_bag_idx[iter*t_data.shape[0]:(iter+1)*t_data.shape[0]] = t_bag_ids
        
        return 0
    def best_threshold(self, precision, recall, thresholds):
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores, nan=0.0, posinf=0.0, neginf=0.0)
        best_f1_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_f1_idx]
        return best_threshold
    def evaluate_teacher(self, epoch, test=False):
        self.model_encoder.eval()
        self.model_teacher.eval()
        self.model_student.eval()
        if test: 
            loader = self.bag_test_dl
        else:
            loader = self.bag_val_dl
        instance_label_gt = []
        instance_label_pred = []
        bag_label_gt = []
        bag_label_pred_withAttnScore = []
        # bag_label_pred_withStudentPred =[]
        total_loss = 0.0
        total_samples = 0
        for i, (t_data, t_bagids, t_labels) in enumerate(loader):
            t_data, t_labels, t_bagids = t_data.to(self.device), t_labels.to(self.device), t_bagids.to(self.device)
                
            inner_ids = t_bagids[len(t_bagids)-1]
            unique, inverse, counts = torch.unique(inner_ids, sorted=True, return_inverse=True, return_counts=True)
            bag_idx = torch.cat([(inverse == x).nonzero()[0] for x in range(len(unique))]).sort()[1]
            bags = unique[bag_idx]
            counts = counts[bag_idx]
            
            batch_instance_label_gt =[]
            batch_instance_label_pred = []
            batch_bag_label_pred_withAttnScore = torch.empty((len(bags),2), dtype=torch.float, device=self.device)
            # batch_bag_label_pred_with_StuPred = torch.empty((len(bags),2), dtype=torch.float, device=self.device)
            with torch.no_grad():
                feat = self.model_encoder(t_data)[:, :self.model_teacher.input_dims]
                for b, bag in enumerate(bags):
                    bag_instances = feat[inner_ids == bag]
                    # instances_prediction_byStudent = self.model_student(bag_instances)[:, 1].unsqueeze(0)
                    bag_prediction_withAttnScore = self.model_teacher(bag_instances,replaceAS=None)
                    # bag_prediction_withStudentPred = self.model_teacher(bag_instances, replaceAS=instances_prediction_byStudent)
                    instances_attn_score = self.model_teacher.attention_module(bag_instances)
                    
                    bag_prediction_withAttnScore  = torch.softmax(bag_prediction_withAttnScore, dim= 0)
                    # bag_prediction_withStudentPred = torch.softmax(bag_prediction_withStudentPred, dim = 0)
                    batch_bag_label_pred_withAttnScore[b] = bag_prediction_withAttnScore
                    # batch_bag_label_pred_with_StuPred[b] = bag_prediction_withStudentPred
                    
                    batch_instance_label_pred.append(instances_attn_score.squeeze(0))
                    batch_instance_label_gt.append(t_labels[b]*torch.ones(len(instances_attn_score.squeeze(0)), dtype=torch.long, device=self.device))
            
            batch_instance_label_pred = torch.cat(batch_instance_label_pred, dim=0)
            batch_instance_label_gt = torch.cat(batch_instance_label_gt, dim=0)
            loss = - 1. * (t_labels * torch.log(batch_bag_label_pred_withAttnScore[:, 1]+1e-5) + (1. - t_labels) * torch.log(1. - batch_bag_label_pred_withAttnScore[:, 1]+1e-5))
            
            total_loss += torch.sum(loss).item() 
            total_samples += len(t_labels)
            
            bag_label_gt.append(t_labels)
            instance_label_gt.append(batch_instance_label_gt)
            instance_label_pred.append(batch_instance_label_pred)
            bag_label_pred_withAttnScore.append(batch_bag_label_pred_withAttnScore)
            # bag_label_pred_withStudentPred.append(batch_bag_label_pred_with_StuPred)
        
        avg_loss = total_loss / total_samples
        # instance_label_gt = torch.cat(instance_label_gt, dim=0)
        # instance_label_pred = torch.cat(instance_label_pred, dim=0)
        bag_label_gt = torch.cat(bag_label_gt, dim=0)
        bag_label_pred_withAttnScore = torch.cat(bag_label_pred_withAttnScore, dim=0)
        # bag_label_pred_withStudentPred = torch.cat(bag_label_pred_withStudentPred, dim=0)
        # instance_label_pred_normed = (instance_label_pred - instance_label_pred.min()) / (instance_label_pred.max() - instance_label_pred.min())
        # instance_auc_ByTeacher = roc_auc_score(instance_label_gt.cpu().detach().numpy(), instance_label_pred_normed.cpu().detach().numpy())
        
        # bag_label_prob_withStudentPred = bag_label_pred_withStudentPred.cpu().detach().numpy()[:,1]
        bag_label_prob_withAttnScore = bag_label_pred_withAttnScore.cpu().detach().numpy()[:,1]
        bag_label_gt_np = bag_label_gt.cpu().detach().numpy()    
        
        if ~test:
            precision_withAttn, recall_withAttn, thresholds_withAttn = precision_recall_curve(bag_label_gt_np, bag_label_prob_withAttnScore)
            # precision_withStu, recall_withStu, thresholds_withStu = precision_recall_curve(bag_label_gt_np, bag_label_prob_withStudentPred)
            self.best_threshold_withAttnScore = self.best_threshold(precision_withAttn, recall_withAttn, thresholds_withAttn)
            # self.best_threshold_withStuPred = self.best_threshold(precision_withStu, recall_withStu, thresholds_withStu)
            
        
        
        bag_pred_withAttnScore = (bag_label_prob_withAttnScore > self.best_threshold_withAttnScore).astype(int)
        # bag_pred_withStudentPred = (bag_label_prob_withStudentPred > self.best_threshold_withStuPred).astype(int)
        
        bag_auc_ByTeacher_withAttnScore = roc_auc_score(bag_label_gt_np, bag_label_prob_withAttnScore)
        bag_accuracy_ByTeacher_withAttnScore = accuracy_score(bag_label_gt_np, bag_pred_withAttnScore)
        bag_f1macro_ByTeacher_withAttnScore = f1_score(bag_label_gt_np, bag_pred_withAttnScore, average='macro')
        # bag_f1macro_ByTeacher_withStuPred = f1_score(bag_label_gt_np, bag_pred_withStudentPred, average='macro')
        # bag_auc_ByTeacher_withStudentPred = roc_auc_score(bag_label_gt_np, bag_label_prob_withStudentPred)
        
        return avg_loss, bag_auc_ByTeacher_withAttnScore, bag_f1macro_ByTeacher_withAttnScore, bag_accuracy_ByTeacher_withAttnScore
        
        
            
