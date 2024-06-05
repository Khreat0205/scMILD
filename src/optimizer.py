import os 
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from termcolor import colored
import numpy as np
import copy
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, f1_score
from sklearn.mixture import GaussianMixture
import warnings
from sklearn.exceptions import ConvergenceWarning

# from torch.utils.tensorboard import SummaryWriter

class GMMLoss(nn.Module):
    def __init__(self, mu1, mu2, sigma1, sigma2, pi1, pi2):
        super(GMMLoss, self).__init__()
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.pi1 = pi1
        self.pi2 = pi2

    def forward(self, x):
        # 가우시안 분포 1
        gaussian1 = torch.exp(-0.5 * ((x - self.mu1) / self.sigma1) ** 2) / (self.sigma1 * torch.sqrt(2 * torch.tensor(3.14159265)))
        
        # 가우시안 분포 2
        gaussian2 = torch.exp(-0.5 * ((x - self.mu2) / self.sigma2) ** 2) / (self.sigma2 * torch.sqrt(2 * torch.tensor(3.14159265)))

        # 가우시안 믹스처 모델
        gmm = self.pi1 * gaussian1 + self.pi2 * gaussian2

        # 음의 로그 우도 (Negative Log Likelihood)
        loss = -torch.log(gmm + 1e-5)

        return loss.mean()
    
class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5, device='cuda'):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma
        self.device = device

    def forward(self, features, labels=None):
        device = self.device

        #  features are normalized
        # print(features.shape)
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim

        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  # TODO: removed abs ### ? 

        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean
        return loss


class Optimizer:
    def __init__(self, exp, model_teacher, model_student, model_encoder, optimizer_teacher, optimizer_student,
                optimizer_encoder,bag_train_dl, bag_val_dl, bag_test_dl, instance_train_dl, 
                # instance_val_dl, instance_test_dl,
                n_epochs, device, val_combined_metric=True, stuOptPeriod=3, stu_loss_weight_neg = 0.3, writer=None, patience=15, csv='tmp.csv', saved_path=None, epoch_warmup=0, train_stud = True, gmm = False):
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
        # self.instance_val_dl = instance_val_dl
        # self.instance_test_dl = instance_test_dl
        
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
        self.epoch_warmup = epoch_warmup
        self.train_stud = train_stud
        self.use_gmm = gmm
        self.gmm_loss = GMMLoss(mu1=0.2, mu2=0.8, sigma1=0.2, sigma2=0.2, pi1=0.7, pi2=0.3)
        self.op_loss = OrthogonalProjectionLoss(gamma=0.5, device= self.device)
    def optimize(self):
        best_combined_metric = float('inf')
        # best_combined_metric = 0
        best_model_wts_teacher = None
        best_model_wts_encoder = None
        best_model_wts_student = None
        no_improvement = 0
        for epoch in tqdm(range(self.n_epochs),desc='Training'):
            loss_training, bag_auc_training = self.optimize_teacher(epoch)
            loss_val, bag_auc_ByTeacher_withAttnScore, bag_f1macro_ByTeacher_withAttnScore, bag_accuracy_ByTeacher_withAttnScore = self.evaluate_teacher(epoch, test=False)
            if self.writer is not None:
                self.writer.add_scalar('Train/Loss', loss_training, epoch)
                self.writer.add_scalar('Train/AUC', bag_auc_training, epoch)
            
                self.writer.add_scalar('Val/Loss', loss_val, epoch)
                self.writer.add_scalar('Val/AUC', bag_auc_ByTeacher_withAttnScore, epoch)
                self.writer.add_scalar('Val/F1-Macro', bag_f1macro_ByTeacher_withAttnScore, epoch)
                self.writer.add_scalar('Val/Accuracy', bag_accuracy_ByTeacher_withAttnScore, epoch)
            
            if epoch % self.stuOptPeriod == 0 and self.train_stud:
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
        # total_loss_gmm = 0.0
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
            # batch_loss_gmm = []
            batch_bag_label_pred = torch.empty((len(bags),2), dtype=torch.float, device=self.device)
            
            for b, bag in enumerate(bags):
                bag_instances = feat[inner_ids == bag]
                bag_pred = self.model_teacher(bag_instances)
                instance_attn_score = self.model_teacher.attention_module(bag_instances)
                batch_bag_label_pred[b] = bag_pred
                batch_instance_label_pred.append(instance_attn_score.squeeze(0))
                batch_instance_label_gt.append(t_labels[b]*torch.ones(len(instance_attn_score.squeeze(0)), dtype=torch.long, device=self.device))
                # if t_labels[b] == 1:
                #     instance_attn_score_normed = (instance_attn_score - instance_attn_score.min())/(instance_attn_score.max() - instance_attn_score.min())
                    # instance_attn_score_normed_np = instance_attn_score_normed.reshape(-1,1).cpu().detach().numpy()
                    # gmm_bag = GaussianMixture(n_components= 2, reg_covar=1e-5,  random_state=self.exp)
                    # gmm_bag.fit(instance_attn_score_normed_np)
                    # log_likelihood = gmm_bag.score_samples(instance_attn_score_normed_np)
                    # print("test log_likelihood:", log_likelihood)
                    # loss_gmm = -np.mean(log_likelihood)
                    # print("loss_gmm:", loss_gmm)
                    # loss_gmm_tensor = torch.tensor(loss_gmm, device=self.device)
                    # batch_loss_gmm.append(loss_gmm_tensor)
                    
                    # loss_gmm_bag = self.gmm_loss(instance_attn_score_normed)
                    # batch_loss_gmm.append(loss_gmm_bag)
            batch_instance_label_pred = torch.cat(batch_instance_label_pred, dim=0)
            batch_instance_label_gt = torch.cat(batch_instance_label_gt, dim=0)
            # batch_loss_gmm = torch.stack(batch_loss_gmm).mean()
            bag_prediction = torch.softmax(batch_bag_label_pred, dim=1)
            # batch_instance_label_pred_normed = (batch_instance_label_pred - batch_instance_label_pred.min()) / (batch_instance_label_pred.max() - batch_instance_label_pred.min())
            loss_teacher = -1. * (t_labels * torch.log(bag_prediction[:, 1]+1e-5) + (1. - t_labels) * torch.log(1. - bag_prediction[:, 1]+1e-5))
            # loss_gmm = self.gmm_loss(batch_instance_label_pred_normed)
            total_loss += torch.sum(loss_teacher).item()
            # total_loss_gmm += torch.sum(batch_loss_gmm).item()
            total_samples += loss_teacher.size(0)
            loss_teacher = loss_teacher.mean()
            # loss_teacher = loss_teacher + batch_loss_gmm
            # loss_teacher = loss_teacher + loss_gmm
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
        #avg_loss_gmm = total_loss_gmm / total_samples
        # print("Classification Loss:" , avg_loss, "GMM Loss:", avg_loss_gmm)
        instance_label_gt = torch.cat(instance_label_gt, dim=0)
        instance_label_pred = torch.cat(instance_label_pred, dim=0)
        bag_label_gt = torch.cat(bag_label_gt, dim=0)
        bag_label_pred = torch.cat(bag_label_pred, dim=0)
        self.estimated_attn_score_norm_param_min = instance_label_pred.min()
        self.estimated_attn_score_norm_param_max = instance_label_pred.max()
        ##############################
        # instance_label_pred_normed = self.norm_AttnScore2Prob(instance_label_pred)
        
        # instance_label_pred_normed[instance_label_gt == 0] = 0
        ##############################
        # if self.use_gmm:
        #     instance_label_pred_normed_np = instance_label_pred_normed[instance_label_gt == 1].cpu().detach().numpy().reshape(-1,1)
        #     self.gmm_model = GaussianMixture(n_components=2, random_state=self.exp)
        #     self.gmm_model.fit(instance_label_pred_normed_np)
        #     print("GMM component means:", self.gmm_model.means_)
        #     probabilities = self.gmm_model.predict_proba(instance_label_pred_normed_np)
        #     component_labels = np.argmax(probabilities, axis =1 )
        #     if self.gmm_model.means_[0] > self.gmm_model.means_[1]:
        #         self.gmm_component_order = [1,0]
        #     else:
        #         self.gmm_component_order = [0,1]
        #     # component_labels = [self.gmm_component_order[label] for label in component_labels]
            
        bag_auc_ByTeacher = roc_auc_score(bag_label_gt.cpu().detach().numpy(), bag_label_pred.cpu().detach().numpy()[:,1])
        
        return avg_loss, bag_auc_ByTeacher
    def norm_AttnScore2Prob(self, attn_score):
        return (attn_score - self.estimated_attn_score_norm_param_min) / (self.estimated_attn_score_norm_param_max - self.estimated_attn_score_norm_param_min)
    
    def optimize_student(self, epoch):
        self.model_teacher.train()
        self.model_encoder.train()
        self.model_student.train()
        loader = self.instance_train_dl
        start_percentile = 90
        end_percentile = 70
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
                # if self.use_gmm:
                #     pseduo_instance_label_np = pseudo_instance_label.cpu().detach().numpy().reshape(-1,1)
                #     component_labels = self.gmm_model.predict(pseduo_instance_label_np)
                #     component_labels = [self.gmm_component_order[label] for label in component_labels]
                #     pseudo_instance_label[t_instance_labels == 1] = torch.tensor(component_labels, dtype=torch.float, device=self.device)[t_instance_labels == 1]
                
            
            instance_prediction = self.model_student(feat)
            instance_prediction = torch.softmax(instance_prediction, dim=1)
            
            
            loss_student = -1. * torch.mean(self.stu_loss_weight_neg * (1-pseudo_instance_label) * torch.log(instance_prediction[:, 0] + 1e-5) + (1-self.stu_loss_weight_neg) * pseudo_instance_label * torch.log(instance_prediction[:, 1] + 1e-5))
            # print("before:", loss_student)
            if self.use_gmm:
                # op_labels, pseudo_instance labels cutoff >0.5, 1, else 0
                op_labels = pseudo_instance_label.clone()
                op_labels_posbag = op_labels[t_instance_labels == 1].clone()
                #if len(op_labels_posbag) >= 100:
                # current_percentile = start_percentile + (end_percentile - start_percentile) * (epoch / self.n_epochs)
                gmm = GaussianMixture(n_components=2, random_state=self.exp)
                with warnings.catch_warnings(record=True) as w:
                    gmm.fit(op_labels_posbag.cpu().detach().numpy().reshape(-1,1))
                    if len(w) > 0 and issubclass(w[-1].category, ConvergenceWarning):
                        print("gmm not converged")
                        op_labels[(t_instance_labels == 1) & (op_labels > 0.5)] = 1
                        op_labels[(t_instance_labels == 1) & (op_labels <= 0.5)] = 0
                        print("Count op_labels == 1: ", torch.sum(op_labels == 1))
                    else:
                        # print("gmm converged")
                        # print("gmm means:", gmm.means_)
                        if gmm.means_[0] > gmm.means_[1]:
                            component_order = [1,0]
                        else :
                            component_order = [0,1]
                        op_labels = torch.tensor([component_order[label] for label in gmm.predict(op_labels.cpu().detach().numpy().reshape(-1,1))], dtype=torch.float, device=self.device)
                        
                    
                op_labels[t_instance_labels == 0] = 0
                op_loss_batch = self.op_loss(feat, op_labels)
                                
                # # print("op_loss_batch: ",op_loss_batch)
                loss_student = loss_student + 0.5 * op_loss_batch
            # print("after:", loss_student)   
            self.optimizer_encoder.zero_grad()
            self.optimizer_student.zero_grad()
            loss_student.backward()
            self.optimizer_encoder.step()
            self.optimizer_student.step()
            
        
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
        
        
            
