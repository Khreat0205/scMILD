import os 
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from tqdm import tqdm
from termcolor import colored
import numpy as np
import copy
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, f1_score
from sklearn.mixture import GaussianMixture
import warnings
from sklearn.exceptions import ConvergenceWarning

# from torch.utils.tensorboard import SummaryWriter

# from https://github.com/kahnchana/opl/blob/master/loss.py
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
            optimizer_encoder, bag_train_dl, bag_val_dl, bag_test_dl, instance_train_dl, 
            n_epochs, device, val_combined_metric=True, stuOptPeriod=3, stu_loss_weight_neg=0.3, 
            writer=None, patience=15, csv='tmp.csv', saved_path=None, epoch_warmup=0, opl_warmup=0, 
            train_stud=True, opl=False, use_loss=True, op_lambda=0.5, op_gamma=0.5, opl_comps=[2,3,4], 
            code_disease_ratio=None, ratio_reg_lambda=0.0, use_conditional_ae=False):
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
        self.opl_warmup = opl_warmup - 1
        self.train_stud = train_stud
        self.use_opl = opl
        self.opl_comps = opl_comps
        self.op_gamma = op_gamma
        self.op_loss = OrthogonalProjectionLoss(gamma=self.op_gamma, device=self.device)
        self.use_loss = use_loss
        self.op_lambda = op_lambda
        self.code_disease_ratio = code_disease_ratio
        self.ratio_reg_lambda = ratio_reg_lambda
        self.use_conditional_ae = use_conditional_ae
    def optimize(self):
        best_combined_metric = float('inf')
        best_model_wts_teacher = None
        best_model_wts_encoder = None
        best_model_wts_student = None
        no_improvement = 0
        # tqdm(,desc='Training')
        for epoch in range(self.n_epochs):
            loss_training = self.optimize_teacher(epoch)
            if epoch % self.stuOptPeriod == 0 and self.train_stud:
                self.optimize_student(epoch)
            loss_val, bag_auc_ByTeacher_withAttnScore, best_threshold_withAttnScore = self.evaluate_teacher(epoch, test=False)
            if self.writer is not None:
                self.writer.add_scalar('Train/Loss', loss_training, epoch)
                self.writer.add_scalar('Val/Loss', loss_val, epoch)

            if self.val_combined_metric:
                combined_metric = (1-bag_auc_ByTeacher_withAttnScore) + loss_val
            else:
                if self.use_loss:
                    combined_metric = loss_val
                else:
                    combined_metric = (1-bag_auc_ByTeacher_withAttnScore)
            
            if epoch > self.epoch_warmup:
                if combined_metric < best_combined_metric:
                    best_combined_metric = combined_metric
                    best_model_wts_teacher = copy.deepcopy(self.model_teacher.state_dict())
                    best_model_wts_student = copy.deepcopy(self.model_student.state_dict())
                    best_model_wts_encoder = copy.deepcopy(self.model_encoder.state_dict())
                    self.best_threshold_withAttnScore = best_threshold_withAttnScore
                    
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
        
        # self.evaluate_teacher(epoch, test=False)
        
        loss_test, test_auc, test_f1macro_withAttn, test_accuracy= self.evaluate_teacher(epoch, test=True)
        result_df = pd.DataFrame({'exp': [self.exp], 'loss':[loss_test], 'AUC':[test_auc], 'F1-macro':[test_f1macro_withAttn], 'Accuracy':[test_accuracy]})
        if not os.path.exists(self.csv):
            result_df.to_csv(self.csv, index=False)
        else:
            result_df.to_csv(self.csv, mode='a', index=False, header=False)
        
        # if test_accuracy < 0.7:
        #     raise ValueError(f"Accuracy is too low: {test_accuracy}")
        
        return 0
    
    def optimize_teacher(self, epoch):
        self.model_encoder.train()
        if self.optimizer_encoder is not None:
            self.model_encoder.train()
        else:
            self.model_encoder.eval()
        self.model_teacher.train()
        self.model_student.eval()
        
        loader = self.bag_train_dl
        instance_label_pred = []
        bag_label_gt = []
        bag_label_pred = []
        total_loss = 0.0
        total_samples = 0
        
        for i, batch in enumerate(loader):
            # collate 반환: (data, bagids, labels) 또는 (data, bagids, labels, study_ids)
            t_data = batch[0].to(self.device)
            t_bagids = batch[1].to(self.device)
            t_labels = batch[2].to(self.device)
            t_study_ids = batch[3].to(self.device) if len(batch) == 4 else None
    
            # 모델 Forward
            if self.use_conditional_ae and t_study_ids is not None:
                feat = self.model_encoder(t_data, t_study_ids)[:, :self.model_teacher.input_dims]
            else:
                feat = self.model_encoder(t_data)[:, :self.model_teacher.input_dims]
            
            inner_ids = t_bagids[len(t_bagids)-1]
            unique, inverse, counts = torch.unique(inner_ids, sorted=True, return_inverse=True, return_counts=True)
            bag_idx = torch.cat([(inverse == x).nonzero()[0] for x in range(len(unique))]).sort()[1]
            bags = unique[bag_idx]
            counts = counts[bag_idx]
            
            batch_instance_label_pred = []
            batch_bag_label_pred = torch.empty((len(bags), 2), dtype=torch.float, device=self.device)
    
            # Attention scores 수집 (ratio regularization용)
            all_attn_scores = []
            
            for b, bag in enumerate(bags):
                bag_instances = feat[inner_ids == bag]
                bag_pred = self.model_teacher(bag_instances)
                instance_attn_score = self.model_teacher.attention_module(bag_instances)
                batch_bag_label_pred[b] = bag_pred
                batch_instance_label_pred.append(instance_attn_score.squeeze(0))
                all_attn_scores.append(instance_attn_score.squeeze())
                
            batch_instance_label_pred = torch.cat(batch_instance_label_pred, dim=0)
            all_attn_scores = torch.cat(all_attn_scores, dim=0)
            
            bag_prediction = torch.softmax(batch_bag_label_pred, dim=1)
            loss_teacher = -1. * (t_labels * torch.log(bag_prediction[:, 1] + 1e-5) + 
                                  (1. - t_labels) * torch.log(1. - bag_prediction[:, 1] + 1e-5))
            total_loss += torch.sum(loss_teacher).item()
            total_samples += loss_teacher.size(0)
            loss_teacher = loss_teacher.mean()
            
            # Disease ratio regularization
            if self.code_disease_ratio is not None and self.ratio_reg_lambda > 0:
                if hasattr(self.model_encoder, 'vq_model'):
                    with torch.no_grad():
                        if self.use_conditional_ae and t_study_ids is not None:
                            code_indices = self.model_encoder.vq_model.get_codebook_indices(t_data, t_study_ids)
                        else:
                            code_indices = self.model_encoder.vq_model.get_codebook_indices(t_data)
                else:
                    code_indices = None
                
                if code_indices is not None:
                    target_ratios = torch.tensor(
                        [self.code_disease_ratio.get(int(c), 0.5) for c in code_indices.cpu().numpy()],
                        dtype=torch.float32, device=self.device
                    )
                    
                    attn_min = all_attn_scores.min()
                    attn_max = all_attn_scores.max()
                    attn_norm = (all_attn_scores - attn_min) / (attn_max - attn_min + 1e-8)
                    
                    ratio_reg_loss = F.mse_loss(attn_norm, target_ratios)
                    loss_teacher = loss_teacher + self.ratio_reg_lambda * ratio_reg_loss
    
            self.optimizer_teacher.zero_grad()
            if self.optimizer_encoder is not None:
                self.optimizer_encoder.zero_grad()
            loss_teacher.backward()
            self.optimizer_teacher.step()
            if self.optimizer_encoder is not None:
                self.optimizer_encoder.step()
            
            instance_label_pred.append(batch_instance_label_pred)
            bag_label_gt.append(t_labels)
            bag_label_pred.append(bag_prediction)
        
        avg_loss = total_loss / total_samples
        instance_label_pred = torch.cat(instance_label_pred, dim=0)
        self.estimated_attn_score_norm_param_min = instance_label_pred.min()
        self.estimated_attn_score_norm_param_max = instance_label_pred.max()
        
        return avg_loss
    def norm_AttnScore2Prob(self, attn_score):
        return (attn_score - self.estimated_attn_score_norm_param_min) / (self.estimated_attn_score_norm_param_max - self.estimated_attn_score_norm_param_min)
    
    def optimize_student(self, epoch):
        self.model_teacher.train()
        if self.optimizer_encoder is not None:
            self.model_encoder.train()
        else:
            self.model_encoder.eval()
        self.model_student.train()
        
        loader = self.instance_train_dl
        for iter, batch in enumerate(loader):
            # InstanceDataset: (data, bag_id, instance_label) 또는 (data, bag_id, instance_label, study_id)
            # InstanceDataset2: (data, bag_id, instance_label, bag_label) 또는 + study_id
            if len(batch) == 5:
                t_data, t_bag_ids, t_instance_labels, t_bag_labels, t_study_ids = batch
                t_study_ids = t_study_ids.to(self.device)
            elif len(batch) == 4:
                t_data, t_bag_ids, t_instance_labels, t_bag_labels = batch
                t_study_ids = None
            else:
                t_data, t_bag_ids, t_instance_labels = batch
                t_bag_labels = None
                t_study_ids = None
            t_data = t_data.to(self.device)
            t_bag_ids = t_bag_ids.to(self.device)
            t_instance_labels = t_instance_labels.to(self.device)
            # 모델 Forward
            if self.use_conditional_ae and t_study_ids is not None:
                feat = self.model_encoder(t_data, t_study_ids)[:, :self.model_student.input_dims]
            else:
                feat = self.model_encoder(t_data)[:, :self.model_student.input_dims]
                
            with torch.no_grad():
                instance_attn_score = self.model_teacher.attention_module(feat)
                pseudo_instance_label = self.norm_AttnScore2Prob(instance_attn_score).clamp(min=1e-5, max=1-1e-5).squeeze(0)
                pseudo_instance_label[t_instance_labels == 0] = 0
            
            instance_prediction = self.model_student(feat)
            instance_prediction = torch.softmax(instance_prediction, dim=1)
            
            loss_student = -1. * torch.mean(
                self.stu_loss_weight_neg * (1 - pseudo_instance_label) * torch.log(instance_prediction[:, 0] + 1e-5) + 
                (1 - self.stu_loss_weight_neg) * pseudo_instance_label * torch.log(instance_prediction[:, 1] + 1e-5)
            )
            
            if self.use_opl and torch.sum(t_instance_labels == 1).item() > 2 and epoch > self.opl_warmup:
                op_labels = pseudo_instance_label.clone()
                op_labels_posbag = op_labels[t_instance_labels == 1].clone()
                
                if isinstance(self.opl_comps, list):
                    n_components_list = self.opl_comps
                else:
                    n_components_list = [self.opl_comps]
                
                best_score = np.inf
                best_gmm = None
                
                for n_components in n_components_list:
                    gmm = GaussianMixture(n_components=n_components, random_state=self.exp)
                    with warnings.catch_warnings(record=True) as w:
                        gmm.fit(op_labels_posbag.cpu().detach().numpy().reshape(-1, 1))
                        if len(w) > 0 and issubclass(w[-1].category, ConvergenceWarning):
                            continue
                        
                        score = gmm.bic(op_labels_posbag.cpu().detach().numpy().reshape(-1, 1))
                        
                        if score < best_score:
                            best_score = score
                            best_gmm = gmm
                
                if best_gmm is not None:
                    component_order = np.argsort(best_gmm.means_.flatten())
                    component_order = component_order.tolist()
                    op_labels = torch.tensor(
                        [component_order[label] for label in best_gmm.predict(op_labels.cpu().detach().numpy().reshape(-1, 1))],
                        dtype=torch.float, device=self.device
                    )
                else:
                    op_labels[(t_instance_labels == 1) & (op_labels > 0.5)] = 1
                    op_labels[(t_instance_labels == 1) & (op_labels <= 0.5)] = 0
                    
                op_labels[t_instance_labels == 0] = 0
                if torch.sum(op_labels == 1).item() > 0:
                    op_loss_batch = self.op_loss(feat, op_labels)
                    loss_student = loss_student + self.op_lambda * op_loss_batch
            
            if self.optimizer_encoder is not None:
                self.optimizer_encoder.zero_grad()
            self.optimizer_student.zero_grad()
            loss_student.backward()
            if self.optimizer_encoder is not None:
                self.optimizer_encoder.step()
            self.optimizer_student.step()
        
        return 0
    def best_threshold(self, precision, recall, thresholds):
        # remove warning message 
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = 2 * (precision * recall) / (precision + recall)
            f1_scores = np.nan_to_num(f1_scores, nan=0.0, posinf=0.0, neginf=0.0)
            best_f1_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_f1_idx]
            
        return best_threshold
    def evaluate_teacher(self, epoch, test=False):
        with torch.no_grad():
            self.model_encoder.eval()
            self.model_teacher.eval()
            self.model_student.eval()
            
            if test:
                loader = self.bag_test_dl
            else:
                loader = self.bag_val_dl
                
            bag_label_gt = []
            bag_label_pred_withAttnScore = []
            total_loss = 0.0
            total_samples = 0
            
            for i, batch in enumerate(loader):
                # collate 반환: (data, bagids, labels) 또는 (data, bagids, labels, study_ids)
                t_data = batch[0].to(self.device)
                t_bagids = batch[1].to(self.device)
                t_labels = batch[2].to(self.device)
                t_study_ids = batch[3].to(self.device) if len(batch) == 4 else None
                
                # 모델 Forward
                if self.use_conditional_ae and t_study_ids is not None:
                    feat = self.model_encoder(t_data, t_study_ids)[:, :self.model_teacher.input_dims]
                else:
                    feat = self.model_encoder(t_data)[:, :self.model_teacher.input_dims]
                    
                inner_ids = t_bagids[len(t_bagids)-1]
                unique, inverse, counts = torch.unique(inner_ids, sorted=True, return_inverse=True, return_counts=True)
                bag_idx = torch.cat([(inverse == x).nonzero()[0] for x in range(len(unique))]).sort()[1]
                bags = unique[bag_idx]
                counts = counts[bag_idx]
                
                batch_bag_label_pred_withAttnScore = torch.empty((len(bags), 2), dtype=torch.float, device=self.device)
                
                for b, bag in enumerate(bags):
                    bag_instances = feat[inner_ids == bag]
                    bag_prediction_withAttnScore = self.model_teacher(bag_instances, replaceAS=None)
                    bag_prediction_withAttnScore = torch.softmax(bag_prediction_withAttnScore, dim=0)
                    batch_bag_label_pred_withAttnScore[b] = bag_prediction_withAttnScore
                    
                loss = -1. * (t_labels * torch.log(batch_bag_label_pred_withAttnScore[:, 1] + 1e-5) + 
                              (1. - t_labels) * torch.log(1. - batch_bag_label_pred_withAttnScore[:, 1] + 1e-5))
                
                total_loss += torch.sum(loss).item()
                total_samples += len(t_labels)
                
                bag_label_gt.append(t_labels)
                bag_label_pred_withAttnScore.append(batch_bag_label_pred_withAttnScore)
            
            avg_loss = total_loss / total_samples
            bag_label_gt = torch.cat(bag_label_gt, dim=0)
            bag_label_pred_withAttnScore = torch.cat(bag_label_pred_withAttnScore, dim=0)
            
            bag_label_prob_withAttnScore = bag_label_pred_withAttnScore.cpu().detach().numpy()[:, 1]
            bag_label_gt_np = bag_label_gt.cpu().detach().numpy()
            bag_auc_ByTeacher_withAttnScore = roc_auc_score(bag_label_gt_np, bag_label_prob_withAttnScore)
            
            if not test:
                precision_withAttn, recall_withAttn, thresholds_withAttn = precision_recall_curve(
                    bag_label_gt_np, bag_label_prob_withAttnScore
                )
                best_threshold_withAttnScore = self.best_threshold(precision_withAttn, recall_withAttn, thresholds_withAttn)
                
                return avg_loss, bag_auc_ByTeacher_withAttnScore, best_threshold_withAttnScore
            else:
                bag_pred_withAttnScore = (bag_label_prob_withAttnScore > self.best_threshold_withAttnScore).astype(int)
                bag_accuracy_ByTeacher_withAttnScore = accuracy_score(bag_label_gt_np, bag_pred_withAttnScore)
                bag_f1macro_ByTeacher_withAttnScore = f1_score(bag_label_gt_np, bag_pred_withAttnScore, average='macro')
                
                return avg_loss, bag_auc_ByTeacher_withAttnScore, bag_f1macro_ByTeacher_withAttnScore, bag_accuracy_ByTeacher_withAttnScore
            
            
                
