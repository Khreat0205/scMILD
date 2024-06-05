library(Seurat)
library(data.table)
library(ggpubr)
library(factoextra)
# library(NbClust) 
library(ComplexHeatmap)
library(dplyr)
library(tidyr)
library(circlize)
library(RColorBrewer)
test_obj  = trqwe::mcreadRDS(file = "PBMC/test_seurat.RDS")

sample_meta = unique(test_obj@meta.data[,c("sample","disease","age_standard","sex_standard","disease_severity_standard","days_since_symptom_onset_standard")])
# test_obj@meta.data$cell_score_minmax = test_obj@meta.data$cell_score_teacher_minmax
# test_obj@meta.data$cell_score = test_obj@meta.data$cell_score_teacher
sample_dat = test_obj@meta.data %>% 
  group_by(sample,predicted.celltype.l1) %>% 
  summarise_at("cell_score_minmax",sum)
sample_dat = sample_dat %>% group_by(sample) %>% 
  mutate(cell_type_ratio = cell_score_minmax / sum(cell_score_minmax)) %>% ungroup()
sample_dat_l2 = test_obj@meta.data %>% 
  group_by(sample,predicted.celltype.l2) %>% 
  summarise_at("cell_score_minmax",sum)
sample_dat_l2 = sample_dat_l2 %>% group_by(sample) %>% 
  mutate(cell_type_ratio = cell_score_minmax / sum(cell_score_minmax)) %>% ungroup()



sample_matrix = dcast(sample_dat,formula = sample ~predicted.celltype.l1,fill = 0,value.var = "cell_type_ratio")
sample_matrix_l2 = dcast(sample_dat_l2,formula = sample ~predicted.celltype.l2,fill = 0,value.var = "cell_type_ratio")
sample_matrix = merge(sample_matrix, sample_matrix_l2,by="sample",suffixes =c(".l1",".l2"))
rownames(sample_matrix) = sample_matrix$sample
sample_table = merge(sample_matrix,sample_meta,by="sample")


softmax = function(x) exp(x)/sum(exp(x))
softmax_within_sample_id = function(obj){
  cell_score = obj$cell_score # this is vector
  sample_id = obj$sample_id_numeric
  sample_id_unique = unique(sample_id)
  softmax_scores = rep(0,length(cell_score))
  for (i in 1:length(sample_id_unique)){
    sample_id_i = sample_id_unique[i]
    cell_score_i = cell_score[sample_id == sample_id_i]
    softmax_scores[sample_id == sample_id_i] = softmax(cell_score_i)
  }
  
  return(softmax_scores)
}

sm_scores = softmax_within_sample_id(test_obj)
test_obj@meta.data$cell_score_softmax = sm_scores


sample_dat_l2_sm = test_obj@meta.data %>% 
  group_by(sample,predicted.celltype.l2) %>% 
  summarise_at("cell_score_softmax",sum)
sample_dat_l2_sm = sample_dat_l2_sm %>% group_by(sample) %>% 
  mutate(cell_type_ratio = cell_score_softmax / sum(cell_score_softmax)) %>% ungroup()

sample_matrix_l2_sm = dcast(sample_dat_l2_sm,formula = sample ~predicted.celltype.l2,fill = 0,value.var = "cell_type_ratio")



sample_meta = sample_meta[match(sample_matrix_l2_sm[,1],sample_meta$sample),]

ha = rowAnnotation(
  severity = sample_meta$disease_severity_standard,
  age = sample_meta$age_standard,
  days = sample_meta$days_since_symptom_onset_standard,
  sex = sample_meta$sex_standard,
  col = list(severity = c("mild" = "blue", "moderate" = "orange", "severe" = "red"), 
             # age is integer value, gradient color
             age = colorRamp2(c(min(sample_meta$age_standard), max(sample_meta$age_standard)), c("white", "red")),
             # days is categorical value ( 7 or 14)
             days = c("7" = "red", "14" = "orange"),
             # sex is categorical value ( female or male)
             sex = c(female = "pink", male = "navy")
  )
)


hcut_res_only_hosp = hcut(sample_matrix_sm_only_hosp[,-1], k = 2, hc_method = "complete", hc_metric = "euclidean")

sample_meta_only_hosp = sample_meta[sample_meta$disease == "Hosp",]
sample_meta_only_hosp$cluster = hcut_res_only_hosp$cluster
ha_only_hosp = rowAnnotation(
  severity = sample_meta_only_hosp$disease_severity_standard,
  age = sample_meta_only_hosp$age_standard,
  days = sample_meta_only_hosp$days_since_symptom_onset_standard,
  sex = sample_meta_only_hosp$sex_standard,
  col = list(severity = c("mild" = "blue", "moderate" = "white", "severe" = "red"), 
             # age is integer value, gradient color
             age = colorRamp2(c(min(sample_meta_only_hosp$age_standard)*1.2, max(sample_meta_only_hosp$age_standard)*0.8), c("white", "red")),
             # days is categorical value ( 7 or 14)
             days = c("7" = "red", "14" = "orange"),
             # sex is categorical value ( female or male)
             sex = c(female = "pink", male = "navy")
  )
)
Heatmap(sample_matrix_sm_only_hosp[,-1], split=hcut_res_only_hosp$cluster,right_annotation = ha_only_hosp)



full_meta = fread("UC/covid_19_pbmc_full_meta.csv")
full_meta$disease_severity_standard = ifelse(full_meta$`Patient Location` == "Hospital", "moderate",ifelse(full_meta$`Patient Location` == "ICU", "severe", "mild"))
sample_meta = merge(sample_meta, full_meta, by="sample",all.x=T, all.y=F)
sample_meta_only_hosp = merge(sample_meta_only_hosp, full_meta, by="sample",all.x=T, all.y=F)
sample_meta_only_hosp$cluster = as.factor(sample_meta_only_hosp$cluster)
# fwrite(sample_meta_only_hosp,file = "PBMC/clustered_sample_meta.csv")
sample_meta_only_hosp= fread(file = "PBMC/clustered_sample_meta.csv")




# ggboxplot(sample_meta_only_hosp, x = "cluster", y = "Temperature", color = "cluster", add = "jitter")+ stat_pwc(method = "t.test")
# ggboxplot(sample_meta_only_hosp, x = "disease_severity_standard", y = "Temperature", color = "disease_severity_standard", add = "jitter")+ stat_pwc(method = "t.test")
# 
# ggboxplot(sample_meta_only_hosp, x = "cluster", y = "Systolic BP", color = "cluster", add = "jitter")+ stat_pwc(method = "t.test")
# ggboxplot(sample_meta_only_hosp, x = "disease_severity_standard", y = "Systolic BP", color = "cluster", add = "jitter")+ stat_pwc(method = "t.test")
# 
# ggboxplot(sample_meta_only_hosp, x = "cluster", y = "Diastolic BP", color = "cluster", add = "jitter")+ stat_pwc(method = "t.test")
# ggboxplot(sample_meta_only_hosp, x = "disease_severity_standard", y = "Diastolic BP", color = "cluster", add = "jitter")+ stat_pwc(method = "t.test")

# AGE
p_cl_age = ggboxplot(sample_meta_only_hosp, x = "cluster", y = "age_standard", fill = "cluster", add = "jitter")+ stat_pwc(method = "t.test")+theme(legend.position = "none")+xlab("Test Dataset")
p_di_age = ggboxplot(sample_meta_only_hosp, x = "Patient Location", y = "age_standard", fill = "Patient Location", add = "jitter", palette =brewer.pal(2,"Set2"))+
  stat_pwc(method="t.test")+
  theme(legend.position = "none")+xlab("Test Dataset")+ylab("")
p_full_age = ggboxplot(full_meta[full_meta$disease_severity_standard %in% c("moderate","severe")], x = "Patient Location", y = "Age", fill = "Patient Location", add = "jitter", palette =brewer.pal(2,"Set2"))+ 
  stat_pwc(method = "t.test") +
  theme(legend.position = "none")+xlab("Whole Dataset")+ylab("")
p_cl_age + p_di_age + p_full_age
# BMI
p_cl_bmi = ggboxplot(sample_meta_only_hosp, x = "cluster", y = "BMI", fill = "cluster", add = "jitter")+ stat_pwc(method = "t.test")+
  theme(legend.position = "none")+xlab("Test Dataset")
p_di_bmi = ggboxplot(sample_meta_only_hosp, x = "Patient Location", y = "BMI", fill = "Patient Location", add = "jitter", palette =brewer.pal(2,"Set2"))+ stat_pwc(method = "t.test")+
  theme(legend.position = "none")+xlab("Test Dataset")+ylab("")

p_full_bmi = ggboxplot(full_meta[full_meta$disease_severity_standard %in% c("moderate","severe")],
                       x = "Patient Location", y = "BMI", fill = "disease_severity_standard", 
                       add = "jitter", palette =brewer.pal(2,"Set2"))+ stat_pwc(method = "t.test")+
  theme(legend.position = "none")+xlab("Whole Dataset")+ylab("") 
p_cl_bmi + p_di_bmi+ p_full_bmi


softmax_within_sample_id_mat = function(mat){
  cell_score = mat$cell_score 
  sample_id = mat$sample_id_numeric
  sample_id_unique = unique(sample_id)
  softmax_scores = rep(0,length(cell_score))
  for (i in 1:length(sample_id_unique)){
    sample_id_i = sample_id_unique[i]
    cell_score_i = cell_score[sample_id == sample_id_i]
    softmax_scores[sample_id == sample_id_i] = softmax(cell_score_i)
  }
  
  return(softmax_scores)
}

cs_exp3= fread("PBMC/cell_score_3.csv")
obs_exp3 = fread("PBMC/obs_3.csv")

mat_exp3 = cbind(obs_exp3, cs_exp3)
rm(cs_exp3)

mat_exp3$cell_score_softmax = softmax_within_sample_id_mat(mat_exp3)
mat_exp3_l2_sm = mat_exp3 %>% group_by(sample, predicted.celltype.l2) %>% 
  summarise_at("cell_score_softmax",sum)

mat_exp3_l2_sm = mat_exp3_l2_sm %>% group_by(sample) %>% 
  mutate(cell_type_ratio = cell_score_softmax / sum(cell_score_softmax)) %>% ungroup()
mat_exp3_matrix_l2_sm = dcast(mat_exp3_l2_sm ,formula = sample ~predicted.celltype.l2,fill = 0,value.var = "cell_type_ratio")
exp3_matrix_only_hosp =mat_exp3_matrix_l2_sm[mat_exp3_matrix_l2_sm$sample %in% full_meta$sample[full_meta$disease_severity_standard %in% c("moderate","severe")],]

hcut_res_only_hosp_exp3 = hcut(exp3_matrix_only_hosp[,-1], k = 2, hc_method = "complete", hc_metric = "euclidean")

sample_meta_exp3 = unique(obs_exp3[,c("sample","age_standard","sex_standard","disease_severity_standard","days_since_symptom_onset_standard")])
sample_meta_exp3 = merge(sample_meta_exp3, full_meta, by="sample",all.x=T, all.y=F,suffixes = c("",".y"))
sample_meta_exp3_only_hosp = sample_meta_exp3[sample_meta_exp3$disease_severity_standard %in% c("moderate","severe"),]
sample_meta_exp3_only_hosp$cluster = as.factor(hcut_res_only_hosp_exp3$cluster)
sample_meta_exp3_only_hosp$cluster
Heatmap(exp3_matrix_only_hosp[,-1],split=hcut_res_only_hosp_exp3$cluster)
ggboxplot(sample_meta_exp3_only_hosp, x = "cluster", y = "age_standard", fill = "cluster", add = "jitter")+ stat_pwc(method = "t.test")+theme(legend.position = "none")+xlab("Test Dataset")
 ggboxplot(sample_meta_exp3_only_hosp, x = "cluster", y = "BMI", fill = "cluster", add = "jitter")+ stat_pwc(method = "t.test")+
  theme(legend.position = "none")+xlab("Test Dataset")


fisher.test(table(sample_meta_only_hosp$cluster,ifelse(sample_meta_only_hosp$BMI >= 30,"obesity", "no")))
fisher.test(table(sample_meta_only_hosp$cluster, sample_meta_only_hosp$disease_severity_standard))
fisher.test(table(sample_meta_only_hosp$cluster, sample_meta_only_hosp$Asthma))
fisher.test(table(sample_meta_only_hosp$disease_severity_standard, sample_meta_only_hosp$Asthma))
# fisher.test(table(sample_meta_only_hosp$cluster, sample_meta_only_hosp$`Mechanical Ventilation`)[1:2,1:2])
# fisher.test(table(sample_meta_only_hosp$disease_severity_standard, sample_meta_only_hosp$`Mechanical Ventilation`)[2:3,1:2])

fisher.test(table(full_meta$disease_severity_standard, full_meta$Asthma)[2:3,])

fisher.test(table(sample_meta_only_hosp$cluster,sample_meta_only_hosp$`Coronary Artery Disease`))
fisher.test(table(sample_meta_only_hosp$cluster,sample_meta_only_hosp$`Cigarette Smoking`)[,1:2]) ###
fisher.test(table(full_meta$disease_severity_standard, full_meta$`Cigarette Smoking`)[2:3,])
table(sample_meta_only_hosp$cluster,sample_meta_only_hosp$`Respiratory Support`)
fisher.test(table(sample_meta_only_hosp$cluster,sample_meta_only_hosp$Ethnicity))
fisher.test(table(sample_meta_only_hosp$cluster,sample_meta_only_hosp$`Mechanical Ventilation`))
fisher.test(table(full_meta$disease_severity_standard, full_meta$`Mechanical Ventilation`)[2:3,])
fisher.test(table(sample_meta_only_hosp$cluster,sample_meta_only_hosp$`Chronic Kidney Disease`))
fisher.test(table(sample_meta_only_hosp$cluster,sample_meta_only_hosp$`Congestive Heart Failure`))
fisher.test(table(sample_meta_only_hosp$cluster,sample_meta_only_hosp$COPD))
sample_meta_only_hosp$Diabetes_2 = ifelse(sample_meta_only_hosp$Diabetes == "No","No","DM")
sample_meta_only_hosp$T2DM = ifelse(sample_meta_only_hosp$Diabetes == "T2DM","T2DM","No")
sample_meta_only_hosp$T1DM = ifelse(sample_meta_only_hosp$Diabetes == "T1DM","T1DM","No")
fisher.test(table(sample_meta_only_hosp$cluster, sample_meta_only_hosp$Diabetes))
fisher.test(table(sample_meta_only_hosp$cluster, sample_meta_only_hosp$T2DM))
fisher.test(table(sample_meta_only_hosp$cluster, sample_meta_only_hosp$T1DM))
fisher.test(table(sample_meta_only_hosp$cluster, sample_meta_only_hosp$Diabetes_2))
fisher.test(table(sample_meta_only_hosp$cluster, sample_meta_only_hosp$sex_standard))


fisher.test(table(sample_meta_only_hosp$disease_severity_standard,sample_meta_only_hosp$Asthma)[2:3,1:2])
fisher.test(table(sample_meta_only_hosp$disease_severity_standard,sample_meta_only_hosp$`Coronary Artery Disease`)[2:3,1:2])
fisher.test(table(sample_meta_only_hosp$disease_severity_standard,sample_meta_only_hosp$`Cigarette Smoking`)[2:3,1:2])
# fisher.test(table(sample_meta_only_hosp$disease_severity_standard,sample_meta_only_hosp$`Respiratory Support`)[2:3,])
fisher.test(table(sample_meta_only_hosp$disease_severity_standard,sample_meta_only_hosp$Ethnicity)[2:3,])
table(sample_meta_only_hosp$disease_severity_standard,sample_meta_only_hosp$`Chronic Kidney Disease`)[2:3,]
table(sample_meta_only_hosp$disease_severity_standard,sample_meta_only_hosp$`Congestive Heart Failure`)[2:3,]
table(sample_meta_only_hosp$disease_severity_standard,sample_meta_only_hosp$COPD)[2:3,]
fisher.test(table(sample_meta_only_hosp$disease_severity_standard, sample_meta_only_hosp$Diabetes)[2:3,])
fisher.test(table(sample_meta_only_hosp$disease_severity_standard, sample_meta_only_hosp$Diabetes_2)[2:3,])
fisher.test(table(sample_meta_only_hosp$disease_severity_standard, sample_meta_only_hosp$sex_standard)[2:3,])
table(sample_meta_only_hosp$disease_severity_standard, sample_meta_only_hosp$`Patient Location`)


weighted_assay = CreateAssayObject(counts = test_obj[['RNA']]@counts * test_obj@meta.data$cell_score_softmax, key = "WeightedRNA")
test_obj[["WeightedRNA"]] = weighted_assay

ps_obj_non_mono = AggregateExpression(subset(test_obj, predicted.celltype.l2 == "CD14 Mono"), assays = "RNA",group.by = "sample",return.seurat = T)
ps_obj_non_mono = NormalizeData(ps_obj_non_mono)
ps_obj_non_mono = ScaleData(ps_obj_non_mono)
ps_obj_non_mono = FindVariableFeatures(ps_obj_non_mono,nfeatures = 2000)
ps_obj_non_mono = RunPCA(ps_obj_non_mono,features = VariableFeatures(ps_obj_non_mono,2000),npcs = 30)
ps_obj_non_mono = RunUMAP(ps_obj_non_mono, dims = 1:30,reduction = "pca")

ps_obj_non_mono@meta.data$sample = gsub("_","-",gsub("g","",ps_obj_non_mono@meta.data$sample))

merged_meta  = merge(ps_obj_non_mono@meta.data,sample_meta, by.x = "sample" ,by.y="sample", all.x=T,all.y=F)
ps_obj_non_mono@meta.data = cbind(ps_obj_non_mono@meta.data,merged_meta[,setdiff(colnames(merged_meta), colnames(ps_obj_non_mono@meta.data)
)])

ps_obj_non_mono_subset  = subset(ps_obj_non_mono,disease=="Hosp")
sample_meta_only_hosp$sample ==  ps_obj_non_mono_subset$sample
ps_obj_non_mono_subset@meta.data$cluster = sample_meta_only_hosp$cluster
ps_obj_non_mono_subset@meta.data = cbind(ps_obj_non_mono_subset@meta.data,sample_meta_only_hosp[,setdiff(colnames(sample_meta_only_hosp), colnames(ps_obj_non_mono_subset@meta.data))])


ps_obj_non_mono_subset = FindVariableFeatures(ps_obj_non_mono_subset,nfeatures = 1000)
ps_obj_non_mono_subset = RunPCA(ps_obj_non_mono_subset,features = VariableFeatures(ps_obj_non_mono_subset,1000),npcs = 10)
ps_obj_non_mono_subset = RunUMAP(ps_obj_non_mono_subset, dims = 1:10,reduction = "pca")

DimPlot(ps_obj_non_mono_subset,group.by = "disease",label = T,repel = T,pt.size =5)
DimPlot(ps_obj_non_mono_subset,group.by = "disease_severity_standard",label = T,repel = T,pt.size =5)
DimPlot(ps_obj_non_mono_subset,group.by = "cluster",label = T,repel = T,pt.size =5)

# trqwe::mcsaveRDS(ps_obj_non_mono_subset,"PBMC/ps_obj_non_mono_subset.rds")
ps_obj_non_mono_subset = trqwe::mcreadRDS("PBMC/ps_obj_non_mono_subset.rds")
cl_markers = FindMarkers(ps_obj_non_mono_subset,group.by = "cluster",ident.1 = "2", ident.2="1")
disease_markers = FindMarkers(ps_obj_non_mono_subset,group.by = "disease_severity_standard",ident.1 = "severe", ident.2="moderate")

cl2_up = rownames(cl_markers[cl_markers$p_val < 0.01 &cl_markers$avg_log2FC > 0.25,])
cl2_dn = rownames(cl_markers[cl_markers$p_val < 0.01 &cl_markers$avg_log2FC <  -0.25,])
di_sig =  rownames(disease_markers[disease_markers$p_val < 0.01 & abs(disease_markers$avg_log2FC) > 0.25 ,])
severe_up =  rownames(disease_markers[disease_markers$p_val < 0.01 & disease_markers$avg_log2FC > 0.25 ,])
severe_dn =  rownames(disease_markers[disease_markers$p_val < 0.01 & disease_markers$avg_log2FC < -0.25 ,])

FeaturePlot(ps_obj_non_mono_subset,"BMI",label = T,repel = T,pt.size=5)
library(enrichplot)
library(org.Hs.eg.db)
require(DOSE)
library(clusterProfiler)

eg_cl2_up = bitr(cl2_up, fromType="SYMBOL", toType="ENTREZID", OrgDb="org.Hs.eg.db")
bitr(cl2_up, fromType="SYMBOL", toType="KEGG", OrgDb="org.Hs.eg.db")
eg_cl2_dn = bitr(cl2_dn, fromType="SYMBOL", toType="ENTREZID", OrgDb="org.Hs.eg.db")
eg_cl_sig = bitr(c(cl2_up,cl2_dn), fromType="SYMBOL", toType="ENTREZID", OrgDb="org.Hs.eg.db")

eg_disease = bitr(di_sig, fromType="SYMBOL", toType="ENTREZID", OrgDb="org.Hs.eg.db")
eg_severe_up = bitr(severe_up, fromType="SYMBOL", toType="ENTREZID", OrgDb="org.Hs.eg.db")
eg_severe_dn = bitr(severe_dn, fromType="SYMBOL", toType="ENTREZID", OrgDb="org.Hs.eg.db")

kegg_cl2_up= enrichKEGG(eg_cl2_up$ENTREZID,qvalueCutoff = 0.5)
kegg_cl2_dn= enrichKEGG(eg_cl2_dn$ENTREZID,qvalueCutoff = 0.5)
kegg_cl= enrichKEGG(eg_cl_sig$ENTREZID,qvalueCutoff = 0.5)

kegg_disease= enrichKEGG(eg_disease$ENTREZID,qvalueCutoff = 0.5)
kegg_severe_up= enrichKEGG(eg_severe_up$ENTREZID,qvalueCutoff = 0.5)
kegg_severe_dn= enrichKEGG(eg_severe_dn$ENTREZID,qvalueCutoff = 0.5)

dotplot(kegg_cl)
dotplot(kegg_cl2_up)
dotplot(kegg_cl2_dn)
dotplot(kegg_disease)
dotplot(kegg_severe_up)
dotplot(kegg_severe_dn)

lfc_cl = cl_markers[,"avg_log2FC"]
genes_cl = rownames(cl_markers)
entrez_cl = bitr(genes_cl, fromType="SYMBOL", toType="ENTREZID", OrgDb="org.Hs.eg.db")

names(lfc_cl) = entrez_cl$ENTREZID
lfc_cl = sort(lfc_cl,decreasing = T)

go_cl = gseGO(lfc_cl,OrgDb = "org.Hs.eg.db",keyType = "ENTREZID",pvalueCutoff = 0.5)
gse_kegg_cl = gseKEGG(lfc_cl,organism = 'hsa',pvalueCutoff = 0.5)


lfc_disease = disease_markers[,"avg_log2FC"]
genes_disease = rownames(disease_markers)
entrez_disease = bitr(genes_disease, fromType="SYMBOL", toType="ENTREZID", OrgDb="org.Hs.eg.db")

names(lfc_disease) = entrez_disease$ENTREZID
lfc_disease = sort(lfc_disease,decreasing = T)

go_disease = gseGO(lfc_disease,OrgDb = "org.Hs.eg.db",keyType = "ENTREZID",pvalueCutoff = 0.5)
gse_kegg_disease = gseKEGG(lfc_disease,organism = 'hsa',pvalueCutoff = 0.5)

dotplot(go_cl,showCategory=15, title="Cluster GO:BP") + dotplot(go_disease,showCategory=15,title="Disease GO:BP")
dotplot(gse_kegg_cl,showCategory=15, title="Cluster KEGG") + dotplot(gse_kegg_disease,showCategory=15, title ="Disease KEGG")



# ps_obj_mono = AggregateExpression(subset(test_obj, predicted.celltype.l2 == "CD14 Mono"), assays = "WeightedRNA",group.by = "sample",return.seurat = T)
# ps_obj_non = AggregateExpression(test_obj, assays = "RNA",group.by = "sample",return.seurat = T)
# ps_obj_non = NormalizeData(ps_obj_non)
# ps_obj_non = ScaleData(ps_obj_non)
# ps_obj_non = FindVariableFeatures(ps_obj_non,nfeatures = 2000)
# 
# ps_obj_non <- RunPCA(ps_obj_non,features = VariableFeatures(ps_obj_non,1500),npcs = 30)
# ps_obj_non <- RunUMAP(ps_obj_non, dims = 1:30,reduction = "pca")
# 
# 
# ps_obj = AggregateExpression(test_obj, assays = "WeightedRNA",group.by = "sample",return.seurat = T)
# ps_obj = NormalizeData(ps_obj)
# ps_obj = ScaleData(ps_obj)
# ps_obj = FindVariableFeatures(ps_obj,nfeatures = 2000)
# 
# ps_obj <- RunPCA(ps_obj,features = VariableFeatures(ps_obj,1500),npcs = 30)
# ps_obj <- RunUMAP(ps_obj, dims = 1:30,reduction = "pca")
# colnames(test_obj@meta.data)
# sample_metadata = unique(test_obj@meta.data[,c("sample","bag_labels","sample_id_numeric","disease","disease_severity_standard","age_standard","sex_standard","disease_status_standard")])
# sample_metadata
# ps_obj@meta.data$sample = gsub("_","-",gsub("g","",ps_obj@meta.data$sample))
# ps_obj_non@meta.data$sample = gsub("_","-",gsub("g","",ps_obj_non@meta.data$sample))
# merged_meta  = merge(ps_obj@meta.data,sample_metadata, by.x = "sample" ,by.y="sample", all.x=T,all.y=F)
# merged_meta
# ps_obj@meta.data
# ps_obj@meta.data = cbind(ps_obj@meta.data,merged_meta[,setdiff(colnames(merged_meta), colnames(ps_obj@meta.data)
# )])
# ps_obj_non@meta.data = cbind(ps_obj_non@meta.data,merged_meta[,setdiff(colnames(merged_meta), colnames(ps_obj_non@meta.data)
# )])
# ps_obj_non[["RNA"]]@layers$scale.data - ps_obj[["WeightedRNA"]]@layers$scale.data
# 
# 
# ps_obj = FindNeighbors(ps_obj, dims = 1:30)
# ps_obj = FindClusters(ps_obj, resolution = 0.8)
# ps_obj_non = FindNeighbors(ps_obj_non, dims = 1:30)
# ps_obj_non = FindClusters(ps_obj_non, resolution = 0.8)
# cor(test_obj@meta.data$cell_score_minmax,test_obj[["RNA"]]@scale.data[1,])
# ##correlation between test_obj@meta.data$cell_score_minmax and all rows in test_obj@scale.data
# cor_allrows = function(vector,matrix){
#   y = sapply(1:nrow(matrix),function(i) cor(vector,matrix[i,]))
#   names(y) = rownames(matrix)
#   return(y)
# }
# score_and_genes = cor_allrows(vector=test_obj@meta.data$cell_score_minmax,matrix=test_obj[["RNA"]]@scale.data)
# score_and_genes_dat = as.data.table(score_and_genes,keep.rownames = 'genes')
# score_and_genes_dat[which.max(score_and_genes_dat$score_and_genes),]
# score_and_genes_dat[which.min(score_and_genes_dat$score_and_genes),]
# plot(density(score_and_genes_dat$score_and_genes))
# sum(abs(score_and_genes_dat$score_and_genes) > 0.30)
# corrgenes = score_and_genes_dat[abs(score_and_genes_dat$score_and_genes) > 0.35,genes]
# FeaturePlot(test_obj,c(corrgenes,"cell_score_minmax"))
# DimPlot(test_obj,group.by = "subgroup")
# sm_and_genes = cor_allrows(vector=test_obj@meta.data$cell_score_softmax,matrix=test_obj[["RNA"]]@scale.data)
# sm_and_genes
# 
# sm_and_genes[abs(sm_and_genes) > 0.15]
# 
# sapply(1:3,function(i) cor(test_obj@meta.data$cell_score_minmax,test_obj[["RNA"]]@scale.data[1:3,][i,]))
# cor()
# 
# 
# ?cor
# (DimPlot(ps_obj) + DimPlot(ps_obj_non)) / (DimPlot(ps_obj,group.by = "disease",label = T) + DimPlot(ps_obj_non,group.by = "disease",label = T))
# FeaturePlot(ps_obj, "S100A12")
# DimPlot(ps_obj,group.by = "disease_severity_standard",label = T) + DimPlot(ps_obj_non,group.by = "disease_severity_standard",label = T)
# FeaturePlot(ps_obj, features = "age_standard",label = F) + FeaturePlot(ps_obj_non, features = "age_standard",label = F)
# DimPlot(ps_obj,group.by = "sex_standard",label = T) + DimPlot(ps_obj_non,group.by = "sex_standard",label = T)
