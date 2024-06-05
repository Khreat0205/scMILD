library(Seurat)
library(SeuratDisk)
library(SeuratObject)
library(ggpubr)
library(patchwork)
library(data.table)
setwd("/home/local/kyeonghunjeong_920205/nipa_bu/COVID19/3.analysis/9.MIL/scAMIL_cell/scMILD/downstream/")
Convert('../data/PBMC/su_2020_processed.h5ad', dest="h5seurat")
whole_obj = LoadH5Seurat("../data/PBMC/su_2020_processed.h5seurat", meta.data=F)
test_samples = fread("PBMC/test_set_barcodes_2.csv")
meta = fread("PBMC/meta_2.csv")

test_cells = meta[meta$sample_id_numeric %in% test_samples$sample_id_numeric,]
test_obj = subset(whole_obj,cells = test_cells$V1)
cell_scores = fread("PBMC/cell_score_2.csv")
colnames(cell_scores)
rm(whole_obj)
rm(meta)
cell_meta_data = cbind(test_cells, cell_scores)
cell_meta_data = as.data.frame(cell_meta_data)
rownames(cell_meta_data) = cell_meta_data$V1

rm(test_cells)
rm(cell_scores)
test_obj@meta.data = cell_meta_data
test_obj = FindVariableFeatures(test_obj)
test_obj[["RNA"]] = NormalizeData(test_obj[["RNA"]])
test_obj = ScaleData(test_obj)
test_obj <- RunPCA(test_obj,features=VariableFeatures(test_obj))
test_obj <- RunUMAP(test_obj, dims = 1:50,reduction = "pca")

test_obj@meta.data$disease_severity_standard = factor(test_obj@meta.data$disease_severity_standard,levels = c("mild","moderate","severe"))
test_obj@meta.data$disease = factor(test_obj@meta.data$disease_numeric,levels=c(0,1),labels = c("Not-hosp","Hosp"))

# trqwe::mcsaveRDS(test_obj, file = "PBMC/test_seurat.RDS")
# test_obj  = trqwe::mcreadRDS(file = "PBMC/test_seurat.RDS")
test_obj


table(test_obj@meta.data$disease,test_obj@meta.data$cell_score_minmax > 0.5)
test_obj@meta.data$subgroup = factor(ifelse(test_obj@meta.data$cell_score_minmax > 0.5, "Positive","Negative"), levels=c("Negative","Positive"))
test_obj@meta.data$disease_association = ifelse(test_obj$disease == "Hosp", 
                                                ifelse(test_obj$subgroup == "Positive", "Hosp. Positive", "Hosp. Negative"),"Not-hosp.") 
test_obj@meta.data$disease_association = factor(test_obj@meta.data$disease_association,levels = c("Not-hosp.","Hosp. Negative","Hosp. Positive"))
table(test_obj@meta.data$disease, test_obj@meta.data$subgroup)
color_disease = c("#2ECC40","#FF4136")
color_subgroup = c("#00FFFF","#FF00FF")
color_disease_asso = c("#2ECC40", "#9ACD32", "#FF851B")
p_ct  = DimPlot(test_obj,group.by = "predicted.celltype.l2",label = F) + ggtitle("")

p_disease = DimPlot(test_obj,group.by="disease",cols=color_disease)  + ggtitle("")
p_score =  FeaturePlot(test_obj,"cell_score_minmax",order = T) + ggtitle("")
p_subgroup = DimPlot(test_obj,group.by = "subgroup",cols=color_subgroup) + ggtitle("")
p_disease_association = DimPlot(test_obj, group.by = "disease_association", cols = c("#2ECC40", "#9ACD32", "#FF851B"),order = T) + ggtitle("")
#p_ct + FeaturePlot(test_obj, "cell_score_softmax",label = F,max.cutoff = 'q90') + ggtitle("")
# test_obj$cell_score_softmax
res_dir = c("PBMC/")
# dir.create(res_dir)
ggsave(plot = p_ct,filename =  file.path(res_dir,"UMAP_exp2_celltype.pdf"),device = 'pdf',width=10, height=8,dpi = 450)
ggsave(plot = p_score,filename =  file.path(res_dir,"UMAP_exp2_cell_attn_score.pdf"),device = 'pdf',width=8, height=8,dpi = 450)
ggsave(plot = p_disease,filename =  file.path(res_dir,"UMAP_exp2_sample_phenotype.pdf"),device = 'pdf',width=8, height=8,dpi = 450)
ggsave(plot = p_subgroup,filename =  file.path(res_dir,"UMAP_exp2_cell_subgroup.pdf"),device = 'pdf',width=8, height=8,dpi = 450)
ggsave(plot = p_disease_association,filename =  file.path(res_dir,"UMAP_exp2_disease_association.pdf"),device = 'pdf',width=8, height=8,dpi = 450)


p_fill = ggplot(test_obj@meta.data, aes(x = predicted.celltype.l2, fill = subgroup)) +
  geom_bar(position = "fill")+ theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + theme(legend.position = "none") +scale_fill_manual(values = color_subgroup)+xlab("Cell type") + ylab('fraction')

p_stack = ggplot(test_obj@meta.data, aes(x = predicted.celltype.l2, fill = subgroup)) +
  geom_bar(position = "stack") + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+scale_fill_manual(values = color_subgroup)+ labs(fill='Cell subgroup')+xlab("Cell type")


p_fill_bag = ggplot(test_obj@meta.data, aes(x = predicted.celltype.l2, fill = disease)) +
  geom_bar(position = "fill")+ theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+ theme(legend.position = "none") +scale_fill_manual(values = color_disease)+xlab("Cell type") + ylab('fraction')
p_stack_bag = ggplot(test_obj@meta.data, aes(x = predicted.celltype.l2, fill =disease)) +
  geom_bar(position = "stack") + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +scale_fill_manual(values = color_disease) + labs(fill='Condition')+xlab("Cell type")

p_fill_asso = ggplot(test_obj@meta.data, aes(x = predicted.celltype.l2, fill = disease_association)) +
  geom_bar(position = "fill")+ theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+ theme(legend.position = "none") +scale_fill_manual(values = color_disease_asso)+xlab("Cell type") + ylab('fraction')
p_stack_asso = ggplot(test_obj@meta.data, aes(x = predicted.celltype.l2, fill =disease_association,legend.title="asdas")) +
  geom_bar(position = "stack") + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+ scale_fill_manual(values = color_disease_asso) + labs(fill='Condition-association group')+xlab("Cell type")
(p_fill_bag + p_stack_bag) 
(p_fill + p_stack) 
(p_fill_asso+ p_stack_asso)

ggsave(plot = (p_fill_bag + p_stack_bag),filename =  file.path(res_dir,"Bar_exp2_bag_celltype_counts.pdf"),device = 'pdf',width=12, height=8,dpi = 450)
ggsave(plot = (p_fill + p_stack) ,filename =  file.path(res_dir,"Bar_exp2_subgroup_celltype_counts.pdf"),device = 'pdf',width=12, height=8,dpi = 450)
ggsave(plot = (p_fill_asso + p_stack_asso) ,filename =  file.path(res_dir,"Bar_exp2_asso_celltype_counts.pdf"),device = 'pdf',width=12, height=8,dpi = 450)


library(ggpubr)


Idents(test_obj) = "predicted.celltype.l2"
subsetMarkers = function(obj, idents,cut.padj = NULL, logfc.threshold = NULL, type="Positive"){
  if(is.null(logfc.threshold)){
    logfc.threshold = 0.25
  }
  subset_obj = subset(obj, idents = idents)
  
  if(type=="Positive"){
    Idents(subset_obj) = "subgroup"
    disease_markers_subset = FindMarkers(subset_obj ,ident.1 = "Positive",ident.2 = "Negative",logfc.threshold = logfc.threshold,only.pos = T)
  } else if(type == "Disease"){
    Idents(subset_obj) = "disease_cov"
    disease_markers_subset = FindMarkers(subset_obj ,ident.1 = "COVID-19",ident.2 = "normal",logfc.threshold = logfc.threshold,only.pos = T)
  } else if(type == "Subtype"){
    Idents(subset_obj) = "disease_association"
    disease_markers_subset = FindAllMarkers(subset_obj,logfc.threshold = logfc.threshold,only.pos = T)
  }
  
  if(!is.null(cut.padj)){
    disease_markers_subset = disease_markers_subset[disease_markers_subset$p_val_adj < cut.padj,]
  }
  return(disease_markers_subset)
}
test_obj$disease_association
disease_markers_mono = subsetMarkers(test_obj, idents = "CD14 Mono", cut.padj = 0.01,type ="Subtype")
head(disease_markers_mono[disease_markers_mono$cluster == "Hosp. Positive",],14)
fwrite(disease_markers_mono,file = file.path(res_dir, "Table_DEG_CD14_mono_disease_asso.csv"), sep=",")
top15 = rownames(disease_markers_mono[disease_markers_mono$cluster == "Hosp. Positive",])[1:15]
p_top15_deg = DotPlot(subset(test_obj,ident="CD14 Mono"), top15, group.by="disease_association")+RotatedAxis()
p_origin_marekrs= VlnPlot(subset(test_obj,ident="CD14 Mono"),features=original_markers, group.by = "disease_association",alpha=0,add.noise=F,ncol = 5,cols = color_disease_asso)

ggsave(plot = p_top15_deg,filename =  file.path(res_dir,"Dot_DEG_Top15_CD14_disease_asso.pdf"),device = 'pdf',width=12, height=8,dpi = 450)
ggsave(plot = p_origin_marekrs,filename =  file.path(res_dir,"Vln_originalMarkers_CD14_disease_asso.pdf"),device = 'pdf',width=15, height=5,dpi = 450)
latent_dat = test_obj@meta.data[,grepl("feature",colnames(test_obj@meta.data))]
latent_pca = prcomp(latent_dat)
ggboxplot(test_obj@meta.data, x="sex_standard", y="cell_score_minmax",fill="sex_standard",facet.by = "predicted.celltype.l2",outlier.shape = F) + stat_compare_means()
ggboxplot(test_obj@meta.data, x="sex_standard", y="cell_score_minmax",fill="sex_standard",facet.by = "predicted.celltype.l2",outlier.shape = F) + stat_compare_means()


library(ggfortify)
dim(latent_pca$x)
autoplot(latent_pca,data = test_obj@meta.data, colour="disease_association")
autoplot(latent_pca,data = test_obj@meta.data, colour="predicted.celltype.l2")
latent_umap = uwot::umap(X= latent_pca$x[,1:10])
colnames(latent_umap) = c("UMAP1","UMAP2")

ggplot(data=latent_umap, aes(x=UMAP1,y=UMAP2,color=test_obj@meta.data$disease_association)) + 
  geom_point(alpha=0.5)
ggplot(data=latent_umap, aes(x=UMAP1,y=UMAP2,color=test_obj@meta.data$predicted.celltype.l2)) + 
  geom_point(alpha=0.5)

library(tidyr)
library(dplyr)
sample_meta = unique(test_obj@meta.data[,c("sample","disease","age_standard","sex_standard","disease_severity_standard","days_since_symptom_onset_standard")])
unique(test_obj@meta.data$disease_severity_standard)
summary(test_obj@meta.data$days_since_hospitalization_standard)
summary(test_obj@meta.data$disease_status_standard)
summary(test_obj@meta.data$days_since_symptom_onset_standard)
summary(test_obj@meta.data$)
test_obj@meta.data$sample
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
unique(sample_meta$sample)
sample_table = merge(sample_matrix,sample_meta,by="sample")
fwrite(sample_table,file = file.path(res_dir,"table_sample_celltype_contribution.csv"),sep=",")

sample_pearson = cor(sample_table[,c(2:ncol(sample_matrix))], sample_table[,c("age_standard", "days_since_symptom_onset_standard")])
sample_rho = cor(sample_table[,c(2:ncol(sample_matrix))], sample_table[,c("age_standard", "days_since_symptom_onset_standard")], method="spearman")

hosp_pearson = cor(sample_table[sample_table$disease =="Hosp",c(2:ncol(sample_matrix))], sample_table[sample_table$disease =="Hosp",c("age_standard", "days_since_symptom_onset_standard")])
hosp_rho = cor(sample_table[sample_table$disease =="Hosp",c(2:ncol(sample_matrix))], sample_table[sample_table$disease =="Hosp",c("age_standard", "days_since_symptom_onset_standard")], method="spearman")

not_hosp_pearson = cor(sample_table[sample_table$disease =="Not-hosp",c(2:ncol(sample_matrix))], sample_table[sample_table$disease =="Not-hosp",c("age_standard", "days_since_symptom_onset_standard")])
not_hosp_rho = cor(sample_table[sample_table$disease =="Not-hosp",c(2:ncol(sample_matrix))], sample_table[sample_table$disease =="Not-hosp",c("age_standard", "days_since_symptom_onset_standard")], method="spearman")
abs(sample_rho) > 0.45
hosp_rho[abs(hosp_rho[,'age_standard']) > 0.4,]
abs(not_hosp_rho) > 0.45

fwrite(data.table(hosp_rho,keep.rownames = "Cell type"),file=file.path(res_dir,"Rho_cell_type_contribution.csv"),sep=",")

p_sct_bnaive = ggscatter(sample_table[sample_table$disease=="Hosp",], x="age_standard",y="B naive",add = 'reg.line',cor.coef = T,cor.method="spearman",cor.coeff.args =list(method="spearman",cor.coef.name= "rho",label.x.npc="left",digits=4) ,title = "B naive",xlab = "Age",ylab = "Contribution",color ="grey",add.params = list(color="black") )

p_sct_cd8naive = ggscatter(sample_table[sample_table$disease=="Hosp",], x="age_standard",y="CD8 Naive",add = 'reg.line',cor.coef = T,cor.method="spearman",cor.coeff.args =list(method="spearman",cor.coef.name= "rho",label.x.npc="left",digits=4), title = "CD8 naive",xlab = "Age",ylab = "Contribution",color ="grey",add.params = list(color="black") )

p_sct_binter= ggscatter(sample_table[sample_table$disease=="Hosp",], x="age_standard",y="B intermediate",add = 'reg.line',cor.coef = T,cor.method="spearman", cor.coeff.args =list(method="spearman",cor.coef.name= "rho",label.x.npc="left",digits=4), title="B intermediate",xlab = "Age",ylab = "Contribution",color ="grey",add.params = list(color="black") )

ggsave(p_sct_bnaive,dpi = 450,device="pdf",file = file.path(res_dir,"Scatter_age_B_naive.pdf"),height = 6)
ggsave(p_sct_cd8naive,dpi = 450,device="pdf",file = file.path(res_dir,"Scatter_age_CD8_naive.pdf"),height=6)
ggsave(p_sct_binter,dpi = 450,device="pdf",file = file.path(res_dir,"Scatter_age_B_intermediate.pdf"),height=6)

ggscatter(sample_table_melt[sample_table_melt$disease=="Hosp" & sample_table_melt$variable %in% c(),], x="age_standard",y="value",add = 'reg.line',cor.coef = T,cor.method="spearman")



sample_wilcoxon_sex = data.table()
sample_wilcoxon_sex_hosp = data.table()
sample_wilcoxon_sex_not_hosp = data.table()
sample_wilcoxon_severity = data.table()
sample_t_sex = data.table()
sample_t_sex_hosp = data.table()
sample_t_sex_not_hosp = data.table()
sample_t_severity = data.table()

for(i in c(2:ncol(sample_matrix))){
  tmp_ct = colnames(sample_table)[[i]]
  tmp_fm =formula(sprintf("`%s` ~ sex_standard",tmp_ct))
  tmp_fm2 =formula(sprintf("`%s` ~ disease_severity_standard",tmp_ct))
  sample_wilcoxon_sex = rbind(sample_wilcoxon_sex,
                              data.table(tmp_ct,wilcox.test(tmp_fm, sample_table )$p.value))
  sample_wilcoxon_sex_hosp = rbind(sample_wilcoxon_sex_hosp,
                                   data.table(tmp_ct,wilcox.test(tmp_fm, sample_table[sample_table$disease =="Hosp",] )$p.value))
  sample_wilcoxon_severity = rbind(sample_wilcoxon_severity,
                                   data.table(tmp_ct,wilcox.test(tmp_fm2, sample_table[sample_table$disease =="Hosp",] )$p.value))
  sample_wilcoxon_sex_not_hosp = rbind(sample_wilcoxon_sex_not_hosp,
                                       data.table(tmp_ct,wilcox.test(tmp_fm, sample_table[sample_table$disease =="Not-hosp",] )$p.value))
  sample_t_sex = rbind(sample_t_sex,
                       data.table(tmp_ct,t.test(tmp_fm, sample_table )$p.value))
  sample_t_sex_hosp = rbind(sample_t_sex_hosp,
                            data.table(tmp_ct,t.test(tmp_fm, sample_table[sample_table$disease =="Hosp",] )$p.value))
  sample_t_severity = rbind(sample_t_severity,
                            data.table(tmp_ct,t.test(tmp_fm2, sample_table[sample_table$disease =="Hosp",] )$p.value))
  sample_t_sex_not_hosp = rbind(sample_t_sex_not_hosp,
                                data.table(tmp_ct,t.test(tmp_fm, sample_table[sample_table$disease =="Not-hosp",] )$p.value))
  
}

sample_wilcoxon_sex[sample_wilcoxon_sex$V2 < 0.01,] # "CD4 T", "CD4 TCM" ,"Mono", "CD14 Mono","CD4 Naive", "NK_CD56bright"
sample_wilcoxon_sex_hosp[sample_wilcoxon_sex_hosp$V2 < 0.01,] # "CD4 T", "CD4 TCM"
sample_wilcoxon_sex_not_hosp[sample_wilcoxon_sex_not_hosp$V2 < 0.05,] # none
sample_wilcoxon_severity[sample_wilcoxon_severity$V2 <0.01,]

sample_t_sex[sample_t_sex$V2 < 0.01,] # "CD4 T", "CD4 TCM" ,"Mono", "CD14 Mono","CD4 Naive", "NK_CD56bright"
sample_t_sex_hosp[sample_t_sex_hosp$V2 < 0.01,] # "CD4 T", "CD4 TCM"
colnames(sample_t_sex_hosp) = c("Cell type", "t.test_p.value")
sample_t_sex_hosp[sample_t_sex_hosp$t.test_p.value < 0.01,] # "CD4 T", "CD4 TCM"
fwrite(sample_t_sex_hosp,file=file.path(res_dir,"T.test_cell_type_contribution.csv"),sep=",")

sample_t_sex_not_hosp[sample_t_sex_not_hosp$V2 < 0.05,] # none
sample_t_severity[sample_t_severity$V2 <0.05,]

sample_table_melt = melt(sample_table[,c("sample","sex_standard","disease","CD4 T", "CD4 TCM")])

p_box_sex = ggboxplot(sample_table_melt[sample_table_melt$disease == "Hosp",], x="sex_standard",y="value",facet.by = "variable",xlab = "Sex",ylab="Contribution") +stat_compare_means(method = "t.test")
ggsave(p_box_sex,dpi = 450,device="pdf",file = file.path(res_dir,"Box_sex_CD4_T_TCM.pdf"),height=6,width=5)


median(sample_table[sample_table$disease=="Hosp" & sample_table$sex_standard=="female" 
                    ,"CD14 Mono"])
median(sample_table[sample_table$disease=="Hosp" & sample_table$sex_standard=="male" 
                    ,"CD14 Mono"])




# sample_ct_score_rank = sample_ct_score %>% group_by(ind_cov) %>% summarise_at(vars(celltype_prop_score),function(x) rank(x))
# sample_ct_score$rank_within_sample = sample_ct_score_rank$celltype_prop_score
# sample_max_celltype = sample_ct_score[sample_ct_score$rank_within_sample == 8,c("ind_cov","ct_cov")]





