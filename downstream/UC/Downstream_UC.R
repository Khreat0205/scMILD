library(Seurat)
library(SeuratDisk)
library(SeuratObject)
library(ggpubr)
library(patchwork)
library(data.table)
library(reticulate)
setwd("/home/local/kyeonghunjeong_920205/nipa_bu/COVID19/3.analysis/9.MIL/scAMIL_cell/scMILD/downstream/")
base_dir = 'UC'
# # set pat to python (needs scanpy so in terminal install via pip: "python -m pip install scanpy")
path_to_python <- "/home/local/kyeonghunjeong_920205/conda_mil/bin/python3"
use_python(path_to_python)
#
# # import scanpy

sc <- import("scanpy")
# # read h5ad
"scAMIL_cell/data_uc/"

whole_data = ReadMtx(mtx="../data/UC/Fib.matrix.mtx",cells ="../data/UC/Fib.barcodes.tsv",features = "../data/UC/Fib.genes.tsv",feature.column = 1)
whole_meta_data = fread("../data/UC/all.meta2.txt")[-1,]
whole_meta_data = whole_meta_data[whole_meta_data$NAME %in% colnames(whole_data),]
whole_obj = CreateSeuratObject(whole_data,meta.data = whole_meta_data,names.field = "NAME")
# trqwe::mcsaveRDS(whole_obj,"UC/whole_obj.RDS")
test_meta = fread(sprintf("%s/obs_2.csv", base_dir))
test_obj = subset(whole_obj,cells = test_meta$cell.names)
cell_scores = fread(sprintf("%s/cell_score_2.csv",base_dir))

# prop.table(table(cell_scores$bag_labels, cell_scores$cell_score_teacher_minmax > 0.5),margin = 1)

test_meta = cbind(test_meta[,1], cell_scores)
test_meta
# test_meta = cbind(test_meta[,-1], cell_scores)
colnames(test_meta)[1]= "NAME"


test_meta = as.data.frame(merge(test_obj@meta.data, test_meta, 
                                by="NAME"))

rownames(test_meta) = test_meta$NAME
test_meta
test_obj@meta.data = test_meta
colnames(test_obj)

test_obj = FindVariableFeatures(test_obj,assay = "RNA")

test_obj = NormalizeData(test_obj)
test_obj = ScaleData(test_obj)
test_obj <- RunPCA(test_obj,features=VariableFeatures(test_obj))
test_obj <- RunUMAP(test_obj, dims = 1:50,reduction = "pca")

test_obj@meta.data$Health = factor(test_obj@meta.data$Health,levels = c("Healthy","Inflamed"))
# trqwe::mcsaveRDS(test_obj, file = sprintf("%s/test_seurat.RDS",base_dir))
# test_obj@meta.data$disease = factor(test_obj@meta.data$disease_numeric,levels=c(0,1),labels = c("Not-hosp","Hosp"))
# meta_binary_class= test_obj@meta.data[test_obj@meta.data$Health %in% c("Healthy","Inflamed"),]
# trqwe::mcsaveRDS(test_obj, file = sprintf("%s/test_and_unseen_seurat.RDS",base_dir))
# test_obj = trqwe::mcreadRDS(file = sprintf("%s/test_and_unseen_seurat.RDS",base_dir))
# meta_binary_class= test_obj@meta.data[test_obj@meta.data$Health %in% c("Healthy","Inflamed"),]
# test_obj@meta.data$cell_score_teacher_minmax == (test_obj@meta.data$cell_score_teacher- min(meta_binary_class$cell_score_teacher))/ (max(meta_binary_class$cell_score_teacher) - min(meta_binary_class$cell_score_teacher))


# test_obj@meta.data$subgroup = factor(ifelse(test_obj@meta.data$cell_score_teacher_minmax > quantile(test_obj@meta.data$cell_score_teacher_minmax)[4], "Positive","Negative"), levels=c("Negative","Positive"))

# medians = test_obj@meta.data %>% group_by(Cluster, Health) %>% summarise_at('cell_score_teacher_minmax', median)
# subgroup_counts = test_obj@meta.data %>% group_by(Health) %>% summarise_at('subgroup', function(x) sum(as.numeric(x)))
# subgroup_counts = test_obj@meta.data %>% group_by(Health) %>% count(subgroup)
# test_obj@meta.data %>% count(subgroup)
# test_obj = trqwe::mcreadRDS(file=sprintf("%s/test_seurat.RDS",base_dir))
test_obj$cell_score_minmax = test_obj$cell_score_teacher_minmax
test_obj$cell_score_teacher_minmax = NULL

ggdensity(test_obj@meta.data,x = "cell_score_minmax",color = "Health",fill="Health",add="mean", rug=TRUE)
ggdensity(test_obj@meta.data,x = "cell_score_minmax",add="mean", rug=TRUE)

test_obj[,test_obj@meta.data$Health %in% c("Healthy","Inflamed")]

################################ grouping 
test_obj@meta.data$subgroup = factor(ifelse(test_obj@meta.data$cell_score_minmax > 0.5, "Positive","Negative"), levels=c("Negative","Positive"))
test_obj@meta.data$disease_association = ifelse(test_obj$Health == "Inflamed", 
                                                ifelse(test_obj$subgroup == "Positive", "Inflamed Positive", "Inflamed Negative"),"Healthy") 
test_obj@meta.data$disease_association = factor(test_obj@meta.data$disease_association,levels = c("Healthy","Inflamed Negative","Inflamed Positive"))

table(test_obj@meta.data$Health, test_obj@meta.data$subgroup_0.5)
table(test_obj@meta.data$Health, test_obj@meta.data$subgroup)
table(test_obj@meta.data$cell_type, test_obj@meta.data$disease_association)
################################ UMAP
color_disease = c("#2ECC40","#FF4136")
color_subgroup = c("#00FFFF","#FF00FF")
color_disease_asso = c("#2ECC40", "#9ACD32", "#FF851B")

p_ct  = DimPlot(test_obj,group.by = "Cluster",label = F) + ggtitle("")
p_disease = DimPlot(test_obj,group.by="Healthy",cols=color_disease)  + ggtitle("")
p_score =  FeaturePlot(test_obj,"cell_score_minmax",order = T) + ggtitle("")
p_subgroup = DimPlot(test_obj,group.by = "subgroup",cols=color_subgroup) + ggtitle("")
p_disease_association = DimPlot(test_obj, group.by = "disease_association", cols = c("#2ECC40", "#9ACD32", "#FF851B"),order = T) + ggtitle("")

(p_ct | p_disease) / (p_score | p_subgroup)
res_dir = c("downstream_scMILD/uc_inflamed")
# dir.create(res_dir)
ggsave(plot = p_ct,filename =  file.path(res_dir,"UMAP_exp2_celltype.pdf"),device = 'pdf',width=10, height=8,dpi = 450)
ggsave(plot = p_score,filename =  file.path(res_dir,"UMAP_exp2_cell_attn_score.pdf"),device = 'pdf',width=8, height=8,dpi = 450)
ggsave(plot = p_disease,filename =  file.path(res_dir,"UMAP_exp2_sample_phenotype.pdf"),device = 'pdf',width=8, height=8,dpi = 450)
ggsave(plot = p_subgroup,filename =  file.path(res_dir,"UMAP_exp2_cell_subgroup.pdf"),device = 'pdf',width=8, height=8,dpi = 450)
ggsave(plot = p_disease_association,filename =  file.path(res_dir,"UMAP_exp2_disease_association.pdf"),device = 'pdf',width=8, height=8,dpi = 450)

################################ Bar plot
library(ggplot2)

p_fill = ggplot(test_obj@meta.data, aes(x = Cluster, fill = subgroup)) +
  geom_bar(position = "fill")+ theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + theme(legend.position = "none") +scale_fill_manual(values = color_subgroup)+xlab("Cell type") + ylab('fraction')

p_stack = ggplot(test_obj@meta.data, aes(x = Cluster, fill = subgroup)) +
  geom_bar(position = "stack") + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+scale_fill_manual(values = color_subgroup)+ labs(fill='Cell subgroup')+xlab("Cell type")


p_fill_bag = ggplot(test_obj@meta.data, aes(x = Cluster, fill = Health)) +
  geom_bar(position = "fill")+ theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+ theme(legend.position = "none") +scale_fill_manual(values = color_disease)+xlab("Cell type") + ylab('fraction')

p_stack_bag = ggplot(test_obj@meta.data, aes(x = Cluster, fill = Health)) +
  geom_bar(position = "stack") + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +scale_fill_manual(values = color_disease) + labs(fill='Condition')+xlab("Cell type")


p_fill_asso = ggplot(test_obj@meta.data, aes(x = Cluster, fill = disease_association)) +
  geom_bar(position = "fill")+ theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+ theme(legend.position = "none") +scale_fill_manual(values = color_disease_asso)+xlab("Cell type") + ylab('fraction')

p_stack_asso = ggplot(test_obj@meta.data, aes(x = Cluster, fill =disease_association,legend.title="asdas")) +
  geom_bar(position = "stack") + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+ scale_fill_manual(values = color_disease_asso) + labs(fill='Condition-association group')+xlab("Cell type")

(p_fill_bag + p_stack_bag) 
(p_fill + p_stack) /
  (p_fill_asso+ p_stack_asso)

ggsave(plot = (p_fill_bag + p_stack_bag),filename =  file.path(res_dir,"Bar_exp2_bag_celltype_counts.pdf"),device = 'pdf',width=12, height=8,dpi = 450)
ggsave(plot = (p_fill + p_stack) ,filename =  file.path(res_dir,"Bar_exp2_subgroup_celltype_counts.pdf"),device = 'pdf',width=12, height=8,dpi = 450)
ggsave(plot = (p_fill_asso + p_stack_asso) ,filename =  file.path(res_dir,"Bar_exp2_asso_celltype_counts.pdf"),device = 'pdf',width=12, height=8,dpi = 450)

write.csv(table(test_obj@meta.data$Cluster,test_obj@meta.data$Health),file = file.path(res_dir,"CountTable_Cluster_Health.csv"))
write.csv(table(test_obj@meta.data$Cluster,test_obj@meta.data$subgroup),file = file.path(res_dir,"CountTable_Cluster_subgroup.csv"))
write.csv(table(test_obj@meta.data$Cluster,test_obj@meta.data$disease_association),file = file.path(res_dir,"CountTable_Cluster_disease_asso.csv"))

################################ Box plot

VlnPlot(test_obj,features = "cell_score_minmax",group.by="Cluster",sort=T,alpha= 0)
library(ggpubr)
library(dplyr)
require(scales)
cluster_score = test_obj@meta.data %>% group_by(Cluster) %>% summarise_at(.vars = "cell_score_minmax", median)
cluster_score = cluster_score[order(cluster_score$cell_score_minmax,decreasing=T),]
Idents(test_obj) = "Cluster"
identities <- levels(Idents(test_obj))
color_celltype <- hue_pal()(length(identities))
# "#F8766D" "#E18A00" "#BE9C00" "#8CAB00" "#24B700" "#00BE70" "#00C1AB" "#00BBDA" "#00ACFC" "#8B93FF" "#D575FE"
# "#F962DD" "#FF65AC"

names(color_celltype) = sort(identities,decreasing = F)
p_box= ggboxplot(test_obj@meta.data, x="Cluster", y="cell_score_minmax", fill='Cluster',facet.by = "Health",order = cluster_score$Cluster,palette = color_celltype,xlab = "Cell type", ylab="Cell attention score")+ggpubr::rotate_x_text(angle=90) +NoLegend() 
ggsave(plot = p_box ,filename =  file.path(res_dir,"Box_exp2_attn_score_celltype.pdf"),device = 'pdf',width=12, height=8,dpi = 450)

################################ Unseen class

meta_unseen = fread(sprintf("%s/unseen_obs_2.csv", base_dir))


unseen_obj = subset(whole_obj,cells = meta_unseen$cell.names)

colnames(whole_obj)
whole_obj@meta.data

cell_scores_unseen = fread(sprintf("%s/unseen_cell_score_2.csv",base_dir))
# colnames(cell_scores_unseen)[135] = "cell_score_minmax"
unseen_meta = cbind(meta_unseen[,1], cell_scores_unseen)
unseen_meta$cell.names == names(unseen_obj@active.ident)
colnames(unseen_meta)[1]= "NAME"
unseen_meta = as.data.frame(merge(unseen_obj@meta.data, unseen_meta, 
                                  by="NAME",sort=F))
rownames(unseen_meta) = unseen_meta$NAME

unseen_obj@meta.data = unseen_meta

names(unseen_obj@active.ident) %in% unseen_meta$NAME
names(unseen_obj@active.ident) == unseen_meta$NAME

unseen_obj = FindVariableFeatures(unseen_obj)
unseen_obj = NormalizeData(unseen_obj)
unseen_obj = ScaleData(unseen_obj)
unseen_obj <- RunPCA(unseen_obj,features=VariableFeatures(unseen_obj))
unseen_obj <- RunUMAP(unseen_obj, dims = 1:50,reduction = "pca")

unseen_obj$cell_score_minmax = (unseen_obj$cell_score_teacher - min(test_obj$cell_score_teacher))/(max(test_obj$cell_score_teacher) - min(test_obj$cell_score_teacher)) 

unseen_obj@meta.data$subgroup = factor(ifelse(unseen_obj@meta.data$cell_score_minmax > 0.5, "Positive","Negative"), levels=c("Negative","Positive"))

unseen_obj@meta.data$disease_association = paste0(unseen_obj$Health," ",unseen_obj$subgroup)


table(unseen_obj@meta.data$Health, unseen_obj@meta.data$subgroup)


################

#trqwe::mcsaveRDS(unseen_obj, file = sprintf("%s/unseen_seurat.RDS",base_dir))
DimPlot(unseen_obj,group.by = "Cluster")+ FeaturePlot(unseen_obj,"cell_score_minmax") +DimPlot(unseen_obj,group.by = "subgroup",cols = color_subgroup)
DimPlot(unseen_obj,group.by = "Cluster")
####################################################################################################
FeaturePlot(unseen_obj,"AHR")
Idents(test_obj) = 'Cluster'
Idents(unseen_obj) = 'Cluster'
unseen_obj

####################################################################################################

# library(MAST)
# library(SingleCellExperiment)
# library(SummarizedExperiment)
test_obj$disease_association
test_subject = unique(test_obj$Subject)
unseen_obj = trqwe::mcreadRDS(file = sprintf("%s/unseen_seurat.RDS",base_dir))
unseen_obj$nGene = as.numeric(unseen_obj$nGene)
unseen_obj@meta.data$nGene_scaled = scale(unseen_obj@meta.data$nGene)
unseen_obj@meta.data$subgroup

Idents(test_obj) = "Cluster"
unseen_obj@active.ident
cell_types = levels(test_obj@active.ident)

cell_types[1]

table(test_obj$disease_association,test_obj$Cluster)
tab_cluster_disease_association = data.table(table(test_obj$disease_association,test_obj$Cluster))
tab_cluster_disease_association = tab_cluster_disease_association[tab_cluster_disease_association$V1 %in% c("Healthy","Inflamed Positive"),]
tab_cluster_disease_association = tab_cluster_disease_association[tab_cluster_disease_association$N >20,]
target_cell_types = tab_cluster_disease_association$V2[duplicated(tab_cluster_disease_association$V2)]
target_cell_types
unseen_obj_subset = subset(unseen_obj,Subject %in% test_subject)

merge_obj = merge(test_obj,unseen_obj_subset)
merge_obj = JoinLayers(merge_obj)
merge_obj = NormalizeData(merge_obj)
merge_obj = ScaleData(merge_obj)
merge_obj = FindVariableFeatures(merge_obj)
merge_obj = RunPCA(merge_obj)
merge_obj = RunUMAP(merge_obj,dims = 1:30)

DimPlot(merge_obj,group.by = "disease_association")

# trqwe::mcsaveRDS(merge_obj,file.path(base_dir,"merged_test_non_inflamed.rds"))
rm(list=ls())
gc()
setwd("/home/local/kyeonghunjeong_920205/nipa_bu/COVID19/3.analysis/9.MIL/scAMIL_cell/scMILD/downstream/")
base_dir = 'UC'
merge_obj = trqwe::mcreadRDS(file.path(base_dir,"merged_test_non_inflamed.rds"))


data.table(table(merge_obj$disease_association,merge_obj$Cluster))
tab_cluster_disease_association = data.table(table(merge_obj$disease_association,merge_obj$Cluster))
tab_cluster_disease_association = data.table(table(merge_obj$Health,merge_obj$Cluster))
tab_cluster_disease_association = tab_cluster_disease_association[tab_cluster_disease_association$V1 %in% c("Healthy","Inflamed"),]
tab_cluster_disease_association = tab_cluster_disease_association[tab_cluster_disease_association$N >100,]
target_cell_types = tab_cluster_disease_association$V2[duplicated(tab_cluster_disease_association$V2)]
tab_cluster_disease_association
library(monocle3)
merge_obj_subset = subset(merge_obj,Cluster %in% target_cell_types[[1]])
merge_cds = SeuratWrappers::as.cell_data_set(merge_obj_subset)
set.seed(3)
merge_cds <- preprocess_cds(merge_cds, num_dim = 50)
merge_cds <- reduce_dimension(merge_cds)

merge_cds <- cluster_cells(merge_cds)
?preprocess_cds
plot_cells(merge_cds,color_cells_by = "Cluster")
plot_cells(merge_cds,color_cells_by = "Cluster",cell_size=5) + plot_cells(merge_cds,color_cells_by = "subgroup",cell_size=5)
plot_cells(merge_cds,color_cells_by = "partition")+plot_cells(merge_cds,color_cells_by = "Cluster")
merge_cds <- learn_graph(merge_cds)
merge_cds = order_cells(merge_cds)
plot_cells(merge_cds,color_cells_by = "Cluster",cell_size=5)+ plot_cells(merge_cds,color_cells_by = "cell_score_minmax",cell_size=5)+ plot_cells(merge_cds,color_cells_by = "Health",cell_size=5)
plot_cells(merge_cds,color_cells_by = "cell_score_minmax",cell_size = 5)+plot_cells(merge_cds,color_cells_by = "pseudotime",cell_size = 5)
merge_cds@colData[["cell_score_minmax"]]
merge_cds@colData@listData
merge_cds@assays
merge_cds
# tmp_disease = subsetMarkers(test_obj,idents=cell_types[i],"Disease")
# table(unseen_obj_subset$subgroup,unseen_obj_subset$Cluster)
i = 1

cds_each_celltype = list()
for( i in 1:length(target_cell_types)){
  merge_obj_subset = subset(merge_obj,Cluster %in% target_cell_types[[i]])
  merge_cds = SeuratWrappers::as.cell_data_set(merge_obj_subset)
  merge_cds <- preprocess_cds(merge_cds, num_dim = 30)
  merge_cds <- reduce_dimension(merge_cds)
  merge_cds <- cluster_cells(merge_cds)
  merge_cds <- learn_graph(merge_cds)
  cds_each_celltype[[i]] = merge_cds
  
}
i = 1
names(cds_each_celltype) = target_cell_types
# trqwe::mcsaveRDS(cds_each_celltype,file.path(base_dir,"cds_each_celltype.rds"))
plot_cells(cds_each_celltype[[1]],color_cells_by = "cell_score_minmax",cell_size = 5,label_principal_points = T)
cds_each_celltype[[1]] = order_cells(cds_each_celltype[[1]],root_pr_nodes  = "Y_1")
color_disease = c("#2ECC40","#FF4136")
cds_each_celltype[[i]]$cell_score_teacher_out =  ifelse(cds_each_celltype[[i]]$cell_score_minmax > quantile(cds_each_celltype[[i]]$cell_score_minmax)[4], "Positive", "Negative")
cds_each_celltype[[i]]$tmp_cluster = paste0(cds_each_celltype[[i]]$Health," ", cds_each_celltype[[i]]$cell_score_teacher_out)
cds_each_celltype[[i]]$tmp_cluster = factor(cds_each_celltype[[i]]$tmp_cluster,levels=c("Healthy Negative","Healthy Positive", "Non-inflamed Negative","Inflamed Negative", "Non-inflamed Positive", "Inflamed Positive" ), labels = c("Healthy","Healthy", "Non-inflamed Negative", "Inflamed Negative" , "Non-inflamed Positive", "Inflamed Positive" ))




plot_cells(cds_each_celltype[[2]],color_cells_by = "cell_score_minmax",cell_size = 5,label_principal_points = T,graph_label_size = 5)
cds_each_celltype[[2]] = order_cells(cds_each_celltype[[2]],root_pr_nodes = c("Y_23","Y_40"))

i = 2
cds_each_celltype[[i]]$cell_score_teacher_out =  ifelse(cds_each_celltype[[i]]$cell_score_minmax > quantile(cds_each_celltype[[i]]$cell_score_minmax)[4], "Positive", "Negative")

cds_each_celltype[[i]]$tmp_cluster = paste0(cds_each_celltype[[i]]$Health," ", cds_each_celltype[[i]]$cell_score_teacher_out)
cds_each_celltype[[i]]$tmp_cluster = factor(cds_each_celltype[[i]]$tmp_cluster,levels=c("Healthy Negative","Healthy Positive", "Non-inflamed Negative","Inflamed Negative", "Non-inflamed Positive", "Inflamed Positive" ), labels = c("Healthy","Healthy", "Non-inflamed Negative", "Inflamed Negative" , "Non-inflamed Positive", "Inflamed Positive" ))






plot_cells(cds_each_celltype[[4]],color_cells_by = "cell_score_minmax",cell_size = 5,label_principal_points = T,graph_label_size = 5)
cds_each_celltype[[4]] <- learn_graph(cds_each_celltype[[4]],learn_graph_control = list(minimal_branch_len = 5))
plot_cells(cds_each_celltype[[4]],color_cells_by = "cell_score_minmax",cell_size = 5,label_principal_points = T,graph_label_size = 5)
cds_each_celltype[[4]]= order_cells(cds_each_celltype[[4]],root_pr_nodes = c("Y_13","Y_58"))
i = 4
cds_each_celltype[[i]]$tmp_cluster = paste0(cds_each_celltype[[i]]$Health," ", cds_each_celltype[[i]]$cell_score_teacher_out)
cds_each_celltype[[i]]$tmp_cluster = factor(cds_each_celltype[[i]]$tmp_cluster,levels=c("Healthy Negative","Healthy Positive", "Non-inflamed Negative","Inflamed Negative", "Non-inflamed Positive", "Inflamed Positive" ), labels = c("Healthy","Healthy", "Non-inflamed Negative", "Inflamed Negative" , "Non-inflamed Positive", "Inflamed Positive" ))





cds_each_celltype[[3]]$Health= factor(cds_each_celltype[[3]]$Health,levels = c("Healthy","Non-inflamed","Inflamed"))
plot_cells(cds_each_celltype[[3]],color_cells_by = "cell_score_minmax",cell_size = 1,label_principal_points = T,graph_label_size = 5) / (plot_cells(cds_each_celltype[[i]],color_cells_by = "tmp_cluster",cell_size = 1,label_cell_groups = F)+scale_color_manual(values =  c("#2ECC40","#90EE90","#98FB98","#FFA07A","#FF4136")))
cds_each_celltype[[3]]= order_cells(cds_each_celltype[[3]],root_pr_nodes = c("Y_38","Y_51","Y_60","Y_25"))
i = 3
cds_each_celltype[[i]]$cell_score_teacher_out =  ifelse(cds_each_celltype[[i]]$cell_score_minmax > quantile(cds_each_celltype[[i]]$cell_score_minmax)[4], "Positive", "Negative")
cds_each_celltype[[i]]$tmp_cluster = paste0(cds_each_celltype[[i]]$Health," ", cds_each_celltype[[i]]$cell_score_teacher_out)
cds_each_celltype[[i]]$tmp_cluster = factor(cds_each_celltype[[i]]$tmp_cluster,levels=c("Healthy Negative","Healthy Positive", "Non-inflamed Negative","Inflamed Negative", "Non-inflamed Positive", "Inflamed Positive" ), labels = c("Healthy","Healthy", "Non-inflamed Negative", "Inflamed Negative" , "Non-inflamed Positive", "Inflamed Positive" ))


i = 1
ps_score_cor =c()
for(i in 1:4){
  tmp_cor = cor(cds_each_celltype[[i]]$cell_score_minmax, cds_each_celltype[[i]]@principal_graph_aux$UMAP$pseudotime,method="spearman")
  ps_score_cor = c(ps_score_cor,tmp_cor)
}
names(ps_score_cor) = target_cell_types
write.table(ps_score_cor,file = file.path(res_dir,"ps_score_cor.txt"),sep = "\t",quote = F,col.names = F,row.names = T)

i=1
p1_ptime = plot_cells(cds_each_celltype[[i]],color_cells_by = "pseudotime",cell_size = 0.8) 
p1_cscore = plot_cells(cds_each_celltype[[i]],color_cells_by = "cell_score_minmax",cell_size = 0.8)  + guides(color=guide_legend("Cell attention score"))+scale_colour_gradientn(colours=c("#00429D", "#EFEFEF","#93003A"))
p1_sublabel = (plot_cells(cds_each_celltype[[i]],color_cells_by = "tmp_cluster",cell_size = 0.8,label_cell_groups = F)+scale_color_manual(values = c("#228B22", "#32CD32", "#7FFF00", "#FFA07A", "#FF4136")))+ guides(color=guide_legend("Condition-association group"))
i=2
p2_ptime = plot_cells(cds_each_celltype[[i]],color_cells_by = "pseudotime",cell_size = 0.8) 
p2_cscore = plot_cells(cds_each_celltype[[i]],color_cells_by = "cell_score_minmax",cell_size = 0.8)  + guides(color=guide_legend("Cell attention score"))+scale_colour_gradientn(colours=c("#1A5C99", "#EFEFEF","#F58F01"))
p2_sublabel = (plot_cells(cds_each_celltype[[i]],color_cells_by = "tmp_cluster",cell_size = 0.8,label_cell_groups = F)+scale_color_manual(values = c("#228B22", "#32CD32", "#7FFF00", "#FFA07A", "#FF4136")))+ guides(color=guide_legend("Condition-association group"))

i=4
p4_ptime = plot_cells(cds_each_celltype[[i]],color_cells_by = "pseudotime",cell_size = 0.8) 
p4_cscore = plot_cells(cds_each_celltype[[i]],color_cells_by = "cell_score_minmax",cell_size = 0.8)  + guides(color=guide_legend("Cell attention score"))+scale_colour_gradientn(colours=c("#1A5C99", "#EFEFEF","#F58F01"))
cds_each_celltype[[i]]$cell_score_teacher_out =  ifelse(cds_each_celltype[[i]]$cell_score_minmax > quantile(cds_each_celltype[[i]]$cell_score_minmax)[4], "Positive", "Negative")
p4_sublabel = (plot_cells(cds_each_celltype[[i]],color_cells_by = "tmp_cluster",cell_size = 0.8,label_cell_groups = F)+scale_color_manual(values = c("#228B22", "#32CD32", "#7FFF00", "#FFA07A", "#FF4136")))+ guides(color=guide_legend("Condition-association group"))
i=3

p3_ptime = plot_cells(cds_each_celltype[[i]],color_cells_by = "pseudotime",cell_size = 0.8) 
p3_cscore = plot_cells(cds_each_celltype[[i]],color_cells_by = "cell_score_minmax",cell_size = 0.8)  + guides(color=guide_legend("Cell attention score"))+scale_colour_gradientn(colours=c("#1A5C99", "#EFEFEF","#F58F01"))
cds_each_celltype[[i]]$cell_score_teacher_out =  ifelse(cds_each_celltype[[i]]$cell_score_minmax > quantile(cds_each_celltype[[i]]$cell_score_minmax)[4], "Positive", "Negative")
p3_sublabel = (plot_cells(cds_each_celltype[[i]],color_cells_by = "tmp_cluster",cell_size = 0.8,label_cell_groups = F)+scale_color_manual(values = c("#228B22", "#32CD32", "#7FFF00", "#FFA07A", "#FF4136")))+ guides(color=guide_legend("Condition-association group"))

p1 = p1_ptime / p1_cscore / p1_sublabel
p2 = p2_ptime / p2_cscore / p2_sublabel
p3 = p3_ptime / p3_cscore / p3_sublabel
p4 = p4_ptime / p4_cscore / p4_sublabel

ggsave(p1,file = sprintf("%s/Trajectory_%s.pdf",res_dir,target_cell_types[[1]]),device = "pdf",width=5, height=7)
ggsave(p2,file = sprintf("%s/Trajectory_%s.pdf",res_dir,target_cell_types[[2]]),device = "pdf",width=5, height=7)
ggsave(p3,file = sprintf("%s/Trajectory_%s.pdf",res_dir,target_cell_types[[3]]),device = "pdf",width=5, height=7)
ggsave(p4,file = sprintf("%s/Trajectory_%s.pdf",res_dir,target_cell_types[[4]]),device = "pdf",width=5, height=7)
library(ggplot2)
library(Seurat)



# cds_each_celltype[[1]]$Health = factor(cds_each_celltype[[1]]$Health,levels = c("Healthy","Non-inflamed","Inflamed"))
# # 
# # plot_cells(cds_each_celltype[[1]],color_cells_by = "pseudotime",cell_size = 5 ) / plot_cells(cds_each_celltype[[1]],color_cells_by = "cell_score_minmax",cell_size = 5) / (plot_cells(cds_each_celltype[[1]],color_cells_by = "Health",cell_size = 5,label_cell_groups = F)+scale_color_manual(values = c("green","orange","red")))
# # plot_cells(cds_each_celltype[[1]],color_cells_by = "Health",cell_size = 5,label_cell_groups = F) +scale_color_manual(values = c("green","orange","red"))
# # plot_cells(cds_each_celltype[[1]],color_cells_by = "subgroup",cell_size = 5,label_cell_groups = F) +scale_color_manual(values = c("green","red"))
# 
# cds_each_celltype[[2]]$Health = factor(cds_each_celltype[[2]]$Health,levels = c("Healthy","Non-inflamed","Inflamed"))
# plot_cells(cds_each_celltype[[2]],color_cells_by = "pseudotime",cell_size = 5 ) / plot_cells(cds_each_celltype[[2]],color_cells_by = "cell_score_minmax",cell_size = 5) / (plot_cells(cds_each_celltype[[2]],color_cells_by = "Health",cell_size = 5,label_cell_groups = F)+scale_color_manual(values = c("green","orange","red")))
# 
# cds_each_celltype[[3]]$Health = factor(cds_each_celltype[[3]]$Health,levels = c("Healthy","Non-inflamed","Inflamed"))
# 
# plot_cells(cds_each_celltype[[3]],color_cells_by = "pseudotime",cell_size = 5 ) / plot_cells(cds_each_celltype[[3]],color_cells_by = "cell_score_minmax",cell_size = 5) / (plot_cells(cds_each_celltype[[3]],color_cells_by = "Health",cell_size = 5,label_cell_groups = F)+scale_color_manual(values = c("green","orange","red")))
# 
# (plot_cells(cds_each_celltype[[1]],color_cells_by = "disease_association",cell_size = 5,label_cell_groups = F)+scale_color_manual(values = c("darkgreen","green","orange","lightgreen","darkred")))
# 
# (plot_cells(cds_each_celltype[[1]],color_cells_by = "disease_association",cell_size = 5,label_cell_groups = F)+scale_color_manual(values = c("grey","grey","grey","lightgreen","darkred")))
# ?scale_color_manual
# 
# 
# 
# cor(cds_each_celltype[[1]]@colData$cell_score_minmax,cds_each_celltype[[1]]@principal_graph_aux@listData[["UMAP"]][["pseudotime"]],method = "pearson")
# cor(cds_each_celltype[[2]]@colData$cell_score_minmax,cds_each_celltype[[2]]@principal_graph_aux@listData[["UMAP"]][["pseudotime"]],method = "pearson")
# cor(cds_each_celltype[[3]]@colData$cell_score_minmax,cds_each_celltype[[3]]@principal_graph_aux@listData[["UMAP"]][["pseudotime"]],method = "pearson")
# target_cell_types
# merge_obj_subset = subset(merge_obj,Cluster %in% target_cell_types)
# merge_obj_subset <- AddMetaData(
#   object = merge_obj_subset,
#   metadata = cds_each_celltype[[1]]@principal_graph_aux@listData$UMAP$pseudotime,
#   col.name = "Pseudotime Entothelial"
# )
# merge_obj_subset <- AddMetaData(
#   object = merge_obj_subset,
#   metadata = cds_each_celltype[[2]]@principal_graph_aux@listData$UMAP$pseudotime,
#   col.name = "Pseudotime Post-capillary Venules"
# )
# merge_obj_subset <- AddMetaData(
#   object = merge_obj_subset,
#   metadata = cds_each_celltype[[3]]@principal_graph_aux@listData$UMAP$pseudotime,
#   col.name = "Pseudotime WNT5B+ 2"
# )


# cor(merge_obj_subset$cell_score_minmax,merge_obj_subset$`Pseudotime Entothelial`,use = "complete.obs")
# cor(merge_obj_subset$cell_score_minmax,merge_obj_subset$`Pseudotime Post-capillary Venules`,use = "complete.obs")
# cor(merge_obj_subset$cell_score_minmax,merge_obj_subset$`Pseudotime Post-capillary Venules`,use = "complete.obs")
# cor(subset(merge_obj_subset,Health=="Non-inflamed")$cell_score_minmax,subset(merge_obj_subset,Health=="Non-inflamed")$`Pseudotime Post-capillary Venules`,use = "complete.obs")

# cor(merge_obj_subset$cell_score_minmax,merge_obj_subset$`Pseudotime WNT5B+ 2`,use = "complete.obs")



# FeatureScatter(subset(merge_obj_subset, Cluster == "Endothelial" & Health == "Non-inflamed"),feature1 = "cell_score_minmax", "Pseudotime Entothelial",group.by = "disease_association",smooth= F)
# 
# FeaturePlot(subset(merge_obj_subset, Cluster == "Endothelial" & Health == "Non-inflamed"),features = "cell_score_minmax")+ggtitle("Cell attention score") + (FeaturePlot(subset(merge_obj_subset, Cluster == "Endothelial" & Health == "Non-inflamed"),features = c("Pseudotime Entothelial"))& scale_color_viridis_c()) + DimPlot(subset(merge_obj_subset, Cluster == "Endothelial" & Health == "Non-inflamed"),group.by = "disease_association")
# merge_obj_subset$Health = factor(merge_obj_subset$Health,levels= c("Healthy", "Non-inflamed","Inflamed"))

# merge_obj_subset = FindVariableFeatures(merge_obj_subset)
# merge_obj_subset = ScaleData(merge_obj_subset)
# merge_obj_subset = NormalizeData(merge_obj_subset)
# merge_obj_subset = RunPCA(merge_obj_subset,features = VariableFeatures(merge_obj_subset))
# merge_obj_subset = RunUMAP(merge_obj_subset, 1:50)
# gene_fits <- fit_models(cds_each_celltype[[1]], model_formula_str = "~cell_score_minmax")
as.Seurat(cds_each_celltype[[1]])
ps_genes = list()


ps_genes[[1]] <- graph_test(cds_each_celltype[[1]], neighbor_graph="principal_graph", cores=4)
ps_genes[[1]] = ps_genes[[1]][order(ps_genes[[1]]$q_value),]
ps_genes[[1]]
pr_deg_ids <- row.names(subset(ps_genes[[1]], q_value < 0.01))
pr_deg_ids
Seurat::GroupCorrelation()
sum(ps_genes[[1]]$status == "OK")
cds_each_celltype[[1]]@assays@data$counts
rowData(cds_each_celltype[[1]])$gene_short_name = rownames(cds_each_celltype[[1]])
plot_cells(cds = cds_each_celltype[[1]],genes = pr_deg_ids[1:5],show_trajectory_graph = F, label_cell_groups = F, label_leaves = F,cell_size = 3)
AFD_genes <- pr_deg_ids[1:5]
cds_each_celltype[[1]]

AFD_lineage_cds <- cds_each_celltype[[1]][rowData(cds_each_celltype[[1]])$gene_short_name %in% AFD_genes,]

AFD_lineage_cds@colData$Cluster
# rownames(AFD_lineage_cds) = 
rowData(AFD_lineage_cds)
plot_genes_in_pseudotime(AFD_lineage_cds,
                         color_cells_by="disease_association",
                         min_expr=0.1)

merge_obj_subset$disease_association2 = ifelse(grepl("Inflamed",merge_obj_subset$disease_association), "Inflamed",merge_obj_subset$disease_association)
unique(merge_obj_subset$disease_association2)
merge_obj_subset$disease_association = factor(merge_obj_subset$disease_association, levels=c("Healthy","Non-inflamed Negative","Inflamed Negative","Non-inflamed Positive","Inflamed Positive"))
merge_obj_subset$disease_association2 = factor(merge_obj_subset$disease_association2,levels = c("Healthy","Non-inflamed Negative","Non-inflamed Positive","Inflamed Positive"))

p_3class = DimPlot(merge_obj_subset,group.by = "disease_association2",cols = c("darkgreen","green","red","darkred"),shuffle = T)
p_cscore = FeaturePlot(merge_obj_subset,features = "cell_score_minmax")+ggtitle("Cell attention score") 
p_ptime = (FeaturePlot(merge_obj_subset,features = c("Pseudotime Entothelial","Pseudotime Post-capillary Venules","Pseudotime WNT5B+ 2"),ncol=3)& scale_color_viridis_c()) 

p_3class = DimPlot(subset(merge_obj_subset, Cluster == "Endothelial"),group.by = "disease_association2",cols = c("darkgreen","green","red","darkred"),shuffle = T)
p_5class = DimPlot(subset(merge_obj_subset, Cluster == "Endothelial"),group.by = "disease_association",cols = c("darkgreen","green","lightgreen","orangered","darkred"),shuffle = T)
p_2class = DimPlot(subset(merge_obj_subset, Cluster == "Endothelial"),group.by = "disease_association",cols = c("darkgreen","green","grey","orangered","darkred"),shuffle = T,pt.size = 5)
merge_obj_subset$`Pseudotime Entothelial`
p_5class = VlnPlot(subset(merge_obj_subset, Cluster == "Endothelial"),features = "Pseudotime Entothelial",group.by = "disease_association",cols = c("darkgreen","green","lightgreen","orangered","darkred"),alpha=0)

VlnPlot(subset(merge_obj_subset, Cluster == "Post-capillary Venules"),features = "Pseudotime Post-capillary Venules",group.by = "disease_association",cols = c("darkgreen","green","lightgreen","orangered","darkred"),alpha=0)
VlnPlot(subset(merge_obj_subset, Cluster == "WNT5B+ 2"),features = "Pseudotime WNT5B+ 2",group.by = "disease_association",cols = c("darkgreen","green","lightgreen","orangered","darkred"),alpha=0)

library(ggpubr)

options(ggpubr.parse_aes= F)
i = 2
target_cell_types
# merge_obj_subset$`Pseudotime Endothelial`= merge_obj_subset$`Pseudotime Entothelial`
tmp_meta = subset(merge_obj_subset, Cluster == target_cell_types[[i]])@meta.data

ggboxplot(tmp_meta, x="disease_association",y= sprintf("Pseudotime %s",target_cell_types[i]),fill = "disease_association" ,palette = c("darkgreen","green","lightgreen","orangered","darkred"))+RotatedAxis() + NoLegend()




p_2class
p_5class
p_cscore = FeaturePlot(subset(merge_obj_subset, Cluster == "Endothelial"),features = "cell_score_minmax")+ggtitle("Cell attention score") 
p_ptime = (FeaturePlot(subset(merge_obj_subset, Cluster == "Endothelial"),features = c("Pseudotime Entothelial"))& scale_color_viridis_c()) 
(p_3class+ p_cscore) / p_ptime




plot(cds_each_celltype[[1]]@colData$cell_score_stud_softmax,cds_each_celltype[[1]]@principal_graph_aux@listData[["UMAP"]][["pseudotime"]])


plot(cds_each_celltype[[2]]@colData$cell_score_minmax,cds_each_celltype[[2]]@principal_graph_aux@listData[["UMAP"]][["pseudotime"]])
cds_each_celltype[[4]]
target_cell_types[[4]]
plot_cells(cds_each_celltype[[4]],color_cells_by = "cell_score_minmax",cell_size = 5) +plot_cells(cds_each_celltype[[4]],color_cells_by = "Health",cell_size = 5,label_cell_groups = F)
target_cell_types[[5]]
plot_cells(cds_each_celltype[[5]],color_cells_by = "cell_score_minmax",cell_size = 5) +plot_cells(cds_each_celltype[[5]],color_cells_by = "Health",cell_size = 5,label_cell_groups = F)
plot_cells(cds_each_celltype[[6]],color_cells_by = "cell_score_minmax",cell_size = 5) +plot_cells(cds_each_celltype[[6]],color_cells_by = "Health",cell_size = 5,label_cell_groups = F)
target_cell_types
target_cells_2
cds_each_celltype[[1]]$cell_score_teacher_scaled = scale(cds_each_celltype[[1]]$cell_score_minmax) + 1




plot_cells(cds_each_celltype[[8]],color_cells_by = "cell_score_minmax",cell_size = 5) +plot_cells(cds_each_celltype[[8]],color_cells_by = "Health",cell_size = 5,label_cell_groups = F)



subsetMarkers = function(obj, idents, cut.padj = NULL, logfc.threshold = NULL, type="Positive"){
  if(is.null(logfc.threshold)){
    logfc.threshold = 0.1
  }
  subset_obj = subset(obj, idents = idents)
  if(type=="Positive"){
    Idents(subset_obj) = "subgroup"
    disease_markers_subset = FindMarkers(subset_obj ,ident.1 = "Positive",ident.2 = "Negative",logfc.threshold = logfc.threshold,only.pos = F)
  } else if(type == "Disease"){
    Idents(subset_obj) = "Health"
    disease_markers_subset = FindMarkers(subset_obj ,ident.1 = "Inflamed",ident.2 = "Healthy",logfc.threshold = logfc.threshold,only.pos = F)
  } else if(type == "Subtype"){
    Idents(subset_obj) = "disease_association"
    disease_markers_subset = FindMarkers(subset_obj,ident.1 = 'Inflamed Positive', ident.2 = 'Healthy', logfc.threshold = logfc.threshold, only.pos = F)
  } else if(type == "Unused"){
    Idents(subset_obj) = "disease_association"
    disease_markers_subset = FindMarkers(subset_obj,ident.1 = 'Non-inflamed Positive', ident.2 = 'Healthy', logfc.threshold = logfc.threshold, only.pos = F)
  }
  
  if(!is.null(cut.padj)){
    disease_markers_subset = disease_markers_subset[disease_markers_subset$p_val_adj < cut.padj,]
  }
  return(disease_markers_subset)
}


names(cds_each_celltype) = target_cell_types
cds_each_celltype
i = 3
target_cell_types[i]
tmp_disease = subsetMarkers(merge_obj,idents=target_cell_types[i],
                            type="Disease")
tmp_disease = data.table(tmp_disease,keep.rownames = "gene")

tmp_disease_asso = subsetMarkers(merge_obj,idents=target_cell_types[i],
                                 type="Subtype")
tmp_disease_asso = data.table(tmp_disease_asso,keep.rownames = "gene")

tmp_subgroup_unseen = subsetMarkers(merge_obj,idents=target_cell_types[i],type="Unused")

# tmp_subgroup_unseen = subsetMarkers(unseen_obj,idents=target_cell_types[i],type="Positive")
tmp_subgroup_unseen = data.table(tmp_subgroup_unseen,keep.rownames = "gene")


tmp_dat = merge(tmp_disease, tmp_subgroup_unseen,suffixes=c("_disease","_asso"),by="gene",all=F)
tmp_dat2 = merge(tmp_disease_asso, tmp_subgroup_unseen,suffixes=c("_disease","_asso"),by="gene",all=F)
cor(-log10(tmp_dat$p_val_adj_disease), -log10(tmp_dat$p_val_adj_asso))
plot(-log10(tmp_dat$p_val_adj_disease), -log10(tmp_dat$p_val_adj_asso),xlab="Healthy vs. Inflamed", ylab = "Healthy vs. Non-inflamed Positive") 
plot(-log10(tmp_dat2$p_val_adj_disease), -log10(tmp_dat2$p_val_adj_asso),xlab="Healthy vs. Inflamed", ylab = "Healthy vs. Non-inflamed Positive") 
plot(tmp_dat$avg_log2FC_disease, tmp_dat$avg_log2FC_asso,xlab="Healthy vs. Inflamed", ylab = "Healthy vs. Non-inflamed Positive",name="log2FC")
cor(tmp_dat$avg_log2FC_disease, tmp_dat$avg_log2FC_asso)

cor(-log10(tmp_dat2$p_val_adj_disease), -log10(tmp_dat2$p_val_adj_asso))
# cor(-log10(tmp_dat$p_val_disease), -log10(tmp_dat$p_val_asso))
# cor(-log10(tmp_dat$p_val_adj_disease), -log10(tmp_dat$p_val_adj_asso),method="spearman")


cor(tmp_dat$avg_log2FC_disease, tmp_dat$avg_log2FC_asso,use="complete.obs")


unseen_obj.sce = as.SingleCellExperiment()

