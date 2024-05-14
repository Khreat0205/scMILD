library(Seurat)
library(SeuratDisk)
library(ggpubr)
library(patchwork)
library(data.table)
library(SeuratObject)

whole_obs = fread("NS/whole_obs.csv")
Convert("NS/anndata_2.h5ad", dest = "h5seurat", overwrite = T) ## mean performance
test_obj = LoadH5Seurat("NS/anndata_2.h5seurat", meta.data=F)

cell_meta_data  = fread("NS/obs_2.csv")
cell_scores = fread("NS/cell_score_2.csv")
cell_meta_data = cbind(cell_meta_data, cell_scores)
cell_meta_data = cell_meta_data[,-1]
cell_meta_data = as.data.frame(cell_meta_data)
rownames(cell_meta_data) = cell_meta_data$cell.names
test_obj = subset(test_obj,cells = colnames(test_obj))
test_obj@meta.data= cell_meta_data
test_obj = NormalizeData(test_obj)
test_obj = ScaleData(test_obj)
test_obj = FindVariableFeatures(test_obj,nfeatures = 3000)
test_obj <- RunPCA(test_obj,features=VariableFeatures(test_obj))
test_obj <- RunUMAP(test_obj, dims = 1:50,reduction = "pca",seed.use = 2)
test_obj@meta.data$disease_cov = factor(test_obj@meta.data$disease_cov,levels = c("normal","COVID-19"))
test_obj@meta.data$cell_score_minmax

test_obj@meta.data$subgroup = factor(ifelse(test_obj@meta.data$cell_score_minmax > 0.5, "Positive","Negative"), levels=c("Negative","Positive"))
test_obj@meta.data$disease_association = ifelse(test_obj$disease_cov == "COVID-19", 
                                                ifelse(test_obj$subgroup == "Positive", "COVID-19 Positive", "COVID-19 Negative"),"Normal") 

test_obj@meta.data$disease_association = factor(test_obj@meta.data$disease_association,levels = c("Normal","COVID-19 Negative","COVID-19 Positive"))
feat_SARS = rownames(test_obj)[grep("^SARSCoV2-",rownames(test_obj))]
test_obj = MetaFeature(test_obj,features = feat_SARS,meta.name = "Meta-SARSCoV2",slot = "scale.data")


res_dir = "NS/plot_table"
dir.create(res_dir)
text_theme = theme(
  aspect.ratio = 1,
  axis.text = element_text(size = unit(10, "pt")),
  axis.title = element_text(size = unit(10, "pt")),
  legend.text = element_text(size = unit(8, "pt")))#  +coord_fixed(ratio = 1)

trimmedPlot = function(p,filename){
  p_legend = cowplot::get_legend(p)
  p = p+ ggtitle("") +text_theme
  
  filename_legend = gsub(".pdf","_legend.pdf",filename)
  ggsave(plot = p+NoLegend()+coord_fixed(ratio = 1),filename =  file.path(res_dir,filename),device = 'pdf',width=47.5, height=47.5,dpi = 450,units = "mm",scale= 2,limitsize = F,)
  ggsave(plot = p_legend,filename =  file.path(res_dir,filename_legend),device = 'pdf',width=NA, height=47.5,dpi = 450,units = "mm",scale=2)
  
}


p_ct  = DimPlot(test_obj,group.by = "ct_cov",label = F,pt.size = 0.1) 
trimmedPlot(p_ct,"UMAP_exp2_celltype.pdf")


p_score =  FeaturePlot(test_obj,"cell_score_minmax") # + ggtitle("")+ text_theme +coord_fixed(ratio = 1) #Cell attention score
trimmedPlot(p_score,"UMAP_exp2_cell_attn_score.pdf")

p_disease = DimPlot(test_obj,group.by="disease_cov",cols=c("#0070C0","#FF4136")) + ggtitle("")  # +coord_fixed(ratio = 1)# Sample phenotype
trimmedPlot(p_disease,"UMAP_exp2_sample_phenotype.pdf")

p_subgroup = DimPlot(test_obj,group.by = "subgroup",cols=c("#00FFFF","#FF00FF")) + ggtitle("")  #+coord_fixed(ratio = 1)# Cell subgroup
trimmedPlot(p_subgroup,"UMAP_exp2_cell_subgroup.pdf")

p_disease_association = DimPlot(test_obj, group.by = "disease_association", cols = c("#90CAF9", "#FFCC80", "#E53935")) # + ggtitle("")+text_theme +coord_fixed(ratio = 1,)
trimmedPlot(p_disease_association,"UMAP_exp2_disease_association.pdf")

p_metaSARS = FeaturePlot(test_obj,c("Meta-SARSCoV2"),max.cutoff = "q75")+ggtitle('')
ggsave(plot = p_metaSARS,filename =  file.path(res_dir,"UMAP_exp2_meta_SARSCoV2.pdf"),device = 'pdf',width=8, height=8,dpi = 450)
########################################################################
p_fill = ggplot(test_obj@meta.data, aes(x = ct_cov, fill = subgroup)) +
  geom_bar(position = "fill")+ theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + theme(legend.position = "none") +scale_fill_manual(values = c("#00FFFF","#FF00FF"))+xlab("Cell type") + ylab('fraction')

p_stack = ggplot(test_obj@meta.data, aes(x = ct_cov, fill = subgroup)) +
  geom_bar(position = "stack") + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+scale_fill_manual(values = c("#00FFFF","#FF00FF"))+ labs(fill='Cell subgroup')+xlab("Cell type")

p_fill_bag = ggplot(test_obj@meta.data, aes(x = ct_cov, fill = disease_cov)) +
  geom_bar(position = "fill")+ theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+ theme(legend.position = "none") +scale_fill_manual(values = c("#0070C0","#FF4136"))+xlab("Cell type") + ylab('fraction')
p_stack_bag = ggplot(test_obj@meta.data, aes(x = ct_cov, fill =disease_cov,legend.title="asdas")) +
  geom_bar(position = "stack") + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+ scale_fill_manual(values = c("#0070C0","#FF4136")) + labs(fill='Sample phenotype')+xlab("Cell type")

p_fill_asso = ggplot(test_obj@meta.data, aes(x = ct_cov, fill = disease_association)) +
  geom_bar(position = "fill")+ 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+ 
  theme(legend.position = "none") +
  scale_fill_manual(values = c("#90CAF9", "#FFCC80", "#E53935"))+xlab("") + ylab('Fraction')+text_theme

p_stack_asso = ggplot(test_obj@meta.data, aes(x = ct_cov, fill =disease_association,legend.title="asdas")) +
  geom_bar(position = "stack") + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), legend.position="none")+ 
  scale_fill_manual(values = c("#90CAF9", "#FFCC80", "#E53935")) + 
  labs(fill='Disease-association group')+xlab("")+ylab("Count")+text_theme

ggexport(p_fill_asso+p_stack_asso,filename = file.path(res_dir,"Bar_exp2_asso_celltype_counts.pdf"),height = 5, width=9 ,res=450)

ggsave(plot = (p_fill_bag + p_stack_bag),filename =  file.path(res_dir,"Bar_exp2_bag_celltype_counts.pdf"),device = 'pdf',width=8, height=5,dpi = 450)
ggsave(plot = (p_fill + p_stack) ,filename =  file.path(res_dir,"Bar_exp2_subgroup_celltype_counts.pdf"),device = 'pdf',width=8, height=5,dpi = 450)


##############################################################################
library(ggpubr)
library(ComplexHeatmap)

whole_ct_count = as.matrix(table(whole_obs$disease_cov, whole_obs$ct_cov))
test_ct_count = as.matrix(table(test_obj@meta.data$disease_cov, test_obj@meta.data$ct_cov))
test_ct_count_asso = as.matrix(table(test_obj@meta.data$disease_association, test_obj@meta.data$ct_cov))
test_ct_count_subgroup = as.matrix(table(test_obj@meta.data$subgroup, test_obj@meta.data$ct_cov))
merge_test_ct_count = rbind(test_ct_count, test_ct_count_subgroup, test_ct_count_asso )
write.csv(whole_ct_count,file =file.path(res_dir,'whole_dataset_celltype_counts.csv'))
write.csv(merge_test_ct_count,file =file.path(res_dir,'test_dataset_celltype_counts.csv'))

subsetMarkers = function(obj, idents,cut.padj = NULL, logfc.threshold = NULL, type="Positive"){
  if(is.null(logfc.threshold)){
    logfc.threshold = 0.25
  }
  subset_obj = subset(obj, idents = idents)
  # subset_obj = subset(subset_obj,disease_cov == "COVID-19")
  print(subset_obj)
  if(type=="Positive"){
    Idents(subset_obj) = "subgroup"
    disease_markers_subset = FindMarkers(subset_obj ,ident.1 = "Positive",ident.2 = "Negative",logfc.threshold = logfc.threshold,only.pos = T)
  } else if(type == "Disease"){
    Idents(subset_obj) = "disease_cov"
    disease_markers_subset = FindMarkers(subset_obj ,ident.1 = "COVID-19",ident.2 = "normal",logfc.threshold = logfc.threshold,only.pos = T)
  } else if(type == "Subtype"){
    Idents(subset_obj) = "disease_association"
    disease_markers_subset = FindMarkers(subset_obj,ident.1="COVID-19 Positive",ident.2="Normal",logfc.threshold = logfc.threshold,only.pos = T)
  }
  
  if(!is.null(cut.padj)){
    disease_markers_subset = disease_markers_subset[disease_markers_subset$p_val_adj < cut.padj,]
  }
  return(disease_markers_subset)
}
test_obj$disease_association
Idents(test_obj) = "ct_cov"
test_obj$ct_cov
table(subset(test_obj, idents= "Ciliated Cells")$disease_cov)

disease_markers_cil = subsetMarkers(test_obj, idents = "Ciliated Cells", cut.padj = 0.01,type ="Disease")

disease_markers_scrt = subsetMarkers(test_obj, idents = "Secretory Cells",cut.padj = 0.01,type ="Disease")
disease_markers_devcil = subsetMarkers(test_obj, idents = "Developing Ciliated Cells",cut.padj = 0.01,type ="Disease")

dag_markers_scrt = subsetMarkers(test_obj, idents = "Secretory Cells",cut.padj = 0.01,type ="Subtype")

dag_markers_devcil = subsetMarkers(test_obj, idents = "Developing Ciliated Cells",cut.padj = 0.01,type ="Subtype")

dag_markers_cil = subsetMarkers(test_obj, idents = "Ciliated Cells",cut.padj = 0.01,type ="Subtype")
positive_markers_devcil = subsetMarkers(test_obj, idents = "Developing Ciliated Cells",cut.padj = 0.01)
positive_markers_scrt = subsetMarkers(test_obj, idents = "Secretory Cells",cut.padj = 0.01)

p_dev_cil_top35= DotPlot(subset(test_obj,idents = "Developing Ciliated Cells" ),features = rownames(positive_markers_devcil)[1:35],group.by = "disease_association") +RotatedAxis()+ylab("")+xlab("")
p_dev_cil_top35
ggsave(plot =p_dev_cil_top35 ,filename =  file.path(res_dir,"Dot_DEG35_Dev_Ciliated.pdf"),device = 'pdf',width=14, height=4,dpi = 450)
positive_markers_devcil
fwrite(positive_markers_devcil,file = file.path(res_dir,"Table_DEG_DevelopingCiliated.csv"),sep=",")
fwrite(positive_markers_scrt,file = file.path(res_dir,"Table_DEG_Secretory.csv"),sep=",")

intersect_cil = intersect(rownames(dag_markers_cil)[1:100], rownames(disease_markers_cil)[1:100])
setdiff_positive_cil = setdiff(rownames(dag_markers_cil)[1:100], rownames(disease_markers_cil)[1:100])
setdiff_disease_cil = setdiff(rownames(disease_markers_cil)[1:100],rownames(dag_markers_cil)[1:100])
writeLines(intersect_cil,con = file(file.path(res_dir,"DEG_Ciliated_both.txt")))
writeLines(setdiff_positive_cil,con = file(file.path(res_dir,"DEG_Ciliated_only_model.txt")))
writeLines(setdiff_disease_cil,con = file(file.path(res_dir,"DEG_Ciliated_only_pheno.txt")))

cil_sars= c(setdiff_positive_cil[grepl("SARSCoV2",setdiff_positive_cil)])
vln_cil_sars = VlnPlot(subset(test_obj,idents = "Ciliated Cells"),cil_sars,group.by = "disease_association",alpha=0,cols = c("#90CAF9", "#FFCC80", "#E53935")) 
class(vln_cil_sars)

ggsave(plot =vln_cil_sars ,filename =  file.path(res_dir,"Vln_SARSCoV2_Ciliated.pdf"),device = 'pdf',width=6, height=5,dpi = 450)


library(ggvenn)
p_venn_cil = ggvenn(
  list("Model-guided DEG"=rownames(dag_markers_cil)[1:100], "Phenotype-based DEG"=rownames(disease_markers_cil)[1:100]),
  fill_color = c("#0073C2FF", "#EFC000FF"),
  stroke_size = 0.5, set_name_size = 5, text_size=6
)

ggsave(plot =p_venn_cil ,filename =  file.path(res_dir,"Venn_DEG_Ciliated.pdf"),device = 'pdf',width=5, height=5,dpi = 450)


library(clusterProfiler)
library(enrichplot)
library(org.Hs.eg.db)
require(DOSE)
setdiff_positive_cil_dat = dag_markers_cil[rownames(dag_markers_cil) %in% setdiff_positive_cil,]
setdiff_positive_cil_sorted = - log(setdiff_positive_cil_dat$p_val_adj)
names(setdiff_positive_cil_sorted) = rownames(setdiff_positive_cil_dat)

setdiff_disease_cil_dat = dag_markers_cil[rownames(dag_markers_cil) %in% setdiff_disease_cil,]
setdiff_disease_cil_sorted = - log(setdiff_disease_cil_dat$p_val_adj)
names(setdiff_disease_cil_sorted) = rownames(setdiff_disease_cil_dat)

tmp_eg = bitr(names(setdiff_positive_cil_sorted), fromType="SYMBOL", toType="ENTREZID", OrgDb="org.Hs.eg.db")
tmp_eg_disease = bitr(setdiff_disease_cil, fromType="SYMBOL", toType="ENTREZID", OrgDb="org.Hs.eg.db")
tmp_eg_intersect = bitr(intersect_cil, fromType="SYMBOL", toType="ENTREZID", OrgDb="org.Hs.eg.db")
tmp_eg_scrt = bitr(rownames(positive_markers_scrt), fromType="SYMBOL", toType="ENTREZID", OrgDb="org.Hs.eg.db")

tmp_genelist = setdiff_positive_cil_sorted
names(tmp_genelist) = tmp_eg$ENTREZID 

# go_only_positive = enrichGO(tmp_eg$ENTREZID,OrgDb = 'org.Hs.eg.db')
# go_only_disease = enrichGO(tmp_eg_disease$ENTREZID,OrgDb = 'org.Hs.eg.db')
# go_intersect = enrichGO(tmp_eg_intersect$ENTREZID,OrgDb = 'org.Hs.eg.db')
kegg_intersect = enrichKEGG(tmp_eg_intersect$ENTREZID,qvalueCutoff = 0.5)
kegg_only_positive = enrichKEGG(tmp_eg$ENTREZID,qvalueCutoff = 0.5)
kegg_only_disease = enrichKEGG(tmp_eg_disease$ENTREZID,qvalueCutoff = 0.5)
kegg_scrt = enrichKEGG(tmp_eg_scrt$ENTREZID[1:200],qvalueCutoff = 0.5)
fwrite(kegg_scrt@result,file = file.path(res_dir,"ORA_DEG200_KEGG_SecretoryCells.csv"),sep=",")
dotplot(kegg_scrt)

p_kegg_both = dotplot(kegg_intersect,x = "GeneRatio")+ggtitle('Both')+theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
p_kegg_only_positive  = dotplot(kegg_only_positive,x = "GeneRatio")+ggtitle('Only model-guided')+theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
p_kegg_only_disease = dotplot(kegg_only_disease,x = "GeneRatio")+  ggtitle('Only phenotype-based')+theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),, axis.text.y= element_text(angle = 90, vjust = 0.5, hjust=0.5))


# (p_kegg_both + p_kegg_only_positive + p_kegg_only_disease)
p_kegg = wrap_plots(list(p_kegg_both, p_kegg_only_positive, p_kegg_only_disease), widths=c(4,4,1.5))
ggsave(plot =p_kegg,filename =  file.path(res_dir,"Dot3_DEG_Ciliated.pdf"),device = 'pdf',width=15, height=5,dpi = 450)


