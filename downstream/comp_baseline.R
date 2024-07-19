library(data.table)
library(ggpubr)
.libPaths("/home/local/kyeonghunjeong_920205/R_lib/")
rm(list=ls())
gc()
setwd("/home/local/kyeonghunjeong_920205/nipa_bu/COVID19/3.analysis/9.MIL/scAMIL_cell/scMILD/downstream/")
getwd()
### baseline 
# dir_lupus_baseline = "scAMIL_cell/NO_Opt_student_WENO_lupus_disease_model_vae_ed128_md64_lr0.0001_500_0.1_500_15_NO_Opt_student_leaktantan_fix_auc433_noflt_EarlyStopLogic_aebatch128_reduceLayer/"

### scMILD
dir_lupus_scmild = "Lupus/"
dir_ns_scmild = "NS/"
dir_pbmc_scmild = "PBMC/"
dir_uc_scmild = "UC/"
# color_lupus = c("#2ECC40", "#9370DB") # Healthy, Disease (SLE)
# color_ns = c("#0070C0","#FF4136") # Normal, Infection
# color_pbmc = c("#2ECC40","#FF4136") # Not-Hosp. , Hosp. 
# color_uc = c("#3D9970","#FF4136") # Healthy, Inflamed

color_lupus = c("#56B4E9", "#E69F00") # Healthy, Disease (SLE)
color_ns = c("#009E73", "#0072B2") # Normal, Infection
color_pbmc = c("#F0E442", "#D55E00") # Not-Hosp. , Hosp.
color_uc = c("#CC79A7", "#E69F00") # Healthy, Inflamed

# readABSCdat = function(dir_scmild,dir_baseline){
#   scmild_files = list.files(dir_scmild,full.names = T)
#   scmild_obs = scmild_files[grepl("/obs_[0-8].csv",scmild_files)]
#   obs_dat= rbindlist(lapply(scmild_obs, fread))
#   scmild_score = scmild_files[grepl("/cell_score_[0-9].csv",scmild_files)]
#   score_dat = rbindlist(lapply(scmild_score, function(x) {
#     y= fread(x)
#     y= y[,!grepl("feature",colnames(y)),with=F]
#     y$exp = gsub("cell_score_|.csv","",basename(x))
#     return(y)
#   }))
#   
#   baseline_files = list.files(dir_baseline,full.names = T)
#   baseline_score = baseline_files[grepl("/cell_score_[0-9].csv",baseline_files)]
#   score_dat_baseline = rbindlist(lapply(baseline_score, function(x) {
#     y= fread(x)
#     y= y[,!grepl("feature",colnames(y)),with=F]
#     return(y)
#     }
#                                      ))
#   colnames(score_dat_baseline) = paste0(colnames(score_dat_baseline),".baseline")
#   
#   dat = cbind(obs_dat,score_dat )
#   dat = cbind(dat,score_dat_baseline)
#   dat = dat[,!duplicated(colnames(dat)),with=F]
#   dat$bag_labels = as.factor(dat$bag_labels)
#   return(dat)
# }

readABSCdat = function(dir_scmild){
  scmild_files = list.files(dir_scmild,full.names = T)
  scmild_obs = scmild_files[grepl("/obs_[0-8].csv",scmild_files)]
  obs_dat= rbindlist(lapply(scmild_obs, fread))
  scmild_score = scmild_files[grepl("/cell_score_[0-9].csv",scmild_files)]
  score_dat = rbindlist(lapply(scmild_score, function(x) {
    y= fread(x)
    y= y[,!grepl("feature",colnames(y)),with=F]
    y$exp = gsub("cell_score_|.csv","",basename(x))
    return(y)
  }))
  
  weno_files = list.files(dir_scmild,full.names = T)
  weno_score = weno_files[grepl("/cell_score_[0-9]_not_op.csv",weno_files)]
  score_dat_weno = rbindlist(lapply(weno_score, function(x) {
    y= fread(x)
    y= y[,!grepl("feature",colnames(y)),with=F]
    return(y)
  }
  ))
  colnames(score_dat_weno) = paste0(colnames(score_dat_weno),".WENO")
  
  
  abmil_files = list.files(dir_scmild,full.names = T)
  abmil_score = abmil_files[grepl("/cell_score_[0-9]_baseline.csv",abmil_files)]
  score_dat_abmil = rbindlist(lapply(abmil_score, function(x) {
    y= fread(x)
    y= y[,!grepl("feature",colnames(y)),with=F]
    return(y)
  }
  ))
  colnames(score_dat_abmil) = paste0(colnames(score_dat_abmil),".ABMIL")
  
  dat = cbind(obs_dat,score_dat)
  dat = cbind(dat,score_dat_weno)
  dat = cbind(dat,score_dat_abmil)
  dat = dat[,!duplicated(colnames(dat)),with=F]
  dat$bag_labels = as.factor(dat$bag_labels)
  return(dat)
}

ns_dat = readABSCdat(dir_ns_scmild)
lupus_dat = readABSCdat(dir_lupus_scmild)
uc_dat = readABSCdat(dir_uc_scmild)
pbmc_dat = readABSCdat(dir_pbmc_scmild)


library(patchwork)
# ns_dat$bag_labels = as.factor(ns_dat$bag_labels)
# p_ns_density = ggdensity(ns_dat, x = "cell_score_minmax.baseline", fill = "bag_labels",palette=color_ns, ylim=c(0,5)) / ggdensity(ns_dat, x = "cell_score_minmax", fill = "bag_labels",palette=color_ns, ylim=c(0,5))

vlnPlot_twoScore = function(dat, title, score1, score2, color, labels = c(0,1),score3=NULL){
  dat$bag_labels = factor(dat$bag_labels, labels = labels)
  dat$bag_labels = relevel(dat$bag_labels, ref = labels[1])
  
  p1 = ggviolin(dat, x = "bag_labels", y = score1,fill="bag_labels",palette = color,add="median_iqr",ylim=c(0,1))+
    theme(legend.position = "none",axis.text.x = element_text(angle = 45, hjust = 1))+xlab("")+ylab("scMILD") +rremove('ticks')+rremove('y.text')
  p2 = ggviolin(dat, x = "bag_labels", y = score2,fill="bag_labels",palette = color,add="median_iqr",ylim=c(0,1),alpha=0.6)+
    theme(legend.position = "none",axis.text.x = element_text(angle = 45, hjust = 1))+xlab("")+ylab("WENO") +rremove('ticks') +rremove('y.text')
  
  if(!is.null(score3)) {
    p3 = ggviolin(dat, x = "bag_labels", y = score3,fill="bag_labels",palette = color,add="median_iqr",ylim=c(0,1),alpha=0.2)+
      theme(legend.position = "none",axis.text.x = element_text(angle = 45, hjust = 1))+xlab("")+ylab("ABMIL") +rremove('ticks')
    y = wrap_plots(p1+ggtitle(title),p2, p3, nrow=1) +rremove('ticks')+rremove('y.text')
  } else { 
    y = wrap_plots(p1+ggtitle(title),p2, nrow=1)
    
    }
  
  
  return(y)
}

res_dir = c("attn_score")
dir.create(res_dir)
vln_ns = vlnPlot_twoScore(ns_dat,"COVID-19 NS", "cell_score_minmax", "cell_score_minmax.WENO", color_ns, c("Normal","COVID-19"))
wrap_plots(vln_ns[[1]], vln_ns[[2]],guides="collect")
melt_ns_dat = melt(ns_dat[,c("bag_labels","cell_score_minmax","cell_score_minmax.WENO","cell_score_minmax.ABMIL")])
melt_ns_dat$model = tstrsplit(melt_ns_dat$variable, ".", fixed=T,fill = "scMILD")[[2]]
melt_ns_dat$bag_labels = relevel(melt_ns_dat$bag_labels, ref = "1")
melt_ns_dat$bag_labels = factor(melt_ns_dat$bag_labels, labels=c("Infection", "Normal"))
ggviolin(melt_ns_dat, x= "model", y="value",fill="bag_labels",combine=F,merge = "asis", palette=color_ns[2:1], legend.title="Condition")+xlab("")+ylab("Cell attention score") +rremove('y.ticks') +rremove('y.text')
?ggpar



vln_lupus = vlnPlot_twoScore(lupus_dat, "Lupus","cell_score_minmax", "cell_score_minmax.WENO", color_lupus, c("Healthy","SLE"))
vln_pbmc = vlnPlot_twoScore(pbmc_dat, "COVID-19 PBMC","cell_score_minmax", "cell_score_minmax.WENO", color_pbmc, c("Not-Hosp.","Hosp."))
vln_uc = vlnPlot_twoScore(uc_dat,"UC", "cell_score_minmax", "cell_score_minmax.WENO", color_uc, c("Healthy","Inflamed"))

vln_collect = ggarrange(plotlist= list(vln_lupus, vln_ns, vln_pbmc, vln_uc), ncol=2, nrow=2)
vln_collect_wide = ggarrange(plotlist= list(vln_lupus, vln_ns, vln_pbmc, vln_uc), ncol=4, nrow=1)

ggexport(vln_collect, filename = paste0(res_dir,"/vlnPlot_twoScore.pdf"), content = "page")
ggexport(vln_collect_wide, filename = paste0(res_dir,"/vlnPlot_twoScore_wide.pdf"), content = "page",width = 16,height=4)

vln_ns3 = vlnPlot_twoScore(ns_dat,"COVID-19 Infection", "cell_score_minmax", "cell_score_minmax.WENO", color_ns, c("Normal","COVID-19"),"cell_score_minmax.ABMIL")
vln_lupus3 = vlnPlot_twoScore(lupus_dat, "Lupus","cell_score_minmax", "cell_score_minmax.WENO", color_lupus, c("Healthy","SLE"),"cell_score_minmax.ABMIL")
vln_pbmc3 = vlnPlot_twoScore(pbmc_dat, "COVID-19 Hosp.","cell_score_minmax", "cell_score_minmax.WENO", color_pbmc, c("Not-Hosp.","Hosp."),"cell_score_minmax.ABMIL")
vln_uc3 = vlnPlot_twoScore(uc_dat,"UC", "cell_score_minmax", "cell_score_minmax.WENO", color_uc, c("Healthy","Inflamed"),"cell_score_minmax.ABMIL")


vln3_collect_wide = ggarrange(plotlist= list(vln_lupus3, vln_ns3, vln_pbmc3, vln_uc3), ncol=4, nrow=1)
vln3_collect = ggarrange(plotlist= list(vln_lupus3, vln_ns3, vln_pbmc3, vln_uc3), ncol=1, nrow=4)

ggexport(vln3_collect, filename = paste0(res_dir,"/vlnPlot_threeScore.pdf"), content = "page", width=8, height=12)
ggexport(vln3_collect_wide, filename = paste0(res_dir,"/vlnPlot_threeScore_wide.pdf"), content = "page",width = 20,height=4)


Cor_twoScore= function(dat, score1, score2){
  cor1 = cor.test(dat$cell_score_minmax, dat$cell_score_minmax.WENO, method = "spearman")
  cor2 = cor.test(dat$cell_score_minmax, dat$cell_score_minmax.WENO, method = "pearson")
  return(c(cor1$estimate, cor2$estimate))
}
Cor_twoScore(ns_dat)
Cor_twoScore(lupus_dat)
Cor_twoScore(pbmc_dat)
Cor_twoScore(uc_dat)
cor_stats = rbind(Lupus=Cor_twoScore(lupus_dat), NS=Cor_twoScore(ns_dat), PBMC=Cor_twoScore(pbmc_dat), UC=Cor_twoScore(uc_dat))
cor_stats
write.csv(cor_stats,file = file.path(res_dir,"Cor_table.csv"))

library(ROCR)
library(yardstick)
AUCPR_twoScore = function(dat){
  pred1 = prediction(dat$cell_score_minmax,dat$bag_labels)
  pred2 = prediction(dat$cell_score_minmax.WENO,dat$bag_labels)
  perf1 = performance(pred1, "aucpr")
  perf2 = performance(pred2, "aucpr")
  return(c(scMILD=perf1@y.values[[1]], baseline=perf2@y.values[[1]]))
}

aucpr_ns = AUCPR_twoScore(ns_dat)
aucpr_pbmc = AUCPR_twoScore(pbmc_dat)
aucpr_lupus = AUCPR_twoScore(lupus_dat)
aucpr_uc = AUCPR_twoScore(uc_dat)

aucpr_stats = rbind(Lupus=aucpr_lupus, NS=aucpr_ns, PBMC=aucpr_pbmc,UC=aucpr_uc)
write.csv(aucpr_stats,file=file.path(res_dir,"AUCPR_table.csv"))

PrecSpec_at = function(dat, at=c(0.5, 0.5)){
  dat$bag_labels = as.factor(dat$bag_labels)
  
  dat$positive.WENO = factor(dat$cell_score_minmax.WENO > at[1], labels = levels(dat$bag_labels),levels = c(FALSE,TRUE))
  dat$positive = factor(dat$cell_score_minmax > at[2], labels = levels(dat$bag_labels),levels = c(FALSE,TRUE))
  
  prec1 = precision(dat, truth= "bag_labels", estimate= "positive.WENO",event_level = "second")$.estimate
  prec2 = precision(dat, truth= "bag_labels", estimate= "positive",event_level = "second")$.estimate
  spec1 = specificity(dat, truth= "bag_labels", estimate= "positive.WENO",event_level = "second")$.estimate
  spec2 = specificity(dat, truth= "bag_labels", estimate= "positive",event_level = "second")$.estimate
  return(c(scMILD = c(Precision=prec2, Specificity=spec2), baseline = c(Precision=prec1, Specificity=spec1)))
}

precspec_ns_0.5 = PrecSpec_at(ns_dat)
precspec_uc_0.5 = PrecSpec_at(uc_dat)
precspec_lupus_0.5 = PrecSpec_at(lupus_dat)
precspec_pbmc_0.5 = PrecSpec_at(pbmc_dat)

precspec_at0.5 = rbind(Lupus=precspec_lupus_0.5, NS=precspec_ns_0.5, PBMC=precspec_pbmc_0.5, UC=precspec_uc_0.5)
write.csv(precspec_at0.5,file = file.path(res_dir,"PrecSpec_0.5_table.csv"))
precspec_at0.5 = fread(file = file.path(res_dir,"PrecSpec_0.5_table.csv"))
precspec_at0.5_melt  = melt(precspec_at0.5)
precspec_at0.5_melt = cbind(precspec_at0.5_melt, transpose(data.table(apply(precspec_at0.5_melt, 1,function(x) { strsplit(x[[2]],split = ".", fixed=T)[[1]]},simplify = T))))
colnames(precspec_at0.5_melt)[1] = 'Dataset'
colnames(precspec_at0.5_melt)[4] = 'Model'
colnames(precspec_at0.5_melt)[5] = 'Metric'
precspec_at0.5_melt$Model = factor(precspec_at0.5_melt$Model,levels = c("scMILD",'baseline'))
ggbarplot(precspec_at0.5_melt,x = 'Dataset',y='value',fill='Model',facet.by = 'Metric',position = position_dodge2(padding=0))+theme_minimal()

# PrecSpec_gmm = function(dat){
#   dat$bag_labels = as.factor(dat$bag_labels)
#   
#   score_gmm = mixtools::normalmixEM2comp(x = dat$cell_score_minmax,mu = c(0.1, 0.9),sigsqrd = c(0.1,0.1),lambda = 0.5,maxit = 100)
#   score_gmm.WENO = mixtools::normalmixEM2comp(x = dat$cell_score_minmax.WENO,mu = c(0.1, 0.9),sigsqrd = c(0.1,0.1),lambda = 0.5, maxit=100)
#   
#   dat$positive = factor(apply(score_gmm$posterior,1, which.max),labels = c(0,1))
#   
#   dat$positive.WENO = factor(apply(score_gmm.WENO$posterior,1, which.max),labels = c(0,1))
#   prec1 = precision(dat, truth= "bag_labels", estimate= "positive.WENO",event_level = "second")$.estimate
#   prec2 = precision(dat, truth= "bag_labels", estimate= "positive",event_level = "second")$.estimate
#   spec1 = specificity(dat, truth= "bag_labels", estimate= "positive.WENO",event_level = "second")$.estimate
#   spec2 = specificity(dat, truth= "bag_labels", estimate= "positive",event_level = "second")$.estimate
#   return(c(scMILD = c(Precision=prec2, Specificity=spec2), baseline = c(Precision=prec1, Specificity=spec1)))
# }
# 
# 
# precspec_ns_gmm = PrecSpec_gmm(ns_dat)
# precspec_uc_gmm = PrecSpec_gmm(uc_dat)
# precspec_lupus_gmm = PrecSpec_gmm(lupus_dat)
# precspec_pbmc_gmm = PrecSpec_gmm(pbmc_dat)
# 
# precspec_gmm = rbind(Lupus=precspec_lupus_gmm, NS=precspec_ns_gmm, PBMC=precspec_pbmc_gmm, UC=precspec_uc_gmm)
# 
# precspec_gmm_melt  = melt(precspec_gmm)
# precspec_gmm_melt = cbind(precspec_gmm_melt, 
#                           transpose(data.table(apply(precspec_gmm_melt, 1,function(x) { strsplit(x[[2]],split = ".", fixed=T)[[1]]},simplify = T))))
# colnames(precspec_gmm_melt)[1] = 'Dataset'
# colnames(precspec_gmm_melt)[4] = 'Model'
# colnames(precspec_gmm_melt)[5] = 'Metric'
# precspec_gmm_melt$Model = factor(precspec_gmm_melt$Model,levels = c("scMILD",'baseline'))
# ggbarplot(precspec_gmm_melt,x = 'Dataset',y='value',fill='Model',facet.by = 'Metric',position = position_dodge2(padding=0))+theme_minimal()
# 


precspec_ns_0.75 = PrecSpec_at(ns_dat, c(0.75,0.75))
precspec_uc_0.75 = PrecSpec_at(uc_dat, c(0.75,0.75))
precspec_lupus_0.75 = PrecSpec_at(lupus_dat, c(0.75,0.75))
precspec_pbmc_0.75 = PrecSpec_at(pbmc_dat, c(0.75,0.75))

precspec_at0.75 = rbind(Lupus=precspec_lupus_0.75, NS=precspec_ns_0.75, PBMC=precspec_pbmc_0.75, UC=precspec_uc_0.75)
write.csv(precspec_at0.75,file = file.path(res_dir,"PrecSpec_0.75_table.csv"))

precspec_at0.75 = fread(file = file.path(res_dir,"PrecSpec_0.75_table.csv"))
precspec_at0.75_melt  = melt(precspec_at0.75)
precspec_at0.75_melt = cbind(precspec_at0.75_melt, transpose(data.table(apply(precspec_at0.75_melt, 1,function(x) { strsplit(x[[2]],split = ".", fixed=T)[[1]]},simplify = T))))
colnames(precspec_at0.75_melt)[1] = 'Dataset'
colnames(precspec_at0.75_melt)[4] = 'Model'
colnames(precspec_at0.75_melt)[5] = 'Metric'
precspec_at0.75_melt$Model = factor(precspec_at0.75_melt$Model,levels = c("scMILD",'baseline'))
bar_prespec = ggbarplot(precspec_at0.75_melt,x = 'Dataset',y='value',fill='Model',facet.by = 'Metric',position = position_dodge2(padding=0),
          palette = c("black",'white'))+theme_pubclean()+ylab('')+xlab('')


ggexport(bar_prespec, filename = paste0(res_dir,"/BarPlot_PrecSpec_0.75.pdf"), content = "page", width=4, height=3)

precspec_ns_0.8 = PrecSpec_at(ns_dat, c(0.8,0.8))
precspec_uc_0.8 = PrecSpec_at(uc_dat, c(0.8,0.8))
precspec_lupus_0.8 = PrecSpec_at(lupus_dat, c(0.8,0.8))
precspec_pbmc_0.8 = PrecSpec_at(pbmc_dat, c(0.8,0.8))

precspec_at0.8= rbind(Lupus=precspec_lupus_0.8, NS=precspec_ns_0.8, PBMC=precspec_pbmc_0.8, UC=precspec_uc_0.8)
write.csv(precspec_at0.8,file = file.path(res_dir,"PrecSpec_0.8_table.csv"))




twoScoreCutoffPrecision = function(dat, score1, score2){
  dat$bag_labels = as.factor(dat$bag_labels)
  pred1 <- prediction(dat[[score1]], dat$bag_labels)
  pred2 <- prediction(dat[[score2]], dat$bag_labels)
  cutoffs = seq(from=0.1,to=0.95,by=0.05)
  tmp_dt = data.table()
  for(cutoff in cutoffs){
    dat$positive.WENO = factor(dat[[score2]] > cutoff,labels = levels(dat$bag_labels),levels = c(FALSE,T))
    tmp_precision.WENO = precision(dat, truth= "bag_labels", estimate= "positive.WENO", event_level="second")$.estimate
    tmp_spec.WENO = specificity(dat, truth= "bag_labels", estimate= "positive.WENO", event_level="second")$.estimate
    dat$positive = factor(dat[[score1]] > cutoff,labels = levels(dat$bag_labels),levels = c(FALSE,TRUE))
    tmp_precision = precision(dat, truth= "bag_labels", estimate= "positive", event_level="second")  $.estimate
    tmp_spec = specificity(dat, truth= "bag_labels", estimate= "positive", event_level="second")  $.estimate
    tmp_dt = rbind(tmp_dt,data.table(cutoff=cutoff,
                                     precision=tmp_precision, specificity=tmp_spec, 
                                     precision.WENO=tmp_precision.WENO, specificity.WENO = tmp_spec.WENO))
  }
  return(tmp_dt)
}

cuts_prc_lupus = twoScoreCutoffPrecision(lupus_dat, "cell_score_minmax", "cell_score_minmax.WENO")
cuts_prc_ns = twoScoreCutoffPrecision(ns_dat, "cell_score_minmax", "cell_score_minmax.WENO")
cuts_prc_pbmc = twoScoreCutoffPrecision(pbmc_dat, "cell_score_minmax", "cell_score_minmax.WENO")
cuts_prc_uc = twoScoreCutoffPrecision(uc_dat, "cell_score_minmax", "cell_score_minmax.WENO")

cuts_prc_lupus$dataset = "Lupus"
cuts_prc_ns$dataset ="NS"
cuts_prc_pbmc$dataset = "PBMC"
cuts_prc_uc$dataset = "UC"

cuts_prc = rbind(cuts_prc_lupus,cuts_prc_ns,cuts_prc_pbmc,cuts_prc_uc)
write.csv(cuts_prc,file = file.path(res_dir,"Precision_Specificity_Cutoff.csv"))

KS_twoScore = function(dat){
  ks1 = ks.test(dat[dat$bag_labels == "1" ,]$cell_score_minmax,
                dat[dat$bag_labels == "0",]$cell_score_minmax)
  
  ks2 = ks.test(dat[dat$bag_labels == "1",]$cell_score_minmax.WENO,
                dat[dat$bag_labels == "0",]$cell_score_minmax.WENO)
  
  
  ks1b = ks.test(dat[dat$bag_labels == "1" & 
                    dat$cell_score_minmax>0.5,]$cell_score_minmax,
                dat[dat$bag_labels == "0"& 
                    dat$cell_score_minmax>0.5,]$cell_score_minmax)
  
  ks2b = ks.test(dat[dat$bag_labels == "1" & 
                    dat$cell_score_minmax.WENO>0.5,]$cell_score_minmax.WENO,
                dat[dat$bag_labels == "0"& 
                    dat$cell_score_minmax.WENO>0.5,]$cell_score_minmax.WENO)
  ks1c = ks.test(dat[dat$bag_labels == "1" & 
                       dat$cell_score_minmax<0.5,]$cell_score_minmax,
                 dat[dat$bag_labels == "0"& 
                       dat$cell_score_minmax<0.5,]$cell_score_minmax)
  
  ks2c = ks.test(dat[dat$bag_labels == "1" & 
                       dat$cell_score_minmax.WENO<0.5,]$cell_score_minmax.WENO,
                 dat[dat$bag_labels == "0"& 
                       dat$cell_score_minmax.WENO<0.5,]$cell_score_minmax.WENO)
  
  return(c(scMILD.ks.all = ks1$statistic, scMILD.ks.all_p = ks1$p.value,
           
           baseline.ks.all = ks2$statistic, baseline.ks.all_p=ks2$p.value,
           
           scMILD.ks.gt.0.5 = ks1b$statistic, scMILD.ks.gt.0.5_p = ks1b$p.value,
           
           baseline.ks.gt.0.5= ks2b$statistic, baseline.ks.gt.0.5_p=ks2b$p.value,
           
           scMILD.ks.lt.0.5 = ks1c$statistic, scMILD.ks.lt.0.5_p = ks1c$p.value,
           
           baseline.ks.lt.0.5= ks2c$statistic, baseline.ks.lt.0.5_p=ks2c$p.value
           
           ))
}
ks_ns = KS_twoScore(ns_dat)
ks_uc = KS_twoScore(uc_dat)
ks_pbmc = KS_twoScore(pbmc_dat)
ks_lupus = KS_twoScore(lupus_dat)
ks_res = rbind(Lupus=ks_lupus, NS=ks_ns, PBMC=ks_pbmc, UC=ks_uc)
write.csv(ks_res, file.path(res_dir,"ks_results.csv"))


KS_twoScore2 = function(dat){
  ks1 = ks.test(dat[dat$bag_labels == "1" ,]$cell_score_minmax,
                dat[dat$bag_labels == "0",]$cell_score_minmax)
  
  ks2 = ks.test(dat[dat$bag_labels == "1",]$cell_score_minmax.WENO,
                dat[dat$bag_labels == "0",]$cell_score_minmax.WENO)
  
  
  ks1b = ks.test(dat[dat$bag_labels == "1" & 
                       dat$cell_score_minmax>0.75,]$cell_score_minmax,
                 dat[dat$bag_labels == "0"& 
                       dat$cell_score_minmax>0.75,]$cell_score_minmax)
  
  ks2b = ks.test(dat[dat$bag_labels == "1" & 
                       dat$cell_score_minmax.WENO>0.75,]$cell_score_minmax.WENO,
                 dat[dat$bag_labels == "0"& 
                       dat$cell_score_minmax.WENO>0.75,]$cell_score_minmax.WENO)
  ks1c = ks.test(dat[dat$bag_labels == "1" & 
                       dat$cell_score_minmax<0.25,]$cell_score_minmax,
                 dat[dat$bag_labels == "0"& 
                       dat$cell_score_minmax<0.25,]$cell_score_minmax)
  
  ks2c = ks.test(dat[dat$bag_labels == "1" & 
                       dat$cell_score_minmax.WENO<0.25,]$cell_score_minmax.WENO,
                 dat[dat$bag_labels == "0"& 
                       dat$cell_score_minmax.WENO<0.25,]$cell_score_minmax.WENO)
  
  return(c(scMILD.ks.all = ks1$statistic, scMILD.ks.all_p = ks1$p.value,
           
           baseline.ks.all = ks2$statistic, baseline.ks.all_p=ks2$p.value,
           
           scMILD.ks.gt.0.75 = ks1b$statistic, scMILD.ks.gt.0.75_p = ks1b$p.value,
           
           baseline.ks.gt.0.75= ks2b$statistic, baseline.ks.gt.0.75_p=ks2b$p.value,
           
           scMILD.ks.lt.0.25 = ks1c$statistic, scMILD.ks.lt.0.25_p = ks1c$p.value,
           
           baseline.ks.lt.0.25= ks2c$statistic, baseline.ks.lt.0.25_p=ks2c$p.value
           
  ))
}

quantile(ns_dat$cell_score_minmax)
quantile(ns_dat$cell_score_minmax.WENO)[4]

ks_ns2 = KS_twoScore2(ns_dat)
ks_uc2 = KS_twoScore2(uc_dat)
ks_pbmc2 = KS_twoScore2(pbmc_dat)
ks_lupus2 = KS_twoScore2(lupus_dat)
ks_res2 = rbind(Lupus=ks_lupus2, NS=ks_ns2, PBMC=ks_pbmc2, UC=ks_uc2)
write.csv(ks_res2, file.path(res_dir,"ks_results2.csv"))




library(bayestestR)
overlap_twoscore = function(dat){
  ov1 = overlap(dat[dat$bag_labels == "1" ,]$cell_score_minmax,
                dat[dat$bag_labels == "0",]$cell_score_minmax, extend = F, extend_scale = 0, precision = 2^16)
  
  ov2 = overlap(dat[dat$bag_labels == "1",]$cell_score_minmax.WENO,
                dat[dat$bag_labels == "0",]$cell_score_minmax.WENO, extend = F, extend_scale = 0, precision = 2^16)
  
  
  ov1b = overlap(dat[dat$bag_labels == "1" & 
                    dat$cell_score_minmax>0.5,]$cell_score_minmax,
                dat[dat$bag_labels == "0"& 
                    dat$cell_score_minmax>0.5,]$cell_score_minmax, extend = F, extend_scale = 0, precision = 2^16)
  
  ov2b = overlap(dat[dat$bag_labels == "1" & 
                    dat$cell_score_minmax.WENO>0.5,]$cell_score_minmax.WENO,
                dat[dat$bag_labels == "0"& 
                    dat$cell_score_minmax.WENO>0.5,]$cell_score_minmax.WENO, extend = F, extend_scale = 0, precision = 2^16)
  
  ov1c = overlap(dat[dat$bag_labels == "1" & 
                       dat$cell_score_minmax<0.5,]$cell_score_minmax,
                 dat[dat$bag_labels == "0"& 
                       dat$cell_score_minmax<0.5,]$cell_score_minmax, extend = F, extend_scale = 0, precision = 2^16)
  
  ov2c = overlap(dat[dat$bag_labels == "1" & 
                       dat$cell_score_minmax.WENO<0.5,]$cell_score_minmax.WENO,
                 dat[dat$bag_labels == "0"& 
                       dat$cell_score_minmax.WENO<0.5,]$cell_score_minmax.WENO, extend = F, extend_scale = 0, precision = 2^16)
  
  return(c(scMILD.ov.all = ov1,
           
           baseline.ov.all = ov2,
           
           scMILD.ov.gt.0.5 = ov1b,
           
           baseline.ov.gt.0.5= ov2b,
           
           scMILD.ov.lt.0.5 = ov1c,
           
           baseline.ov.lt.0.5= ov2c))
  
}
ov_lupus=overlap_twoscore(lupus_dat)
ov_ns = overlap_twoscore(ns_dat)
ov_pbmc = overlap_twoscore(pbmc_dat)
ov_pbmc
ov_uc = overlap_twoscore(uc_dat)
ov_res = rbind(Lupus=ov_lupus, NS=ov_ns, PBMC=ov_pbmc, UC=ov_uc)
ov_res

ov_lupus[['scMILD.ov.all']]
write.csv(ov_res,file.path(res_dir,"overlap.csv"))


overlap_twoscore2 = function(dat){
  ov1 = overlap(dat[dat$bag_labels == "1" ,]$cell_score_minmax,
                dat[dat$bag_labels == "0",]$cell_score_minmax, extend = F, extend_scale = 0, precision = 2^16)
  
  ov2 = overlap(dat[dat$bag_labels == "1",]$cell_score_minmax.WENO,
                dat[dat$bag_labels == "0",]$cell_score_minmax.WENO, extend = F, extend_scale = 0, precision = 2^16)
  
  
  ov1b = overlap(dat[dat$bag_labels == "1" & 
                       dat$cell_score_minmax>0.75,]$cell_score_minmax,
                 dat[dat$bag_labels == "0"& 
                       dat$cell_score_minmax>0.75,]$cell_score_minmax, extend = F, extend_scale = 0, precision = 2^16)
  
  ov2b = overlap(dat[dat$bag_labels == "1" & 
                       dat$cell_score_minmax.WENO>0.75,]$cell_score_minmax.WENO,
                 dat[dat$bag_labels == "0"& 
                       dat$cell_score_minmax.WENO>0.75,]$cell_score_minmax.WENO, extend = F, extend_scale = 0, precision = 2^16)
  
  ov1c = overlap(dat[dat$bag_labels == "1" & 
                       dat$cell_score_minmax<0.25,]$cell_score_minmax,
                 dat[dat$bag_labels == "0"& 
                       dat$cell_score_minmax<0.25,]$cell_score_minmax, extend = F, extend_scale = 0, precision = 2^16)
  
  ov2c = overlap(dat[dat$bag_labels == "1" & 
                       dat$cell_score_minmax.WENO<0.25,]$cell_score_minmax.WENO,
                 dat[dat$bag_labels == "0"& 
                       dat$cell_score_minmax.WENO<0.25,]$cell_score_minmax.WENO, extend = F, extend_scale = 0, precision = 2^16)
  
  return(c(scMILD.ov.all = ov1,
           
           baseline.ov.all = ov2,
           
           scMILD.ov.gt.0.75 = ov1b,
           
           baseline.ov.gt.0.75= ov2b,
           
           scMILD.ov.lt.0.25 = ov1c,
           
           baseline.ov.lt.0.25= ov2c))
  
}


ov_lupus2=overlap_twoscore2(lupus_dat)
ov_ns2= overlap_twoscore2(ns_dat)
ov_pbmc2 = overlap_twoscore2(pbmc_dat)
ov_uc2 = overlap_twoscore2(uc_dat)
ov_res2 = rbind(Lupus=ov_lupus2, NS=ov_ns2, PBMC=ov_pbmc2, UC=ov_uc2)
# ov_res2[,3] - ov_res2[,4]
# ov_res2
# ov_res2[,5] - ov_res2[,6]
write.csv(ov_res2,file.path(res_dir,"overlap2.csv"))

ks_res2[,5] - ks_res2[,7]
ks_res2_tidy = ks_res2[,c(1,3,5,7,9,11)]
ks_res2_tidy[,3]- ks_res2_tidy[,4]
ks_res2_tidy[,5]- ks_res2_tidy[,6]

ov_res2[,3]- ov_res2[,4]
ov_res2[,5]- ov_res2[,6]
library(RColorBrewer)
library(circlize)
library(ComplexHeatmap)
ov_col_fun = colorRamp2(c(0.8, 0.9, 1), rev(brewer.pal(3,"YlOrRd")))
ov_col_fun = colorRampPalette(rev(brewer.pal(9, "YlOrRd")))(100)
ov_col_fun = colorRampPalette((brewer.pal(9, "YlOrRd")))(100)
ks_col_fun = colorRamp2(c(0, 0.1, 0.2), brewer.pal(3,"BuGn"))
ks_col_fun = colorRampPalette((brewer.pal(9, "BuGn")))(100)
# overlap coefficient ( High - Bad/ Low - Good)
ht_ov = Heatmap(ov_res2,
                col = ov_col_fun,
                column_split = factor(c("All","All",">0.75",">0.75","<0.25","<0.25"), levels = c("All",">0.75","<0.25")),
        cluster_columns=F,
        cluster_rows=F,
        row_labels = c("Lupus", "COVID-19\nInfection", "COVID-19\nHosp.","UC"),
        column_labels=c("scMILD","baseline","scMILD","baseline","scMILD","baseline"),
        cluster_column_slices = F,
        width = unit(2.5,'inch'),height=unit(2,'inch'),
        name = "Overlap\nCoefficient", column_order = c(5,6,3,4,1,2),
        cell_fun = function(j, i, x, y, width, height, fill) {
          grid.text(sprintf("%.3f", ov_res2[i,j]), x, y, gp = gpar(fontsize = 10))
          }
)
# KS statistics ( High - Good / Low -Bad)

ht_ks  = Heatmap(ks_res2_tidy,
                 col =  ks_col_fun,
                 column_split = factor(c("All","All",">0.75",">0.75","<0.25","<0.25"), levels = c("All",">0.75","<0.25")),
        cluster_columns=F,
        cluster_rows=F,
        row_labels = c("Lupus", "COVID-19\nInfection", "COVID-19\nHosp.","UC"),
        column_labels=c("scMILD","baseline","scMILD","baseline","scMILD","baseline"),
        cluster_column_slices = F,
        name = "KS Statistics", column_order = c(3,4,5,6,1,2),
        width = unit(2.5,'inch'),height=unit(2,'inch'),
        cell_fun = function(j, i, x, y, width, height, fill) {
          grid.text(sprintf("%.3f", ks_res2_tidy[i,j]), x, y, gp = gpar(fontsize = 10))
        }
)
ht_ov_wide = Heatmap(ov_res2,
                col = ov_col_fun,
                column_split = factor(c("All","All",">0.75",">0.75","<0.25","<0.25"), levels = c("All",">0.75","<0.25")),
                cluster_columns=F,
                cluster_rows=F,
                row_labels = c("Lupus", "COVID-19\nInfection", "COVID-19\nHosp.","UC"),
                column_labels=c("scMILD","baseline","scMILD","baseline","scMILD","baseline"),
                cluster_column_slices = F,
                width = unit(4,'inch'),height=unit(1.5,'inch'),
                name = "Overlap\nCoefficient", column_order = c(5,6,3,4,1,2),
                cell_fun = function(j, i, x, y, width, height, fill) {
                  grid.text(sprintf("%.3f", ov_res2[i,j]), x, y, gp = gpar(fontsize = 10))
                }
)
# KS statistics ( High - Good / Low -Bad)

ht_ks_wide  = Heatmap(ks_res2_tidy,
                 col =  ks_col_fun,
                 column_split = factor(c("All","All",">0.75",">0.75","<0.25","<0.25"), levels = c("All",">0.75","<0.25")),
                 cluster_columns=F,
                 cluster_rows=F,
                 row_labels = c("Lupus", "COVID-19\nInfection", "COVID-19\nHosp.","UC"),
                 column_labels=c("scMILD","baseline","scMILD","baseline","scMILD","baseline"),
                 cluster_column_slices = F,
                 name = "KS Statistics", column_order = c(3,4,5,6,1,2),
                 width = unit(4,'inch'),height=unit(1.5,'inch'),
                 cell_fun = function(j, i, x, y, width, height, fill) {
                   grid.text(sprintf("%.3f", ks_res2_tidy[i,j]), x, y, gp = gpar(fontsize = 10))
                 }
)


# ht_stat = ht_ov %v% ht_ks
# 
# pdf(file.path(res_dir, "Heatmap_statistics.pdf"), width = 5, height = 4)
# draw(ht_stat)
# dev.off()
# 
# ht_stat_wide = ht_ov + ht_ks

pdf(file.path(res_dir, "Heatmap_ks_statistics.pdf"), width = 5, height = 4)
draw(ht_ks)
dev.off()

pdf(file.path(res_dir, "Heatmap_overlap_coeff.pdf"), width = 5, height = 4)
draw(ht_ov)
dev.off()


pdf(file.path(res_dir, "Heatmap_ks_statistics_wide.pdf"), width = 7, height = 4)
draw(ht_ks_wide)
dev.off()

pdf(file.path(res_dir, "Heatmap_overlap_coeff_wide.pdf"), width = 7, height = 4)
draw(ht_ov_wide)
dev.off()




js_res2 # JS
js_res2[,3] - js_res2[,4]

colnames(ks_res2)




gghistogram(ns_dat, x = "cell_score_minmax",fill="bag_labels",palette = color_ns,position="dodge",bins=100,ylim=c(0,5000))+
  theme(legend.position = "none",axis.text.x = element_text(angle = 45, hjust = 1))+xlab("scMILD")+ylab("Frequency")
gghistogram(ns_dat, x = "cell_score_minmax.WENO",fill="bag_labels",palette = color_ns,position="dodge",bins=100,ylim=c(0,5000))+
  theme(legend.position = "none",axis.text.x = element_text(angle = 45, hjust = 1))+xlab("baseline")+ylab("Frequency")


gghistogram(pbmc_dat, x = "cell_score_minmax",fill="bag_labels",palette = color_ns,position="dodge",bins=100)+
  theme(legend.position = "none",axis.text.x = element_text(angle = 45, hjust = 1))+xlab("scMILD")+ylab("Frequency")
gghistogram(pbmc_dat, x = "cell_score_minmax.WENO",fill="bag_labels",palette = color_ns,position="dodge",bins=100)+
  theme(legend.position = "none",axis.text.x = element_text(angle = 45, hjust = 1))+xlab("scMILD")+ylab("Frequency")

ggdensity(pbmc_dat, x = "cell_score_minmax",fill="bag_labels",palette = color_ns)+
  theme(legend.position = "none",axis.text.x = element_text(angle = 45, hjust = 1))+xlab("scMILD")+ylab("Density")
ggdensity(pbmc_dat, x = "cell_score_minmax.WENO",fill="bag_labels",palette = color_ns)+
  theme(legend.position = "none",axis.text.x = element_text(angle = 45, hjust = 1))+xlab("baseline")+ylab("Density")

gc()
rm(list = ls())
gc()



library(ggridges)


library(patchwork)


# library(LDLcalc)
# # JS divergence
# JS_twoScore = function(dat){
#   js1 = LDLcalc::JSD(dat[dat$bag_labels == "1" ,]$cell_score_minmax,
#                 dat[dat$bag_labels == "0",]$cell_score_minmax)
#   
#   js2 = LDLcalc::JSD(dat[dat$bag_labels == "1",]$cell_score_minmax.WENO,
#                 dat[dat$bag_labels == "0",]$cell_score_minmax.WENO)
#   
#   
#   js1b = LDLcalc::JSD(dat[dat$bag_labels == "1" & 
#                     dat$cell_score_minmax>0.5,]$cell_score_minmax,
#                 dat[dat$bag_labels == "0"& 
#                     dat$cell_score_minmax>0.5,]$cell_score_minmax)
#   
#   js2b = LDLcalc::JSD(dat[dat$bag_labels == "1" & 
#                     dat$cell_score_minmax.WENO>0.5,]$cell_score_minmax.WENO,
#                 dat[dat$bag_labels == "0"& 
#                     dat$cell_score_minmax.WENO>0.5,]$cell_score_minmax.WENO)
#   js1c = LDLcalc::JSD(dat[dat$bag_labels == "1" & 
#                        dat$cell_score_minmax<0.5,]$cell_score_minmax,
#                  dat[dat$bag_labels == "0"& 
#                        dat$cell_score_minmax<0.5,]$cell_score_minmax)
#   
#   js2c = LDLcalc::JSD(dat[dat$bag_labels == "1" & 
#                        dat$cell_score_minmax.WENO<0.5,]$cell_score_minmax.WENO,
#                  dat[dat$bag_labels == "0"& 
#                        dat$cell_score_minmax.WENO<0.5,]$cell_score_minmax.WENO)
#   
#   return(c(scMILD.js.all = js1$JSD,
#            
#            baseline.js.all = js2$JSD,
#            
#            scMILD.js.gt.0.5 = js1b$JSD,
#            
#            baseline.js.gt.0.5= js2b$JSD,
#            
#            scMILD.js.lt.0.5 = js1c$JSD,
#            
#            baseline.js.lt.0.5= js2c$JSD))
# }
# 
# js_lupus = JS_twoScore(lupus_dat)
# js_ns = JS_twoScore(ns_dat)
# js_pbmc = JS_twoScore(pbmc_dat)
# js_uc = JS_twoScore(uc_dat)
# js_res = rbind(Lupus=js_lupus, NS=js_ns, PBMC=js_pbmc, UC=js_uc)
# js_res
# 
# write.csv(js_res,file.path(res_dir,"JS_divergence.csv"))
# 
# 
# JS_twoScore2 = function(dat){
#   js1 = LDLcalc::JSD(dat[dat$bag_labels == "1" ,]$cell_score_minmax,
#                      dat[dat$bag_labels == "0",]$cell_score_minmax)
#   
#   js2 = LDLcalc::JSD(dat[dat$bag_labels == "1",]$cell_score_minmax.WENO,
#                      dat[dat$bag_labels == "0",]$cell_score_minmax.WENO)
#   
#   
#   js1b = LDLcalc::JSD(dat[dat$bag_labels == "1" & 
#                             dat$cell_score_minmax>0.75,]$cell_score_minmax,
#                       dat[dat$bag_labels == "0"& 
#                             dat$cell_score_minmax>0.75,]$cell_score_minmax)
#   
#   js2b = LDLcalc::JSD(dat[dat$bag_labels == "1" & 
#                             dat$cell_score_minmax.WENO>0.75,]$cell_score_minmax.WENO,
#                       dat[dat$bag_labels == "0"& 
#                             dat$cell_score_minmax.WENO>0.75,]$cell_score_minmax.WENO)
#   js1c = LDLcalc::JSD(dat[dat$bag_labels == "1" & 
#                             dat$cell_score_minmax<0.25,]$cell_score_minmax,
#                       dat[dat$bag_labels == "0"& 
#                             dat$cell_score_minmax<0.25,]$cell_score_minmax)
#   
#   js2c = LDLcalc::JSD(dat[dat$bag_labels == "1" & 
#                             dat$cell_score_minmax.WENO<0.25,]$cell_score_minmax.WENO,
#                       dat[dat$bag_labels == "0"& 
#                             dat$cell_score_minmax.WENO<0.25,]$cell_score_minmax.WENO)
#   
#   return(c(scMILD.js.all = js1$JSD,
#            
#            baseline.js.all = js2$JSD,
#            
#            scMILD.js.gt.0.75 = js1b$JSD,
#            
#            baseline.js.gt.0.75= js2b$JSD,
#            
#            scMILD.js.lt.0.25 = js1c$JSD,
#            
#            baseline.js.lt.0.25= js2c$JSD))
# }
# 
# js_lupus2 = JS_twoScore2(lupus_dat)
# js_ns2 = JS_twoScore2(ns_dat)
# js_pbmc2 = JS_twoScore2(pbmc_dat)
# js_uc2 = JS_twoScore2(uc_dat)
# js_res2 = rbind(Lupus=js_lupus2, NS=js_ns2, PBMC=js_pbmc2, UC=js_uc2)
# 
# 
# write.csv(js_res2,file.path(res_dir,"JS_divergence2.csv"))


## by AUC 
vln_lupus_mxdelta = vlnPlot_twoScore(lupus_dat[exp==3], "Lupus","cell_score_minmax", "cell_score_minmax.WENO", color_lupus, c("Healthy","SLE"))

vln_ns_mxdelta = vlnPlot_twoScore(ns_dat[exp==4],"COVID-19 NS", "cell_score_minmax", "cell_score_minmax.WENO", color_ns, c("Normal","COVID-19"))
vln_pbmc_mxdelta = vlnPlot_twoScore(pbmc_dat[exp==2], "COVID-19 PBMC","cell_score_minmax", "cell_score_minmax.WENO", color_pbmc, c("Not-Hosp.","Hosp."))

vln_uc_mxdelta = vlnPlot_twoScore(uc_dat[exp==4],"UC", "cell_score_minmax", "cell_score_minmax.WENO", color_uc, c("Healthy","Inflamed"))


vln_collect_mxdelta = ggarrange(plotlist= list(vln_lupus_mxdelta, vln_ns_mxdelta, vln_pbmc_mxdelta, vln_uc_mxdelta), ncol=2, nrow=2)

vln_collect_wide_mxdelta = ggarrange(plotlist= list(vln_lupus_mxdelta, vln_ns_mxdelta, vln_pbmc_mxdelta, vln_uc_mxdelta), ncol=4, nrow=1)
ggexport(vln_collect_mxdelta, filename = paste0(res_dir,"/vlnPlot_twoScore_maxDeltaAUC.pdf"), content = "page")
ggexport(vln_collect_wide_mxdelta, filename = paste0(res_dir,"/vlnPlot_twoScore_wide_maxDeltaAUC.pdf"), content = "page",width = 16,height=4)



cortables = fread(file.path(res_dir, "Cor_table.csv"),sep=",")
library(ComplexHeatmap)
library(circlize)
cor_mat = as.matrix(cortables[,-1])
rownames(cor_mat) = cortables$V1
colnames(cor_mat) = c("spearman", "pearson")
cor_mat2= cor_mat[,-2]
cor_mat

ht_cor = Heatmap(cor_mat2, name = "Correlation", col = colorRamp2(c(0, 1), c("white", "red")), show_row_names = T, show_column_names = F, cluster_rows = FALSE, cluster_columns = FALSE,
        cell_fun = function(j, i, x, y, width, height, fill) {
          grid.text(sprintf("%.3f", cor_mat2[i]), x, y, gp = gpar(fontsize = 10))
        })
pdf(file.path(res_dir, "Heatmap_pearosn.pdf"), width = 2, height = 2)
draw(ht_cor)
dev.off()

rm(list = ls())
gc()
