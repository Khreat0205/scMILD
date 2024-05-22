library(data.table)
library(ggpubr)
rm(list=ls())
gc()
getwd()
setwd("../../")
# setwd("/home/local/kyeonghunjeong_920205/nipa_bu/COVID19/3.analysis/9.MIL/scAMIL_cell/scMILD/downstream/")
getwd()
### scMILD
dir_lupus_scmild = "Lupus/"
dir_ns_scmild = dir_ns_baseline = "NS/"
dir_pbmc_scmild = "PBMC/"
dir_uc_scmild = "UC/"
color_lupus = c("#2ECC40", "#9370DB") # Healthy, SLE
color_ns = c("#0070C0","#FF4136") # Normal, COVID-19
color_pbmc = c("#2ECC40","#FF4136") # Not-Hosp. , Hosp. 
color_uc = c("#3D9970","#FF4136") # Healthy, Inflamed

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
  
  baseline_files = list.files(dir_scmild,full.names = T)
  baseline_score = baseline_files[grepl("/cell_score_[0-9]_baseline.csv",baseline_files)]
  score_dat_baseline = rbindlist(lapply(baseline_score, function(x) {
    y= fread(x)
    y= y[,!grepl("feature",colnames(y)),with=F]
    return(y)
  }
  ))
  colnames(score_dat_baseline) = paste0(colnames(score_dat_baseline),".baseline")
  
  dat = cbind(obs_dat,score_dat )
  dat = cbind(dat,score_dat_baseline)
  dat = dat[,!duplicated(colnames(dat)),with=F]
  dat$bag_labels = as.factor(dat$bag_labels)
  return(dat)
}

ns_dat = readABSCdat(dir_ns_scmild)
lupus_dat = readABSCdat(dir_lupus_scmild)
uc_dat = readABSCdat(dir_uc_scmild)
pbmc_dat = readABSCdat(dir_pbmc_scmild)


library(patchwork)
vlnPlot_twoScore = function(dat, title, score1, score2, color, labels = c(0,1)){
  dat$bag_labels = factor(dat$bag_labels, labels = labels)
  dat$bag_labels = relevel(dat$bag_labels, ref = labels[1])
  p1 = ggviolin(dat, x = "bag_labels", y = score1,fill="bag_labels",palette = color,add="median_iqr",ylim=c(0,1))+
    theme(legend.position = "none",axis.text.x = element_text(angle = 45, hjust = 1))+xlab("")+ylab("scMILD")
  p2 = ggviolin(dat, x = "bag_labels", y = score2,fill="bag_labels",palette = color,add="median_iqr",ylim=c(0,1),alpha=0.6)+
    theme(legend.position = "none",axis.text.x = element_text(angle = 45, hjust = 1))+xlab("")+ylab("Baseline")
  
  return(wrap_plots(p1+ggtitle(title),p2, nrow=1))
}
res_dir = c("attn_score")
dir.create(res_dir)
vln_ns = vlnPlot_twoScore(ns_dat,"COVID-19 NS", "cell_score_minmax", "cell_score_minmax.baseline", color_ns, c("Normal","COVID-19"))
vln_lupus = vlnPlot_twoScore(lupus_dat, "Lupus","cell_score_minmax", "cell_score_minmax.baseline", color_lupus, c("Healthy","SLE"))
vln_pbmc = vlnPlot_twoScore(pbmc_dat, "COVID-19 PBMC","cell_score_minmax", "cell_score_minmax.baseline", color_pbmc, c("Not-Hosp.","Hosp."))
vln_uc = vlnPlot_twoScore(uc_dat,"UC", "cell_score_minmax", "cell_score_minmax.baseline", color_uc, c("Healthy","Inflamed"))

vln_collect = ggarrange(plotlist= list(vln_lupus, vln_ns, vln_pbmc, vln_uc), ncol=2, nrow=2)
vln_collect_wide = ggarrange(plotlist= list(vln_lupus, vln_ns, vln_pbmc, vln_uc), ncol=4, nrow=1)

ggexport(vln_collect, filename = paste0(res_dir,"/vlnPlot_twoScore.pdf"), content = "page")
ggexport(vln_collect_wide, filename = paste0(res_dir,"/vlnPlot_twoScore_wide.pdf"), content = "page",width = 16,height=4)
vln_collect_wide


Cor_twoScore= function(dat, score1, score2){
  cor1 = cor.test(dat$cell_score_minmax, dat$cell_score_minmax.baseline, method = "spearman")
  cor2 = cor.test(dat$cell_score_minmax, dat$cell_score_minmax.baseline, method = "pearson")
  return(c(cor1$estimate, cor2$estimate))
}
Cor_twoScore(ns_dat)
Cor_twoScore(lupus_dat)
Cor_twoScore(pbmc_dat)
Cor_twoScore(uc_dat)
cor_stats = rbind(Lupus=Cor_twoScore(lupus_dat), NS=Cor_twoScore(ns_dat), PBMC=Cor_twoScore(pbmc_dat), UC=Cor_twoScore(uc_dat))
write.csv(cor_stats,file = file.path(res_dir,"Cor_table.csv"))

library(ROCR)
library(yardstick)
AUCPR_twoScore = function(dat){
  pred1 = prediction(dat$cell_score_minmax,dat$bag_labels)
  pred2 = prediction(dat$cell_score_minmax.baseline,dat$bag_labels)
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
  dat$positive.baseline = factor(dat$cell_score_minmax.baseline > at[1], labels = levels(dat$bag_labels),levels = c(FALSE,TRUE))
  dat$positive = factor(dat$cell_score_minmax > at[2], labels = levels(dat$bag_labels),levels = c(FALSE,TRUE))
  prec1 = precision(dat, truth= "bag_labels", estimate= "positive.baseline",event_level = "second")$.estimate
  prec2 = precision(dat, truth= "bag_labels", estimate= "positive",event_level = "second")$.estimate
  spec1 = specificity(dat, truth= "bag_labels", estimate= "positive.baseline",event_level = "second")$.estimate
  spec2 = specificity(dat, truth= "bag_labels", estimate= "positive",event_level = "second")$.estimate
  return(c(scMILD = c(Precision=prec2, Specificity=spec2), baseline = c(Precision=prec1, Specificity=spec1)))
}

precspec_ns_0.5 = PrecSpec_at(ns_dat)
precspec_uc_0.5 = PrecSpec_at(uc_dat)
precspec_lupus_0.5 = PrecSpec_at(lupus_dat)
precspec_pbmc_0.5 = PrecSpec_at(pbmc_dat)

precspec_at0.5 = rbind(Lupus=precspec_lupus_0.5, NS=precspec_ns_0.5, PBMC=precspec_pbmc_0.5, UC=precspec_uc_0.5)
write.csv(precspec_at0.5,file = file.path(res_dir,"PrecSpec_0.5_table.csv"))



twoScoreCutoffPrecision = function(dat, score1, score2){
  dat$bag_labels = as.factor(dat$bag_labels)
  pred1 <- prediction(dat[[score1]], dat$bag_labels)
  pred2 <- prediction(dat[[score2]], dat$bag_labels)
  cutoffs = seq(from=0.1,to=0.95,by=0.05)
  tmp_dt = data.table()
  for(cutoff in cutoffs){
    dat$positive.baseline = factor(dat[[score2]] > cutoff,labels = levels(dat$bag_labels),levels = c(FALSE,T))
    tmp_precision.baseline = precision(dat, truth= "bag_labels", estimate= "positive.baseline", event_level="second")$.estimate
    tmp_spec.baseline = specificity(dat, truth= "bag_labels", estimate= "positive.baseline", event_level="second")$.estimate
    dat$positive = factor(dat[[score1]] > cutoff,labels = levels(dat$bag_labels),levels = c(FALSE,TRUE))
    tmp_precision = precision(dat, truth= "bag_labels", estimate= "positive", event_level="second")  $.estimate
    tmp_spec = specificity(dat, truth= "bag_labels", estimate= "positive", event_level="second")  $.estimate
    tmp_dt = rbind(tmp_dt,data.table(cutoff=cutoff,
                                     precision=tmp_precision, specificity=tmp_spec, 
                                     precision.baseline=tmp_precision.baseline, specificity.baseline = tmp_spec.baseline))
  }
  return(tmp_dt)
}

cuts_prc_lupus = twoScoreCutoffPrecision(lupus_dat, "cell_score_minmax", "cell_score_minmax.baseline")
cuts_prc_ns = twoScoreCutoffPrecision(ns_dat, "cell_score_minmax", "cell_score_minmax.baseline")
cuts_prc_pbmc = twoScoreCutoffPrecision(pbmc_dat, "cell_score_minmax", "cell_score_minmax.baseline")
cuts_prc_uc = twoScoreCutoffPrecision(uc_dat, "cell_score_minmax", "cell_score_minmax.baseline")

cuts_prc_lupus$dataset = "Lupus"
cuts_prc_ns$dataset ="NS"
cuts_prc_pbmc$dataset = "PBMC"
cuts_prc_uc$dataset = "UC"

cuts_prc = rbind(cuts_prc_lupus,cuts_prc_ns,cuts_prc_pbmc,cuts_prc_uc)
write.csv(cuts_prc,file = file.path(res_dir,"Precision_Specificity_Cutoff.csv"))

KS_twoScore = function(dat){
  ks1 = ks.test(dat[dat$bag_labels == "1" ,]$cell_score_minmax,
                dat[dat$bag_labels == "0",]$cell_score_minmax)
  
  ks2 = ks.test(dat[dat$bag_labels == "1",]$cell_score_minmax.baseline,
                dat[dat$bag_labels == "0",]$cell_score_minmax.baseline)
  
  
  ks1b = ks.test(dat[dat$bag_labels == "1" & 
                    dat$cell_score_minmax>0.5,]$cell_score_minmax,
                dat[dat$bag_labels == "0"& 
                    dat$cell_score_minmax>0.5,]$cell_score_minmax)
  
  ks2b = ks.test(dat[dat$bag_labels == "1" & 
                    dat$cell_score_minmax.baseline>0.5,]$cell_score_minmax.baseline,
                dat[dat$bag_labels == "0"& 
                    dat$cell_score_minmax.baseline>0.5,]$cell_score_minmax.baseline)
  ks1c = ks.test(dat[dat$bag_labels == "1" & 
                       dat$cell_score_minmax<0.5,]$cell_score_minmax,
                 dat[dat$bag_labels == "0"& 
                       dat$cell_score_minmax<0.5,]$cell_score_minmax)
  
  ks2c = ks.test(dat[dat$bag_labels == "1" & 
                       dat$cell_score_minmax.baseline<0.5,]$cell_score_minmax.baseline,
                 dat[dat$bag_labels == "0"& 
                       dat$cell_score_minmax.baseline<0.5,]$cell_score_minmax.baseline)
  
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
  
  ks2 = ks.test(dat[dat$bag_labels == "1",]$cell_score_minmax.baseline,
                dat[dat$bag_labels == "0",]$cell_score_minmax.baseline)
  
  
  ks1b = ks.test(dat[dat$bag_labels == "1" & 
                       dat$cell_score_minmax>0.75,]$cell_score_minmax,
                 dat[dat$bag_labels == "0"& 
                       dat$cell_score_minmax>0.75,]$cell_score_minmax)
  
  ks2b = ks.test(dat[dat$bag_labels == "1" & 
                       dat$cell_score_minmax.baseline>0.75,]$cell_score_minmax.baseline,
                 dat[dat$bag_labels == "0"& 
                       dat$cell_score_minmax.baseline>0.75,]$cell_score_minmax.baseline)
  ks1c = ks.test(dat[dat$bag_labels == "1" & 
                       dat$cell_score_minmax<0.25,]$cell_score_minmax,
                 dat[dat$bag_labels == "0"& 
                       dat$cell_score_minmax<0.25,]$cell_score_minmax)
  
  ks2c = ks.test(dat[dat$bag_labels == "1" & 
                       dat$cell_score_minmax.baseline<0.25,]$cell_score_minmax.baseline,
                 dat[dat$bag_labels == "0"& 
                       dat$cell_score_minmax.baseline<0.25,]$cell_score_minmax.baseline)
  
  return(c(scMILD.ks.all = ks1$statistic, scMILD.ks.all_p = ks1$p.value,
           
           baseline.ks.all = ks2$statistic, baseline.ks.all_p=ks2$p.value,
           
           scMILD.ks.gt.0.75 = ks1b$statistic, scMILD.ks.gt.0.75_p = ks1b$p.value,
           
           baseline.ks.gt.0.75= ks2b$statistic, baseline.ks.gt.0.75_p=ks2b$p.value,
           
           scMILD.ks.lt.0.25 = ks1c$statistic, scMILD.ks.lt.0.25_p = ks1c$p.value,
           
           baseline.ks.lt.0.25= ks2c$statistic, baseline.ks.lt.0.25_p=ks2c$p.value
           
  ))
}

quantile(ns_dat$cell_score_minmax)
quantile(ns_dat$cell_score_minmax.baseline)[4]

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
  
  ov2 = overlap(dat[dat$bag_labels == "1",]$cell_score_minmax.baseline,
                dat[dat$bag_labels == "0",]$cell_score_minmax.baseline, extend = F, extend_scale = 0, precision = 2^16)
  
  
  ov1b = overlap(dat[dat$bag_labels == "1" & 
                    dat$cell_score_minmax>0.5,]$cell_score_minmax,
                dat[dat$bag_labels == "0"& 
                    dat$cell_score_minmax>0.5,]$cell_score_minmax, extend = F, extend_scale = 0, precision = 2^16)
  
  ov2b = overlap(dat[dat$bag_labels == "1" & 
                    dat$cell_score_minmax.baseline>0.5,]$cell_score_minmax.baseline,
                dat[dat$bag_labels == "0"& 
                    dat$cell_score_minmax.baseline>0.5,]$cell_score_minmax.baseline, extend = F, extend_scale = 0, precision = 2^16)
  
  ov1c = overlap(dat[dat$bag_labels == "1" & 
                       dat$cell_score_minmax<0.5,]$cell_score_minmax,
                 dat[dat$bag_labels == "0"& 
                       dat$cell_score_minmax<0.5,]$cell_score_minmax, extend = F, extend_scale = 0, precision = 2^16)
  
  ov2c = overlap(dat[dat$bag_labels == "1" & 
                       dat$cell_score_minmax.baseline<0.5,]$cell_score_minmax.baseline,
                 dat[dat$bag_labels == "0"& 
                       dat$cell_score_minmax.baseline<0.5,]$cell_score_minmax.baseline, extend = F, extend_scale = 0, precision = 2^16)
  
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
  
  ov2 = overlap(dat[dat$bag_labels == "1",]$cell_score_minmax.baseline,
                dat[dat$bag_labels == "0",]$cell_score_minmax.baseline, extend = F, extend_scale = 0, precision = 2^16)
  
  
  ov1b = overlap(dat[dat$bag_labels == "1" & 
                       dat$cell_score_minmax>0.75,]$cell_score_minmax,
                 dat[dat$bag_labels == "0"& 
                       dat$cell_score_minmax>0.75,]$cell_score_minmax, extend = F, extend_scale = 0, precision = 2^16)
  
  ov2b = overlap(dat[dat$bag_labels == "1" & 
                       dat$cell_score_minmax.baseline>0.75,]$cell_score_minmax.baseline,
                 dat[dat$bag_labels == "0"& 
                       dat$cell_score_minmax.baseline>0.75,]$cell_score_minmax.baseline, extend = F, extend_scale = 0, precision = 2^16)
  
  ov1c = overlap(dat[dat$bag_labels == "1" & 
                       dat$cell_score_minmax<0.25,]$cell_score_minmax,
                 dat[dat$bag_labels == "0"& 
                       dat$cell_score_minmax<0.25,]$cell_score_minmax, extend = F, extend_scale = 0, precision = 2^16)
  
  ov2c = overlap(dat[dat$bag_labels == "1" & 
                       dat$cell_score_minmax.baseline<0.25,]$cell_score_minmax.baseline,
                 dat[dat$bag_labels == "0"& 
                       dat$cell_score_minmax.baseline<0.25,]$cell_score_minmax.baseline, extend = F, extend_scale = 0, precision = 2^16)
  
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
ov_res2[,3] - ov_res2[,4]
ov_res2
ov_res2[,5] - ov_res2[,6]
write.csv(ov_res2,file.path(res_dir,"overlap2.csv"))

ks_res2[,5] - ks_res2[,7]
ks_res2_tidy = ks_res2[,c(1,3,5,7,9,11)]
ks_res2_tidy[,3]- ks_res2_tidy[,4]
ks_res2_tidy[,5]- ks_res2_tidy[,6]

ov_res2[,3]- ov_res2[,4]
ov_res2[,5]- ov_res2[,6]
library(RColorBrewer)
library(circlize)
ov_col_fun = colorRamp2(c(0.8, 0.9, 1), rev(brewer.pal(3,"YlOrRd")))
ov_col_fun = colorRampPalette(rev(brewer.pal(9, "YlOrRd")))(100)
ks_col_fun = colorRamp2(c(0, 0.1, 0.2), brewer.pal(3,"BuGn"))
ks_col_fun = colorRampPalette((brewer.pal(9, "BuGn")))(100)
# overlap coefficient ( High - Bad/ Low - Good)
ht_ov = Heatmap(ov_res2,
                col = ov_col_fun,
                column_split = factor(c("All","All",">0.75",">0.75","<0.25","<0.25"), levels = c("All",">0.75","<0.25")),
        cluster_columns=F,
        cluster_rows=F,
        row_labels = c("Lupus", "COVID-19\nNS", "COVID-19\nPBMC","UC"),
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
        row_labels = c("Lupus", "COVID-19\nNS", "COVID-19\nPBMC","UC"),
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
                row_labels = c("Lupus", "COVID-19\nNS", "COVID-19\nPBMC","UC"),
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
                 row_labels = c("Lupus", "COVID-19\nNS", "COVID-19\nPBMC","UC"),
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
gghistogram(ns_dat, x = "cell_score_minmax.baseline",fill="bag_labels",palette = color_ns,position="dodge",bins=100,ylim=c(0,5000))+
  theme(legend.position = "none",axis.text.x = element_text(angle = 45, hjust = 1))+xlab("baseline")+ylab("Frequency")


gghistogram(pbmc_dat, x = "cell_score_minmax",fill="bag_labels",palette = color_ns,position="dodge",bins=100)+
  theme(legend.position = "none",axis.text.x = element_text(angle = 45, hjust = 1))+xlab("scMILD")+ylab("Frequency")
gghistogram(pbmc_dat, x = "cell_score_minmax.baseline",fill="bag_labels",palette = color_ns,position="dodge",bins=100)+
  theme(legend.position = "none",axis.text.x = element_text(angle = 45, hjust = 1))+xlab("scMILD")+ylab("Frequency")

ggdensity(pbmc_dat, x = "cell_score_minmax",fill="bag_labels",palette = color_ns)+
  theme(legend.position = "none",axis.text.x = element_text(angle = 45, hjust = 1))+xlab("scMILD")+ylab("Density")
ggdensity(pbmc_dat, x = "cell_score_minmax.baseline",fill="bag_labels",palette = color_ns)+
  theme(legend.position = "none",axis.text.x = element_text(angle = 45, hjust = 1))+xlab("baseline")+ylab("Density")

gc()
rm(list = ls())
gc()



library(ggridges)
library(patchwork)

## by AUC 
vln_lupus_mxdelta = vlnPlot_twoScore(lupus_dat[exp==7], "Lupus","cell_score_minmax", "cell_score_minmax.baseline", color_lupus, c("Healthy","SLE"))

vln_ns_mxdelta = vlnPlot_twoScore(ns_dat[exp==3],"COVID-19 NS", "cell_score_minmax", "cell_score_minmax.baseline", color_ns, c("Normal","COVID-19"))
vln_pbmc_mxdelta = vlnPlot_twoScore(pbmc_dat[exp==4], "COVID-19 PBMC","cell_score_minmax", "cell_score_minmax.baseline", color_pbmc, c("Not-Hosp.","Hosp."))

vln_uc_mxdelta = vlnPlot_twoScore(uc_dat[exp==8],"UC", "cell_score_minmax", "cell_score_minmax.baseline", color_uc, c("Healthy","Inflamed"))


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
