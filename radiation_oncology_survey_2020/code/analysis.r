#Analyze survey data results with nnet_survival model predictions
#survey prognosis groups: 1=0-3 months, 2=3.1-6, 3=6.1-9, 4=9.1-12, 5=12.1-18, 6=18.1-24, 7=24.1+

rm(list=ls())
graphics.off()
options(digits=10)
library(Hmisc)
library(survival)
library(rms)
library(ggplot2)
library(dplyr)
library(survminer)
library(lubridate)
library(readxl)
library(tidyr)
library(survivalROC)
library(boot)
library(stringr)
library(arrow)

source('mfg_utils.r')

data_dir <- ''
fig_dir <- ''

sites_frame <- data.frame(site_name=c('Brain','Bone','Abdomen/pelvis','Thorax','Soft tissue/node'))
sites_frame$site_nums <- list(c(1),c(2,3),c(4,8,9,10,11,13,14,18,19),c(7,12),c(5,6,15,16,17))

data1=read.csv(paste(data_dir,'survey_outcomes.csv',sep=''),stringsAsFactors=FALSE)
data1 <- subset(data1, select = -c(lfu_date, death_date)) #remove outdated follow-up info
data1$tx_date <- mdy(data1$tx_date)
data1$birth_date <- mdy(data1$birth_date)
data1$age <- as.numeric(data1$tx_date-data1$birth_date)/365
data1$patient_id <- as.integer(data1$patient_id)
names(data1)[names(data1) == 'patient_id'] <- 'patient_id_int'
data1$primsite_f <- factor(data1$primsite_coded, levels=seq(12), labels=c('CNS', 'Head and neck', 'Thorax', 'Breast', 'Gastrointestinal', 'Genitourinary', 'Gynecologic', 'Hematologic', 'Bone', 'Skin', 'Endocrine', 'Other/unknown'))
data1$primsite_f[data1$primsite_coded %in% c(1, 11)] <- 'Other/unknown'

data2=read_excel(paste(data_dir,'survey_outcomes_updated.xlsx',sep='')) #load updated follow-up info
data3 <- data2[c('Patient ID','Alive','Date of death','Last Follow-up')]
names(data3) <- c('patient_id_int','alive','death_date','lfu_date')
data3$patient_id_int <- as.integer(data3$patient_id_int)
data3$death_date <- ymd(data3$death_date)
data3$lfu_date <- ymd(data3$lfu_date)
data3 <- unique(data3)
temp <- data3 %>% group_by(patient_id_int) %>%  filter(n() > 1)
data3 <- data3 %>% group_by(patient_id_int) %>% filter(row_number()==1)

attending_grad_year=read.csv(paste(data_dir,'attending_grad_year.csv',sep=''),stringsAsFactors=FALSE)

#data quality checks
nrow(unique(data1[c('patient_id_int','tx_date')]))

#join tables to add updated follow-up info
data <- inner_join(data1,data3,by="patient_id_int")
data <- inner_join(data,attending_grad_year,by="dr")
data$yr_after_grad <- 2016-data$yr_res_grad

#more data quality checks
length(unique(data$patient_id_int))
summary(data$alive & !is.na(data$death_date))

#load computer model predictions
model_predsurv <- read_parquet(paste(data_dir,'model_predsurv.parquet',sep=''))
model_results <- read_parquet(paste(data_dir,'model_results.parquet',sep=''))
attr(model_results$eval_date,'tzone') <- 'UTC'
model_results$eval_date <- as.Date(model_results$eval_date)
model_results$patient_id_int = as.integer(model_results$patient_id)
names(model_results)[names(model_results) == 'eval_date'] <- 'tx_date'
model_results_for_join <- model_results[,c('patient_id_int','tx_date','median_pred_surv','improve_surv_feats','worsen_surv_feats')]
data_w_model_results <- inner_join(data,model_results_for_join,by=c("patient_id_int",'tx_date'))
data_w_model_results$time_death <- as.numeric(data_w_model_results$death_date-data_w_model_results$tx_date)
data_w_model_results$time_fu <- as.numeric(data_w_model_results$lfu_date-data_w_model_results$tx_date)
data_w_model_results$time_death_fu <- apply(data_w_model_results[,c('time_death','time_fu')],1,max,na.rm=TRUE)
data_w_model_results$dead <- !is.na(data_w_model_results$death_date)
data_w_model_results <- data_w_model_results[data_w_model_results$time_death_fu>0 | data_w_model_results$dead==TRUE,] #1 removed w/ 9 days f/u
data_analyze <- data_w_model_results[!is.na(data_w_model_results$prognosis_fac),]

cuts <- 365*c(-1, 1/4,1/2,3/4,1,1.5,2,999999)
data_analyze$prognosis_ml <- cut(data_analyze$median_pred_surv, cuts, include = TRUE,labels=FALSE)
data_analyze$pred_surv_bin <- as.factor(data_analyze$prognosis_ml)
data_analyze$ecog <- 5-cut(data_analyze$kps,c(0, 35, 55, 75, 95,105),include=TRUE,labels=FALSE) #https://www.ncbi.nlm.nih.gov/pubmed/20674334
data_analyze$ecog_factor <- as.factor(cut(as.numeric(data_analyze$ecog), c(-1, 0.5, 1.5, 2.5, 999), include = TRUE,labels=c('0','1','2','3-4')))
data_analyze$ecog_neg <- -data_analyze$ecog

#####
#data summary
length(unique(data_analyze$patient_id_int))
courses_per_pt <- data_analyze %>% group_by(patient_id_int) %>% tally(sort=T)
summary(courses_per_pt$n)
summary(data_analyze$time_death_fu)
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
#5.0000  115.0000  331.0000  422.0842  709.5000 1190.0000
summary(data_analyze$time_death_fu[data_analyze$dead==F])
summary(data_analyze$dead)
#   Mode   FALSE    TRUE 
#logical     270     609 
mysurv <- npsurv(Surv(data_analyze$time_death_fu, data_analyze$dead) ~ 1)
mysurv
#median 355 (321-388)
print(summary(mysurv,times=365*c(0.5,1,2)))
# time n.risk n.event survival   std.err lower 95% CI upper 95% CI
# 182.5    569     291 0.664686 0.0160397     0.633981     0.696879
# 365.0    410     150 0.487670 0.0170851     0.455308     0.522333
# 730.0    207     135 0.320381 0.0162506     0.290063     0.353869


tableone(tables=c('data','data_analyze'), vars=c('age','male','primsite_f','tx_type==0','total_dose','fx'),categorical=c(0,1,1,1,0,0))

for (i in seq(nrow(sites_frame))) {
  print(as.character(sites_frame$site_name[i]))
  print(summary(data_analyze$tx_site1 %in% sites_frame$site_nums[i][[1]] | data_analyze$tx_site2 %in% sites_frame$site_nums[i][[1]] | data_analyze$tx_site3 %in% sites_frame$site_nums[i][[1]]))
}

#####
#combine faculty and model
data_analyze$combined <- cut( (data_analyze$prognosis_fac+data_analyze$prognosis_ml)/2 , seq(8)-0.5, include = TRUE,labels=FALSE) #rounding down

#####
#AUC taking physician level clustering into account
library(fastAUC)
fu_time <- 365
data_temp <- data_analyze[data_analyze$time_death_fu>=fu_time | data_analyze$dead,]
data_temp$died_later <- data_temp$time_death_fu>fu_time

#physicians vs computer
auc_result <- auc(test_1=data_temp$prognosis_fac,
                  test_2=data_temp$prognosis_ml,
                  status=data_temp$died_later,
                  cluster=data_temp$dr)
auc_result$p_value
#0.1149344392
se<-auc_result$var[1,1]^0.5
low <- auc_result$auc[1]-1.96*se; hi <- auc_result$auc[1]+1.96*se #assume normal dist due to large n
print(cbind(auc_result$auc[1], low, hi)) #physician AUC
#0.7199186992 0.6309126467 0.8089247517

se<-auc_result$var[2,2]^0.5
low <- auc_result$auc[2]-1.96*se; hi <- auc_result$auc[2]+1.96*se #assume normal dist due to large n
print(cbind(auc_result$auc[2], low, hi)) #computer AUC
#0.7704994193 0.728049291 0.8129495475

#physicians vs combined (physician and computer averaged)
auc_result <- auc(test_1=data_temp$prognosis_fac,
                  test_2=data_temp$combined,
                  status=data_temp$died_later,
                  cluster=data_temp$dr)
auc_result$p_value
#0.0006246882811
se<-auc_result$var[2,2]^0.5
low <- auc_result$auc[2]-1.96*se; hi <- auc_result$auc[2]+1.96*se #assume normal dist due to large n
print(cbind(auc_result$auc[2], low, hi)) #combined AUC
#0.7891045849 0.733713725 0.8444954448

#computer vs combined (physician and computer averaged)
auc_result <- auc(test_1=data_temp$prognosis_ml,
                  test_2=data_temp$combined,
                  status=data_temp$died_later,
                  cluster=data_temp$dr)
auc_result$p_value
#0.3391964479

#physicians vs Jang ECOG
auc_result <- fastAUC::auc(test_1=data_temp$prognosis_fac,
                  test_2=-data_temp$ecog,
                  status=data_temp$died_later,
                  cluster=data_temp$dr)
auc_result$p_value

#computer vs Jang ECOG
auc_result <- fastAUC::auc(test_1=data_temp$prognosis_ml,
                  test_2=data_temp$ecog_neg,
                  status=data_temp$died_later,
                  cluster=data_temp$dr)
auc_result$p_value

#all comparisons
preds <- list(data_temp$prognosis_fac, data_temp$prognosis_ml, data_temp$combined, data_temp$ecog_neg)
preds_names <- c('fac','ml','combined','ecog')
for (i in seq(length(preds))) {
  for (j in seq(i)) {
    if (i!=j) {
      auc_result <- auc(test_1=preds[[i]],
                        test_2=preds[[j]],
                        status=data_temp$died_later,
                        cluster=data_temp$dr)
      cat(preds_names[i], preds_names[j])
      print(auc_result$p_value)
    }
  }
}


####
#ROC plots
library(pROC)
svg(filename = paste(fig_dir,'roc_12mo.svg',sep=''), width = 4.5, height = 4.5)
roc_fac <- plot.roc(data_temp$died_later, data_temp$prognosis_fac,
                    main="12 month survival",
                    percent=TRUE,
                    col=cbPalette[1],xaxs="i",yaxs="i")
roc_ml <- lines.roc(data_temp$died_later, data_temp$prognosis_ml, 
                    percent=TRUE, 
                    col=cbPalette[2])
roc_combined <- lines.roc(data_temp$died_later, data_temp$combined, 
                          percent=TRUE, 
                          col=cbPalette[3])

legend("bottomright", legend=c("Physicians", "Computer","Combined"),
       col=cbPalette[1:3], lwd=2,cex=1)
dev.off()


#####
#C-index faculty vs model (CIs too narrow since doesn't account for clustering within physicians)
w<-rcorr.cens(as.numeric(data_analyze$prognosis_fac),Surv(data_analyze$time_death_fu, data_analyze$dead))
w<-rcorr.cens(as.numeric(data_analyze$prognosis_ml),Surv(data_analyze$time_death_fu, data_analyze$dead))
w<-rcorr.cens(as.numeric(data_analyze$combined),Surv(data_analyze$time_death_fu, data_analyze$dead))
w<-rcorr.cens(as.numeric(-data_analyze$ecog),Surv(data_analyze$time_death_fu, data_analyze$dead))

C <- w['C Index']
se <- w['S.D.']/2
low <- C-1.96*se; hi <- C+1.96*se
print(cbind(C, low, hi))

#calibration plots Kaplan-Meier
cuts <- c(0.5,2.5,4.5,6.5,7.5) #divides into 0-6, 6.1-12, 12.1-24, 24.1+
#cuts <- c(0.5,1.5,2.5,4.5,6.5,7.5) #divides into 0-3, 3.1-6, 6.1-12, 12.1-24, 24.1+
data_analyze$temp_bin <- as.factor(cut(as.numeric(data_analyze$prognosis_fac), cuts, include = TRUE,labels=FALSE))
summary(data_analyze$temp_bin)
#result:  1   2   3   4 
#        255 314 225  85
mysurv <- npsurv(Surv(data_analyze$time_death_fu, data_analyze$dead) ~ data_analyze$temp_bin)
svg(filename = paste(fig_dir,'dr_model_calib_km.svg',sep=''), width = 5, height = 4.5)
par(yaxt="n")
survplot(fit=mysurv,conf="none",xlim=c(0,365*2),time.inc=365/4,lty=1,label.curves = FALSE,col=cbPalette,lwd=2,
         n.risk=F,
         adj.n.risk=.5,cex.n.risk=.8,
         xlab="Follow-up, months",
         ylab="Survival")
par(yaxt="s") ; axis(2, at=seq(0, 1, 0.25))
data_analyze$temp_bin <- as.factor(cut(as.numeric(data_analyze$pred_surv_bin), cuts, include = TRUE,labels=FALSE))
summary(data_analyze$temp_bin)
#result:  1   2   3   4 
#        249 211 203 216 
mysurv <- npsurv(Surv(data_analyze$time_death_fu, data_analyze$dead) ~ data_analyze$temp_bin)
lines(mysurv,lty=3,lwd=2,col=cbPalette)
#legend(400,0.8,c("Physicians: 0-6", "Computer: 0-6", "Physicians: 6.1-12","Computer: 6.1-12","Physicians: 12.1-24","Computer: 12.1-24","Physiciians: >24","Computer: >24"),lty=c(1,3,1,3,1,3,1,3),lwd=2,col=cbPalette[c(1,1,2,2,3,3,4,4)],cex=1,title='Predicted survival (months)')
legend(200,0.8,c("Physicians: 0-6", "Physicians: 6.1-12","Physicians: 12.1-24","Physicians: >24","Computer:  0-6","Computer:  6.1-12", "Computer:  12.1-24","Computer:  >24"),lty=c(1,1,1,1,3,3,3,3),lwd=2,col=cbPalette[c(1,2,3,4,1,2,3,4)],cex=1,title='Predicted survival (months)')
title("Blue computer; black physican. 0-6, 6.1-12, 12.1-24, 24.1+")
dev.off()

#calib plots pred vs acutal median surv w/ CIs
n <- 8
median_results <- data.frame(pred_method=c(rep('Physician',n/2),rep('Computer',n/2)),pred_range=rep(c('0-6 mo.','6.1-12 mo.','12.1-24 mo.','>24 mo.'),2),
                             median=rep(NA,n),median_lower=rep(NA,n),median_upper=rep(NA,n))
median_results$pred_method <- factor(median_results$pred_method, levels=c('Physician','Computer'))
median_results$method_and_range <- paste(median_results$pred_method,median_results$pred_range)
median_results$method_and_range <- factor(median_results$method_and_range, levels = median_results$method_and_range[c(1,5,2,6,3,7,4,8)])
cuts <- c(0.5,2.5,4.5,6.5,7.5) #divides into 0-6, 6.1-12, 12.1-24, 24.1+
data_analyze$temp_bin <- as.factor(cut(as.numeric(data_analyze$prognosis_fac), cuts, include = TRUE,labels=FALSE))
mysurv <- survfit(Surv(data_analyze$time_death_fu, data_analyze$dead) ~ data_analyze$temp_bin)
median_results[1:4,'median'] <- summary(mysurv)$table[,'median']
median_results[1:4,'median_lower'] <- summary(mysurv)$table[,'0.95LCL']
median_results[1:4,'median_upper'] <- summary(mysurv)$table[,'0.95UCL']
median_results[1:4,'n'] <- summary(mysurv)$table[,'n.start']
data_analyze$temp_bin <- as.factor(cut(as.numeric(data_analyze$pred_surv_bin), cuts, include = TRUE,labels=FALSE))
mysurv <- survfit(Surv(data_analyze$time_death_fu, data_analyze$dead) ~ data_analyze$temp_bin)
median_results[5:8,'median'] <- summary(mysurv)$table[,'median']
median_results[5:8,'median_lower'] <- summary(mysurv)$table[,'0.95LCL']
median_results[5:8,'median_upper'] <- summary(mysurv)$table[,'0.95UCL']
median_results[5:8,'n'] <- summary(mysurv)$table[,'n.start']
median_results[is.na(median_results$median_upper),'median_upper'] <- median_results[is.na(median_results$median_upper),'median']
myplot <- ggplot(data=median_results,aes(x=method_and_range,y=median,color=pred_method,fill=pred_method))+ #,label=paste("n=",count,sep="")))+
  #geom_point(aes(size=count))+#geom_text(y=1)+
  geom_errorbar(aes(x=method_and_range,ymin=median_lower,ymax=median_upper),width=0.3)+
  geom_point(shape=21,size=2)+scale_color_manual(values=cbPalette[c(1,3)])+scale_fill_manual(values=cbPalette[c(1,3)])+
  scale_y_continuous(limits=c(0,365*4),breaks=seq(0,1460,365/2), expand = c(0, 0))+
  labs(y="Actual median survival",x="Method")+
  theme_bw(base_size = 12)+theme(panel.grid.major.x=element_blank(),panel.grid.minor.x=element_blank())+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  theme(legend.position = "none")
svg(filename = paste(fig_dir,'dr_model_calib_median.svg',sep=''), width = 5, height = 4.5)
myplot
dev.off()
median_results_months <- median_results
median_results_months$median <- median_results_months$median / 30.4
median_results_months$median_lower <- median_results_months$median_lower / 30.4
median_results_months$median_upper <- median_results_months$median_upper / 30.4

#performance of specific cutoffs
time_array <- c(365/2, 365)
for(mytime in time_array) {
  print(summary(data_analyze$median_pred_surv>mytime))
  mysurv <- npsurv(Surv(data_analyze$time_death_fu, data_analyze$dead) ~ data_analyze$median_pred_surv>mytime)
  print(summary(mysurv,times=mytime))
}

#predictions for patients who lived longer/shorter than cutoff
fu_time <- 365
data_temp <- data_analyze[data_analyze$time_death_fu>=fu_time | data_analyze$dead,]
data_temp$died_later <- factor(data_temp$time_death_fu>fu_time,labels=c('Actual survival 0-12 months (n=441)','Actual survival >12 months (n=410)'))
data_temp$prognosis_fac_f <- factor(data_temp$prognosis_fac, levels=seq(7), labels=c('0-3', '3.1-6','6.1-9','9.1-12', '12.1-18', '18.1-24', '>24'))
data_temp$prognosis_ml_f <- factor(data_temp$prognosis_ml, levels=seq(7), labels=c('0-3', '3.1-6','6.1-9','9.1-12', '12.1-18', '18.1-24', '>24'))
data_temp_long <- data_temp %>% pivot_longer(c('prognosis_fac_f','prognosis_ml_f'), names_to='Method', values_to='predsurv')
data_temp_long$Method <- factor(data_temp_long$Method, levels=c('prognosis_fac_f','prognosis_ml_f'), labels=c('Physician','Computer'))

svg(filename = paste(fig_dir,'actual_surv_12mo.svg',sep=''), width = 8, height = 3.7)
ggplot(data_temp_long, aes(x=predsurv, fill=Method)) + geom_bar(width=0.65,stat="count", position=position_dodge()) +
  facet_grid(cols = vars(died_later)) + scale_fill_manual(values=cbPalette[c(1,3)]) +
  labs(y="No. patients",x="Predicted survival (months)")+theme_bw()+
  theme(panel.grid.major.x = element_blank(), axis.text.x = element_text(angle = 45, hjust = 1))+
  scale_y_continuous(expand = expand_scale(add = c(0,10)))+
  theme(panel.spacing = unit(1, "lines"))
dev.off()

ggplot(data_temp_long, aes(x=predsurv)) + geom_bar() +
  facet_grid(cols = vars(died_later), rows=vars(method))

ggplot(data_temp_long, aes(x=predsurv)) + geom_bar()

ggplot(data_temp_long, aes(x=predsurv,fill=method,color=method,alpha=method)) + geom_bar() + scale_alpha_manual(values=c(0.3,0.3))

sums <- data_temp_long %>% group_by(predsurv, method) %>% tally()
ggplot(sums, aes(x=predsurv, y=n, color=method)) + geom_point()

#####
#Patients who lived longer/shorter: fractionation etc.
data_temp2 <- data_temp[data_temp$tx_site1 %in% c(2,3),]
data_temp2$fx_f <- cut(data_temp2$fx, c(-999, 1.5, 5.5, 10.5, 999,9999), include = TRUE,labels=c('1','2-5','6-10','>10','SBRT'))
data_temp2$fx_f[data_temp2$tx_type>0] <- 'SBRT'
data_temp2$fac_longer <- factor(data_temp2$prognosis_fac>4, labels=c('Physician predicts 0-12 mo.','Physician predicts >12  mo.'))
data_temp2$ml_longer <- factor(data_temp2$prognosis_ml>4, labels=c('Computer predicts 0-12 mo.','Computer predicts >12 mo.'))
ggplot(data_temp2, aes(x=fx_f)) + geom_bar(width=0.65) +
  facet_grid(cols = vars(fac_longer), rows=vars(ml_longer))+
  labs(y="No. patients",x="Radiation fractions")+theme_bw()+
  theme(panel.spacing = unit(1, "lines"),panel.grid.major.x = element_blank())+
  scale_y_continuous(expand = expand_scale(add = c(0,5)))+ggtitle('Fractions used for bone metastases')


#####
#per physician
data_analyze$dr_f <- as.factor(data_analyze$dr)
dr_freq <- data_analyze %>% group_by(dr_f) %>% summarise(count=n())
dr_freq <- dr_freq[order(dr_freq$count,decreasing=TRUE),]
dr_freq$id <- seq(nrow(dr_freq))
n_indiv <- 12 #12 = 20 or more.
dr_freq$id[-(1:n_indiv)] <- n_indiv+1 #group doctors with <20 patients
dr_dict <- dr_freq$id
names(dr_dict) <- dr_freq$dr_f
for (i in seq(nrow(data_analyze))) {
  data_analyze$dr_id[i] <- dr_dict[data_analyze$dr[i]]
}

#dr_results <- data.frame(cindex_fac=rep(NA,n_indiv+1),cindex_fac_lower=rep(NA,n_indiv+1),cindex_fac_upper=rep(NA,n_indiv+1),cindex_machine=rep(NA,n_indiv+1),cindex_machine_lower=rep(NA,n_indiv+1),cindex_machine_upper=rep(NA,n_indiv+1),yr_after_grad=rep(NA,n_indiv+1))
dr_results <- data.frame(cindex_fac=rep(NA,n_indiv),cindex_fac_lower=rep(NA,n_indiv),cindex_fac_upper=rep(NA,n_indiv),cindex_machine=rep(NA,n_indiv),cindex_machine_lower=rep(NA,n_indiv),cindex_machine_upper=rep(NA,n_indiv),yr_after_grad=rep(NA,n_indiv))
#for (dr_id in seq(n_indiv+1)) {
for (dr_id in seq(n_indiv)) {
  data_dr <- data_analyze[data_analyze$dr_id==dr_id,]
  dr_results$yr_after_grad[dr_id]<-data_dr$yr_after_grad[1]
  w <- rcorr.cens(as.numeric(data_dr$prognosis_ml),Surv(data_dr$time_death_fu, data_dr$dead))
  C <- w['C Index']
  se <- w['S.D.']/2
  dr_results$cindex_machine[dr_id]<-C
  dr_results$cindex_machine_lower[dr_id]<-C-1.96*se
  dr_results$cindex_machine_upper[dr_id]<-C+1.96*se
  w <- rcorr.cens(as.numeric(data_dr$prognosis_fac),Surv(data_dr$time_death_fu, data_dr$dead))
  C <- w['C Index']
  se <- w['S.D.']/2
  dr_results$cindex_fac[dr_id]<-C
  dr_results$cindex_fac_lower[dr_id]<-C-1.96*se
  dr_results$cindex_fac_upper[dr_id]<-C+1.96*se
}
#dr_results$dr_id <- seq(n_indiv+1)
dr_results$dr_id <- seq(n_indiv)
temp <- dr_freq %>% group_by(id) %>% summarise(count = sum(count))
dr_results$count <- temp$count[1:n_indiv]
dr_results$machine_minus_dr <- dr_results$cindex_machine-dr_results$cindex_fac

weighted.mean(dr_results$machine_minus_dr, dr_results$count)
#0.01425783794
summary(dr_results$machine_minus_dr)
#Min.     1st Qu.      Median        Mean     3rd Qu.        Max. 
#-0.11301370 -0.05924226 -0.03374087 -0.01522706  0.03913194  0.10763217 
summary(dr_results$cindex_machine>=dr_results$cindex_fac)
#Mode   FALSE    TRUE 
#logical       7       5
plot(dr_results$count, dr_results$machine_minus_dr)
#cor(dr_results$count, dr_results$machine_minus_dr)
cor.test(dr_results$count, dr_results$machine_minus_dr)
# Pearson's product-moment correlation
# 
# data:  dr_results$count and dr_results$machine_minus_dr
# t = 2.9791034, df = 10, p-value = 0.01382887
# alternative hypothesis: true correlation is not equal to 0
# 95 percent confidence interval:
# 0.1843620824 0.9039003649
# sample estimates:
# cor 
# 0.6857115146 

plot(dr_results$yr_after_grad, dr_results$machine_minus_dr)
cor.test(dr_results$yr_after_grad, dr_results$machine_minus_dr)
# Pearson's product-moment correlation
# 
# data:  dr_results$yr_after_grad and dr_results$machine_minus_dr
# t = 0.67074228, df = 10, p-value = 0.517576
# alternative hypothesis: true correlation is not equal to 0
# 95 percent confidence interval:
# -0.4159403949  0.6982460148
# sample estimates:
# cor 
# 0.2074912144 

#Doctor vs ML by # of pts
myplot=ggplot(dr_results,aes(cindex_fac,cindex_machine))+ #area of point proportional to # pts 
  scale_x_continuous(limits=c(0.5,1),expand = c(0,0))+scale_y_continuous(limits=c(0.5,1),expand = c(0,0))+
  geom_point(aes(size=count),alpha=0.7,fill=cbPalette[6],colour='black',shape=21)+geom_abline(linetype=2, color='gray')+theme_bw()+
  xlab("Physician c-index")+ylab("Computer model c-index")
svg(filename = paste(fig_dir,'per_dr_cindex_pt_count.svg',sep=''), width = 5, height = 4)
myplot
dev.off()

#Doctor vs ML by years experience since residency graduation
myplot=ggplot(dr_results,aes(yr_after_grad,machine_minus_dr))+
  scale_x_continuous()+scale_y_continuous()+
  geom_point(fill=cbPalette[6],colour='black',shape=21)+geom_hline(yintercept=0,linetype=2,color='gray')+theme_bw()+
  xlab("Years since residency graduation")+ylab("Computer model c-index minus physician c-index")
svg(filename = paste(fig_dir,'per_dr_cindex_years_experience.svg',sep=''), width = 5, height = 4)
myplot
dev.off()


#####
#results by primary site
primsites <- as.character(unique(data_analyze$primsite_f))
primsite_results <- data.frame(primsite_c=primsites, cindex_fac=rep(NA,length(primsites)),cindex_fac_lower=rep(NA,length(primsites)),cindex_fac_upper=rep(NA,length(primsites)),cindex_machine=rep(NA,length(primsites)),cindex_machine_lower=rep(NA,length(primsites)),cindex_machine_upper=rep(NA,length(primsites)))
for (i in seq(length(primsites))) {
  data_primsite <- data_analyze[data_analyze$primsite_f==primsites[i],]
  w <- rcorr.cens(as.numeric(data_primsite$prognosis_ml),Surv(data_primsite$time_death_fu, data_primsite$dead))
  C <- w['C Index']
  se <- w['S.D.']/2
  primsite_results$cindex_machine[i]<-C
  primsite_results$cindex_machine_lower[i]<-C-1.96*se
  primsite_results$cindex_machine_upper[i]<-C+1.96*se
  w <- rcorr.cens(as.numeric(data_primsite$prognosis_fac),Surv(data_primsite$time_death_fu, data_primsite$dead))
  C <- w['C Index']
  se <- w['S.D.']/2
  primsite_results$cindex_fac[i]<-C
  primsite_results$cindex_fac_lower[i]<-C-1.96*se
  primsite_results$cindex_fac_upper[i]<-C+1.96*se
}
temp <- data_analyze %>% group_by(primsite_f) %>% tally()
temp$primsite_c <- as.character(temp$primsite_f)
primsite_results <- inner_join(primsite_results,temp[,c('primsite_c','n')],by="primsite_c")
primsite_results$machine_minus_dr <- primsite_results$cindex_machine-primsite_results$cindex_fac
#Doctor vs ML by primsite, showing # of pts
myplot=ggplot(primsite_results,aes(cindex_fac,cindex_machine))+ #area of point proportional to # pts 
  scale_x_continuous(limits=c(0.5,1),expand = c(0,0))+scale_y_continuous(limits=c(0.5,1),expand = c(0,0))+
  geom_point(aes(color=primsite_c, size=n),alpha=0.7)+geom_abline(linetype=2, color='gray')+theme_bw()+
  xlab("Physician c-index")+ylab("Computer model c-index")+
  scale_color_manual(values=cbPalette, name='Primary site')+ #scale_color_brewer(palette='Set1')
  scale_size(name='Patients', breaks=c(50,100,150,200))
svg(filename = paste(fig_dir,'primsite_cindex_pt_count.svg',sep=''), width = 6, height = 4)
myplot
dev.off()

#####
#ML predictions within doctor prediction groups
myplots <- list()
p_values <- c()
bins <- c(1,2)
data_subset <- data_analyze[data_analyze$prognosis_fac %in% bins,]
data_subset$temp_ml_bin <- as.factor(cut(data_subset$prognosis_ml, c(-999, 2.5, 999), include = TRUE,labels=c('Same','Longer')))
print(c(bins,summary(data_subset$temp_ml_bin)))
mysurv <- npsurv(Surv(time_death_fu*12/365, dead) ~ temp_ml_bin, data=data_subset)
logrank_result <- survdiff(Surv(time_death_fu*12/365, dead) ~ temp_ml_bin, data=data_subset)
p_values[1] <- 1 - pchisq(logrank_result$chisq, length(logrank_result$n) - 1)
myplots[[1]] <- ggsurvplot(mysurv, risk.table = FALSE,censor=TRUE,break.x.by=6,break.y.by=0.25,
                           xlim=c(0,36),palette=cbPalette[c(2,3)],axes.offset=FALSE,
                           xlab='Follow-up (months)',ylab='Overall survival',title='Physician predicts 0-6 month survival')
bins <- c(3,4)
data_subset <- data_analyze[data_analyze$prognosis_fac %in% bins,]
data_subset$temp_ml_bin <- as.factor(cut(data_subset$prognosis_ml, c(-999, 2.5, 4.5, 999), include = TRUE,labels=c('Shorter','Same','Longer')))
print(c(bins,summary(data_subset$temp_ml_bin)))
mysurv <- npsurv(Surv(time_death_fu*12/365, dead) ~ temp_ml_bin, data=data_subset)
logrank_result <- survdiff(Surv(time_death_fu*12/365, dead) ~ temp_ml_bin, data=data_subset)
p_values[2] <- 1 - pchisq(logrank_result$chisq, length(logrank_result$n) - 1)
myplots[[3]] <- ggsurvplot(mysurv, risk.table = FALSE,censor=TRUE,break.x.by=6,break.y.by=0.25,
                           xlim=c(0,36),palette=cbPalette,axes.offset=FALSE,
                           xlab='Follow-up (months)',ylab='Overall survival',title='Physician predicts 6.1-12 month survival')
bins <- c(5,6)
data_subset <- data_analyze[data_analyze$prognosis_fac %in% bins,]
data_subset$temp_ml_bin <- as.factor(cut(data_subset$prognosis_ml, c(-999, 4.5, 6.5, 999), include = TRUE,labels=c('Shorter','Same','Longer')))
print(c(bins,summary(data_subset$temp_ml_bin)))
mysurv <- npsurv(Surv(time_death_fu*12/365, dead) ~ temp_ml_bin, data=data_subset)
logrank_result <- survdiff(Surv(time_death_fu*12/365, dead) ~ temp_ml_bin, data=data_subset)
p_values[3] <- 1 - pchisq(logrank_result$chisq, length(logrank_result$n) - 1)
myplots[[2]] <- ggsurvplot(mysurv, risk.table = FALSE,censor=TRUE,break.x.by=6,break.y.by=0.25,
                           xlim=c(0,36),palette=cbPalette,axes.offset=FALSE,
                           xlab='Follow-up (months)',ylab='Overall survival',title='Physician predicts 12.1-24 month survival')
bins <- c(7)
data_subset <- data_analyze[data_analyze$prognosis_fac %in% bins,]
data_subset$temp_ml_bin <- as.factor(cut(data_subset$prognosis_ml, c(-999, 6.5, 999), include = TRUE,labels=c('Shorter','Same')))
print(c(bins,summary(data_subset$temp_ml_bin)))
mysurv <- npsurv(Surv(time_death_fu*12/365, dead) ~ temp_ml_bin, data=data_subset)
logrank_result <- survdiff(Surv(time_death_fu*12/365, dead) ~ temp_ml_bin, data=data_subset)
p_values[4] <- 1 - pchisq(logrank_result$chisq, length(logrank_result$n) - 1)
myplots[[4]] <- ggsurvplot(mysurv, risk.table = FALSE,censor=TRUE,break.x.by=6,break.y.by=0.25,
                           xlim=c(0,36),palette=cbPalette,axes.offset=FALSE,
                           xlab='Follow-up (months)',ylab='Overall survival',title='Physician predicts >24 month survival')

svg(filename = paste(fig_dir,'ml_add_to_phys.svg',sep=''), width = 9, height = 9)
arrange_ggsurvplots(myplots,print=T,ncol=2,nrow=2)
dev.off()

data_subset <- data_analyze[data_analyze$prognosis_fac<3,]
mysurv <- npsurv(Surv(time_death_fu*12/365, dead) ~ prognosis_ml<3, data=data_subset)
#survplot(mysurv)
mysurv
print(summary(mysurv,times=c(6)))


data_analyze$complex <- data_analyze$tx_type>0 | data_analyze$fx>5

#>6mo
pred_cutoff <- 2
fac_long <- data_analyze[data_analyze$prognosis_fac>pred_cutoff,]
prop.table(table(fac_long$prognosis_ml>pred_cutoff,fac_long$complex,dnn=c('ml>cutoff','complex')),1)
prop.table(table(data_analyze$prognosis_fac>pred_cutoff,data_analyze$complex,dnn=c('fac>cutoff','complex')),1)

temp <- data_analyze %>% filter(complex==T) %>% group_by(prognosis_fac_f, prognosis_ml_f) %>% tally()
temp <- data_analyze %>% group_by(prognosis_fac_f, prognosis_ml_f) %>% tally()

chisq.test(data_analyze$prognosis_fac_f,data_analyze$complex)

data_subset <- data_analyze[data_analyze$prognosis_fac %in% c(1,2),]
chisq.test(data_subset$prognosis_ml_f,data_subset$complex)
data_subset <- data_analyze[data_analyze$prognosis_fac %in% c(3,4),]
chisq.test(data_subset$prognosis_ml_f,data_subset$complex)
data_subset <- data_analyze[data_analyze$prognosis_fac %in% c(5,6),]
chisq.test(data_subset$prognosis_ml_f,data_subset$complex)
data_subset <- data_analyze[data_analyze$prognosis_fac %in% c(7),]
chisq.test(data_subset$prognosis_ml_f,data_subset$complex)


logit_faconly <- glm(complex ~ prognosis_fac_f, data = data_analyze, family = "binomial")
summary(logit_faconly)
logit_fac_ml <- glm(complex ~ prognosis_fac_f + prognosis_ml_f, data = data_analyze, family = "binomial")
summary(logit_fac_ml)
anova(logit_faconly, logit_fac_ml,test="Chisq")

#>12mo
pred_cutoff <- 4
fac_long <- data_analyze[data_analyze$prognosis_fac>pred_cutoff,]
prop.table(table(fac_long$prognosis_ml>pred_cutoff,fac_long$complex,dnn=c('ml>cutoff','complex')),1)
prop.table(table(data_analyze$prognosis_fac>pred_cutoff,data_analyze$complex,dnn=c('fac>cutoff','complex')),1)

#>24mo
pred_cutoff <- 6
fac_short <- data_analyze[data_analyze$prognosis_fac<=pred_cutoff,]
prop.table(table(fac_short$prognosis_ml>pred_cutoff,fac_short$complex,dnn=c('ml>cutoff','complex')),1)
prop.table(table(data_analyze$prognosis_fac>pred_cutoff,data_analyze$complex,dnn=c('fac>cutoff','complex')),1)

#####
#patients w/ biggest discrepancy between attending and model

cuts <- c(0.5,2.5,4.5,6.5,7.5) #divides into 0-6 months, 6.1-12, 12.1-24, 24.1+
data_analyze$temp_bin <- as.factor(cut(as.numeric(data_analyze$prognosis_fac), cuts, include = TRUE,labels=FALSE))
data_analyze$prognosis_fac_f <- factor(data_analyze$temp_bin, levels=seq(4), labels=c('0-6', '6.1-12','12.1-24', '>24'))
data_analyze$temp_bin <- as.factor(cut(as.numeric(data_analyze$prognosis_ml), cuts, include = TRUE,labels=FALSE))
data_analyze$prognosis_ml_f <- factor(data_analyze$temp_bin, levels=seq(4), labels=c('0-6', '6.1-12','12.1-24', '>24'))

counts <- data_analyze %>% group_by(prognosis_fac_f,prognosis_ml_f, .drop=FALSE) %>% tally()
svg(filename = paste(fig_dir,'physician_comp_table.svg',sep=''), width = 4.5, height = 3)
ggplot(data = counts, aes(x = prognosis_fac_f, y = prognosis_ml_f)) +
  geom_tile(aes(fill = n)) +
  geom_text(aes(label = n), color = "black") +
  scale_fill_gradient("Counts Legend \n", low = "lightblue", high = "blue") +
  labs(x = "\nPhysician prediction (months)", y = "Computer prediction (months)") +
  theme_bw()
dev.off()

large_diff <- data_analyze[abs(data_analyze$prognosis_fac-data_analyze$prognosis_ml)>=5,]
large_diff$diff <- abs(large_diff$prognosis_fac-large_diff$prognosis_ml)
large_diff$surv_mo <- large_diff$time_death_fu*12/365

large_diff$tx_site1_text <- ''
for (i in seq(nrow(sites_frame))) {
  large_diff$tx_site1_text[large_diff$tx_site1 %in% sites_frame$site_nums[i][[1]]] <- as.character(sites_frame$site_name[i])
}

write.table(format(large_diff[order(large_diff$prognosis_fac,large_diff$prognosis_ml),c('prognosis_fac_f','prognosis_ml_f','alive','surv_mo','primsite_f','tx_site1_text','kps','improve_surv_feats','worsen_surv_feats')],digits=2),file=paste(data_dir,'large_diff.csv',sep=''),sep=',',row.names=FALSE)

comp_more_optimistic <- data_analyze[data_analyze$prognosis_ml-data_analyze$prognosis_fac>=5,]
mysurv <- npsurv(Surv(comp_more_optimistic$time_death_fu, comp_more_optimistic$dead) ~ 1)

#####
#for pts who died, distribution of errors
data_temp <- data_analyze[data_analyze$dead==1,]
cuts <- 365*c(-1, 1/4,1/2,3/4,1,1.5,2,999999)
data_temp$actual_bin <- cut(data_temp$time_death_fu, cuts, include = TRUE,labels=FALSE)
data_temp$phys_actual_diff <- data_temp$prognosis_fac - data_temp$actual_bin
data_temp$ml_actual_diff <- data_temp$prognosis_ml - data_temp$actual_bin

data_temp_long <- data_temp %>% pivot_longer(c('phys_actual_diff','ml_actual_diff'), names_to='Method', values_to='actual_diff')
data_temp_long$Method <- factor(data_temp_long$Method, levels=c('phys_actual_diff','ml_actual_diff'), labels=c('Physician','Computer'))

svg(filename = paste(fig_dir,'pred_actual_bin_diff.svg',sep=''), width = 5, height = 3)
ggplot(data_temp_long, aes(x=actual_diff, fill=Method)) +
  geom_bar(width=0.65,stat="count", position=position_dodge2(preserve='single')) +
  scale_fill_manual(values=cbPalette[c(1,3)]) +
  labs(y="No. patients",x="Predicted minus actual survival bin")+theme_bw()+
  scale_x_continuous(breaks=seq(-6,6))+
  theme(panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(), axis.text.x = element_text(angle = 45, hjust = 1))+
  scale_y_continuous(expand = expand_scale(add = c(0,10)))
dev.off()

#####
#most common features for good and poor-prognosis pts
improve_surv_feats<-strsplit(data_analyze$improve_surv_feats,' \\+| -')
worsen_surv_feats<-strsplit(data_analyze$worsen_surv_feats,' \\+| -')
summary(data_analyze$prognosis_ml>5)
summary(data_analyze$prognosis_ml<3)
improve_surv_in_good_prog <- unlist(improve_surv_feats[data_analyze$prognosis_ml>5])
worsen_surv_in_poor_prog <- unlist(worsen_surv_feats[data_analyze$prognosis_ml<3])
View(summary(as.factor(improve_surv_in_good_prog)))
View(summary(as.factor(worsen_surv_in_poor_prog)))

###
#subgroup analysis by sex and race ethnicity: C-index and calibration plots
summary(as.factor(data_analyze$RACE))
summary(as.factor(data_analyze$ETHNICITY))
summary(as.factor(data_analyze$male))

subset_indices <- list(data_analyze$male==1, data_analyze$male==0, data_analyze$RACE=='ASIAN', data_analyze$RACE=='WHITE', data_analyze$ETHNICITY=='HISPANIC/LATINO', data_analyze$ETHNICITY=='NON-HISPANIC/NON-LATINO')
subset_names <- c('male','female','Asian','white','Hispanic','not Hispanic')
method_names <- c('prognosis_fac','prognosis_ml','combined')
for (i in seq(length(subset_names))) {
  data_subset <- data_analyze[subset_indices[[i]],]
  print(subset_names[i])
  for (j in seq(length(method_names))) {
    print(method_names[j])
    w<-rcorr.cens(as.numeric(data_subset[,method_names[j]]),Surv(data_subset$time_death_fu, data_subset$dead))
    C <- w['C Index']
    se <- w['S.D.']/2
    low <- C-1.96*se; hi <- C+1.96*se
    print(cbind(C, low, hi))
  }
  cuts <- c(0.5,2.5,4.5,6.5,7.5) #divides into 0-6, 6.1-12, 12.1-24, 24.1+
  data_subset$temp_bin <- as.factor(cut(as.numeric(data_subset$pred_surv_bin), cuts, include = TRUE,labels=FALSE))
  mysurv <- npsurv(Surv(data_subset$time_death_fu, data_subset$dead) ~ data_subset$temp_bin)
  print(mysurv)
  svg(filename = paste(fig_dir,'model_calib_subset_',subset_names[i],'.svg',sep=''), width = 5, height = 4.5)
  par(yaxt="n")
  survplot(fit=mysurv,conf="none",xlim=c(0,365.25*2),time.inc=365.25/4,lty=1,label.curves = FALSE,col=cbPalette,lwd=2,
           n.risk=F,
           adj.n.risk=.5,cex.n.risk=.8,
           xlab="Follow-up, months",
           ylab="Survival")
  par(yaxt="s") ; axis(2, at=seq(0, 1, 0.25))
  title("0-6, 6.1-12, 12.1-24, 24.1+")
  dev.off()
}

#AUC sex/race/ethnicity subsets
library(fastAUC)
fu_time <- 365
data_temp <- data_analyze[data_analyze$time_death_fu>=fu_time | data_analyze$dead,]
data_temp$died_later <- data_temp$time_death_fu>fu_time
method_names <- c('prognosis_fac','prognosis_ml','combined', 'ecog_neg')
subset_indices <- list(data_temp$male==0, data_temp$male==1, data_temp$RACE=='ASIAN', data_temp$RACE=='WHITE', data_temp$ETHNICITY=='HISPANIC/LATINO', data_temp$ETHNICITY=='NON-HISPANIC/NON-LATINO')
subset_names <- c('female','male','Asian','white','Hispanic','not Hispanic')

primsites <- sort(as.character(unique(data_analyze$primsite_f)))
primsite_indices <- list()
for (i in seq(length(primsites))) {
  primsite_indices[[i]] <- data_temp$primsite_f==primsites[i]
}

subset_indices <- c(subset_indices, primsite_indices)
subset_names <- c(subset_names, primsites)

subset_results <- matrix(nrow=length(subset_names),ncol=length(method_names)+2)
for (i in seq(length(subset_names))) {
  data_subset <- data_temp[subset_indices[[i]],]
  #print(subset_names[i])
  #print(nrow(data_subset))
  subset_results[i,1] <- subset_names[i]
  subset_results[i,2] <- nrow(data_subset)
  for (j in seq(length(method_names))) {
    #print(method_names[j])
    auc_result <- auc(test_1=data_subset[,method_names[j]],
                    status=data_subset$died_later,
                    cluster=data_subset$dr)
    se<-auc_result$var^0.5
    low <- auc_result$auc-1.96*se; hi <- auc_result$auc+1.96*se #assume normal dist due to large n
    #print(cbind(auc_result$auc, low, hi))
    subset_results[i,j+2] <- paste(format(auc_result$auc, digits=2), ' (',format(low, digits=2), '-', format(hi, digits=2),')',sep='')
  }
}
subset_results <- as.data.frame(subset_results)
colnames(subset_results) <- c('subset','n',method_names)
write.table(subset_results,file=paste(data_dir,'subset_results.csv',sep=''),sep=',',row.names=FALSE, quote=F)



#####
#calibrate Jang ECOG performance status model to pts getting palliative RT
pt_info_pall_rt=read.csv(paste(data_dir,'pt_info.csv',sep=''),stringsAsFactors=FALSE)
pall_rt=read.csv(paste(data_dir,'pall_rt_outcomes_w_site.csv',sep=''),stringsAsFactors=FALSE)
pall_rt$tx_date <- mdy(pall_rt$tx_date)
pall_rt <- pall_rt[pall_rt$tx_date >= as.Date("2008-02-28"),]
pall_rt$patient_id <- as.integer(pall_rt$patient_id)
pall_rt <- pall_rt %>% distinct(patient_id, tx_date, .keep_all = TRUE)
pall_rt$duplicate <- 0
for (i in seq(nrow(pall_rt))) { # remove duplicate courses (within 14 days)
  tempTable2 <- pall_rt[-i,]
  tempTable <- tempTable2[tempTable2$patient_id==pall_rt[i,'patient_id'],]
  tempDiff <- as.numeric(tempTable$tx_date - pall_rt[i,'tx_date'])
  tempDiff=tempDiff[tempDiff > -14 & tempDiff < 1]
  if (length(tempDiff)) {
    pall_rt[i,'duplicate'] <- 1
  }
}
pall_rt <- pall_rt[pall_rt$duplicate==0,]
pt_info_pall_rt$death_date <- ymd(substr(pt_info_pall_rt$death_date,1,10))
pt_info_pall_rt$lfu_date <- ymd(substr(pt_info_pall_rt$lfu_date,1,10))
pt_info_pall_rt <- pt_info_pall_rt[,c('patient_id','death_date','lfu_date')]
pall_rt <- inner_join(pall_rt,pt_info_pall_rt,by="patient_id")

pall_rt$time_death <- as.numeric(pall_rt$death_date-pall_rt$tx_date)
pall_rt$time_fu <- as.numeric(pall_rt$lfu_date-pall_rt$tx_date)
pall_rt$time_death_fu <- apply(pall_rt[,c('time_death','time_fu')],1,max,na.rm=TRUE)
pall_rt$dead <- !is.na(pall_rt$death_date)
pall_rt <- pall_rt[pall_rt$time_death_fu>0,]
pall_rt <- pall_rt[pall_rt$ecog != 999,]
pall_rt$ecog_factor <- as.factor(cut(as.numeric(pall_rt$ecog), c(-1, 0.5, 1.5, 2.5, 999), include = TRUE,labels=c('0','1','2','3-4')))
summary(as.factor(pall_rt$ecog))
mysurv <- npsurv(Surv(pall_rt$time_death_fu, pall_rt$dead) ~ 1, data = pall_rt)
mysurv <- npsurv(Surv(pall_rt$time_death_fu, pall_rt$dead) ~ ecog_factor, data = pall_rt)
survplot(mysurv)
mysurv
#983 pts
#n events median 0.95LCL 0.95UCL
#ecog_factor=0    30     20    745     259    1179
#ecog_factor=1   457    393    294     249     345
#ecog_factor=2   402    368    151     135     187
#ecog_factor=3-4  94     87     79      60      96

c(745,294,151,79)*12/365 #median surv in months
#24.493150685  9.665753425  4.964383562  2.597260274

#calibration plot for Jang ECOG model
summary(as.factor(data_analyze$ecog))
#0   1   2   3   4
#67 498 244  64   6 
mysurv <- npsurv(Surv(data_analyze$time_death_fu, data_analyze$dead) ~ data_analyze$ecog_factor)
svg(filename = paste(fig_dir,'calib_jang_ecog.svg',sep=''), width = 5, height = 4.5)
par(yaxt="n")
survplot(fit=mysurv,conf="none",xlim=c(0,365.25*2),time.inc=365.25/4,lty=1,label.curves = FALSE,col=cbPalette,lwd=2,
         n.risk=F,
         adj.n.risk=.5,cex.n.risk=.8,
         xlab="Follow-up, months",
         ylab="Survival")
par(yaxt="s") ; axis(2, at=seq(0, 1, 0.25))
legend(200,0.8,c("24.5 (ECOG 0)", "9.7 (ECOG 1)", "5.0 (ECOG 2)", "2.6 (ECOG 3-4)"),lwd=2,col=cbPalette[c(1,2,3,4)],cex=1,title='Predicted survival (months)')
dev.off()

#####
#add procedure name to feature_names.csv
feature_coefs=read.csv(paste(data_dir,'feature_names.csv',sep=''),stringsAsFactors=FALSE)
proc_codes_names=read_excel(paste(data_dir,'proc_codes_names.xlsx',sep=''))
proc_codes_names <- proc_codes_names %>% group_by(CPT_CODE) %>% filter(row_number()==1)
proc_codes_names$feature_name <- paste('proc_',proc_codes_names$CPT_CODE,sep='')
names(proc_codes_names)[names(proc_codes_names) == 'PROC_NAME'] <- 'description'
proc_codes_names <- proc_codes_names[,c('feature_name','description')]
feature_coefs <- left_join(feature_coefs,proc_codes_names,by="feature_name")
is_diag = which(substr(feature_coefs$feature_name,1,4)=='diag')
temp <- explain_code(substr(feature_coefs[is_diag,'feature_name'],6,999),condense=F)
feature_coefs$description[is_diag] <- temp
write.table(feature_coefs,file=paste(data_dir,'feature_names_w_descriptions.csv',sep=''),sep=',',row.names=FALSE)
