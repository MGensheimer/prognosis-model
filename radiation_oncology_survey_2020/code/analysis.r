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
data1$mrn_full <- as.integer(data1$mrn_full)
names(data1)[names(data1) == 'mrn_full'] <- 'mrn_int'
data1$primsite_f <- factor(data1$primsite_coded, levels=seq(12), labels=c('CNS', 'Head and neck', 'Thorax', 'Breast', 'Gastrointestinal', 'Genitourinary', 'Gynecologic', 'Hematologic', 'Bone', 'Skin', 'Endocrine', 'Other/unknown'))
data1$primsite_f[data1$primsite_coded %in% c(1, 11)] <- 'Other/unknown'

data2=read_excel(paste(data_dir,'survey_outcomes_updated.xlsx',sep='')) #load updated follow-up info
data3 <- data2[c('MRN','Alive','Date of death','Last Follow-up')]
names(data3) <- c('mrn_int','alive','death_date','lfu_date')
data3$mrn_int <- as.integer(data3$mrn_int)
data3$death_date <- ymd(data3$death_date)
data3$lfu_date <- ymd(data3$lfu_date)
data3 <- unique(data3)
temp <- data3 %>% group_by(mrn_int) %>%  filter(n() > 1)
data3 <- data3 %>% group_by(mrn_int) %>% filter(row_number()==1)

attending_grad_year=read.csv(paste(data_dir,'attending_grad_year.csv',sep=''),stringsAsFactors=FALSE)

#data quality checks
nrow(unique(data1[c('mrn_int','tx_date')]))

#join tables to add updated follow-up info
data <- inner_join(data1,data3,by="mrn_int")
data <- inner_join(data,attending_grad_year,by="dr")
data$yr_after_grad <- 2016-data$yr_res_grad

#more data quality checks
length(unique(data$mrn_int))
summary(data$alive & !is.na(data$death_date))

#load computer model predictions
model_predsurv <- read_parquet(paste(data_dir,'model_predsurv.parquet',sep=''))
model_results <- read_parquet(paste(data_dir,'model_results.parquet',sep=''))
attr(model_results$eval_date,'tzone') <- 'UTC'
model_results$eval_date <- as.Date(model_results$eval_date)
model_results$mrn_int = as.integer(model_results$MRN_FULL)
names(model_results)[names(model_results) == 'eval_date'] <- 'tx_date'
model_results_for_join <- model_results[,c('mrn_int','tx_date','median_pred_surv','median_pred_surv_2wkshift','improve_surv_feats','worsen_surv_feats')]
data_w_model_results <- inner_join(data,model_results_for_join,by=c("mrn_int",'tx_date'))
data_w_model_results$time_death <- as.numeric(data_w_model_results$death_date-data_w_model_results$tx_date)
data_w_model_results$time_fu <- as.numeric(data_w_model_results$lfu_date-data_w_model_results$tx_date)
data_w_model_results$time_death_fu <- apply(data_w_model_results[,c('time_death','time_fu')],1,max,na.rm=TRUE)
data_w_model_results$dead <- !is.na(data_w_model_results$death_date)
data_w_model_results <- data_w_model_results[data_w_model_results$time_death_fu>0 | data_w_model_results$dead==TRUE,] #1 removed w/ 9 days f/u
data_analyze <- data_w_model_results[!is.na(data_w_model_results$prognosis_fac),]

cuts <- 365*c(-1, 1/4,1/2,3/4,1,1.5,2,999999)
data_analyze$prognosis_ml <- cut(data_analyze$median_pred_surv, cuts, include = TRUE,labels=FALSE)
data_analyze$pred_surv_bin <- as.factor(data_analyze$prognosis_ml)
data_analyze$pred_surv_2wkshift_bin <- as.factor(cut(data_analyze$median_pred_surv_2wkshift, cuts, include = TRUE,labels=FALSE))

#####
#data summary
length(unique(data_analyze$mrn_int))
courses_per_pt <- data_analyze %>% group_by(mrn_int) %>% tally(sort=T)
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
#C-index faculty vs model
w<-rcorr.cens(as.numeric(data_analyze$prognosis_fac),Surv(data_analyze$time_death_fu, data_analyze$dead))
w<-rcorr.cens(as.numeric(data_analyze$prognosis_ml),Surv(data_analyze$time_death_fu, data_analyze$dead))
w<-rcorr.cens(as.numeric(data_analyze$pred_surv_2wkshift_bin),Surv(data_analyze$time_death_fu, data_analyze$dead))
w<-rcorr.cens(as.numeric(data_analyze$combined),Surv(data_analyze$time_death_fu, data_analyze$dead))
C <- w['C Index']
se <- w['S.D.']/2
low <- C-1.96*se; hi <- C+1.96*se
print(cbind(C, low, hi))
#Results:                     C          low           hi
#fac                   0.6711895009 0.6491436573 0.6932353445
#model                 0.7028146768 0.6830854157 0.722543938
#model w/ 2wk old data 0.6560264769 0.6349100975 0.6771428562
#fac + model           0.7211983928 0.701738948 0.7406578376

#C-index significantly different?
temp<-rcorrp.cens(as.numeric(data_analyze$prognosis_fac),
                  as.numeric(data_analyze$prognosis_ml),
                  Surv(data_analyze$time_death_fu, data_analyze$dead),
                  method=1)
2*pnorm(-abs(temp[1]/temp[2])) #2 sided p value
#2.459522645e-06

temp<-rcorrp.cens(as.numeric(data_analyze$prognosis_ml),
                  as.numeric(data_analyze$combined),
                  Surv(data_analyze$time_death_fu, data_analyze$dead),
                  method=1)
2*pnorm(-abs(temp[1]/temp[2])) #2 sided p value


#C-index faculty vs model, randomly pick 1 treatment per pt
reps <- 1000
pvalue_a <- c()
for (i in seq(reps)) {
  data_analyze_oneperpt <- data_analyze %>% group_by(mrn_int) %>% sample_n(1)
  temp<-rcorrp.cens(as.numeric(data_analyze_oneperpt$prognosis_fac),
                    as.numeric(data_analyze_oneperpt$prognosis_ml),
                    Surv(data_analyze_oneperpt$time_death_fu, data_analyze_oneperpt$dead),
                    method=1)
  pvalue_a <- append(pvalue_a, 2*pnorm(-abs(temp[1]/temp[2]))) #2 sided p value
}
median(pvalue_a)
#0.0004251060924
mean(pvalue_a)
#0.0006348942465

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
survplot(fit=mysurv,conf="none",xlim=c(0,365.25*2),time.inc=365.25/4,lty=1,label.curves = FALSE,col=cbPalette,lwd=2,
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
#ROC
survauc <- function(data, indices, method, fu_time, auc_only=T) {
  d <- data[indices,] # allows boot to select sample
  surv.res <- with(d,
                   survivalROC(Stime        = time_death_fu,
                               status       = dead,
                               marker       = -eval(parse(text=method)),
                               predict.time = fu_time,
                               method       = "KM"))
  if(auc_only) {
    return(surv.res$AUC)
  }
  else {
    return(surv.res)
  }
}

fu_time_m_a <- c(6, 12)
method_a <- c('prognosis_fac','prognosis_ml')
for (fu_time_m in fu_time_m_a) {
  print(fu_time_m)
  fu_time <- fu_time_m*365/12
  ROCphys <- survauc(data=data_analyze, indices=seq(nrow(data_analyze)), method='prognosis_fac',fu_time=fu_time, auc_only=F)
  ROCml <- survauc(data=data_analyze, indices=seq(nrow(data_analyze)), method='prognosis_ml',fu_time=fu_time, auc_only=F)
  print(ROCphys$AUC)
  print(ROCml$AUC)
  svg(filename = paste(fig_dir,'roc_',fu_time_m,'mo.svg',sep=''), width = 5, height = 5)
  with(ROCphys, plot(TP ~ FP,type="l",col=cbPalette[1],lwd=2, xaxs="i",yaxs="i",xlab='False positive rate',ylab='True positive rate'))
  with(ROCml, lines(TP ~ FP,col=cbPalette[3],lwd=2))
  lines(c(0,1),c(0,1),lty=2,col='#999999')
  legend(0.4,0.35,c(
    paste("Physicians: AUC ",format(ROCphys$AUC,digits=3),sep=""),
    paste("Computer: AUC ",format(ROCml$AUC,digits=3),sep="")
  ),lty=1,lwd=2,col=cbPalette[c(1,3)],cex=0.8)
  title(paste(format(fu_time/30.44,digits=1)," month survival",sep=""))
  dev.off()
  #bootstrapped confidence intervals for time-dependent AUC
  for (method in method_a) {
    boot.res <- boot(data=data_analyze, statistic=survauc, R=10000, method=method, fu_time=fu_time)
    temp <- boot.ci(boot.res)
    print(paste(method,': bootstrapped 95% CI (percentile method)'))
    print(temp$percent)
  }
}

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
large_diff$surv_mo <- large_diff$time_death_fu/30

large_diff$tx_site1_text <- ''
for (i in seq(nrow(sites_frame))) {
  large_diff$tx_site1_text[large_diff$tx_site1 %in% sites_frame$site_nums[i][[1]]] <- as.character(sites_frame$site_name[i])
}

write.table(format(large_diff[order(large_diff$prognosis_fac,large_diff$prognosis_ml),c('prognosis_fac_f','prognosis_ml_f','alive','surv_mo','primsite_f','tx_site1_text','kps','improve_surv_feats','worsen_surv_feats')],digits=2),file=paste(data_dir,'large_diff.csv',sep=''),sep=',',row.names=FALSE)

comp_more_optimistic <- data_analyze[data_analyze$prognosis_ml-data_analyze$prognosis_fac>=5,]
mysurv <- npsurv(Surv(comp_more_optimistic$time_death_fu, comp_more_optimistic$dead) ~ 1)

######
#plot computer prediction distribution
summary(data_analyze$median_pred_surv < 3*365/12) #6%
summary(data_analyze$median_pred_surv < 6*365/12) #28%
summary(data_analyze$median_pred_surv < 12*365/12) #52%

temp <- data_analyze
temp$median_pred_surv[temp$median_pred_surv>1825] <- 1826
temp$months <- floor(temp$median_pred_surv/30)+0.5

ggplot(temp, aes(x=months)) + 
  geom_histogram(binwidth=1,color="black", fill="white") +
  scale_x_continuous(breaks=seq(0,61,3),limits=c(0,61),expand=c(0,0)) +
  scale_y_continuous(expand=c(0,5)) +
  theme_bw() +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())
