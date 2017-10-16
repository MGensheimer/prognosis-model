# Load simulated patient data from Python and process it
#
# Tested with R version 3.3.2
#
# Author: Michael Gensheimer, 10/16/2017
# michael.gensheimer@gmail.com

rm(list=ls())
graphics.off()
library(Hmisc)
library(survival)
library(rms)
library(ggplot2)
library(MASS)
library(lubridate)
library(caret)
library(dplyr)
library(psych)
library(kappaSize)
library(survminer)
library(rhdf5)
library(survIDINRI)
library(Matrix)
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7") #color palette

data_directory <- 'C:/Users/Michael/Documents/research/machine learning/prognosis/code/github/python_output/'
visits=read.csv(paste(data_directory,'visits.csv',sep=''),stringsAsFactors=FALSE)
visits_tfidf <- h5read(paste(data_directory,'text_labs.h5',sep=''),'text_labs',compoundAsDataFrame=FALSE)
visits_tfidf <- visits_tfidf$table
visits_tfidf <- t(visits_tfidf$values_block_0)
visits$visit_date <- ymd(visits$visit_date)
visits$dob <- ymd(visits$dob)
visits$date_last_contact_or_death <- ymd(visits$date_last_contact_or_death)
visits$age <- as.numeric(visits$visit_date-visits$dob)/365.25

n_CoursesPallRtPreStudy <- 300 #number of palliative radiation courses prior to prospective study
n_CoursesPallRtStudy <- 200    #number of radiationcourses on palliative radiation study

temp_sample <- sample(x=nrow(visits),size=n_CoursesPallRtPreStudy,replace=FALSE) #assign palliative radiation courses to random visits
choices <- which(visits$set==2)
choices <- setdiff(choices,temp_sample)
temp_sample2 <- sample(x=choices,size=n_CoursesPallRtStudy,replace=FALSE) #assign study radiation courses to random visits, but only test set visits
visits$pallRtPreStudyIndex <- NA
visits$pallRtStudyIndex <- NA
visits[temp_sample,'pallRtPreStudyIndex'] <- seq(1,n_CoursesPallRtPreStudy)
visits[temp_sample2,'pallRtStudyIndex'] <- seq(1,n_CoursesPallRtStudy)

mysurv <- npsurv(Surv(visits$days_to_last_contact_or_death, visits$dead) ~ 1)
survplot(fit=mysurv,xlim=c(0,365*5))
title('Survival for visits in all patients')

visits_train <- visits[visits$set==0 & visits$days_to_last_contact_or_death>0,]
visits_tfidf_train <- data.frame(visits_tfidf[visits$set==0 & visits$days_to_last_contact_or_death>0,])
visits_valid <- visits[visits$set==1 & visits$days_to_last_contact_or_death>0,]
visits_tfidf_valid <- data.frame(visits_tfidf[visits$set==1 & visits$days_to_last_contact_or_death>0,])
visits_test <- visits[visits$set==2 & visits$days_to_last_contact_or_death>0,]
visits_tfidf_test <- data.frame(visits_tfidf[visits$set==2 & visits$days_to_last_contact_or_death>0,])

myPsm <- psm(Surv(visits_train$days_to_last_contact_or_death, visits_train$dead) ~ .,data=visits_tfidf_train,dist='weibull')

visits$progIndex <- predict(myPsm,type='lp',visits_tfidf)
visits_train$progIndex <- predict(myPsm,type='lp',visits_tfidf_train)
visits_valid$progIndex <- predict(myPsm,type='lp',visits_tfidf_valid)
visits_test$progIndex <- predict(myPsm,type='lp',visits_tfidf_test)

n_bins <- 5 #number of bins for calibration plot
visits_train$progIndexBin <- with(visits_train, cut(progIndex, quantile(progIndex, probs = seq(0, 1, 1/n_bins)), include = TRUE))
visits_valid$progIndexBin <- with(visits_valid, cut(progIndex, quantile(progIndex, probs = seq(0, 1, 1/n_bins)), include = TRUE))

surv_times <- seq(0,365*5,365/16)
pred_surv <- survest(myPsm,visits_tfidf_train,times=surv_times)
mysurv <- npsurv(Surv(visits_train$days_to_last_contact_or_death, visits_train$dead) ~ visits_train$progIndexBin)
survplot(fit=mysurv,conf="none",xlim=c(0,365*5),label.curves = FALSE)
for (bin in seq(n_bins)) {
  bin_surv <- pred_surv[visits_train$progIndexBin==levels(visits_train$progIndexBin)[bin],]
  y <- colMeans(bin_surv)
  smoothingSpline = smooth.spline(surv_times, y, spar=0.2)
  lines(smoothingSpline, col='red', lwd=1)
}
title('PSM training set calibration: red predicted, black actual')

pred_surv <- survest(myPsm,visits_tfidf_valid,times=surv_times)
mysurv <- npsurv(Surv(visits_valid$days_to_last_contact_or_death, visits_valid$dead) ~ visits_valid$progIndexBin)
survplot(fit=mysurv,conf="none",xlim=c(0,365*5),label.curves = FALSE)
for (bin in seq(n_bins)) {
  bin_surv <- pred_surv[visits_valid$progIndexBin==levels(visits_valid$progIndexBin)[bin],]
  y <- colMeans(bin_surv)
  smoothingSpline = smooth.spline(surv_times, y, spar=0.2)
  lines(smoothingSpline, col='red', lwd=1)
}
title('PSM validation set calibration: red predicted, black actual')

w<-rcorr.cens(visits_train$progIndex,Surv(visits_train$days_to_last_contact_or_death, visits_train$dead))
C <- w['C Index']
se <- w['S.D.']/2
low <- C-1.96*se; hi <- C+1.96*se
'Training set C-index (95% CI)'
cbind(C, low, hi)
w<-rcorr.cens(visits_valid$progIndex,Surv(visits_valid$days_to_last_contact_or_death, visits_valid$dead))
C <- w['C Index']
se <- w['S.D.']/2
low <- C-1.96*se; hi <- C+1.96*se
'Validation set C-index (95% CI)'
cbind(C, low, hi)
w<-rcorr.cens(visits_test$progIndex,Surv(visits_test$days_to_last_contact_or_death, visits_test$dead))
C <- w['C Index']
se <- w['S.D.']/2
low <- C-1.96*se; hi <- C+1.96*se
'Test set C-index (95% CI)'
cbind(C, low, hi)

visits_trainvalid_PallRtPreStudy <- visits[(visits$set==0 | visits$set==1) & !is.na(visits$pallRtPreStudyIndex) & visits$days_to_last_contact_or_death>0,]

# Choose cut-points to discretize prognostic index, based on best performance for palliative radiation courses that were given prior to prospective study
n_bins <- 4 # With real patients, 7 bins are used but for the smaller simulated dataset 4 bins illustrate the procedure better
cuts <- c(-999, 5.6, 6.2, 6.9, 999) #These cut-points were chosen manually to produce median survival close to intended for each bin, and place the most patients in the correct bin
visits_trainvalid_PallRtPreStudy$progIndexPallRtBin <- cut(visits_trainvalid_PallRtPreStudy$progIndex, cuts, include = TRUE,labels=FALSE)
mysurv <- npsurv(Surv(visits_trainvalid_PallRtPreStudy$days_to_last_contact_or_death, visits_trainvalid_PallRtPreStudy$dead) ~ visits_trainvalid_PallRtPreStudy$progIndexPallRtBin)
correctBin <- data.frame(shorter=rep(NA,n_bins),longer=rep(NA,n_bins),median=rep(NA,n_bins),row.names=c('0-3','3.1-12','12.1-24','24.1-'))
lowLimit <- c(0,3,12,24)*30.44 #Patients in bin 1 are predicted to live 0-3 months, bin 2 3.6-12 months, bin 3 12.1-24 months, bin 4 24.1 or more months
upLimit <- c(3,12,24,999)*30.44
mySumm <- summary(mysurv)
for (i in seq(n_bins)) {
  myTime <- mySumm$time[mySumm$strata==levels(mySumm$strata)[i]]
  mySurvprob <- mySumm$surv[mySumm$strata==levels(mySumm$strata)[i]]
  correctBin$shorter[i]<-1-mySurvprob[which(myTime>=lowLimit[i])[1]]
  correctBin$longer[i]<-mySurvprob[which(myTime>upLimit[i])[1]]
  correctBin$median[i]<-myTime[which(mySurvprob<0.5)[1]]/30.44
}
correctBin$shorter[1]<-0
correctBin$longer[n_bins]<-0
correctBin$correct <- 1-correctBin$shorter-correctBin$longer
correctBin$n <- summary(as.factor(visits_trainvalid_PallRtPreStudy$progIndexPallRtBin))

correctBin$n %*% correctBin$correct/nrow(visits_trainvalid_PallRtPreStudy) #proportion of patients assigned to correct bin
survplot(fit=mysurv,conf="none",xlim=c(0,365*3),time.inc=365.25/4,lty=1,label.curves=list(keys="lines"),col=cbPalette) #plot survival of patients in the four bins

# Test model performance using radiation courses on prospective study
visits_PallRtStudy <- visits[!is.na(visits$pallRtStudyIndex),]
visits_PallRtStudy$progIndexBin <- as.factor(cut(visits_PallRtStudy$progIndex, cuts, include = TRUE,labels=FALSE))
visits_PallRtStudy$progPhysicianBin <- as.factor(sample(x=4,size=n_CoursesPallRtStudy,replace=TRUE)) #Generate simulated physician survival predictions (randomly assign to bins 1-4)

# Plot study patients' survival for the 4 bins of physician and model predictions
mysurv <- npsurv(Surv(visits_PallRtStudy$days_to_last_contact_or_death, visits_PallRtStudy$dead) ~ visits_PallRtStudy$progPhysicianBin)
survplot(fit=mysurv,conf="none",xlim=c(0,365.25*2),time.inc=365.25/4,lty=1,label.curves = FALSE,col=cbPalette[2],lwd=2,
         xlab="Follow-up, months",
         ylab="Survival")
mysurv <- npsurv(Surv(visits_PallRtStudy$days_to_last_contact_or_death, visits_PallRtStudy$dead) ~ visits_PallRtStudy$progIndexBin)
lines(mysurv,lty=1,lwd=2,col=cbPalette[3])
legend(400,0.8,c("Physicians", "Machine learning model"),lty=c(1,1),lwd=c(2,2),col=cbPalette[2:3],cex=1)
title("Blue machine learning; orange physican. 0-3, 3.1-12, 12.1-24, >24 months")

# Display C-index for physicians, machine learning model
w<-rcorr.cens(as.numeric(visits_PallRtStudy$progPhysicianBin),Surv(visits_PallRtStudy$days_to_last_contact_or_death, visits_PallRtStudy$dead))
C <- w['C Index']
se <- w['S.D.']/2
low <- C-1.96*se; hi <- C+1.96*se
'Palliative radiation study courses physician C-index (95% CI)'
cbind(C, low, hi)

w<-rcorr.cens(as.numeric(visits_PallRtStudy$progIndexBin),Surv(visits_PallRtStudy$days_to_last_contact_or_death, visits_PallRtStudy$dead))
C <- w['C Index']
se <- w['S.D.']/2
low <- C-1.96*se; hi <- C+1.96*se
'Palliative radiation study courses model C-index (95% CI)'
cbind(C, low, hi)

# Test for difference in discrimination between physicians and machine learning model
temp<-rcorrp.cens(as.numeric(visits_PallRtStudy$progPhysicianBin),
                  as.numeric(visits_PallRtStudy$progIndexBin),
                  Surv(visits_PallRtStudy$days_to_last_contact_or_death, visits_PallRtStudy$dead),
                  method=1)
2*pnorm(-abs(temp[1]/temp[2])) #2 sided p value

# Continuous net reclassification improvement from adding model prediction to physician prediction
outcome <- visits_PallRtStudy[,c('days_to_last_contact_or_death','dead')]
covs1 <- model.matrix(~visits_PallRtStudy$progPhysicianBin)[,-1] # Create dummy variables from factor. The "-1" removes the intercept term
covs2 <- cbind(covs1,model.matrix(~visits_PallRtStudy$progIndexBin)[,-1]) # Create dummy variables from factor. The "-1" removes the intercept term
x<-IDI.INF(outcome, covs1, covs2, 30.44*3)
IDI.INF.OUT(x) #M1 indicates IDI;  M2 indicates NRI;  M3 indicates median difference

# Per physician NRI from adding model to physician prediction
n_phys <- 5
visits_PallRtStudy$phys_id <- as.factor(sample(x=n_phys,size=n_CoursesPallRtStudy,replace=TRUE))
phys_results <- data.frame(nri=rep(NA,n_phys),nri_lower=rep(NA,n_phys),nri_upper=rep(NA,n_phys))
for (phys_id in seq(n_phys)) {
  visits_phys <- visits_PallRtStudy[visits_PallRtStudy$phys_id==phys_id,]
  outcome <- visits_phys[,c('days_to_last_contact_or_death','dead')]
  covs1 <- model.matrix(~visits_phys$progPhysicianBin)[,-1] # Create dummy variables from factor. The "-1" removes the intercept term
  covs2 <- cbind(covs1,model.matrix(~visits_phys$progIndexBin)[,-1]) # Create dummy variables from factor. The "-1" removes the intercept term
  x<-IDI.INF(outcome, covs1, covs2, 30.44*3)
  phys_results$nri[phys_id] <- 2*x$m2[1]
  phys_results$nri_lower[phys_id] <- 2*x$m2[2]
  phys_results$nri_upper[phys_id] <- 2*x$m2[3]
}
phys_results$phys_id <- seq(n_phys)
phys_freq <- visits_PallRtStudy %>% group_by(phys_id) %>% summarise(count=n())
phys_results$count <- phys_freq$count

myplot = ggplot(data=phys_results,aes(x=reorder(phys_id,nri),y=nri,size=count))+
  geom_errorbar(aes(x=reorder(phys_id,nri),ymin=nri_lower,ymax=nri_upper,size=1),width=0.3)+
  geom_point(shape=21,colour="black",fill=cbPalette[6])+scale_size_area()+  
  scale_y_continuous(limits=c(-2,2),breaks=seq(-2,2,0.5))+
  geom_hline(yintercept=0,linetype=2)+
  labs(y="Continuous NRI",x="Physician")+
  theme_bw(base_size = 12)+theme(panel.grid.major.x=element_blank(),panel.grid.minor.x=element_blank())
myplot # plot per-physician NRI with 95% CIs

#Performance of high/low risk groups according to physicians and machine learning model
#For true/false positive rates, cases=died within 3 months, controls=survived >3 months

fu_time <- 30.44*3 # Analyze probability of 3 month survival
mysurv <- npsurv(Surv(visits_PallRtStudy$days_to_last_contact_or_death, visits_PallRtStudy$dead) ~ 1)
temp <- summary(mysurv,times=fu_time)
case_prop <- 1 - temp$surv

#Compute performance characteristics of physicians
visits_temp <- visits_PallRtStudy[visits_PallRtStudy$progPhysicianBin==1,]
mysurv <- npsurv(Surv(visits_temp$days_to_last_contact_or_death, visits_temp$dead) ~ 1)
temp <- summary(mysurv,times=fu_time)
ppv_phys <- 1-temp$surv
visits_temp <- visits_PallRtStudy[visits_PallRtStudy$progPhysicianBin!=1,]
mysurv <- npsurv(Surv(visits_temp$days_to_last_contact_or_death, visits_temp$dead) ~ 1)
temp <- summary(mysurv,times=fu_time)
npv_phys <- temp$surv
tp <- ppv_phys*sum(visits_PallRtStudy$progPhysicianBin==1)/nrow(visits_PallRtStudy)
fp <- (1-ppv_phys)*sum(visits_PallRtStudy$progPhysicianBin==1)/nrow(visits_PallRtStudy)
tn <- npv_phys*sum(visits_PallRtStudy$progPhysicianBin!=1)/nrow(visits_PallRtStudy)
fn <- (1-npv_phys)*sum(visits_PallRtStudy$progPhysicianBin!=1)/nrow(visits_PallRtStudy)
tpr_phys <- tp/(tp+fn)
fpr_phys <- fp/(fp+tn)
prop_hr_phys <- (tpr_phys*case_prop + fpr_phys*(1-case_prop)) #proportion of patients put in high risk group

#Compute performance characteristics of model
visits_temp <- visits_PallRtStudy[visits_PallRtStudy$progIndex>5.6,]
mysurv <- npsurv(Surv(visits_temp$days_to_last_contact_or_death, visits_temp$dead) ~ 1)
temp <- summary(mysurv,times=fu_time)
ppv_model <- 1-temp$surv
visits_temp <- visits_PallRtStudy[visits_PallRtStudy$progIndexBin!=1,]
mysurv <- npsurv(Surv(visits_temp$days_to_last_contact_or_death, visits_temp$dead) ~ 1)
temp <- summary(mysurv,times=fu_time)
npv_model <- temp$surv
tp <- ppv_phys*sum(visits_PallRtStudy$progIndexBin==1)/nrow(visits_PallRtStudy)
fp <- (1-ppv_model)*sum(visits_PallRtStudy$progIndexBin==1)/nrow(visits_PallRtStudy)
tn <- npv_phys*sum(visits_PallRtStudy$progIndexBin!=1)/nrow(visits_PallRtStudy)
fn <- (1-npv_model)*sum(visits_PallRtStudy$progIndexBin!=1)/nrow(visits_PallRtStudy)
tpr_model <- tp/(tp+fn)
fpr_model <- fp/(fp+tn)
prop_hr_model <- (tpr_model*case_prop + fpr_model*(1-case_prop)) #proportion of patients put in high risk group

#Compute category-based NRI of substituting model prediction for physician prediction
nri <- (tpr_model-tpr_phys) + (fpr_phys-fpr_model) #category-based net reclassification improvement
print(cbind(fu_time, case_prop, nri))
print(cbind(fpr_phys,tpr_phys,ppv_phys,npv_phys,prop_hr_phys))
print(cbind(fpr_model,tpr_model,ppv_model,npv_model,prop_hr_model))
