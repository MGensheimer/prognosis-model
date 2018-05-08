# Load simulated patient data from Python and process it
# Outputs:
#   glmnet_cv.svg: Cross validation likelihood for various lambda values (regularization strength)
#   log.txt: Log listing C-index for test set and other information
#   cindex_test.svg: C-index for test set at various landmark time points
#   landmark_test_calib_1.svg, etc.: Calibration plots for test set at different landmark time points
#   test_calib_medsurv_1.svg: Calibration plot for landmark time t0, binning patients into 4 groups based on median predicted survival
#
# Tested with R version 3.4.3
#
# Author: Michael Gensheimer, Stanford University, May 8, 2018
# mgens@stanford.edu

rm(list=ls())
graphics.off()
options(digits=10)
library(feather)
library(rhdf5)
library(survival)
library(rms)
library(Hmisc)
library(glmnet)
library(dplyr)
library(survminer)
library(stringr)
library(ggplot2)

data_dir = '~/Documents/research/machine learning/prognosis/data/github/python_output/'
output_dir = '~/Documents/research/machine learning/prognosis/data/github/r_output/'
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
cox_probs <- c(0,.16,.5,.84,1) # as suggested in https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/1471-2288-13-33

visits <- read_feather(paste(data_dir,'visits.feather',sep=''))
visits$age <- rnorm(nrow(visits),mean=60,sd=10)
visits$sex <- sample(0:1,size=nrow(visits),replace=TRUE)
text <- h5read(paste(data_dir,'text.h5',sep=''),'visits_tfidf_lasso',compoundAsDataFrame=FALSE)
text <- t(text)
labsvitals <- h5read(paste(data_dir,'labsvitals.h5',sep=''),'visits_labs',compoundAsDataFrame=FALSE)
labsvitals <- t(labsvitals)
diag_proc_medi <- h5read(paste(data_dir,'diag_proc_medi.h5',sep=''),'diag_proc_medi',compoundAsDataFrame=FALSE)
diag_proc_medi <- t(diag_proc_medi)
data <- cbind(visits$age, visits$sex, text, labsvitals, diag_proc_medi)
text <- 0
labsvitals <- 0
diag_proc_medi <- 0
gc()

data <- data[visits$age>=18 & visits$days_to_last_contact_or_death>0,] #exclude children and visits with no f/u
visits <- visits[visits$age>=18 & visits$days_to_last_contact_or_death>0,]

visits$visit_date <- as.Date(visits$visit_date)
first_visit <- visits %>% group_by(patient_id) %>% summarise(first_visit=min(visit_date))
visits <- inner_join(visits,first_visit,by='patient_id')
visits$time1 <- as.numeric(visits$visit_date-visits$first_visit) #time1 = time since landmark time t0 (first visit after metastatic cancer diagnosis)
visits_train <- visits[visits$set==0,]
visits_test <- visits[visits$set==2,]
dataMean <- apply(data[visits$set==0,], 2, mean)
dataStd <- apply(data[visits$set==0,], 2, sd)
data <- (data - do.call('rbind',rep(list(dataMean),dim(data)[1]))) / do.call('rbind',rep(list(dataStd),dim(data)[1]))

############################################################
#Train Glmnet Cox model using dynamic prediction/landmarking
landmark_gap <- 365 #Would be 365/2 for real data, but for small simulated dataset, better to have fewer bins
landmarks <- seq(0,365*5,landmark_gap)
t_hor <- 365*5 #administrative censoring time horizon (see Putter slides)
log <- 1
time_varying <- 0
max_landmark_diff <- landmark_gap #max time a visit can be separated from landmark time and still be included in that landmark (visit must be earlier than landmark time)
time_inc<-365/4
end_time <- 365*5
surv_times=seq(0,end_time,time_inc)
logfilename <- paste(output_dir,'log.txt',sep='')

if(log) {write(paste('Run time:',Sys.time()),file=logfilename,append=TRUE)}
landmark_n_visits <- numeric()
visits_train$visitID <- seq(nrow(visits_train))
data_train_landmark <- c()
fu_time_train <- c()
dead_train <- c()
landmark_train <- c()
patient_id_train <- c()
for(i in seq(length(landmarks))) {
  visits_train$landmark_diff <- landmarks[i] - visits_train$time1
  #find each patient's visit closest to landmark date but not after it
  rows <- visits_train %>% group_by(patient_id) %>% filter(landmark_diff>=0 & days_to_last_contact_or_death+time1>landmarks[i]) %>% top_n(1, time1) %>% ungroup() %>% pull(visitID)
  data_train_landmark[[i]] <- data[which(visits$set==0)[rows],]
  fu_time_train[[i]] <- visits_train$days_to_last_contact_or_death[rows] - visits_train$landmark_diff[rows]
  dead_train[[i]] <- visits_train$dead[rows]
  landmark_train[[i]] <- rep(landmarks[i], length(rows))
  patient_id_train[[i]] <- visits_train$patient_id[rows]
}
data_train_landmark <- do.call(rbind,data_train_landmark)
fu_time_train <- unlist(fu_time_train)
dead_train <- unlist(dead_train)
dead_train[fu_time_train>t_hor] <- FALSE #administrative censoring if follow-up time > horizon time 
fu_time_train[fu_time_train>t_hor] <- t_hor
landmark_train <- unlist(landmark_train)
patient_id_train <- unlist(patient_id_train)
data_train_landmark <- cbind(data_train_landmark,landmark_train,landmark_train^2)
colnames(data_train_landmark) <- NULL

pts <- data.frame(patient_id = unique(patient_id_train))
n_folds <- 10
set.seed(0)
pts$fold <- sample(1:n_folds, nrow(pts), replace=T)
fold <- inner_join(data.frame(patient_id=patient_id_train), pts, by='patient_id')
lambda_seq <- 10^seq(5, -3, -0.1)
cv_glmnet <- cv.glmnet(data_train_landmark,
                       Surv(fu_time_train, dead_train),
                       family = "cox", alpha=0,standardize=FALSE,
                       foldid=fold$fold, lambda=lambda_seq, penalty.factor=c(rep(1,dim(data_train_landmark)[2]-2),0,0) ) #do not apply shrinkage to landmark time variables

svg(filename = paste(output_dir,'glmnet_cv.svg',sep=''), width = 7, height = 6)
plot(cv_glmnet)
dev.off()

glmnet_coefs <- as.vector(coef(cv_glmnet, s = "lambda.min"))
data_train_landmark <- data.frame(data_train_landmark)
cph.model <- cph(Surv(fu_time_train, dead_train) ~ ., data=data_train_landmark, #Use cph function in rms module to estimate baseline hazard function
                 iter.max=0, init=glmnet_coefs, surv=TRUE,residuals=FALSE,time.inc=1000)

visits_test$visitID <- seq(nrow(visits_test))
landmark_n_visits <- numeric()
data_test_landmark <- c()
fu_time_test <- c()
dead_test <- c()
landmark_test <- c()
patient_id_test <- c()
for(i in seq(length(landmarks))) {
  visits_test$landmark_diff <- landmarks[i] - visits_test$time1
  #find each patient's visit closest to landmark date but not after it
  rows <- visits_test %>% group_by(patient_id) %>% filter(landmark_diff>=0 & days_to_last_contact_or_death+time1>landmarks[i]) %>% top_n(1, time1) %>% ungroup() %>% pull(visitID)
  data_test_landmark[[i]] <- data[which(visits$set==2)[rows],]
  fu_time_test[[i]] <- visits_test$days_to_last_contact_or_death[rows] - visits_test$landmark_diff[rows]
  dead_test[[i]] <- visits_test$dead[rows]
  landmark_test[[i]] <- rep(landmarks[i], length(rows))
  patient_id_test[[i]] <- visits_test$patient_id[rows]
  landmark_n_visits[i] <- length(dead_test[[i]])
}
data_test_landmark <- do.call(rbind,data_test_landmark)
fu_time_test <- unlist(fu_time_test)
dead_test <- unlist(dead_test)
dead_test[fu_time_test>t_hor] <- FALSE
fu_time_test[fu_time_test>t_hor] <- t_hor
landmark_test <- unlist(landmark_test)
patient_id_test <- unlist(patient_id_test)
data_test_landmark <- cbind(data_test_landmark,landmark_test,landmark_test^2)
colnames(data_test_landmark) <- NULL
data_test_landmark <- data.frame(data_test_landmark)
pred_surv <- survest(fit=cph.model, newdata=data_test_landmark, times=surv_times, se.fit=FALSE)$surv
cindex_test <- c()
se_test <- c()
for(i in seq(length(landmarks))) {
  w<-rcorr.cens(pred_surv[landmark_test==landmarks[i],5],Surv(fu_time_test[landmark_test==landmarks[i]], dead_test[landmark_test==landmarks[i]]))
  cindex_test[i] <- w['C Index']
  se_test[i] <- w['S.D.']/2
  n_bins <- 4
  progIndexBin <- cut(pred_surv[landmark_test==landmarks[i],5], quantile(pred_surv[landmark_test==landmarks[i],5], probs = cox_probs), include = TRUE)
  mysurv <- npsurv(Surv(fu_time_test[landmark_test==landmarks[i]], dead_test[landmark_test==landmarks[i]]) ~ progIndexBin)
  svg(filename = paste(output_dir,'landmark_test_calib_',i,'.svg',sep=''), width = 7, height = 6)
  survplot(fit=mysurv,conf="none",xlim=c(0,365*5),label.curves = FALSE,lty=1,
           time.inc=365/4,col=cbPalette[2],lwd=2,n.risk=FALSE,xlab="Follow-up, years",ylab="Survival")
  for (bin in seq(n_bins)) {
    bin_surv <- pred_surv[which(landmark_test==landmarks[i])[progIndexBin==levels(progIndexBin)[bin]],]
    y <- colMeans(bin_surv)
    lines(surv_times, y, col=cbPalette[3], lwd=2)
  }
  title(paste('Landmark time (years): ',landmarks[i]/365,' No. patients',landmark_n_visits[i]))
  dev.off()
}

results <- data.frame(landmark=landmarks/365, n_visits=landmark_n_visits, cindex=cindex_test,cindex_lower=cindex_test - 1.96*se_test,cindex_upper=cindex_test + 1.96*se_test)
write.table(results,file=logfilename,append=TRUE,row.names=FALSE)

myplot = ggplot(data=results,aes(x=landmark,y=cindex))+ #,label=paste("n=",count,sep="")))+
  geom_errorbar(aes(x=landmark,ymin=cindex_lower,ymax=cindex_upper),width=0.3)+#+
  geom_point()+  
  scale_y_continuous()+
  scale_x_continuous(breaks=seq(0,5,0.5))+
  labs(y="C-index",x="Landmark time (years from metastatic diagnosis)")+
  theme_bw(base_size = 12)+theme(panel.grid.major.x=element_blank(),panel.grid.minor.x=element_blank())+
  #theme(plot.margin = unit(c(1,1,5,3), "lines"))
  theme(plot.margin = unit(c(10,10,50,100), "pt"))
library(grid)
vert_pos <- -0.05
for(i in seq(nrow(results))) {
  text <- textGrob(results$n_visits[i], gp=gpar(fontsize=10))
  myplot = myplot + annotation_custom(text,xmin=results$landmark[i],xmax=results$landmark[i],ymin=vert_pos,ymax=vert_pos)
}
text <- textGrob('No. of \n patients', gp=gpar(fontsize=12),just='right')
myplot = myplot + annotation_custom(text,xmin=-0.5,xmax=-0.5,ymin=vert_pos,ymax=vert_pos)
svg(filename = paste(output_dir,'cindex_test.svg',sep=''), width = 7, height = 6)
gt <- ggplot_gtable(ggplot_build(myplot))
gt$layout$clip[gt$layout$name == "panel"] <- "off"
grid.draw(gt)
dev.off()

time1_temp <- t(do.call(rbind, replicate(length(landmarks), visits_test$time1,simplify=FALSE)))
landmarks_temp <- do.call(rbind, replicate(nrow(visits_test), landmarks,simplify=FALSE))
time_diff_temp <- abs(time1_temp-landmarks_temp)
visits_test$landmark <- landmarks[apply(time_diff_temp, 1, which.min)]
data_temp <- data[which(visits$set==2),]
data_temp <- cbind(data_temp,visits_test$landmark,visits_test$landmark^2)
colnames(data_temp) <- NULL
data_temp <- data.frame(data_temp)
pred_surv_1yr_test <- survest(fit=cph.model, newdata=data_temp, times=365,se.fit=FALSE)$surv
pred_surv_1yr_test[is.na(pred_surv_1yr_test)] <- 1
visits_test$pred_surv_1yr <- pred_surv_1yr_test

surv_times_fine=seq(0,365*5,365/16)
pred_surv_test <- survest(fit=cph.model, newdata=data_temp, times=surv_times_fine,se.fit=FALSE)$surv
visits_test$median_surv <- max(surv_times_fine)
for(i in seq(nrow(visits_test))) {
  pred_surv_test[i,is.na(pred_surv_test[i,])] <- min(pred_surv_test[i,],na.rm=TRUE)
  if(min(pred_surv_test[i,]) < 0.5) { #if median survival reached for this visit
    visits_test[i, 'median_surv'] <- surv_times_fine[which(pred_surv_test[i,]<0.5)[1]]
  }
}
visits_test$median_surv_bin <- cut(visits_test$median_surv, 365*c(0, 1/4, 1/2, 1, 999), include.lowest = TRUE)
visits_test$median_surv_bin <- factor(visits_test$median_surv_bin, #reorder levels for plots
                                      levels=c("(365,3.65e+05]","(182,365]","(91.2,182]","[0,91.2]"))

#survival curves at specific landmark times
for(i in c(1)) {
  visits_test$landmark_diff <- landmarks[i] - visits_test$time1
  visits_temp <- visits_test %>% group_by(patient_id) %>% filter(landmark_diff>=0 & days_to_last_contact_or_death+time1>landmarks[i]) %>% top_n(1, time1) %>% ungroup()
  rows <- visits_temp$visitID
  fu_time <- visits_test$days_to_last_contact_or_death[rows] - visits_test$landmark_diff[rows]
  mysurv <- survfit(Surv(fu_time, dead) ~ median_surv_bin, data=visits_temp)
  svg(filename = paste(output_dir,'test_calib_medsurv_',i,'.svg',sep=''), width = 10.5, height = 8)
  print(ggsurvplot(mysurv, risk.table = TRUE,censor=FALSE,break.x.by=365/2,break.y.by=0.1,xlim=c(0,365*5),palette=cbPalette[1:4],axes.offset=FALSE))
  dev.off()
  if(i==1) {
    sink(logfilename, append=TRUE)
    print('Test set calibration landmark time 0:')
    print(mysurv)
    sink()
  }
}

