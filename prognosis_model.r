# Load simulated patient data from Python and process it
# Outputs:
#   glmnet_fit_all.RData: L2 regularized Cox models for 3 follow-up time periods
#   log.txt: Log listing C-index on training, validation, and test sets and other information
#   lambda_all.txt: Lists best value of lambda regularization parameter for each of 3 follow-up time periods
#   calib_valid_all.svg: Calibration plots for validation set
#   calib_test_all.svg: Calibration plots for test set
#   test_surv_bins.svg: Actual survival for 4 bins of model-predicted survival
#   coefs.csv: Final model coefficients for each predictor variable (for 0-6 months follow-up time) 
#   roc.svg: ROC curves for 3 specified follow-up times
#
# Tested with R version 3.4.3
#
# Author: Michael Gensheimer, Stanford University, Mar. 8, 2018
# michael.gensheimer@gmail.com

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
library(ggplot2)
library(survminer)
library(pROC)

convert_timeperiod <- function(olddata,start_time,end_time) {
  newdata <- olddata[olddata$days_to_last_contact_or_death>=start_time,]
  newdata$dead[newdata$days_to_last_contact_or_death>end_time] <- FALSE
  newdata$days_to_last_contact_or_death[newdata$days_to_last_contact_or_death>end_time] <- end_time
  newdata$days_to_last_contact_or_death <- newdata$days_to_last_contact_or_death - start_time
  return(newdata)
}

data_dir = '/Users/michael/Documents/research/machine learning/prognosis/code/github/python_output/'
output_dir = '/Users/michael/Documents/research/machine learning/prognosis/code/github/r_output/'
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

#Glmnet for prognosis study
visits <- read_feather(paste(data_dir,'visits.feather',sep=''))
text <- h5read(paste(data_dir,'text.h5',sep=''),'visits_tfidf_lasso',compoundAsDataFrame=FALSE)
text <- t(text)
labsvitals <- h5read(paste(data_dir,'labsvitals.h5',sep=''),'visits_labs',compoundAsDataFrame=FALSE)
labsvitals <- t(labsvitals)
diag_proc_medi <- h5read(paste(data_dir,'diag_proc_medi.h5',sep=''),'diag_proc_medi',compoundAsDataFrame=FALSE)
diag_proc_medi <- t(diag_proc_medi)
data <- cbind(text, labsvitals, diag_proc_medi)
text <- 0
labsvitals <- 0
diag_proc_medi <- 0
gc()

data <- data[visits$days_to_last_contact_or_death>0,] #exclude visits with no f/u
visits <- visits[visits$days_to_last_contact_or_death>0,]

visits_train <- visits[visits$set==0,]
visits_valid <- visits[visits$set==1,]
visits_test <- visits[visits$set==2,]

dataMean <- apply(data[visits$set==0,], 2, mean)
dataStd <- apply(data[visits$set==0,], 2, sd)
data <- (data - do.call('rbind',rep(list(dataMean),dim(data)[1]))) / do.call('rbind',rep(list(dataStd),dim(data)[1]))
data[is.na(data)] <- 0 #if standard deviation 0, will have NAs

start_time <- 365.25*c(0,0.5,2)
end_time <- 365.25*c(0.5,2,5)
cox_probs <- c(0,.16,.5,.84,1) # as suggested in https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/1471-2288-13-33

set.seed(0)
n_timebins <- length(start_time)
visits_train_timeperiod = NULL
visits_valid_timeperiod = NULL
visits_test_timeperiod = NULL
for(i in seq(n_timebins)) {
  visits_train_timeperiod[[i]] <- convert_timeperiod(visits_train,start_time[i],end_time[i])
  visits_valid_timeperiod[[i]] <- convert_timeperiod(visits_valid,start_time[i],end_time[i])
  visits_test_timeperiod[[i]] <- convert_timeperiod(visits_test,start_time[i],end_time[i])
}

model_name <- c('all') #include all features
model_features <- list(seq(418)) #include all features

for(which_model in seq(length(model_name))) {
  logfilename <- paste(output_dir,'log.txt',sep='')
  write(paste('Run time:',Sys.time(),'Features:',model_name[which_model]),file=logfilename,append=TRUE)

  data_subset <- data[,model_features[[which_model]]]
  data_train <- data_subset[visits$set==0,]
  data_valid <- data_subset[visits$set==1,]
  data_test <- data_subset[visits$set==2,]
  data_subset <- 0
  gc()

  glmnet_file <- paste(output_dir,'glmnet_fit_', model_name[which_model], '.RData',sep='')
  if(file.exists(glmnet_file)) {
    load(paste(output_dir,'glmnet_fit_', model_name[which_model], '.RData',sep=''))
  } else {
    fit_timeperiod = NULL
    for(i in seq(n_timebins)) {
      fit_timeperiod[[i]] <- glmnet(data_train[visits_train$days_to_last_contact_or_death>=start_time[i],], Surv(visits_train_timeperiod[[i]]$days_to_last_contact_or_death, visits_train_timeperiod[[i]]$dead), family = "cox", alpha=0,standardize=FALSE)
    }
    save(fit_timeperiod, file=glmnet_file)
  }
    
  lambda_file <- paste(output_dir,'lambda_',model_name[which_model],'.txt',sep='')
  if(file.exists(lambda_file)) {
    tempFrame <- read.csv(lambda_file, stringsAsFactors=FALSE)
    best_lambda <- tempFrame$lambda
  } else {
    best_lambda=numeric(n_timebins)
    for(i in seq(n_timebins)) {
      results_frame <- data.frame(lambda=fit_timeperiod[[i]]$lambda, cindex_train=0, cindex_valid=0)
      for (j in seq(1,nrow(results_frame))) {
        pred_coef_train <- predict(fit_timeperiod[[i]],newx=data_train,s=fit_timeperiod[[i]]$lambda[j],type="link")
        pred_coef_valid <- predict(fit_timeperiod[[i]],newx=data_valid,s=fit_timeperiod[[i]]$lambda[j],type="link")
        results_frame[j,'cindex_train'] <- rcorr.cens(-pred_coef_train,Surv(visits_train$days_to_last_contact_or_death, visits_train$dead))[1]
        results_frame[j,'cindex_valid'] <- rcorr.cens(-pred_coef_valid,Surv(visits_valid$days_to_last_contact_or_death, visits_valid$dead))[1]
      }
      best_lambda[i] <- results_frame$lambda[which(results_frame$cindex_valid-max(results_frame$cindex_valid) > -0.005)[1]]
      write(paste('Time period:',i),file=logfilename,append=TRUE)
      write.table(results_frame,file=logfilename,append=TRUE,row.names=FALSE)
    }
    write.table(data.frame(timeperiod=seq(n_timebins),lambda=best_lambda),file=lambda_file,append=FALSE,row.names=FALSE,sep=',')
  }

  n_knots <- 5
  cph_timeperiod = NULL
  glmnet_lp_timeperiod_allpts_train = NULL
  glmnet_lp_timeperiod_valid = NULL
  glmnet_lp_timeperiod_allpts_valid = NULL
  glmnet_lp_timeperiod_test = NULL
  glmnet_lp_timeperiod_allpts_test = NULL
  for(i in seq(n_timebins)) {
    glmnet_coefs_timeperiod <- as.vector(coef(fit_timeperiod[[i]],s=best_lambda[i]))
    glmnet_lp_timeperiod_allpts_train[[i]] <- as.vector(data.matrix(data_train) %*% glmnet_coefs_timeperiod)
    glmnet_lp_timeperiod_valid[[i]] <- as.vector(data.matrix(data_valid[visits_valid$days_to_last_contact_or_death>=start_time[i],]) %*% glmnet_coefs_timeperiod)
    glmnet_lp_timeperiod_allpts_valid[[i]] <- as.vector(data.matrix(data_valid) %*% glmnet_coefs_timeperiod)
    glmnet_lp_timeperiod_test[[i]] <- as.vector(data.matrix(data_test[visits_test$days_to_last_contact_or_death>=start_time[i],]) %*% glmnet_coefs_timeperiod)
    glmnet_lp_timeperiod_allpts_test[[i]] <- as.vector(data.matrix(data_test) %*% glmnet_coefs_timeperiod)
    v_temp <- visits_valid_timeperiod[[i]] #shorten text due to error in cph with long text
    v_temp$d <- v_temp$days_to_last_contact_or_death
    g_lp <- glmnet_lp_timeperiod_valid[[i]]
    cph_timeperiod[[i]] <- cph(Surv(v_temp$d, v_temp$dead) ~ rcs(g_lp,n_knots), surv=TRUE)
  }

  time_inc <- 365.25/16
  lp=NULL
  pred_surv_timebin_train=NULL
  pred_surv_timebin_valid=NULL
  pred_surv_timebin_test=NULL
  surv_times = NULL
  for(i in seq(n_timebins)) {
    pred_surv_timebin_train[[i]] <- survest(fit=cph_timeperiod[[i]], newdata=glmnet_lp_timeperiod_allpts_train[[i]], times=seq(0,end_time[i]-start_time[i],time_inc), se.fit=FALSE)$surv
    pred_surv_timebin_valid[[i]] <- survest(fit=cph_timeperiod[[i]], newdata=glmnet_lp_timeperiod_allpts_valid[[i]], times=seq(0,end_time[i]-start_time[i],time_inc), se.fit=FALSE)$surv
    pred_surv_timebin_test[[i]] <- survest(fit=cph_timeperiod[[i]], newdata=glmnet_lp_timeperiod_allpts_test[[i]], times=seq(0,end_time[i]-start_time[i],time_inc), se.fit=FALSE)$surv
    if(i>1) {
      pred_surv_timebin_train[[i]] <- pred_surv_timebin_train[[i]] * do.call(cbind,rep(list(pred_surv_timebin_train[[i-1]][,dim(pred_surv_timebin_train[[i-1]])[2]]),dim(pred_surv_timebin_train[[i]])[2]))
      pred_surv_timebin_valid[[i]] <- pred_surv_timebin_valid[[i]] * do.call(cbind,rep(list(pred_surv_timebin_valid[[i-1]][,dim(pred_surv_timebin_valid[[i-1]])[2]]),dim(pred_surv_timebin_valid[[i]])[2]))
      pred_surv_timebin_test[[i]] <- pred_surv_timebin_test[[i]] * do.call(cbind,rep(list(pred_surv_timebin_test[[i-1]][,dim(pred_surv_timebin_test[[i-1]])[2]]),dim(pred_surv_timebin_test[[i]])[2]))
    }
    surv_times=c(surv_times,start_time[i]+seq(0,end_time[i]-start_time[i],time_inc))
  }
  pred_surv_train <- do.call(cbind,pred_surv_timebin_train)
  pred_surv_valid <- do.call(cbind,pred_surv_timebin_valid)
  pred_surv_test <- do.call(cbind,pred_surv_timebin_test)
  pred_surv1yr_train <- pred_surv_train[,min(which(surv_times >= 365.25))]
  pred_surv1yr_valid <- pred_surv_valid[,min(which(surv_times >= 365.25))]
  pred_surv1yr_test <- pred_surv_test[,min(which(surv_times >= 365.25))]
  cindex_train <- rcorr.cens(pred_surv1yr_train,Surv(visits_train$days_to_last_contact_or_death, visits_train$dead))[1]
  cindex_valid <- rcorr.cens(pred_surv1yr_valid,Surv(visits_valid$days_to_last_contact_or_death, visits_valid$dead))[1]
  write(paste('Piecewise model 1yr survival C-index train:',cindex_train),file=logfilename,append=TRUE)
  write(paste('Piecewise model 1yr survival C-index valid:',cindex_valid),file=logfilename,append=TRUE)
}

#Calculations using final model

#general statistics
cbind(length(unique(visits$patient_id)),length(unique(visits_train$patient_id)), length(unique(visits_valid$patient_id)), length(unique(visits_test$patient_id)))
cbind(nrow(visits_train), nrow(visits_valid), nrow(visits_test))
mysurv <- npsurv(Surv(days_to_last_contact_or_death, dead) ~ 1, data=visits)
mysurv
summary(visits$days_to_last_contact_or_death)
temp <- visits %>% group_by(patient_id) %>% top_n(1)
summary(temp$dead)

cindex_test <- rcorr.cens(pred_surv1yr_test,Surv(visits_test$days_to_last_contact_or_death, visits_test$dead))[1]
write(paste('Piecewise model 1yr survival C-index test:',cindex_test),file=logfilename,append=TRUE)

#calibration plots
n_bins <- 4
time_inc <- 365.25/16
svg(filename = paste(output_dir,'calib_valid_',model_name[which_model],'.svg',sep=''), width = 15, height = 4.5)
par(mfcol=c(1,n_timebins))
for(i in seq(n_timebins)) {
  surv_times_temp=seq(0,end_time[i]-start_time[i],time_inc)
  pred_surv_timebin <- survest(fit=cph_timeperiod[[i]], newdata=glmnet_lp_timeperiod_valid[[i]], times=surv_times_temp, se.fit=FALSE)$surv
  pred_surv_endofbin <- pred_surv_timebin[,dim(pred_surv_timebin)[2]]
  progIndexBin <- cut(pred_surv_endofbin, quantile(pred_surv_endofbin, probs = cox_probs), include = TRUE)
  mysurv <- npsurv(Surv(visits_valid_timeperiod[[i]]$days_to_last_contact_or_death, visits_valid_timeperiod[[i]]$dead) ~ progIndexBin)
  plot(mysurv,axes=FALSE,xlim=c(0,end_time[i]-start_time[i]),xlab='Follow-up (years)',ylab='Overall survival',xaxs="i", yaxs="i",col=cbPalette[2],lwd=2)
  axis(side=1, at=seq(0,end_time[i]-start_time[i],365.25/2), labels=seq(0,end_time[i]-start_time[i],365.25/2))
  axis(side=2, at=seq(0,1,0.1), labels=c('0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'))
  rug(x = seq(0,end_time[i]-start_time[i],365.25/4), ticksize = -0.01, side = 1)
  for (bin in seq(n_bins)) {
    bin_surv <- pred_surv_timebin[progIndexBin==levels(progIndexBin)[bin],]
    y <- colMeans(bin_surv)
    lines(surv_times_temp, y, col=cbPalette[3], lwd=2)
  }
  title(paste('Validation set calibration for timeperiod',i))
}
dev.off()

svg(filename = paste(output_dir,'calib_test_',model_name[which_model],'.svg',sep=''), width = 15, height = 4.5)
par(mfcol=c(1,n_timebins))
for(i in seq(n_timebins)) {
  surv_times_temp=seq(0,end_time[i]-start_time[i],time_inc)
  pred_surv_timebin <- survest(fit=cph_timeperiod[[i]], newdata=glmnet_lp_timeperiod_test[[i]], times=surv_times_temp, se.fit=FALSE)$surv
  pred_surv_endofbin <- pred_surv_timebin[,dim(pred_surv_timebin)[2]]
  progIndexBin <- cut(pred_surv_endofbin, quantile(pred_surv_endofbin, probs = cox_probs), include = TRUE)
  mysurv <- npsurv(Surv(visits_test_timeperiod[[i]]$days_to_last_contact_or_death, visits_test_timeperiod[[i]]$dead) ~ progIndexBin)
  plot(mysurv,axes=FALSE,xlim=c(0,end_time[i]-start_time[i]),xlab='Follow-up (years)',ylab='Overall survival',xaxs="i", yaxs="i",col=cbPalette[2],lwd=2)
  axis(side=1, at=seq(0,end_time[i]-start_time[i],365.25/2), labels=seq(0,end_time[i]-start_time[i],365.25/2))
  axis(side=2, at=seq(0,1,0.1), labels=c('0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'))
  rug(x = seq(0,end_time[i]-start_time[i],365.25/4), ticksize = -0.01, side = 1)
  for (bin in seq(n_bins)) {
    bin_surv <- pred_surv_timebin[progIndexBin==levels(progIndexBin)[bin],]
    y <- colMeans(bin_surv)
    lines(surv_times_temp, y, col=cbPalette[3], lwd=2)
  }
  title(paste('Test set calibration for timeperiod',i))
}
dev.off()

#ROC curves at specific follow-up time points
fu_time_array <- 365.25*c(1/12, 1/4, 1)
svg(filename = paste(output_dir,'roc.svg',sep=''), width = 10, height = 3)
par(mfcol=c(1,length(fu_time_array)))
for(fu_time in fu_time_array) {
  subset <- visits_test$days_to_last_contact_or_death>fu_time | visits_test$dead==TRUE
  roc_obj <- roc(c(visits_test[subset,'days_to_last_contact_or_death']>fu_time), pred_surv1yr_test[subset])
  print(c(fu_time, auc(roc_obj)))
  plot(roc_obj,xaxs="i", yaxs="i",col=cbPalette[3])
  title(paste('ROC curve for follow-up time:',fu_time))
}
dev.off()

#model coefficients for 0-6 month follow-up time period
glmnet_coefs <- coef(fit_timeperiod[[1]],s=best_lambda[1])
write.table(data.frame(coef=as.numeric(glmnet_coefs)),file=paste(output_dir,'coefs.csv',sep=''),append=FALSE,row.names=FALSE,sep=',')

#plot actual survival for four bins (predicted survival 0-3 ,3-6, 6-12, >12 months)
visits_test$median_surv <- max(surv_times)
for(i in seq(nrow(visits_test))) {
  if(max(pred_surv_test[i,] < 0.5)) { #if median survival reached for this visit
    visits_test[i, 'median_surv'] <- surv_times[which(pred_surv_test[i,]<0.5)[1]]
  }
}
visits_test$median_surv_bin <- cut(visits_test$median_surv, 365.25*c(0, 1/4, 1/2, 1, 999), include.lowest = TRUE)
mysurv <- survfit(Surv(days_to_last_contact_or_death, dead) ~ median_surv_bin, data=visits_test)
svg(filename = paste(output_dir,'test_surv_bins.svg',sep=''), width = 10.5, height = 8)
ggsurvplot(mysurv, data = visits_test, risk.table = TRUE,censor=FALSE,break.x.by=365.25/2,break.y.by=0.1,xlim=c(0,365.25*5),palette=cbPalette[1:4],axes.offset=FALSE)
dev.off()
