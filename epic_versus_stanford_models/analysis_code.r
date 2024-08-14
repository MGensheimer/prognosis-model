rm(list=ls())
graphics.off()
library(lubridate)
library(dplyr)
library(readxl)
library(ggplot2)
library(stringr)
library(tidyr)
library(rms)
library(survminer)
library(pROC)
library(gtsummary)
library(survival)

source('~/Documents/software/r/mfg_utils.r')
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

data_preds_dir <- '/Users/michael/Documents/research/machine_learning/prognosis/data/cba/'
data_epic_dir <- '/Users/michael/Documents/research/machine_learning/prognosis/data/epic/'
fig_dir <- '/Users/michael/Documents/research/machine_learning/prognosis/figures/epic_eoli/'

preds_data <- data.frame(epic_pred_date=as.Date(c('2021-06-14','2021-11-16','2022-01-31','2022-03-28')),
                         epic_preds_files=c('DM_END_OF_LIFE_Extracted20210614.txt','DM_END_OF_LIFE_Extracted20211116.txt','DM_END_OF_LIFE_Extracted20220131.txt','DM_END_OF_LIFE_Extracted20220328.txt'),
                         cancer_preds_files=I(list(c('preds_medonc_pilot_2021-06-13.csv','preds_medonc_randomized_study_2021-06-13.csv'),
                                                   c('preds_medonc_pilot_2021-11-14.csv','preds_medonc_randomized_study_2021-11-14.csv'),
                                                   c('preds_medonc_pilot_2022-01-30.csv','preds_medonc_randomized_study_2022-01-30.csv'),
                                                   c('preds_medonc_pilot_2022-03-27.csv','preds_medonc_randomized_study_2022-03-27.csv'))))

#load in survival/follow-up data from Epic
actual_surv <- read.csv(paste(data_preds_dir,'Medical_Oncology_Historical_Data 2023-04-18.csv',sep=''))
actual_surv <- actual_surv[,c('patient_id','lfu_date','death_date')]
actual_surv$lfu_date[actual_surv$lfu_date=='NULL'] <- NA
actual_surv$death_date[actual_surv$death_date=='NULL'] <- NA
actual_surv$lfu_date <- as.Date(actual_surv$lfu_date)
actual_surv$death_date <- as.Date(actual_surv$death_date)

#load Epic and Stanford model predictions
joined_list <- list()
for (pred_row in seq(nrow(preds_data))) {
  preds_list <- c()
  i<-1
  for (preds_file in preds_data$cancer_preds_files[[pred_row]]) {
    preds_date <- str_match(preds_file,'\\d\\d\\d\\d-\\d\\d-\\d\\d')
    temp_frame <- read.csv(paste(data_preds_dir,preds_file,sep=''),stringsAsFactors=FALSE, colClasses=c("mrn_full"="character"))
    columns <- c('patient_id','mrn_full','DepartmentNM','ProviderNM','VisitTypeDSC','sex','race','ethnicity','dob','median_pred_surv','pred_12mo_surv')
    if('medonc_attending' %in% colnames(temp_frame)) {
      columns <- append(columns,'medonc_attending')
    }
    if('AppointmentDTS' %in% colnames(temp_frame)) {
      columns <- append(columns,'AppointmentDTS')
    }
    temp_frame <- temp_frame[,columns]
    preds_list[[i]] <- temp_frame
    if(!'AppointmentDTS' %in% colnames(preds_list[[i]])) { #for a few weeks at beginning of data collection, appointment date not recorded, so estimate appointment date from report date.
      preds_list[[i]]$AppointmentDTS <- ymd(preds_date) + 4 #if report run on Sunday, this is average date of clinic visits in that week
    } else {
      preds_list[[i]]$AppointmentDTS <- ymd(substr(preds_list[[i]]$AppointmentDTS,1,10))
    }
    i<-i+1
  }
  cancer_preds <- bind_rows(preds_list)
  epic_preds <- read.table(paste(data_epic_dir,preds_data$epic_preds_files[[pred_row]],sep=''),stringsAsFactors=FALSE,header=T)
  colnames(epic_preds)[2] <- 'patient_id'
  epic_preds <- epic_preds[!duplicated(epic_preds$patient_id), ]
  cancer_preds <- cancer_preds[!duplicated(cancer_preds$patient_id), ]
  joined_list[[pred_row]] <- left_join(cancer_preds,epic_preds,by="patient_id")
  joined_list[[pred_row]] <- left_join(joined_list[[pred_row]],actual_surv,by="patient_id")
  joined_list[[pred_row]]$epic_pred_date <- preds_data$epic_pred_date[[pred_row]]
}
joined <- bind_rows(joined_list)

# data processing
joined <- joined[!duplicated(joined$patient_id),]

#add in updated STARR follow-up data that includes Social Security Administration Death Master File data
starr_fu <- read.csv(paste(data_epic_dir,'demographics-47101-mgens-confidential.csv',sep=''))
starr_mrns <- read.csv(paste(data_epic_dir,'patientCodebook-47101-mgens-confidential.csv',sep=''),colClasses=c("MRN"="character"))
colnames(starr_mrns)[1] <- 'Patient.Id'
starr_fu <- left_join(starr_fu, starr_mrns, by='Patient.Id')
starr_fu <- starr_fu %>% select(MRN, Date.of.Death, Death.Date.SSA.Do.Not.Disclose, Recent.Encounter.Date)
starr_fu$Date.of.Death <- mdy(substr(starr_fu$Date.of.Death,1,10))
starr_fu$Death.Date.SSA.Do.Not.Disclose <- mdy(substr(starr_fu$Death.Date.SSA.Do.Not.Disclose,1,10))
starr_fu$Recent.Encounter.Date <- mdy(substr(starr_fu$Recent.Encounter.Date,1,10))
starr_fu$Date.of.Death[!is.na(starr_fu$Death.Date.SSA.Do.Not.Disclose)] <- starr_fu$Death.Date.SSA.Do.Not.Disclose[!is.na(starr_fu$Death.Date.SSA.Do.Not.Disclose)]
starr_fu <- starr_fu %>% select(MRN, Date.of.Death, Recent.Encounter.Date)
colnames(starr_fu) <- c('mrn_full','death_date_starr','lfu_date_starr')
joined <- left_join(joined,starr_fu,by='mrn_full')
table(!is.na(joined$death_date), !is.na(joined$death_date_starr),dnn=c('dead','dead STARR'))
joined$death_date[!is.na(joined$death_date_starr)] <- joined$death_date_starr[!is.na(joined$death_date_starr)]
joined$lfu_date[!is.na(joined$lfu_date_starr)] <- joined$lfu_date_starr[!is.na(joined$lfu_date_starr)]

# more processing
joined_hassurv <- joined[joined$patient_id %in% actual_surv$patient_id,] #restrict to patients with survival follow-up data
joined_hassurv$time_death <- as.numeric(joined_hassurv$death_date-joined_hassurv$epic_pred_date)
joined_hassurv$time_fu <- as.numeric(joined_hassurv$lfu_date-joined_hassurv$epic_pred_date)
joined_hassurv <- joined_hassurv[!is.na(joined_hassurv$time_fu),]
joined_hassurv$time_death_fu <- apply(joined_hassurv[,c('time_death','time_fu')],1,max,na.rm=TRUE)
joined_hassurv$dead <- !is.na(joined_hassurv$death_date)
joined_hassurv <- joined_hassurv[joined_hassurv$time_death_fu>0,]
joined_hassurv <- joined_hassurv %>% filter(!is.na(END_OF_LIFE_CARE_INDEX_SCORE))
joined_hassurv$dob <- as.Date(joined_hassurv$dob)
joined_hassurv$age <- as.numeric((joined_hassurv$epic_pred_date-joined_hassurv$dob))/365
joined_hassurv$race_recoded <- 'Other/multiple'
joined_hassurv$race_recoded[joined_hassurv$race=='Asian'] <- 'Asian'
joined_hassurv$race_recoded[joined_hassurv$race=='White' | joined_hassurv$race=='White, non-Hispanic'] <- 'White'
joined_hassurv$race_recoded[joined_hassurv$race=='Unknown' | joined_hassurv$race=='Patient Refused'] <- 'Unknown'
joined_hassurv$race_recoded[joined_hassurv$race=='Black or African American'] <- 'Black'
joined_hassurv$ethnicity_recoded <- 'Unknown'
joined_hassurv$ethnicity_recoded[joined_hassurv$ethnicity=='Hispanic/Latino'] <- 'Hispanic'
joined_hassurv$ethnicity_recoded[joined_hassurv$ethnicity=='Non-Hispanic/Non-Latino'] <- 'Not Hispanic'
joined_hassurv$pred_12mo_risk <- 1-joined_hassurv$pred_12mo_surv

#patient characteristics table
mysummary <- joined_hassurv %>% tbl_summary(include=c('DepartmentNM','sex','race_recoded','ethnicity_recoded','age','pred_12mo_risk','END_OF_LIFE_CARE_INDEX_SCORE','dead','time_death_fu'),
                                            label=c(DepartmentNM ~ 'department',pred_12mo_risk ~ 'Stanford model-predicted 1-year mortality risk',time_death_fu~'Follow-up time (days)'))
mysummary %>% as_flex_table() %>% flextable::save_as_docx(path=paste(fig_dir,'visits_summary.docx',sep=''))
summary(joined_hassurv$time_death_fu)
mysurv <- npsurv(Surv(time_death_fu*12/365, dead) ~ 1, data=joined_hassurv)
mysurv
summary(mysurv, times = c(12, 24))

joined_hassurv$prog_bin_canc <- cut2(joined_hassurv$pred_12mo_surv, g=4)
levels(joined_hassurv$prog_bin_canc) <- c(1,2,3,4)
joined_hassurv$prog_bin_epic <- cut2(-joined_hassurv$END_OF_LIFE_CARE_INDEX_SCORE, g=4)
levels(joined_hassurv$prog_bin_epic) <- c(1,2,3,4)

#Kaplan-Meier curves for patients in quartiles of Stanford model-predicted survival
cancer_surv <- npsurv(Surv(time_death_fu*12/365, dead) ~ prog_bin_canc, data=joined_hassurv)
svg(filename = paste(fig_dir,'kaplan_meier_cancer_2021-2022_data.svg',sep=''), width = 6.5, height = 6)
cancer_plot <- ggsurvplot(cancer_surv, risk.table = TRUE,censor=FALSE,break.x.by=3,break.y.by=0.25,
                     xlim=c(0,25),palette=cbPalette,axes.offset=FALSE,
                     xlab='Follow-up (months)',ylab='Overall survival')
print(cancer_plot)
dev.off()

#Kaplan-Meier curves for patients in quartiles of EOLCI-predicted survival
epic_surv <- npsurv(Surv(time_death_fu*12/365, dead) ~ prog_bin_epic, data=joined_hassurv)
svg(filename = paste(fig_dir,'kaplan_meier_epic_2021-2022_data.svg',sep=''), width = 6.5, height = 6)
epic_plot <- ggsurvplot(epic_surv, risk.table = TRUE,censor=FALSE,break.x.by=3,break.y.by=0.25,
                          xlim=c(0,25),palette=cbPalette,axes.offset=FALSE,
                          xlab='Follow-up (months)',ylab='Overall survival')
print(epic_plot)
dev.off()

joined_hassurv$tertile_canc <- cut2(joined_hassurv$pred_12mo_surv, g=3)
levels(joined_hassurv$tertile_canc) <- c(1,2,3)
joined_hassurv$tertile_epic <- cut2(-joined_hassurv$END_OF_LIFE_CARE_INDEX_SCORE, g=3)
levels(joined_hassurv$tertile_epic) <- c(1,2,3)

cancer_surv <- npsurv(Surv(time_death_fu*12/365, dead) ~ tertile_canc, data=joined_hassurv)
epic_surv <- npsurv(Surv(time_death_fu*12/365, dead) ~ tertile_epic, data=joined_hassurv)
fit <- list(CANCER_SURV = cancer_surv, EPIC_surv = epic_surv)
svg(filename = paste(fig_dir,'kaplan_meier_tertiles_2021-2022_data.svg',sep=''), width = 6.5, height = 6)
ggsurvplot(fit,combine=T, risk.table = T,censor=F,break.x.by=3,break.y.by=0.25,
           xlim=c(0,25),axes.offset=FALSE,linetype=c(1,1,1,2,2,2),
           xlab='Follow-up (months)',ylab='Overall survival')
dev.off()

#C-index for Stanford model
w<-rcorr.cens(as.numeric(joined_hassurv$pred_12mo_surv),Surv(joined_hassurv$time_death_fu, joined_hassurv$dead))
C <- w['C Index']
se <- w['S.D.']/2
low <- C-1.96*se; hi <- C+1.96*se
print(cbind(C, low, hi))

#C-index for EOLCI
w<-rcorr.cens(as.numeric(-joined_hassurv$END_OF_LIFE_CARE_INDEX_SCORE),Surv(joined_hassurv$time_death_fu, joined_hassurv$dead))
C <- w['C Index']
se <- w['S.D.']/2
low <- C-1.96*se; hi <- C+1.96*se
print(cbind(C, low, hi))

#AUC and ROC curve of EOLCI vs. Stanford model, for various follow-up time points
timeperiods <- c(180,365)
for (timeperiod in timeperiods) {
  print(paste('time period: ',timeperiod,'days'))
  joined_hassurv_timeperiod <- joined_hassurv[(joined_hassurv$dead==T) | (joined_hassurv$time_death_fu>timeperiod),]
  joined_hassurv_timeperiod$dead_timeperiod <- joined_hassurv_timeperiod$dead & joined_hassurv_timeperiod$time_death_fu<=timeperiod
  print(paste('Evaluable patients: ',nrow(joined_hassurv_timeperiod)))
  print('Stanford model')
  print(auc(joined_hassurv_timeperiod$dead_timeperiod, -joined_hassurv_timeperiod$pred_12mo_surv))
  print(ci.auc(joined_hassurv_timeperiod$dead_timeperiod, -joined_hassurv_timeperiod$pred_12mo_surv))
  print('Epic EOLCI')
  print(auc(joined_hassurv_timeperiod$dead_timeperiod, joined_hassurv_timeperiod$END_OF_LIFE_CARE_INDEX_SCORE))
  print(ci.auc(joined_hassurv_timeperiod$dead_timeperiod, joined_hassurv_timeperiod$END_OF_LIFE_CARE_INDEX_SCORE))
  
  svg(filename = paste(fig_dir,'roc_',timeperiod,'day_surv_2021-2022_data.svg',sep=''), width = 5.5, height = 5)
  my_roc <- plot(roc(joined_hassurv_timeperiod$dead_timeperiod, -joined_hassurv_timeperiod$pred_12mo_surv),col=cbPalette[2],asp=NA,main=paste(timeperiod,' day survival',sep=''))
  my_roc <- plot(roc(joined_hassurv_timeperiod$dead_timeperiod, joined_hassurv_timeperiod$END_OF_LIFE_CARE_INDEX_SCORE),col=cbPalette[3],add=T)
  legend("bottomright",
         legend=c("Stanford cancer model","Epic EOLCI"),
         col=c(cbPalette[2], cbPalette[3]),
         lty=1,
         cex=1)
  dev.off()
}

#Positive predictive value of Epic-specified high risk group
summary(joined_hassurv_timeperiod$dead_timeperiod)
summary(joined_hassurv_timeperiod$END_OF_LIFE_CARE_INDEX_SCORE>=45)
summary(joined_hassurv_timeperiod$dead_timeperiod[joined_hassurv_timeperiod$END_OF_LIFE_CARE_INDEX_SCORE>=45])
sum(joined_hassurv_timeperiod$dead_timeperiod[joined_hassurv_timeperiod$END_OF_LIFE_CARE_INDEX_SCORE>=45])/sum(joined_hassurv_timeperiod$END_OF_LIFE_CARE_INDEX_SCORE>=45)
summary(joined_hassurv_timeperiod$dead_timeperiod[joined_hassurv_timeperiod$END_OF_LIFE_CARE_INDEX_SCORE<45])
sum(joined_hassurv_timeperiod$dead_timeperiod[joined_hassurv_timeperiod$END_OF_LIFE_CARE_INDEX_SCORE<45])/sum(joined_hassurv_timeperiod$END_OF_LIFE_CARE_INDEX_SCORE<45)

#PPV of Stanford high risk group
temp <- sort(joined_hassurv_timeperiod$pred_12mo_risk)
cutpoint <- temp[1283-484] #pick cut point so that there are the same # of Stanford model high-risk patients as Epic model high-risk patients
stanford_highrisk <- joined_hassurv_timeperiod[joined_hassurv_timeperiod$pred_12mo_risk>cutpoint,]
summary(stanford_highrisk$dead)

#Stanford model: most common features influencing survival
split_string_on_plus_or_minus <- function(input_string) {
  substrings <- c()
  current_substring <- ""
  for (i in seq_along(strsplit(input_string, "")[[1]])) {
    char <- substr(input_string, i, i)
    if (char == "+" || char == "-") {
      if (current_substring != "") {
        substrings <- c(substrings, current_substring)
      }
      current_substring <- char
    } else {
      current_substring <- paste0(current_substring, char)
    }
  }
  if (current_substring != "") {
    substrings <- c(substrings, current_substring)
  }
  return(substrings)
}

result <- joined_hassurv %>%
  mutate(splits = lapply(improve_surv_feats, split_string_on_plus_or_minus)) %>%
  unnest(splits)

common_improve_surv <- result$splits
common_improve_surv <- common_improve_surv[common_improve_surv != " "]
common_improve_surv <- trimws(common_improve_surv)
common_improve_surv_frame <- data.frame(common_improve_surv)
colnames(common_improve_surv_frame) <- 'term'
most_common_improve_terms <- common_improve_surv_frame %>% count(term, sort=T)

result <- joined_hassurv %>%
  mutate(splits = lapply(worsen_surv_feats, split_string_on_plus_or_minus)) %>%
  unnest(splits)

common_worsen_surv <- result$splits
common_worsen_surv <- common_worsen_surv[common_worsen_surv != " "]
common_worsen_surv <- trimws(common_worsen_surv)
common_worsen_surv_frame <- data.frame(common_worsen_surv)
colnames(common_worsen_surv_frame) <- 'term'
most_common_worsen_terms <- common_worsen_surv_frame %>% count(term, sort=T)
