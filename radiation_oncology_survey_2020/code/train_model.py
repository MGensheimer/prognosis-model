import pandas as pd
import numpy as np
from numpy.random import seed
import os
import json
import pdb
import multiprocessing
import subprocess
import gc
import datetime
import random
import logging
from sys import stdout

from process_data import *

patient_set='cba20200114'
test_chang=1 #if test_chang=1, held-out test set is only Chang study patients
model_name='chang1'

base_dir=''
data_dir=base_dir+'data/'
config_dir=base_dir+'config/'
output_dir=base_dir+'output/'
first_epic=pd.Timestamp(2008,2,29)
num_cpus = min(16,multiprocessing.cpu_count()-2)

#####
#configure logger
logger = logging.getLogger('Model training')
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler(stdout)
f_handler = logging.FileHandler(output_dir+'models/'+model_name+'/train_log.txt')
formatter = logging.Formatter('%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s')
c_handler.setFormatter(formatter)
f_handler.setFormatter(formatter)
logger.addHandler(c_handler)
logger.addHandler(f_handler)

#####
#load config file

with open(config_dir+'data_devel_'+patient_set+'.json') as config_file:
  config = json.load(config_file)
  config['patient_set']=patient_set

with open(config_dir+'model_config_'+model_name+'.json') as config_file:
  model_config = json.load(config_file)
  model_config['model_name']=model_name

#####
#load data

logger.info('Starting model training')
logger.info('Patient set: %s',config['patient_set'])
logger.info('Model name: %s',model_config['model_name'])
logger.info('Loading database tables')

pt_data={}
for table_name, table_details in config['tables'].items():
  if table_details['process']==True:
    pt_data[table_name] = pd.read_parquet(data_dir+'devel/'+config['patient_set']+'/'+table_name+'_devel.parquet')
    if 'PatientID' in pt_data[table_name].columns:
      pt_data[table_name].rename(columns={'PatientID':'patient_id'},inplace=True)
    logger.info('Table name: '+table_name+'   Length: '+str(len(pt_data[table_name])))
  else:
    logger.info('Skipping table '+table_name)

#####
#data preprocessing

if ('fu_info' in config['tables']) and config['tables']['fu_info']['process']:
  pt_data['fu_info'] = pt_data['fu_info'].groupby('patient_id',as_index=False).first() #fu_info sometimes has duplicate patients

pt_data['pts_scirdb'].drop('PATIENT_ID',axis=1,inplace=True)
pt_data['pts_scirdb'].rename(columns={'EPIC_ID':'patient_id'},inplace=True)
pt_data['pts_scirdb']=pt_data['pts_scirdb'][~pt_data['pts_scirdb'].LFU_DATE.isna()]
pt_data['pts_scirdb']=pt_data['pts_scirdb'][pt_data['pts_scirdb'].LFU_DATE>first_epic]
pt_data['pts_scirdb']=pt_data['pts_scirdb'][pt_data['pts_scirdb'].LFU_DATE>pt_data['pts_scirdb'].EARLIEST_METS]
pt_data['pts_scirdb']=pt_data['pts_scirdb'].groupby('patient_id').first().reset_index() #remove a few duplicate patients
pt_data['pts_scirdb']['dead']=pt_data['pts_scirdb'].DEATH_DATE.notnull()
lfu_after_death = pt_data['pts_scirdb'].LFU_DATE>pt_data['pts_scirdb'].DEATH_DATE
pt_data['pts_scirdb'].loc[lfu_after_death,'LFU_DATE']=pt_data['pts_scirdb'].loc[lfu_after_death,'DEATH_DATE']

pts_w_labs=pd.merge(pt_data['pts_scirdb'],pt_data['labs'],on='patient_id').query('result_time>=EARLIEST_METS').patient_id.unique()
pts_w_notes=pd.merge(pt_data['pts_scirdb'],pt_data['notes'],on='patient_id').query('pat_enc_contact_date>=EARLIEST_METS').patient_id.unique()
pts_w_proc=pd.merge(pt_data['pts_scirdb'],pt_data['proc'],on='patient_id').query('ordering_date>=EARLIEST_METS').patient_id.unique()
pts_include=set(pts_w_labs).intersection(set(pts_w_proc)).intersection(set(pts_w_notes))
if 0:
  random.seed(0)
  pts_include=set(random.sample(pts_include,100))

for table_name, table_data in pt_data.items():
  if table_name!='patients':
    pt_data[table_name] = table_data[table_data.patient_id.isin(pts_include)]

include_imp = ('impressions' in config['tables']) and config['tables']['impressions']['process']
logger.info('Parsing notes')
parse_notes(pt_data, data_dir, include_imp=include_imp)
logger.info('Done parsing notes')

pt_data['notes'] = pd.merge(pt_data['notes'],pt_data['pts_scirdb'],on='patient_id')
pt_data['notes'] = pt_data['notes'][(pt_data['notes'].pat_enc_contact_date < pt_data['notes'].DEATH_DATE) | ~pt_data['notes'].dead]
pt_data['notes'] = pt_data['notes'][(pt_data['notes'].note_last_update_dttm < pt_data['notes'].DEATH_DATE) | ~pt_data['notes'].dead]
pt_data['notes'] = pt_data['notes'][pt_data['notes'].pat_enc_contact_date >= pt_data['notes'].EARLIEST_METS-np.timedelta64(365, 'D')]
pt_data['notes'] = pt_data['notes'][pt_data['notes'].note_last_update_dttm>=pt_data['notes'].pat_enc_contact_date]

pt_data['notes'].loc[:,'fu_days'] = pt_data['notes'].LFU_DATE-pt_data['notes'].note_last_update_dttm
pt_data['notes'].fu_days=pt_data['notes'].fu_days /  np.timedelta64(1, 'D')
pt_data['notes'] = pt_data['notes'][pt_data['notes'].fu_days >= 1]
pt_data['notes']=pt_data['notes'].reset_index(drop=True) #make index consecutive so can easily map notes dataframe to feature arrays later  

pt_data['notes']['include_surv_12mo'] = (pt_data['notes'].note_last_update_dttm >= pt_data['notes'].EARLIEST_METS) & (pt_data['notes'].dead | (pt_data['notes'].LFU_DATE-pt_data['notes'].note_last_update_dttm>datetime.timedelta(days=365)))
pt_data['notes']['surv_12mo']         = ~pt_data['notes'].dead | (pt_data['notes'].DEATH_DATE-pt_data['notes'].note_last_update_dttm>datetime.timedelta(days=365))

#####
#divide randomly into train/test sets on patient level

pts_all=pd.Series(pt_data['notes'].patient_id.unique())
if test_chang:
  chang_pts = pd.read_excel(data_dir+'misc/chang_outcomes_w_mrn0s.xlsx', dtype={'mrn_full':str})
  chang_pts.rename(columns={'mrn_full':'MRN_FULL'},inplace=True)
  chang_pts = pd.merge(chang_pts,pt_data['pts_scirdb'].loc[:,['patient_id','MRN_FULL']],on='MRN_FULL')
  pts_chang=pts_all.loc[pts_all.isin(chang_pts.patient_id)]
  pts_train_test=pts_all[~pts_all.isin(pts_chang)]
  pts_train=pts_train_test.sample(n=int(len(pts_train_test)*0.8),random_state=1)
  pts_test=pts_train_test[~pts_train_test.isin(pts_train)]
  pt_data['notes'].loc[:,'set']=2 #0=train, 2=test/validate, 3=chang (final held-out test set)
  pt_data['notes'].loc[pt_data['notes'].patient_id.isin(pts_train),'set']=0
  pt_data['notes'].loc[pt_data['notes'].patient_id.isin(pts_chang),'set']=3
  train_sample = pt_data['notes'].index[(pt_data['notes'].set==0) & pt_data['notes'].include_surv_12mo].values
  test_sample = pt_data['notes'].index[(pt_data['notes'].set==2) & pt_data['notes'].include_surv_12mo].values
else:
  test_prop=0.2
  pts_test=pts_all.sample(n=int(len(pts_all)*test_prop),random_state=1)
  pts_train=pts_all[~pts_all.isin(pts_test)]
  pt_data['notes'].loc[:,'set']=2 #0=train, 2=test
  pt_data['notes'].loc[pt_data['notes'].patient_id.isin(pts_train),'set']=0
  train_sample = pt_data['notes'].index[(pt_data['notes'].set==0) & pt_data['notes'].include_surv_12mo].values
  test_sample = pt_data['notes'].index[(pt_data['notes'].set==2) & pt_data['notes'].include_surv_12mo].values


#####
#use lasso to do variable selection on note terms
max_features=100000
count_vect = CountVectorizer(max_features=max_features,ngram_range=(1, 2))
count_vect.fit(pt_data['notes'].loc[train_sample].note_text)
all_counts=count_vect.transform(pt_data['notes'].note_text)
all_counts = scipy.sparse.csc_matrix(all_counts).astype('float64')
note_sums = scipy.sparse.csc_matrix(pt_data['notes'].word_count).transpose().astype('float64')
all_counts = all_counts.multiply(note_sums.power(-1))
feature_sums = scipy.sparse.csc_matrix(all_counts.sum(axis=0))
all_counts = all_counts.multiply(feature_sums.power(-1))
all_counts = scipy.sparse.csc_matrix(all_counts)

from glmnet_py import glmnet
from glmnetPrint import glmnetPrint; from glmnetCoef import glmnetCoef; from glmnetPredict import glmnetPredict

fit = glmnet(x = all_counts[train_sample,:], y = pt_data['notes'].loc[train_sample].surv_12mo.values*1.0, family = 'binomial', alpha=1.0)

chunksize=1000
num_s = fit['lambdau'].shape[0]
predictions = np.zeros([len(test_sample), num_s])
for i in range(int(len(test_sample)/chunksize)): #looping avoids MemoryError
  predictions[(i*chunksize):((i+1)*chunksize),:] = glmnetPredict(fit, all_counts[test_sample[(i*chunksize):((i+1)*chunksize)],:], ptype = 'response')

predictions[((i+1)*chunksize):,:] = glmnetPredict(fit, all_counts[test_sample[((i+1)*chunksize):],:], ptype = 'response')

for i in range(num_s):
  test_auc=metrics.roc_auc_score(pt_data['notes'].loc[test_sample].surv_12mo,predictions[:,i])
  print(i, fit['lambdau'][i], fit['df'][i], test_auc)

best_lambda_i=30
coefs=glmnetCoef(fit, s=scipy.float64([fit['lambdau'][best_lambda_i]]))[1:].flatten()
features=pd.DataFrame({'feature' : count_vect.get_feature_names(), 'coef' : coefs})
features.loc[:,'coef_abs'] = abs(features.coef)
features_sorted = features.sort_values(by='coef_abs',ascending=False)
features_sorted=features_sorted.reset_index()

selected_terms = features.feature.loc[abs(coefs)>0].tolist()
selected_terms_frame=pd.DataFrame({'feature' : selected_terms})
selected_terms_frame.loc[:,'exclude']=0
selected_terms_frame.to_csv(output_dir+'models/'+model_config['model_name']+'/text_features_unedited.csv',index=False)

#now, manually remove any undesirable features (physician names, city names, etc.) and save as text_features_edited.csv


#####
#find landmark times

landmark_gap = 365./2
landmarks = np.arange(0,365*5+0.1,landmark_gap)
t_hor = 365*5 #administrative censoring time horizon (see Putter slides)

#find t0 for each patient (date of first note on or after date of metastatic diagnosis)
t0 = pt_data['notes'].query('note_last_update_dttm>=EARLIEST_METS').groupby('patient_id',as_index=False)['note_last_update_dttm'].min()
landmarks_pot_list = []
for landmark in landmarks:
  landmarks_pot_list.append(pd.DataFrame({'patient_id':t0.patient_id, 'landmark':landmark, 'eval_date':t0.note_last_update_dttm+datetime.timedelta(days=landmark)}))

landmarks_pot = pd.concat(landmarks_pot_list)
landmarks_pot = landmarks_pot.merge(pt_data['notes'], on='patient_id')
#only include landmark time points that have at least one note/visit within landmark_gap prior to time point:
landmarks_pot = (landmarks_pot[(landmarks_pot.note_last_update_dttm<=landmarks_pot.eval_date)
  & (landmarks_pot.note_last_update_dttm>landmarks_pot.eval_date-datetime.timedelta(days=landmark_gap))
  & (landmarks_pot.eval_date < landmarks_pot.LFU_DATE)
  ])
pt_landmarks = landmarks_pot.loc[:,['patient_id','landmark','eval_date','dead','LFU_DATE']].drop_duplicates().reset_index(drop=True)
pt_landmarks.loc[:,'fu_days'] = (pt_landmarks.LFU_DATE-pt_landmarks.eval_date)
pt_landmarks.fu_days=pt_landmarks.fu_days /  np.timedelta64(1, 'D')
pt_landmarks = pt_landmarks.merge(pt_data['fu_info'].loc[:,['patient_id','dob','sex']],on='patient_id')
pt_landmarks['age'] = (pt_landmarks.eval_date-pt_landmarks.dob)/np.timedelta64(365, 'D')
pt_landmarks = pt_landmarks[pt_landmarks.age>=18] #exclude children
pt_landmarks.loc[:,'date_int'] = ((pt_landmarks.eval_date-pd.Timestamp(2000,1,1)) / np.timedelta64(1, 'D')).astype('int32')


#####
#automatically pick most important labs/vitals using lasso

labs = pt_data['labs'].copy(deep=True)
labs.drop_duplicates(inplace=True)
labs=labs.groupby('component_id').filter(lambda x: len(x)>100)
temp1 = labs.loc[:,['component_id','component_name']].drop_duplicates().reset_index(drop=True).sort_values(by='component_id')
temp2 = pd.DataFrame({'component_id':[-5,-4,-3,-2,-1], 'component_name':['bp_diastolic','bp_systolic','pulse','temperature','weight_kg']}) #add in vital signs
lab_features=pd.concat([temp2,temp1],ignore_index=True)
selected_labs = lab_features.component_id.tolist()
lab_dict=dict(zip(selected_labs,range(0,len(selected_labs)))) #dictionary maps from component ID to index
labs_vitals_bypt = labs_vitals_combine(pt_data, selected_labs).groupby('patient_id')
landmark_labs_vitals = labs_vitals_combine_weight(labs_vitals_bypt, lab_dict, pt_landmarks.copy(deep=True))
if 0:
  np.save(data_dir+'temp/landmark_labs_vitals.npy', landmark_labs_vitals)
  landmark_labs_vitals=np.load(data_dir+'temp/landmark_labs_vitals.npy')

feature_means = np.nanmean(landmark_labs_vitals[pt_landmarks.patient_id.isin(pts_train),:], axis=0)
feature_std = np.nanstd(landmark_labs_vitals[pt_landmarks.patient_id.isin(pts_train),:], axis=0)
feature_include = feature_std != 0 #if standard deviation is 0, then all values are the same and feature should not be used
landmark_labs_vitals_scaled = standardize_data_and_impute_mean(landmark_labs_vitals,feature_means,feature_std,feature_include)
lab_features_after_exclude=lab_features.iloc[feature_include]

pt_landmarks.reset_index(inplace=True)
pt_landmarks['include_surv_12mo'] = pt_landmarks.dead | (pt_landmarks.fu_days>=365)
pt_landmarks['surv_12mo']         = ~pt_landmarks.dead | (pt_landmarks.fu_days>=365)
train_sample = pt_landmarks.index[pt_landmarks.patient_id.isin(pts_train) & pt_landmarks.include_surv_12mo].values
test_sample = pt_landmarks.index[pt_landmarks.patient_id.isin(pts_test) & pt_landmarks.include_surv_12mo].values

from glmnet_py import glmnet
from glmnetPrint import glmnetPrint; from glmnetCoef import glmnetCoef; from glmnetPredict import glmnetPredict

fit = glmnet(x = landmark_labs_vitals_scaled[train_sample,:].astype('float64'), y = pt_landmarks.loc[train_sample].surv_12mo.values*1.0, family = 'binomial', alpha=1.0)

chunksize=1000
num_s = fit['lambdau'].shape[0]
predictions = np.zeros([len(test_sample), num_s])
for i in range(int(len(test_sample)/chunksize)): #looping avoids MemoryError
  predictions[(i*chunksize):((i+1)*chunksize),:] = glmnetPredict(fit, landmark_labs_vitals_scaled[test_sample[(i*chunksize):((i+1)*chunksize)],:], ptype = 'response')

predictions[((i+1)*chunksize):,:] = glmnetPredict(fit, landmark_labs_vitals_scaled[test_sample[((i+1)*chunksize):],:], ptype = 'response')

best_lambda_i = -1
best_auc = -1
for i in range(num_s):
  test_auc=metrics.roc_auc_score(pt_landmarks.loc[test_sample].surv_12mo,predictions[:,i])
  if test_auc > best_auc:
    best_lambda_i=i
    best_auc=test_auc
  print(i, fit['lambdau'][i], fit['df'][i], test_auc)

coefs=glmnetCoef(fit, s=scipy.float64([fit['lambdau'][best_lambda_i]]))[1:].flatten()
features=pd.DataFrame({'feature' : lab_features_after_exclude.component_id, 'feature_desc':lab_features_after_exclude.component_name,'coef' : coefs})
features.loc[:,'coef_abs'] = abs(features.coef)
features_sorted = features.sort_values(by='coef_abs',ascending=False)
features_sorted=features_sorted.reset_index()
selected_terms = features.loc[abs(features.coef)>0,['feature','feature_desc']]
selected_terms.to_csv(output_dir+'models/'+model_config['model_name']+'/labsvitals_features.csv',index=False)

#####
#pick most important diagnoses, procedures, medications using lasso
pt_landmarks.reset_index(inplace=True)
pt_landmarks['include_surv_12mo'] = pt_landmarks.dead | (pt_landmarks.fu_days>=365)
pt_landmarks['surv_12mo']         = ~pt_landmarks.dead | (pt_landmarks.fu_days>=365)
train_sample = pt_landmarks.index[pt_landmarks.patient_id.isin(pts_train) & pt_landmarks.include_surv_12mo].values
test_sample = pt_landmarks.index[pt_landmarks.patient_id.isin(pts_test) & pt_landmarks.include_surv_12mo].values

diag_proc_medi, item_dict = diag_proc_medi_combine(pt_data, include_hosp=1)
diag_proc_medi=diag_proc_medi.groupby('feature').filter(lambda x: len(x)>100)
diag_proc_medi_bypt = sortedcontainers.SortedDict({pt:group for (pt, group) in diag_proc_medi.groupby('patient_id')})
diag_proc_medi_array=diag_proc_medi_weight(pt_landmarks, diag_proc_medi_bypt, len(item_dict),
  sparse_output=True, timeperiod_time=[182],timeperiod_weight=[1.])

from glmnet_py import glmnet
from glmnetPrint import glmnetPrint; from glmnetCoef import glmnetCoef; from glmnetPredict import glmnetPredict

fit_diag_proc_medi = glmnet(x = diag_proc_medi_array[train_sample,:], y = pt_landmarks.loc[train_sample].surv_12mo.values*1.0, family = 'binomial', alpha=1.0)

chunksize=1000
num_s = fit_diag_proc_medi['lambdau'].shape[0]
predictions = np.zeros([len(test_sample), num_s])
for i in range(int(len(test_sample)/chunksize)): #looping avoids MemoryError
  predictions[(i*chunksize):((i+1)*chunksize),:] = glmnetPredict(fit_diag_proc_medi, diag_proc_medi_array[test_sample[(i*chunksize):((i+1)*chunksize)],:], ptype = 'response')

predictions[((i+1)*chunksize):,:] = glmnetPredict(fit_diag_proc_medi, diag_proc_medi_array[test_sample[((i+1)*chunksize):],:], ptype = 'response')

best_lambda_i = -1
best_auc = -1
for i in range(num_s):
  test_auc=metrics.roc_auc_score(pt_landmarks.loc[test_sample].surv_12mo,predictions[:,i])
  if test_auc > best_auc:
    best_lambda_i=i
    best_auc=test_auc
  print(i, fit_diag_proc_medi['lambdau'][i], fit_diag_proc_medi['df'][i], test_auc)

coefs=glmnetCoef(fit_diag_proc_medi, s=scipy.float64([fit_diag_proc_medi['lambdau'][best_lambda_i]]))[1:].flatten()
features=pd.DataFrame({'feature':list(item_dict.values()),'feature_desc' : list(item_dict.keys()), 'coef' : coefs})
features.loc[:,'coef_abs'] = abs(features.coef)
features_sorted = features.sort_values(by='coef_abs',ascending=False)
features_sorted=features_sorted.reset_index()
selected_terms = features.loc[abs(features.coef)>0,['feature_desc']]
selected_terms.to_csv(output_dir+'models/'+model_config['model_name']+'/diagprocmedi_features.csv',index=False)

#####
#create landmark data

if 0:
  pt_landmarks_copy = copy.copy(pt_landmarks)
  random.seed(0)
  pt_landmarks = pt_landmarks_copy.sample(1000)
  pt_landmarks.sort_values(by=['patient_id','eval_date'], inplace=True)

landmark_data, feature_names = make_feat_array(pt_data, config, model_config, pt_landmarks.loc[:,['patient_id','eval_date','date_int']], config_dir, data_dir, output_dir)

pt_landmarks_train = pt_landmarks.loc[pt_landmarks.patient_id.isin(pts_train),:]
pt_landmarks_train.reset_index(inplace=True,drop=True)
pt_landmarks_test = pt_landmarks.loc[pt_landmarks.patient_id.isin(pts_test),:]
pt_landmarks_test.reset_index(inplace=True,drop=True)
landmark_data_train = landmark_data[pt_landmarks.patient_id.isin(pts_train),:]
landmark_data_test = landmark_data[pt_landmarks.patient_id.isin(pts_test),:]

feature_means = np.nanmean(landmark_data_train, axis=0)
feature_std = np.nanstd(landmark_data_train, axis=0)
feature_include = feature_std != 0 #if standard deviation is 0, then all values are the same and feature should not be used

output_full_dir=output_dir+'models/'+model_config['model_name']
if not os.path.exists(output_full_dir):
    os.makedirs(output_full_dir)

np.save(output_dir+'models/'+model_config['model_name']+'/feature_means.npy', feature_means)
np.save(output_dir+'models/'+model_config['model_name']+'/feature_std.npy', feature_std)
np.save(output_dir+'models/'+model_config['model_name']+'/feature_include.npy', feature_include)

for data_source, data_source_details in model_config['data_sources'].items():
  if data_source_details['process']:
    np.save(output_dir+'models/'+model_config['model_name']+'/feature_means_'+data_source+'.npy', feature_means[feature_names.feature_category==data_source])
    np.save(output_dir+'models/'+model_config['model_name']+'/feature_std_'+data_source+'.npy', feature_std[feature_names.feature_category==data_source])
    np.save(output_dir+'models/'+model_config['model_name']+'/feature_include_'+data_source+'.npy', feature_include[feature_names.feature_category==data_source])

feature_names['feature_means'] = feature_means
feature_names['feature_std'] = feature_std
feature_names['feature_include'] = feature_include
feature_names.to_csv(output_dir+'models/'+model_config['model_name']+'/feature_names.csv',index=False)

landmark_data_scaled = standardize_data_and_impute_mean(landmark_data,feature_means,feature_std,feature_include)
landmark_data_train_scaled = standardize_data_and_impute_mean(landmark_data_train,feature_means,feature_std,feature_include)
landmark_data_test_scaled = standardize_data_and_impute_mean(landmark_data_test,feature_means,feature_std,feature_include)

logger.info('Patients in train set: ' + str(len(pts_train)))
logger.info('Landmark dates in train set: ' + str(len(pt_landmarks_train)))
logger.info('Patients in test set: ' + str(len(pts_test)))
logger.info('Landmark dates in test set: ' + str(len(pt_landmarks_test)))
logger.info('Number of features before features with all 0s excluded: ' + str(len(feature_names)))
logger.info('Number of features after features with all 0s excluded: ' + str(landmark_data_train_scaled.shape[1]))


#####
#train model

from nnet_survival import *
import tensorflow.keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, LSTM, GRU, Embedding, Concatenate, Conv1D, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization, TimeDistributed
from tensorflow.keras import optimizers, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from scipy import stats
from lifelines.utils import concordance_index

breaks=np.array(model_config['breaks'])
n_intervals=len(breaks)-1
timegap = breaks[1:] - breaks[:-1]

def train_surv_model(input_x, input_y, l2, verbose=0, random_seed=0):
  seed(random_seed)
  tensorflow.random.set_seed(random_seed)
  model = Sequential()
  model.add(Dense(1, input_dim=input_x.shape[1],bias_initializer='zeros', kernel_regularizer=regularizers.l2(l2)))
  model.add(Dense(4))
  model.add(Activation('relu'))
  model.add(Dense(n_intervals))
  model.add(Activation('sigmoid'))
  model.compile(loss=surv_likelihood(n_intervals), optimizer=optimizers.RMSprop())
  early_stopping = EarlyStopping(monitor='loss', patience=5)
  history=model.fit(input_x, input_y, batch_size=256, epochs=100000, callbacks=[early_stopping],verbose=verbose)
  return model

#cross-validation to pick regularization

from sklearn.model_selection import GroupKFold
n_folds = 5
l2_array=[0.01, 0.1, 1., 10.]
gkf = GroupKFold(n_splits=n_folds)
early_stopping = EarlyStopping(monitor='loss', patience=20)
grid_search_train_loss = np.zeros((len(l2_array),n_folds))
grid_search_test_loss = np.zeros((len(l2_array),n_folds))
grid_search_train_cindex = np.zeros((len(l2_array),n_folds))
grid_search_test_cindex = np.zeros((len(l2_array),n_folds))
cv_folds = gkf.split(landmark_data_train_scaled, groups=pt_landmarks_train.patient_id.values)
y_train = make_surv_array(pt_landmarks_train.fu_days.values,pt_landmarks_train.dead.values,breaks)
j=0
for traincv, testcv in cv_folds:
  print(str(j+1) + '/' + str(n_folds))
  x_train_cv = landmark_data_train_scaled[traincv,:]
  y_train_cv = y_train[traincv,:]
  x_test_cv = landmark_data_train_scaled[testcv,:]
  y_test_cv = y_train[testcv,:]
  for i in range(len(l2_array)):
    model = train_surv_model(input_x=x_train_cv, input_y=y_train_cv, l2=l2_array[i])
    grid_search_train_loss[i,j] = model.evaluate(x_train_cv,y_train_cv,verbose=0)
    grid_search_test_loss[i,j] = model.evaluate(x_test_cv,y_test_cv,verbose=0)
    y_pred=model.predict_proba(x_train_cv,verbose=0)
    oneyr_surv=np.cumprod(y_pred[:,0:np.nonzero(breaks>=365)[0][0]], axis=1)[:,-1]
    grid_search_train_cindex[i,j] = concordance_index(pt_landmarks_train.loc[traincv,'fu_days'],oneyr_surv,pt_landmarks_train.loc[traincv,'dead'])
    y_pred=model.predict_proba(x_test_cv,verbose=0)
    oneyr_surv=np.cumprod(y_pred[:,0:np.nonzero(breaks>=365)[0][0]], axis=1)[:,-1]
    grid_search_test_cindex[i,j] = concordance_index(pt_landmarks_train.loc[testcv,'fu_days'],oneyr_surv,pt_landmarks_train.loc[testcv,'dead'])
  j=j+1

print(np.average(grid_search_train_loss,axis=1))
print(np.average(grid_search_test_loss,axis=1))
print(np.average(grid_search_train_cindex,axis=1))
print(np.average(grid_search_test_cindex,axis=1))
l2_final = l2_array[np.argmax(-np.average(grid_search_test_loss,axis=1))]

#final model using train data
y_train = make_surv_array(pt_landmarks_train.fu_days.values,pt_landmarks_train.dead.values,breaks)
y_test = make_surv_array(pt_landmarks_test.fu_days.values,pt_landmarks_test.dead.values,breaks)

model = train_surv_model(input_x=landmark_data_train_scaled, input_y=y_train, l2=0.1)

with open(output_dir+'models/'+model_config['model_name']+'/model_survival_trainset_arch.json', "w") as text_file:
    text_file.write("%s" % model.to_json())

model.save_weights(output_dir+'models/'+model_config['model_name']+'/model_survival_trainset_weights.h5')


#test set loss
print(model.evaluate(landmark_data_test_scaled, y_test, verbose=0))

#Discrimination performance
y_pred=model.predict_proba(landmark_data_train_scaled,verbose=0)
oneyr_surv=np.cumprod(y_pred[:,0:np.nonzero(breaks>=365)[0][0]], axis=1)[:,-1]
logger.info('Train set C-index: ' + str(concordance_index(pt_landmarks_train.fu_days,oneyr_surv,pt_landmarks_train.dead)))

y_pred=model.predict_proba(landmark_data_test_scaled,verbose=0)
oneyr_surv=np.cumprod(y_pred[:,0:np.nonzero(breaks>=365)[0][0]], axis=1)[:,-1]
logger.info('Test set C-index: ' + str(concordance_index(pt_landmarks_test.fu_days,oneyr_surv,pt_landmarks_test.dead)))

for landmark in landmarks:
  y_pred=model.predict_proba(landmark_data_test_scaled[pt_landmarks_test.landmark==landmark,:],verbose=0)
  oneyr_surv=np.cumprod(y_pred[:,0:np.nonzero(breaks>=365)[0][0]], axis=1)[:,-1]
  logger.info('Landmark %s %s %s',landmark,oneyr_surv.shape[0],concordance_index(pt_landmarks_test.fu_days[pt_landmarks_test.landmark==landmark],
    oneyr_surv,pt_landmarks_test.dead[pt_landmarks_test.landmark==landmark]))

#partial models for sanity check
landmark_data_copy=copy.copy(landmark_data)
for data_source, data_source_details in model_config['data_sources'].items():
  if data_source_details['process']:
    landmark_data_partial=landmark_data[:,feature_names.feature_category==data_source]
    landmark_data_partial_train = landmark_data_partial[pt_landmarks.patient_id.isin(pts_train),:]
    landmark_data_partial_test = landmark_data_partial[pt_landmarks.patient_id.isin(pts_test),:]
    feature_means_partial = np.nanmean(landmark_data_partial_train, axis=0)
    feature_std_partial = np.nanstd(landmark_data_partial_train, axis=0)
    feature_include_partial = feature_std_partial != 0
    landmark_data_partial_train_scaled = standardize_data_and_impute_mean(landmark_data_partial_train,feature_means_partial,feature_std_partial,feature_include_partial)
    landmark_data_partial_test_scaled = standardize_data_and_impute_mean(landmark_data_partial_test,feature_means_partial,feature_std_partial,feature_include_partial)
    model_partial=train_surv_model(input_x=landmark_data_partial_train_scaled, input_y=y_train, l2=0.1, verbose=0)
    y_pred=model_partial.predict_proba(landmark_data_partial_test_scaled,verbose=0)
    oneyr_surv=np.cumprod(y_pred[:,0:np.nonzero(breaks>=365)[0][0]], axis=1)[:,-1]
    logger.info('Model trained with just %s: C-index %f',data_source,concordance_index(pt_landmarks_test.fu_days,oneyr_surv,pt_landmarks_test.dead))

#Calibration plots

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def calib_plot(fu_time, n_bins, pred_surv, time, dead, color, label, error_bars=0,alpha=1., markersize=1., markertype='o'):
  cuts = np.concatenate((np.array([-1e6]),np.percentile(pred_surv, np.arange(100/n_bins,100,100/n_bins)),np.array([1e6])))
  bin = pd.cut(pred_surv,cuts,labels=False)
  kmf = KaplanMeierFitter()
  est = []
  ci_upper = []
  ci_lower = []
  mean_pred_surv = []
  for which_bin in range(max(bin)+1):
    kmf.fit(time[bin==which_bin], event_observed=dead[bin==which_bin])
    est.append(np.interp(fu_time, kmf.survival_function_.index.values, kmf.survival_function_.KM_estimate))
    ci_upper.append(np.interp(fu_time, kmf.survival_function_.index.values, kmf.confidence_interval_.loc[:,'KM_estimate_upper_0.95']))
    ci_lower.append(np.interp(fu_time, kmf.survival_function_.index.values, kmf.confidence_interval_.loc[:,'KM_estimate_lower_0.95']))
    mean_pred_surv.append(np.mean(pred_surv[bin==which_bin]))
  est = np.array(est)
  ci_upper = np.array(ci_upper)
  ci_lower = np.array(ci_lower)
  if error_bars:
    plt.errorbar(mean_pred_surv, est, yerr = np.transpose(np.column_stack((est-ci_lower,ci_upper-est))), fmt='o',c=color,label=label)
  else:
    plt.plot(mean_pred_surv, est, markertype, c=color,label=label, alpha=alpha, markersize=markersize)
  return (mean_pred_surv, est)

def calib_plot_km(fu_time_break, n_bins, pred_surv, time, dead):
  cuts = np.concatenate((np.array([-1e6]),np.percentile(pred_surv[:,fu_time_break], np.arange(100/n_bins,100,100/n_bins)),np.array([1e6])))
  bin = pd.cut(pred_surv[:,fu_time_break],cuts,labels=False)
  kmf = KaplanMeierFitter()
  actual = []
  predicted = []
  for which_bin in range(max(bin)+1):
    kmf.fit(time[bin==which_bin], event_observed=dead[bin==which_bin])
    plt.plot(kmf.survival_function_.index.values, kmf.survival_function_.KM_estimate,ls='--',c='C'+str(which_bin))
    bin_pred_surv=np.mean(pred_surv[bin==which_bin,:], axis=0)
    plt.plot(breaks,bin_pred_surv,ls='-',c='C'+str(which_bin))

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

#Calibration plot at specific follow-up time
n_bins = 10
my_alpha = 0.7
my_markersize = 5.
fu_time_array = np.array([0.5, 1, 3])*365.
fu_time_label_array = ['6 months', '1 year', '3 years']

for landmark in landmarks:
  _ = plt.figure(figsize=(4.5,10))
  _ = plt.clf()
  for fu_time_i in range(len(fu_time_array)):
    fu_time = fu_time_array[fu_time_i]
    _ = plt.subplot(3, 1, 1+fu_time_i)
    _ = plt.plot([0,1], [0,1], ls="--", c=".7")
    pred_surv = nnet_pred_surv(model.predict_proba(landmark_data_test_scaled[pt_landmarks_test.landmark==landmark,:],verbose=0), breaks, fu_time)
    (pred, actual)=calib_plot(fu_time, n_bins, pred_surv,pt_landmarks_test.fu_days[pt_landmarks_test.landmark==landmark].values, pt_landmarks_test.dead[pt_landmarks_test.landmark==landmark].values,
      CB_color_cycle[1],'Nnet-survival', alpha=my_alpha, markersize=my_markersize, markertype='o')
    _ = plt.xlim([0,1])
    _ = plt.ylim([0,1])
    _ = plt.legend()
    _ = plt.xlabel('Predicted survival rate')
    _ = plt.ylabel('Actual survival rate')
    _ = plt.title(fu_time_label_array[fu_time_i])
  _ = plt.tight_layout()
  #plt.show()
  _ = plt.savefig(output_dir+'models/'+model_config['model_name']+'/calib_'+str(landmark)+'.pdf')
  _ = plt.close('all')


#calibration plot with Kaplan-Meier curves
days_plot = 365*5
for landmark in landmarks:
  y_pred=model.predict_proba(landmark_data_test_scaled[pt_landmarks_test.landmark==landmark,:],verbose=0)
  pred_surv=np.concatenate((np.ones((y_pred.shape[0],1)),np.cumprod(y_pred,axis=1)),axis=1)
  matplotlib.style.use('default')
  _ = plt.figure(figsize=(8,6))
  _ = plt.clf()
  calib_plot_km(fu_time_break=5,n_bins=4,pred_surv=pred_surv,time=pt_landmarks_test.fu_days[pt_landmarks_test.landmark==landmark].values,
    dead=pt_landmarks_test.dead[pt_landmarks_test.landmark==landmark].values)
  _ = plt.xticks(np.arange(0, days_plot+0.0001, 365/4))
  _ = plt.yticks(np.arange(0, 1.0001, 0.125))
  _ = plt.xlim([0,days_plot])
  _ = plt.ylim([0,1])
  _ = plt.xlabel('Follow-up time (days)')
  _ = plt.ylabel('Proportion surviving')
  _ = plt.savefig(output_dir+'models/'+model_config['model_name']+'/calib_km_'+str(landmark)+'.pdf')
  _ = plt.close('all')
