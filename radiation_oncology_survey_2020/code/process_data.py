import pandas as pd
import pickle
import gzip
import datetime
import numpy as np
import string
import time
import copy
import re
import dask
import dask.dataframe as dd
import math
import sortedcontainers
import scipy.sparse
import os
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import scale
import gc
import multiprocessing
import subprocess
import pdb

code_dir=''
num_cpus = min(16,multiprocessing.cpu_count()-2)

def work(chunk):
  return subprocess.call('python '+code_dir+'lemmatize.py '+str(chunk), shell=True)

def parse_notes(pt_data, data_dir, include_imp=1):
  if include_imp:
    impressions=pt_data['impressions'].copy()
    impressions=impressions.loc[:,['patient_id','result_time','impression']]
    impressions.rename(columns={'result_time':'pat_enc_contact_date','impression':'note_text'},inplace=True)
    impressions['note_last_update_dttm']=impressions.pat_enc_contact_date
    impressions['department_name']='impression'
    pt_data['notes']=pd.concat([pt_data['notes'],impressions], ignore_index=True, sort=False)

  pt_data['notes'] = pt_data['notes'][~pt_data['notes'].pat_enc_contact_date.isna()]
  pt_data['notes'] = pt_data['notes'][~pt_data['notes'].note_last_update_dttm.isna()]
  pt_data['notes'].pat_enc_contact_date=pt_data['notes'].pat_enc_contact_date.dt.floor('D')
  pt_data['notes'].note_last_update_dttm=pt_data['notes'].note_last_update_dttm.dt.floor('D')
  pt_data['notes'] = pt_data['notes'].groupby('patient_id',as_index=True).apply(lambda x: x.drop_duplicates(subset='note_text')) #remove duplicate notes
  pt_data['notes'] = pt_data['notes'].reset_index(drop=True)
  pt_data['notes'] = pt_data['notes'][pt_data['notes'].pat_enc_contact_date > pd.Timestamp(2008,2,29)] #notes before this date are pre-Epic and have unreliable dates

  #concatenate notes from same day that were finalized on same day
  temp = pt_data['notes'].groupby(['patient_id','pat_enc_contact_date','note_last_update_dttm']).note_text.apply(lambda x: x.str.cat(sep=' '))
  temp=temp.reset_index()
  temp2 = pt_data['notes'].groupby(['patient_id','pat_enc_contact_date','note_last_update_dttm'],as_index=False).first()
  temp2 = temp2.drop(['patient_id', 'pat_enc_contact_date', 'note_last_update_dttm','note_text'], axis=1)
  pt_data['notes'] = pd.concat([temp,temp2],axis=1)
  pt_data['notes']=pt_data['notes'].reset_index(drop=True) #make index consecutive so can easily map notes dataframe to feature arrays later  
  temp=0
  temp2=0
  gc.collect()

  #remove patient name from notes
  pt_data['notes'] = pd.merge(pt_data['notes'],pt_data['fu_info'].loc[:,['patient_id','firstname','lastname']],on='patient_id')
  dask.config.set(scheduler='processes')
  df=dd.from_pandas(pt_data['notes'],npartitions=num_cpus)
  result=df.apply(lambda x: re.sub(x.firstname+'|'+x.lastname,'',x.note_text,flags=re.IGNORECASE),axis=1,meta=('string')).compute()
  pt_data['notes'].note_text=result

  def n_to_w(inString):
    digits=['0','1','2','3','4','5','6','7','8','9',r'1\d','2\d','3\d','4\d','5\d','6\d','7\d','8\d','9\d','1\d\d']
    digit_words=['zero','one','two','three','four','five','six','seven','eight','nine','ten','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety','onehundred']
    inString=re.sub(r'\.\d+','',inString) #remove digits after decimal point
    inString=re.sub(r'(?<=[\d]),(?=[\d])','',inString) #remove commas/periods in between digits to make sure this function does not misinterpret the first part of 3.2 or 3,200 as the number 3 
    for i in range(len(digits)):
      inString=re.sub(r'(?<=[^\d])'+digits[i]+r'(?=[^\d])',digit_words[i],inString)
    inString=re.sub(r'\d','',inString) #remove numbers > 199
    return inString

  df=dd.from_pandas(pt_data['notes'].note_text,npartitions=num_cpus)
  result=df.apply(n_to_w,meta=('string'))
  pt_data['notes'].note_text=result.compute()
  df=0
  result=0

  #lemmatize notes
  if len(pt_data['notes']) >= 100000:
    partitions=1000
  elif len(pt_data['notes']) >= 10000:
    partitions=100
  elif len(pt_data['notes']) >= 1000:
    partitions=10
  else:
    partitions=1

  lemmatize_compute=1
  if lemmatize_compute:
    temp=np.array_split(pt_data['notes'].note_text, partitions)
    for i in range(partitions):
      temp[i].to_pickle(data_dir+'temp/raw'+str(i)+'.pkl')
    temp=0
    pool = multiprocessing.Pool(processes=num_cpus)
    result=pool.map(work, range(partitions))
    if max(result)>0:
      print('one of the lemmatize processes had an error')

  lemmatized=[]
  for i in range(partitions):
    lemmatized=lemmatized + pd.read_pickle(data_dir+'temp/lemma'+str(i)+'.pkl').tolist()
  
  gc.collect()
  os.system('rm /home/mgens/cancer_prognosis/data/temp/*')

  chunksize=min(1000,len(pt_data['notes']))
  for i in range(int(len(pt_data['notes'])/chunksize)): #looping avoids MemoryError
    pt_data['notes'].loc[pt_data['notes'].index[i*chunksize:(i+1)*chunksize-1],'note_text']=lemmatized[i*chunksize:(i+1)*chunksize-1]

  pt_data['notes'].loc[pt_data['notes'].index[(i+1)*chunksize:],'note_text']=lemmatized[(i+1)*chunksize:]
  lemmatized=0
  gc.collect()
  pt_data['notes'].loc[:,'word_count'] = pt_data['notes'].note_text.apply(lambda x: x.count(' ')-1)
  pt_data['notes']=pt_data['notes'][pt_data['notes'].word_count>=10]
  pt_data['notes']=pt_data['notes'].reset_index() #make index consecutive so can easily map notes dataframe to feature arrays later  
  pt_data['notes'] = pt_data['notes'].drop(['index'], axis=1)

def notes_to_freq(pt_data, selected_terms):
  count_vect = CountVectorizer(ngram_range=(1,2),vocabulary=dict(zip(selected_terms, range(len(selected_terms)))))
  all_counts=count_vect.transform(pt_data['notes'].note_text)
  all_counts = scipy.sparse.csc_matrix(all_counts).astype('float64')
  note_sums = scipy.sparse.csc_matrix(pt_data['notes'].word_count).transpose().astype('float64')
  all_counts = all_counts.multiply(note_sums.power(-1))
  return all_counts.toarray()

def notes_combine_weight(pt_data, notes_freqs, eval_dates):
#expects eval_dates for each patient to be consecutive (i.e. patient 1's dates must all be in a row, then patient 2's dates)  
  half_life=30
  max_days_back=365
  visit_index=0
  visit_notes_freqs=np.zeros([len(eval_dates),notes_freqs.shape[1]], dtype='float32')
  pt_data['notes']=pt_data['notes'].reset_index(drop=True) #make index consecutive as it will be used in next step
  for which_pt, dates in eval_dates.groupby('patient_id',sort=False):
    this_pt_notes = copy.copy(pt_data['notes'].loc[pt_data['notes'].patient_id==which_pt,:])
    for eval_date in dates.eval_date:
      in_window=(this_pt_notes.note_last_update_dttm<=eval_date) & (this_pt_notes.note_last_update_dttm>eval_date - np.timedelta64(max_days_back,'D'))
      daysback=(eval_date-this_pt_notes.loc[in_window,'note_last_update_dttm']) / np.timedelta64(1, 'D')
      weight=pow(2,-daysback/half_life)
      weight=weight/sum(weight)
      notes_in_window=this_pt_notes[in_window].index.values
      visit_notes_freqs[visit_index,:]=weight.values.dot(notes_freqs[notes_in_window,:])
      visit_index=visit_index+1
      if visit_index % 40000 == 0:
        print(float(visit_index)/len(eval_dates))
  return visit_notes_freqs

def labs_vitals_combine(pt_data, selected_labs):
  temp=pt_data['vitals'][['patient_id','contact_date','bp_systolic','bp_diastolic','temperature','pulse','weight_kg']]
  temp=pd.melt(temp,id_vars=['patient_id','contact_date'],var_name='component_name',value_name='ord_num_value')
  temp.rename(columns={'contact_date':'result_time'},inplace=True)
  temp.loc[temp.component_name=='bp_diastolic','component_id']=-5
  temp.loc[temp.component_name=='bp_systolic','component_id']=-4
  temp.loc[temp.component_name=='pulse','component_id']=-3
  temp.loc[temp.component_name=='temperature','component_id']=-2
  temp.loc[temp.component_name=='weight_kg','component_id']=-1
  temp.loc[:,'component_id']=temp.component_id.astype('int64')
  labs_vitals=pd.concat([pt_data['labs'], temp], ignore_index=True, sort=True)
  labs_vitals=labs_vitals.loc[labs_vitals.component_id.isin(selected_labs),:]
  labs_vitals.result_time=labs_vitals.result_time.dt.normalize()
  labs_vitals.drop_duplicates(inplace=True)
  return labs_vitals

def labs_vitals_combine_weight(labs_vitals_bypt, lab_dict, eval_dates):
#expects eval_dates for each patient to be consecutive (i.e. patient 1's dates must all be in a row, then patient 2's dates)
  half_life=30
  max_days_back=365
  visits_labs=np.zeros([len(eval_dates),len(lab_dict)], dtype='float32')
  visits_labs[:]=np.nan #if patient did not have that lab, will leave that entry as NaN
  visits_labs_index=0
  count=0
  for which_pt, dates in eval_dates.groupby('patient_id',sort=False):
    count=count+1
    if count % 500 == 0:
      print(float(count)/len(eval_dates.patient_id.unique()))
    if which_pt in labs_vitals_bypt.groups.keys():
      this_pt_labs = copy.copy(labs_vitals_bypt.get_group(which_pt))
      for eval_date in dates.eval_date:
        in_window=(this_pt_labs.result_time<=eval_date) & (this_pt_labs.result_time>eval_date - np.timedelta64(max_days_back,'D'))
        for component_id, lab_frame in this_pt_labs.loc[in_window,:].groupby('component_id'):
          daysback=(eval_date-lab_frame.result_time) / np.timedelta64(1, 'D')        
          weight=pow(2,-daysback/half_life)
          weight=weight/sum(weight)
          weighted_lab=weight.values.dot(lab_frame.ord_num_value)
          visits_labs[visits_labs_index,lab_dict[component_id]]=weighted_lab
        visits_labs_index=visits_labs_index+1
    else:
      visits_labs_index=visits_labs_index+len(dates)
  return visits_labs

def diag_proc_medi_combine(pt_data, include_hosp, item_dict=0):
  #can be run with an item_dict supplied, in which case only features in that dict are kept
  #or without item_dict supplied, in which case all features are retained and an item_dict is created
  dx = pt_data['dx_all'].copy(deep=True)
  if 'DiagnosisDTS' in dx.columns:
    dx.rename(columns={'DiagnosisDTS':'item_date'}, inplace=True)
  elif 'diagnosis_date' in dx.columns:
    dx.rename(columns={'diagnosis_date':'item_date'}, inplace=True)
  dx['CurrentICD9ListTXT']=dx.CurrentICD9ListTXT.str.replace(pat=r',.*',repl='') #if multiple diagnoses in entry, remove all but first
  dx.loc[:,'feature_name'] = dx.CurrentICD9ListTXT.apply(lambda x: 'diag_'+x)
  dx = dx.groupby(['patient_id','feature_name'],as_index=False)['item_date'].min() #only use first instance of each diagnosis
  proc = pt_data['proc'].copy(deep=True)
  proc.rename(columns={'ordering_date':'item_date'}, inplace=True)
  proc.loc[:,'feature_name'] = proc.cpt_code.apply(lambda x: 'proc_'+x)
  meds = pt_data['meds'].copy(deep=True)
  meds.rename(columns={'ordering_date':'item_date'}, inplace=True)
  meds = meds.loc[~meds.name.isna()]
  meds.loc[:,'feature_name'] = meds.name.str.replace(' .*', '',regex=True)
  meds.loc[:,'feature_name'] = meds.feature_name.apply(lambda x: 'medi_'+x)
  dx=dx.loc[:,['patient_id', 'item_date','feature_name']]
  proc=proc.loc[:,['patient_id', 'item_date','feature_name']]
  meds=meds.loc[:,['patient_id', 'item_date','feature_name']]
  diag_proc_medi = pd.concat([dx,proc,meds])
  if include_hosp:
    temp = pt_data['admissions'].loc[:,['patient_id','hosp_dischrg_time']]
    temp.rename(columns={'hosp_dischrg_time':'item_date'}, inplace=True)
    temp.loc[:,'feature_name']='hosp_disch'
    diag_proc_medi = pd.concat([diag_proc_medi,temp])
  diag_proc_medi = diag_proc_medi.loc[~diag_proc_medi.item_date.isnull()]
  if item_dict==0:
    item_names=np.sort(diag_proc_medi.feature_name.unique()).tolist()
    item_dict=dict(zip(item_names,range(0,len(item_names))))
  else:
    diag_proc_medi=diag_proc_medi.loc[diag_proc_medi.feature_name.isin(item_dict.keys()),:]
  diag_proc_medi.loc[:,'feature'] = diag_proc_medi.feature_name.apply(lambda x: item_dict[x])
  diag_proc_medi.loc[:,'date_int'] = ((diag_proc_medi.item_date-pd.Timestamp(2000,1,1)) / np.timedelta64(1, 'D')).astype('int32')
  diag_proc_medi.sort_values(by=['patient_id', 'item_date'], inplace=True)
  diag_proc_medi.reset_index(inplace=True, drop=True)
  diag_proc_medi.drop_duplicates(subset=['patient_id','date_int','feature_name'], inplace=True)
  return diag_proc_medi, item_dict

def diag_proc_medi_weight(eval_dates, diag_proc_medi_bypt, n_features, sparse_output=False,
  timeperiod_time = [999999, 120, 60, 30],# visits within this many days of the datapoint are affected
  timeperiod_weight = [0.125, 0.25, 0.5, 1.0]): # weight of feature depending on time gap from datapoint to visit
  n_timeperiods=len(timeperiod_time)
  eval_dates_bypt= [group for _, group in eval_dates.groupby('patient_id',sort=False)]
  if sparse_output:
    diag_proc_medi_array=scipy.sparse.lil_matrix((len(eval_dates),n_features), dtype='float64')
  else:
    diag_proc_medi_array=np.zeros([len(eval_dates),n_features], dtype='float16')
  counter=0
  for pt in range(len(eval_dates_bypt)):
    if pt % 1000 == 0:
      print('patient %s of %s' % (pt, len(eval_dates_bypt)))
    patient_id=eval_dates_bypt[pt].patient_id.iloc[0]
    if (patient_id in diag_proc_medi_bypt) and (len(diag_proc_medi_bypt[patient_id])>0):
      date_array_diag = np.transpose(np.tile(diag_proc_medi_bypt[patient_id].date_int, (len(eval_dates_bypt[pt]),1)))
      date_array_eval = np.tile(eval_dates_bypt[pt].date_int, (len(diag_proc_medi_bypt[patient_id]),1))
      date_diff = date_array_eval-date_array_diag
      feature_array = np.transpose(np.tile(diag_proc_medi_bypt[patient_id].feature, (len(eval_dates_bypt[pt]),1)))
      for j in range(n_timeperiods):
        weight_array= np.logical_and(date_diff>=0, date_diff<=timeperiod_time[j])
        for date_i in range(len(eval_dates_bypt[pt])):
          diag_proc_medi_array[counter+date_i,feature_array[np.nonzero(weight_array[:,date_i])[0],date_i]]=timeperiod_weight[j]
    counter=counter+len(eval_dates_bypt[pt])
  if sparse_output:
    return scipy.sparse.csc_matrix(diag_proc_medi_array)
  else:
    return diag_proc_medi_array

def make_feat_array(pt_data, config, model_config, eval_dates, config_dir, data_dir, output_dir, text_fts_custom=''):
#expects eval_dates for each patient to be consecutive (i.e. patient 1's dates must all be in a row, then patient 2's dates)
  #import logging
  logger = logging.getLogger('Model training')
  if not eval_dates.patient_id.is_monotonic_increasing:
    raise Exception('Patient IDs in eval_dates are not sorted and increasing')
  temp_arrays=[]
  temp_feature_names=[]
  for data_source, data_source_details in model_config['data_sources'].items():
    if data_source_details['process']:
      logger.info('Adding data source to feature array: %s',data_source)
      if data_source=='demographics':
        temp = eval_dates.merge(pt_data['fu_info'].loc[:,['patient_id','dob','sex']],on='patient_id')
        temp['age'] = (temp.eval_date-temp.dob)/np.timedelta64(365, 'D')
        temp_arrays.append(np.transpose(np.array([temp.age, temp.sex=='Female']))) #1=female, 0=male
        temp_feature_names.append(pd.DataFrame({'feature_name':['age','female'], 'feature_category':data_source}))
      if data_source=='dx_proc_medi':
        dx_proc_medi_features = pd.read_csv(output_dir+'models/'+model_config['model_name']+'/diagprocmedi_features.csv')
        item_names=dx_proc_medi_features.feature_desc.tolist()
        item_dict=dict(zip(item_names,range(len(item_names))))
        diag_proc_medi, _ = diag_proc_medi_combine(pt_data, include_hosp=1, item_dict=item_dict)
        diag_proc_medi_bypt = sortedcontainers.SortedDict({pt:group for (pt, group) in diag_proc_medi.groupby('patient_id')})
        landmark_diag_proc_medi = diag_proc_medi_weight(eval_dates, diag_proc_medi_bypt, len(item_dict))
        temp_arrays.append(landmark_diag_proc_medi)
        temp_feature_names.append(pd.DataFrame({'feature_name':item_names, 'feature_category':data_source}))
      if data_source=='labs_vitals':
        lab_features = pd.read_csv(output_dir+'models/'+model_config['model_name']+'/labsvitals_features.csv')
        lab_dict=dict(zip(lab_features.feature.tolist(),range(0,len(lab_features)))) #dictionary maps from component ID to index
        labs_vitals_bypt = labs_vitals_combine(pt_data, lab_features.feature.tolist()).groupby('patient_id')
        landmark_labs_vitals = labs_vitals_combine_weight(labs_vitals_bypt, lab_dict, eval_dates)
        temp_arrays.append(landmark_labs_vitals)
        temp_feature_names.append(pd.DataFrame({'feature_name':lab_features.feature_desc, 'feature_category':data_source}))
      if data_source=='notes':
        selected_terms_frame = pd.read_csv(output_dir+'models/'+model_config['model_name']+'/text_features_edited'+text_fts_custom+'.csv')
        selected_terms_frame = selected_terms_frame.loc[selected_terms_frame.exclude==0]
        notes_freqs = notes_to_freq(pt_data, selected_terms_frame.feature.tolist())
        landmark_notes_freqs = notes_combine_weight(pt_data, notes_freqs, eval_dates)
        temp_arrays.append(landmark_notes_freqs)
        temp_feature_names.append(pd.DataFrame({'feature_name':selected_terms_frame.feature, 'feature_category':data_source}))
  landmark_data=np.concatenate(temp_arrays,axis=1)
  feature_names=pd.concat(temp_feature_names, ignore_index=True)
  logger.info('Finished adding data to feature array.')
  return (landmark_data, feature_names)

def standardize_data_and_impute_mean(data_in, feature_means, feature_std,feature_include):
  data_in_copy = copy.deepcopy(data_in)
  for i in range(len(feature_means)):
    data_in_copy[np.isnan(data_in_copy[:,i]),i] = feature_means[i] #mean imputation
  data_out = (data_in_copy-feature_means)/feature_std
  data_out = np.nan_to_num(data_out)
  data_out = data_out[:,feature_include]
  return data_out

def calc_median_surv(y_pred,breaks):
#y_pred = array with predicted conditional probability of surviving that timepoint
#breaks = array with timepoints for survival estimation, including 0
  interp_times=np.arange(0,breaks[-1]+0.001,1)
  interp_surv=np.interp(interp_times,breaks,np.concatenate(([1],np.cumprod(y_pred))))
  interp_times=np.append(interp_times,999999) #in case patient is predicted to live past last timepoint with prediction
  interp_surv=np.append(interp_surv,0)
  return interp_times[np.argwhere(interp_surv<0.5)[0][0]]

def calc_timepoint_surv(y_pred,breaks,timepoint):
#y_pred = array with predicted conditional probability of surviving that timepoint
#breaks = array with timepoints for survival estimation, including 0
  interp_times=np.arange(0,breaks[-1]+0.001,1)
  interp_surv=np.interp(interp_times,breaks,np.concatenate(([1],np.cumprod(y_pred))))
  interp_times=np.append(interp_times,999999) #in case patient is predicted to live past last timepoint with prediction
  interp_surv=np.append(interp_surv,0)
  return interp_surv[np.argwhere(interp_times>=timepoint)[0][0]]

