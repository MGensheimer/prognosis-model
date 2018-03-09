# Generate simulated patient data, process data, and export to R for further analysis
# Outputs:
#   visits.feather: File that lists date of each visit for each patient, and time to death or last follow-up
#   text.h5: HDF5 file with array of note text features for each visit
#   text_features.csv: List of the note text terms that were selected by lasso
#   labsvitals.h5: HDF5 file with array of labs/vitals features for each visit
#   diag_proc_medi.h5: HDF5 file with array of diagnosis/procedure/medication features for each visit
#
# Tested with Python 3.5.2
#
# Author: Michael Gensheimer, Stanford University, Mar. 8, 2018
# michael.gensheimer@gmail.com

import pandas as pd
import numpy as np
import datetime
import time
import copy
import re
import dask
import dask.dataframe as dd
import spacy
from spacy import attrs
import sortedcontainers
import glob
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import feather
import itertools
import tables

np.random.seed(0)

#Populate notes dataframe with data from IMDB sentiment analysis dataset
#The data can be downloaded from: http://ai.stanford.edu/~amaas/data/sentiment/ 

data_dir = '/home/michael/prognosis/code/github/aclImdb/train/' #change this to the directory where you extracted the IMDB sentiment analysis training files
output_dir = '/home/michael/prognosis/code/github/python_output/' #change this to your preferred output directory
files = glob.glob(data_dir+'/neg/*.txt')
neg_reviews = []
for filename in files:
  file = open(filename)
  neg_reviews.append(file.read())

files = glob.glob(data_dir+'/pos/*.txt')
pos_reviews = []
for filename in files:
  file = open(filename)
  pos_reviews.append(file.read())

n_pts = 1000  #1000 simulated patients
n_notes=25000 #25000 simulated notes

#Create simulated distributions of death and censoring times. The first half of the patients have a good prognosis, the second half have a poor prognosis. 
time = np.floor(np.random.exponential(365*4/np.log(2),int(n_pts/2)))
censtime = np.floor(np.random.exponential((365*3)/np.log(2),int(n_pts/2)))
event_goodprog = time<censtime
t_goodprog = np.minimum(time,censtime)
time = np.floor(np.random.exponential(365*.5/np.log(2),int(n_pts/2)))
censtime = np.floor(np.random.exponential((365*3)/np.log(2),int(n_pts/2)))
event_poorprog = time<censtime
t_poorprog = np.minimum(time,censtime)
event = np.concatenate((event_goodprog,event_poorprog))
t = np.concatenate((t_goodprog,t_poorprog))

pt_info = pd.DataFrame({
  'patient_id' : range(n_pts),
  'dead' : event,
  'date_last_contact_or_death' : pd.to_timedelta(t,unit='d')+pd.Timestamp(datetime.date(2008,1,2)),
  'dob' : [ pd.Timestamp(datetime.date(1950,1,1)) ] * n_pts
  })

notes_list = []
for i in range(n_pts): #generate several notes for each patient
  time_from_2008 = (pt_info.loc[i,'date_last_contact_or_death'] - pd.Timestamp(datetime.date(2008,1,1))) / np.timedelta64(1,'D')
  notes_temp = pd.DataFrame({
    'patient_id' : [i] * int(n_notes/n_pts), 
    'firstname' : ['Jane'] * int(n_notes/n_pts), #give all patients the sane name
    'lastname' : ['Smith'] * int(n_notes/n_pts),
    'visit_date' : pd.to_timedelta(np.random.randint(0,time_from_2008,int(n_notes/n_pts)),unit='d')+pd.Timestamp(datetime.date(2008,1,1)), #random date from 1/1/2008 to this patient's date of last contact or death
    'earliest_mets' : [pd.Timestamp(datetime.date(2005,1,1))] * int(n_notes/n_pts) #all patients diagnosed with metastatic disease on 1/1/2005
    })
  notes_temp.loc[:,'days_to_last_contact_or_death'] = (pt_info.loc[i,'date_last_contact_or_death'] - notes_temp.visit_date) / np.timedelta64(1,'D')
  notes_list.append(notes_temp)

notes=pd.concat(notes_list)
notes.loc[:,'note_text'] = pos_reviews + neg_reviews # assign positive text to the good prognosis patients, and negative text to the poor prognosis patients
notes=pd.merge(notes,pt_info,on='patient_id')
notes = notes[notes.visit_date >= notes.earliest_mets] # only examine visits from after the date of diagnosis of metastatic cancer 
notes = notes[notes.days_to_last_contact_or_death >= 0] # exclude notes from after the death date
pt_info=pt_info[pt_info.patient_id.isin(notes.patient_id)] #only include patients with at least 1 note
notes['has_fu'] = notes.days_to_last_contact_or_death>0 # does patient have any follow-up data
notes['include_surv_12mo'] = (notes.has_fu & notes.dead) | (notes.days_to_last_contact_or_death>365) # include in 12 month survival analysis
notes['surv_12mo'] = ~notes.dead | (notes.days_to_last_contact_or_death>365) # did pt survive for 12 months or more

#Note text processing
#remove patient name from notes
dask.set_options(get=dask.multiprocessing.get)
df=dd.from_pandas(notes,npartitions=4)
result=df.apply(lambda x: re.sub(x.firstname+'|'+x.lastname,'',x.note_text,flags=re.IGNORECASE),axis=1) 
notes.note_text=result.compute()

def n_to_w(inString):
  digits=['0','1','2','3','4','5','6','7','8','9',r'1\d','2\d','3\d','4\d','5\d','6\d','7\d','8\d','9\d','1\d\d']
  digit_words=['zero','one','two','three','four','five','six','seven','eight','nine','ten','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety','onehundred']
  inString=re.sub(r'\.\d+','',inString) #remove digits after decimal point
  inString=re.sub(r'(?<=[\d]),(?=[\d])','',inString) #remove commas/periods in between digits to make sure this function does not misinterpret the first part of 3.2 or 3,200 as the number 3 
  for i in range(len(digits)):
    inString=re.sub(r'(?<=[^\d])'+digits[i]+r'(?=[^\d])',digit_words[i],inString)
  inString=re.sub(r'\d','',inString) #remove numbers > 199
  return inString

df=dd.from_pandas(notes.note_text,npartitions=4)
result=df.apply(n_to_w)
notes.note_text=result.compute()
df=0
result=0

#parse notes using Spacy NLP. Restrict to most common 1000 words (when using real data, this is 20,000 words)
n_common_words=1000
en_nlp = spacy.load('en')
def count_words(inSeries):
  counts=np.zeros(4000000)
  for doc in en_nlp.pipe(inSeries,n_threads=4,batch_size=1000):
  #for doc in en_nlp.pipe(inSeries.values.astype(unicode),n_threads=1,batch_size=1000):
    doc_array=doc.to_array([attrs.LEMMA,attrs.IS_ALPHA])
    counts[doc_array[doc_array[:,1]==True,0]]=counts[doc_array[doc_array[:,1]==True,0]]+1
  return counts

most_common=np.argsort(-count_words(notes.note_text))[:n_common_words]
most_common_words=np.array(list(map(lambda x: en_nlp.vocab[int(x)].lower_,most_common)))

def parse_notes(inSeries):
  outList=list()
  counter=0
  for doc in en_nlp.pipe(inSeries,n_threads=4,batch_size=5000):
    doc_array=doc.to_array([attrs.LEMMA])
    doc_array_clean=doc_array[np.in1d(doc_array,most_common)]
    outList.append(str.join(' ',list(map(lambda x: en_nlp.vocab.strings[int(x)],doc_array_clean))))
    counter=counter+1
  return outList

temp=parse_notes(notes.note_text)
notes.loc[:,'note_text_parsed']=' ' 
chunksize=1000
for i in range(int(len(notes)/chunksize)): #looping avoids MemoryError
  notes.loc[notes.index[i*chunksize:(i+1)*chunksize],'note_text_parsed']=temp[i*chunksize:(i+1)*chunksize]

notes.loc[notes.index[(i+1)*chunksize:],'note_text_parsed']=temp[(i+1)*chunksize:] #last chunk
temp=0
notes=notes.reset_index() #make index consecutive so can easily map notes dataframe to feature arrays later  

#Create random train/validate/test split. Assign 70% of pts to train set, 15% to validation set, 15% to test set.
train_prop=0.7
valid_prop=0.15
test_prop=0.15
pts_all=pd.Series(notes.patient_id[notes.has_fu==True].unique())
pts_test=pts_all.sample(n=int(len(pts_all)*test_prop),random_state=4)
pts_train=pts_all[~pts_all.isin(pts_test)].sample(frac=train_prop/(1-test_prop),random_state=1)
pts_validate=pts_all[~pts_all.isin(pts_train) & ~pts_all.isin(pts_test)]
notes.loc[:,'set']=2 #0=train, 1=validate, 2=test
notes.loc[notes.patient_id.isin(pts_train),'set']=0
notes.loc[notes.patient_id.isin(pts_validate),'set']=1
train_sample = notes.index[notes.patient_id.isin(pts_train) & notes.include_surv_12mo].values #for GLMNET note term variable selection
validate_sample = notes.index[notes.patient_id.isin(pts_validate) & notes.include_surv_12mo].values

#GLMNET for selection of most useful note text terms
max_features=10000
tfidf_vect = TfidfVectorizer(max_features=max_features,ngram_range=(1, 2))
tfidf_vect.fit(notes.loc[train_sample].note_text_parsed)
all_tfidf=tfidf_vect.transform(notes.note_text_parsed)
all_tfidf = scipy.sparse.csc_matrix(all_tfidf)

from glmnet_py import glmnet; from glmnetPlot import glmnetPlot 
from glmnetPrint import glmnetPrint; from glmnetCoef import glmnetCoef; from glmnetPredict import glmnetPredict
fit = glmnet(x = all_tfidf[train_sample,:], y = notes.loc[train_sample].surv_12mo.values*1.0, family = 'binomial', alpha=1.0)

chunksize=1000
num_s = fit['lambdau'].shape[0]
predictions = np.zeros([len(validate_sample), num_s])
for i in range(int(len(validate_sample)/chunksize)): #looping avoids MemoryError when using real dataset
  predictions[i*chunksize:(i+1)*chunksize,:] = glmnetPredict(fit, all_tfidf[validate_sample[i*chunksize:(i+1)*chunksize],:], ptype = 'response')
predictions[(i+1)*chunksize:,:] = glmnetPredict(fit, all_tfidf[validate_sample[(i+1)*chunksize:],:], ptype = 'response')

print('Lambda_index Lambda #vars ValidationAUC')
for i in range(num_s):
  validate_auc=metrics.roc_auc_score(notes.loc[validate_sample].surv_12mo,predictions[:,i])
  print(i, fit['lambdau'][i], fit['df'][i], validate_auc)

#best lambda is index 19, with validation AUC around 0.70
#save the selected note text terms
coefs=glmnetCoef(fit, s=scipy.float64([fit['lambdau'][19]]))[1:].flatten()
temp=tfidf_vect.get_feature_names()
temp2=list(itertools.compress(temp,abs(coefs)>0))
features=pd.DataFrame({'feature' : temp2})
features.to_csv(output_dir+'text_features.csv',index=False)

all_tfidf_lasso = all_tfidf[:,abs(coefs)>0].toarray().astype('float32')
visits_tfidf_lasso=np.zeros([len(notes),all_tfidf_lasso.shape[1]], dtype='float32')
half_life=30 #influence of notes decays over time with this half-life
tfidf_index=0
for which_pt, dates in notes.groupby('patient_id'):
  this_pt_notes = copy.copy(notes.loc[notes.patient_id==which_pt,:])
  for visit_date in dates.visit_date:
    in_window=this_pt_notes.visit_date<=visit_date
    daysback=(visit_date-this_pt_notes.loc[in_window,'visit_date']) / np.timedelta64(1, 'D')
    weight=pow(2,-daysback/half_life)
    weight=weight/sum(weight)
    notes_in_window=this_pt_notes[this_pt_notes.visit_date<=visit_date].index.values
    visits_tfidf_lasso[tfidf_index,:]=weight.values.dot(all_tfidf_lasso[notes_in_window,:])
    tfidf_index=tfidf_index+1
    if tfidf_index % 40000 == 0:
      print(float(tfidf_index)/len(notes))

#export GLMNET to R
h5file = tables.open_file(output_dir+'text.h5', mode='w')
data_storage = h5file.create_array(h5file.root, 'visits_tfidf_lasso', visits_tfidf_lasso)
h5file.close()

visits = notes.copy()
visits.drop(['note_text_parsed', 'note_text'],axis=1,inplace=True)
feather.write_dataframe(visits, output_dir+'visits.feather')

#Process laboratory data
n_labs = 100000
num_lab_cats = 50 #only analyze most common labs 
labs = pd.DataFrame({ #simulated lab data
  'patient_id' : np.random.randint(0,n_pts,n_labs),
  'component_id' : 5+np.floor(np.random.exponential(100/np.log(2),n_labs)), #component IDs 0-4 are reserved for vital signs
  'value' : np.random.normal(size=n_labs),
  'date' : pd.to_timedelta(np.random.randint(0,365.25*9,n_labs),unit='d')+pd.Timestamp(datetime.date(2008,1,1)), #random date from 2008 to 2016
  })
  
n_vitals=50000
vitals = pd.DataFrame({ #simulated vital signs data
  'patient_id' : np.random.randint(0,n_pts,n_vitals),
  'component_id' : np.random.randint(5,size=n_vitals), #5 vital signs are recorded: systolic BP, diastolic BP, pulse, temperature, weight
  'value' : np.random.normal(size=n_vitals),
  'date' : pd.to_timedelta(np.random.randint(0,365.25*9,n_vitals),unit='d')+pd.Timestamp(datetime.date(2008,1,1)), #random date from 2008 to 2016
  })

labs_grouped=labs.groupby('component_id')
components=labs_grouped.size().sort_values(ascending=False)
labs=labs.loc[labs.component_id.isin(components.index[0:num_lab_cats]),:] #only analyze most common types of labs
labs_vitals=pd.concat([labs,vitals],ignore_index=True)

labs_train=labs_vitals.loc[labs_vitals.patient_id.isin(pts_train),:]
labs_train_grouped=labs_train.groupby('component_id')
labs_vitals_stats=labs_train_grouped['value'].agg(['std','mean'])
for i in labs_vitals.index.values:
  labs_vitals.at[i,'value_standardized']=(labs_vitals.at[i,'value']-labs_vitals_stats.at[labs_vitals.at[i,'component_id'],'mean'])/labs_vitals_stats.at[labs_vitals.at[i,'component_id'],'std']

lab_ids=labs_vitals.component_id.sort_values().unique() #list of component IDs
lab_dict=dict(zip(lab_ids,range(0,len(lab_ids)))) #dictionary maps from component ID to index
labs_vitals_bypt = sortedcontainers.SortedDict({k:copy.copy(labs_vitals.loc[labs_vitals.patient_id==k,:]) for k in labs_vitals.patient_id.unique()})

#Find most recent labs at time of each visit and do exponential time decay weighting
visits_labs=np.zeros([len(visits),len(lab_ids)], dtype='float32')
visits_labs_index=0
for which_pt, dates in visits.groupby('patient_id'):
  if which_pt in labs_vitals_bypt:
    this_pt_labs = copy.copy(labs_vitals_bypt[which_pt])
    for visit_date in dates.visit_date:
      in_window=this_pt_labs.date<=visit_date
      for component_id, lab_frame in this_pt_labs.loc[in_window,:].groupby('component_id'):
        daysback=(visit_date-lab_frame.date) / np.timedelta64(1, 'D')        
        weight=pow(2,-daysback/half_life)
        weight=weight/sum(weight)
        weighted_lab=weight.values.dot(lab_frame.value_standardized)
        visits_labs[visits_labs_index,lab_dict[component_id]]=weighted_lab
      visits_labs_index=visits_labs_index+1
  else:
    visits_labs_index=visits_labs_index+len(dates)

#export labs/vitals data to R
h5file = tables.open_file(output_dir+'labsvitals.h5', mode='w')
data_storage = h5file.create_array(h5file.root, 'visits_labs', visits_labs)
h5file.close()


#procedures, diagnoses, and medications
n_diagnoses=10000
n_procedures=10000
n_medi=10000
features=50
diagnoses = pd.DataFrame({ #simulated data
  'patient_id' : np.random.randint(0,n_pts,n_diagnoses),
  'feature_name' : np.random.randint(0, features, size=n_diagnoses),
  'item_date' : pd.to_timedelta(np.random.randint(0,365.25*9,n_diagnoses),unit='d')+pd.Timestamp(datetime.date(2008,1,1)), #random date from 2008 to 2016
  })
procedures = pd.DataFrame({ #simulated data
  'patient_id' : np.random.randint(0,n_pts,n_procedures),
  'feature_name' : features+np.random.randint(0, features, size=n_procedures),
  'item_date' : pd.to_timedelta(np.random.randint(0,365.25*9,n_procedures),unit='d')+pd.Timestamp(datetime.date(2008,1,1)), #random date from 2008 to 2016
  })
medi = pd.DataFrame({ #simulated data
  'patient_id' : np.random.randint(0,n_pts,n_medi),
  'feature_name' : 2*features+np.random.randint(0, features, size=n_medi),
  'item_date' : pd.to_timedelta(np.random.randint(0,365.25*9,n_medi),unit='d')+pd.Timestamp(datetime.date(2008,1,1)), #random date from 2008 to 2016
  })

#for each patient, only keep earliest instance of each diagnosis
diagnoses = diagnoses.groupby(['patient_id','item_date'],as_index=False).first()

#combine diagnoses, procedures, medications
diag_proc_medi = pd.concat([diagnoses,procedures,medi])
item_names=diag_proc_medi.feature_name.sort_values().unique()
item_dict=dict(zip(item_names,range(0,len(item_names))))
diag_proc_medi.loc[:,'feature'] = diag_proc_medi.feature_name.apply(lambda x: item_dict[x])
diag_proc_medi.sort_values(by=['patient_id', 'item_date'], inplace=True)
diag_proc_medi.reset_index(inplace=True,drop=True)
diag_proc_medi.loc[:,'date_int'] = ((diag_proc_medi.item_date-datetime.date(2000,1,1)) / np.timedelta64(1, 'D')).astype('int32')
visits.loc[:,'date_int'] = ((visits.visit_date-datetime.date(2000,1,1)) / np.timedelta64(1, 'D')).astype('int32')

diag_proc_medi_bypt = sortedcontainers.SortedDict({k:copy.copy(diag_proc_medi.loc[diag_proc_medi.patient_id==k,:]) for k in diag_proc_medi.patient_id.unique()})

timeperiod_time = [999999, 120, 60, 30] # visits within this many days of the datapoint are affected
timeperiod_weight = [0.125, 0.25, 0.5, 1.0] # weight of feature depending on time gap from datapoint to visit
n_timeperiods=len(timeperiod_time)
diag_proc_medi_array=np.zeros([len(visits),len(item_names)], dtype='float32')
data_index=0
for which_pt, dates in visits.groupby('patient_id'):
  if which_pt in diag_proc_medi_bypt:
    for i in range(len(diag_proc_medi_bypt[which_pt])):
      for j in range(n_timeperiods):
        within_timeperiod = np.flatnonzero( ((dates.date_int-diag_proc_medi_bypt[which_pt].date_int.iloc[i]>=0) & (dates.date_int-diag_proc_medi_bypt[which_pt].date_int.iat[i]<=timeperiod_time[j])) )
        diag_proc_medi_array[data_index+within_timeperiod, diag_proc_medi_bypt[which_pt].feature.iat[i]] = timeperiod_weight[j]
  data_index=data_index+len(dates)

h5file = tables.open_file(output_dir+'diag_proc_medi.h5', mode='w')
data_storage = h5file.create_array(h5file.root, 'diag_proc_medi', diag_proc_medi_array)
h5file.close()
