# Generate simulated patient data, process data, and export to R for further analysis
# Outputs:
#   visits.csv:   CSV file that lists date of each visit for each patient, and time to death or last follow-up
#   text_labs.h5: HDF5 file with array of features for each visit
#   coefs.csv:    CSV file that lists logistic regression survival model coefficients for each note text term 
#
# Tested with Python 3.5.2
#
# Author: Michael Gensheimer, 10/11/2017
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
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

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
n_pal_rt_study_pts = 100  #100 simulated patients in palliative radiation study
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
pall_rt_study_id = np.random.choice(n_pts,size=n_pal_rt_study_pts,replace=False) # randomly assign some patients to be in the palliative radiation study
notes = notes[notes.visit_date >= notes.earliest_mets] # only examine visits from after the date of diagnosis of metastatic cancer 
notes = notes[notes.days_to_last_contact_or_death >= 0] # exclude notes from after the death date
pt_info=pt_info[pt_info.patient_id.isin(notes.patient_id)] #only include patients with at least 1 note
notes['has_fu'] = notes.days_to_last_contact_or_death>0 # does patient have any follow-up data
notes['include_surv_3mo'] = (notes.has_fu & notes.dead) | (notes.days_to_last_contact_or_death>90) # include in 3 month survival analysis
notes['surv_3mo'] = ~notes.dead | (notes.days_to_last_contact_or_death>90) # did pt survive for 3 months or more

#Note text processing
#remove patient name from notes
dask.set_options(get=dask.multiprocessing.get)
df=dd.from_pandas(notes,npartitions=30)
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

df=dd.from_pandas(notes.note_text,npartitions=30)
result=df.apply(n_to_w)
notes.note_text=result.compute()
df=0
result=0

#parse notes using Spacy NLP. Restrict to most common 100 words (when using real data, this is 20,000 words)
n_common_words=100
en_nlp = spacy.load('en')
def count_words(inSeries):
  counts=np.zeros(4000000)
  for doc in en_nlp.pipe(inSeries,n_threads=24,batch_size=1000):
  #for doc in en_nlp.pipe(inSeries.values.astype(unicode),n_threads=1,batch_size=1000):
    doc_array=doc.to_array([attrs.LEMMA,attrs.IS_ALPHA])
    counts[doc_array[doc_array[:,1]==True,0]]=counts[doc_array[doc_array[:,1]==True,0]]+1
  return counts

most_common=np.argsort(-count_words(notes.note_text))[:n_common_words]
most_common_words=np.array(list(map(lambda x: en_nlp.vocab[int(x)].lower_,most_common)))

def parse_notes(inSeries):
  outList=list()
  counter=0
  for doc in en_nlp.pipe(inSeries,n_threads=24,batch_size=5000):
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
#Ensure that all palliative radiation study patients are in test set  
train_prop=0.7
valid_prop=0.15
test_prop=0.15
pts_all=pd.Series(notes.patient_id[notes.has_fu==True].unique())
pts_pall_rt_study=pts_all[pts_all.isin(pall_rt_study_id)]
pts_not_pall_rt_study=pts_all[~pts_all.isin(pts_pall_rt_study)]
pts_test_subset2=pts_not_pall_rt_study.sample(n=int(len(pts_all)*test_prop)-len(pts_pall_rt_study),random_state=4)
pts_test=pd.concat([pts_pall_rt_study,pts_test_subset2])
pts_train=pts_all[~pts_all.isin(pts_test)].sample(frac=train_prop/(1-test_prop),random_state=1)
pts_validate=pts_all[~pts_all.isin(pts_train) & ~pts_all.isin(pts_test)]
pts_train_validate=pd.concat([pts_train,pts_validate])
notes.loc[:,'set']=2 #0=train, 1=validate, 2=test
notes.loc[notes.patient_id.isin(pts_train),'set']=0
notes.loc[notes.patient_id.isin(pts_validate),'set']=1

#Find influence of note terms on survival using term frequency/inverse document frequency and logistic regression 
max_features=500 #100,000 when using real patient data
C=0.1 #0.5 when using real patient data
max_df=1.0
train_sample = notes.index[notes.patient_id.isin(pts_train) & notes.include_surv_3mo].values
validate_sample = notes.index[notes.patient_id.isin(pts_validate) & notes.include_surv_3mo].values
tfidf_vect = TfidfVectorizer(max_features=max_features,ngram_range=(1, 2),max_df=max_df)
tfidf_vect.fit(notes.loc[train_sample].note_text_parsed)
all_tfidf=tfidf_vect.transform(notes.note_text_parsed) #sparse matrix with term frequency/inverse document frequency vector for each note
model = LogisticRegression(C=C)
model=model.fit(all_tfidf[train_sample,:],notes.loc[train_sample].surv_3mo)
train_auc=metrics.roc_auc_score(notes.loc[train_sample].surv_3mo,model.predict_proba(all_tfidf[train_sample,:])[:,1])
validate_auc=metrics.roc_auc_score(notes.loc[validate_sample].surv_3mo,model.predict_proba(all_tfidf[validate_sample,:])[:,1])
print(train_auc,validate_auc) # AUC for training set shold be around 0.67, AUC for validation set should be around 0.65
coefs=pd.DataFrame({'feature' : tfidf_vect.get_feature_names(), 'coef' : model.coef_.squeeze()})
coefs.sort_values(by='coef').to_csv(output_dir+'coefs.csv') # negative coefficient means shorter survival when term more frequent

#SVD dimensionality reduction to reduce number of features
svd_components=100
svd=TruncatedSVD(n_components=svd_components)
svd.fit(all_tfidf[train_sample,:])
all_svd=svd.transform(all_tfidf)

#Find patient visits to analyze (a visit is defined as a date the patient had one more notes written) 
visits = pd.DataFrame(notes.loc[notes.patient_id.isin(pts_all),:].groupby('patient_id',as_index=True).visit_date.apply(lambda x: pd.Series(x.sort_values().unique())))
visits=visits.reset_index(drop=False)
visits=pd.merge(visits,pt_info,on='patient_id')
visits.loc[:,'days_to_last_contact_or_death'] = (visits.loc[:,'date_last_contact_or_death']-visits.loc[:,'visit_date']) /  np.timedelta64(1, 'D')
visits['has_fu'] = visits.days_to_last_contact_or_death > 0
visits['include_surv_3mo'] = (visits.dead &  visits.has_fu) | (visits.date_last_contact_or_death-visits.visit_date>datetime.timedelta(days=90))
visits['surv_3mo'] = ~visits.dead | (visits.date_last_contact_or_death-visits.visit_date>datetime.timedelta(days=90))
visits.loc[:,'train_3mo']=visits.patient_id.isin(pts_train) & (visits.include_surv_3mo==True)
visits.loc[:,'valid_3mo']=visits.patient_id.isin(pts_validate) & (visits.include_surv_3mo==True)
visits.loc[:,'set']=3 #0=train, 1=validate, 2=test #3=NA d/t no follow-up information
visits.loc[visits.patient_id.isin(pts_train),'set']=0
visits.loc[visits.patient_id.isin(pts_validate),'set']=1
visits.loc[visits.patient_id.isin(pts_test),'set']=2
visits.loc[~visits.has_fu,'set']=3

#Exponential time decay feature weighting using SVD
visits_tfidf=np.zeros([len(visits),svd_components], dtype='float32')
half_life=30
max_days_back=half_life*3
tfidf_index=0
for which_pt, dates in visits.groupby('patient_id'):
  this_pt_notes = copy.copy(notes.loc[notes.patient_id==which_pt,:])
  for visit_date in dates.visit_date:
    in_window=(this_pt_notes.visit_date<=visit_date) & (this_pt_notes.visit_date>visit_date - np.timedelta64(max_days_back,'D'))
    daysback=(visit_date-this_pt_notes.loc[in_window,'visit_date']) / np.timedelta64(1, 'D')
    weight=pow(2,-daysback/half_life)
    weight=weight/sum(weight)
    notes_in_window=this_pt_notes[(this_pt_notes.visit_date<=visit_date) & (this_pt_notes.visit_date>visit_date - np.timedelta64(max_days_back,'D'))].index.values
    visits_tfidf[tfidf_index,:]=weight.values.dot(all_svd[notes_in_window,:])
    tfidf_index=tfidf_index+1

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

#Find most recent labs at time of each note and do exponential time decay weighting
visits_labs=np.zeros([len(visits),len(lab_ids)], dtype='float32')
visits_labs_index=0
for which_pt, dates in visits.groupby('patient_id'):
  if which_pt in labs_vitals_bypt:
    this_pt_labs = copy.copy(labs_vitals_bypt[which_pt])
    for visit_date in dates.visit_date:
      in_window=(this_pt_labs.date<=visit_date) & (this_pt_labs.date>visit_date - np.timedelta64(max_days_back,'D'))
      for component_id, lab_frame in this_pt_labs.loc[in_window,:].groupby('component_id'):
        daysback=(visit_date-lab_frame.date) / np.timedelta64(1, 'D')        
        weight=pow(2,-daysback/half_life)
        weight=weight/sum(weight)
        weighted_lab=weight.values.dot(lab_frame.value_standardized)
        visits_labs[visits_labs_index,lab_dict[component_id]]=weighted_lab
      visits_labs_index=visits_labs_index+1
  else:
    visits_labs_index=visits_labs_index+len(dates)

text_labs=np.concatenate((visits_tfidf,visits_labs),axis=1) #First 100 dimensions are note text terms (transformed by SVD); next 5 dimensions are vital signs; final 50 dimensions are labs
scaler = StandardScaler().fit(text_labs[visits.set==0,:]) 
text_labs=scaler.transform(text_labs) #standardize data to unit variance and 0 mean

#export data to R

(visits*1).to_csv(output_dir+'visits.csv',date_format='%Y-%m-%d')
store_export = pd.HDFStore(output_dir+'text_labs.h5')
store_export.append('text_labs', pd.DataFrame(text_labs))
store_export.close()
