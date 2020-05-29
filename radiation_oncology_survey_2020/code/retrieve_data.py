import jaydebeapi
import pandas as pd
import numpy as np
import os
import json
import pdb

patient_set='cba20200114'

base_dir='/home/mgens/cancer_prognosis/'
data_dir=base_dir+'data/'
config_dir=base_dir+'config/'
output_dir=base_dir+'output/'
date_len=[19,27]

os.system('export CLASSPATH=/home/mgens/cancer_prognosis/code')

with open(config_dir+'data_devel_'+patient_set+'.json') as config_file:
  config = json.load(config_file)
  config['patient_set']=patient_set

def get_edw_credentials(): 
    filepath = "/home/mgens/.secrets/.edw_creds"
    with open(filepath, "r") as fd : 
        user, passwd = fd.readline().rstrip().split('/')
    return (user, passwd)

user, passwd = get_edw_credentials()
conn = jaydebeapi.connect('net.sourceforge.jtds.jdbc.Driver', 'jdbc:jtds:sqlserver://redacted/SAM;domain=ENTERPRISE;useLOBs=false', [user, passwd],'/home/mgens/cancer_prognosis/code/jtds-1.2.8.jar')

if not os.path.exists(data_dir+'devel/'+config['patient_set']):
    os.makedirs(data_dir+'devel/'+config['patient_set'])

for table, table_details in config['tables'].items():
	if table_details['download']==True:
		print('downloading table '+table)
		query = 'select * from '+table_details['db_table_name']
		df = pd.read_sql(query, con=conn)
		for column in df.columns:
			not_na_rows = df.loc[~df[column].isna()].index
			if isinstance(df.loc[not_na_rows[0],column],str):
				if len(df.loc[not_na_rows[0],column]) in date_len: #date+time columns are strings w/ specific number of chars
					temp=df[column][~df[column].isna()]
					if ((min(temp.apply(lambda x: len(x))) in date_len) & #now verify the rest of the rows
					(max(temp.apply(lambda x: len(x))) in date_len)):
						temp=temp.apply(lambda x: x[0:16]) #discard seconds onward avoids PyArrow error
						temp=pd.to_datetime(temp)
						df.loc[:,column]=np.nan
						df[column]=df[column].astype('datetime64[ns]')
						df.loc[temp.index,column]=temp
						print('Converting date column from str to datetime: ',table,'/ '+column)
		df.to_parquet(data_dir+'devel/'+config['patient_set']+'/'+table+'_devel.parquet')
	else:
		print('skipping table '+table)

conn.close()

