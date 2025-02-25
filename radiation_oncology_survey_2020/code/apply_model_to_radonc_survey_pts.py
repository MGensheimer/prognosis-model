logger.info('Patients in Daniel Chang survey study set: ' + str(len(pts_chang)))
pt_landmarks_notchang = pt_landmarks.loc[pt_landmarks.patient_id.isin(pts_train_test),:]
landmark_data_notchang = landmark_data[pt_landmarks.patient_id.isin(pts_train_test),:]
landmark_data_notchang_scaled = standardize_data_and_impute_mean(landmark_data_notchang,feature_means,feature_std,feature_include)
y_notchang = make_surv_array(pt_landmarks_notchang.fu_days.values,pt_landmarks_notchang.dead.values,breaks)
model_notchang = train_surv_model(input_x=landmark_data_notchang_scaled, input_y=y_notchang, l2=0.1)
with open(output_dir+'models/'+model_config['model_name']+'/model_survival_notchang_arch.json', "w") as text_file:
    text_file.write("%s" % model_notchang.to_json())

model_notchang.save_weights(output_dir+'models/'+model_config['model_name']+'/model_survival_notchang_weights.h5')

weights = model_notchang.get_weights()
feature_names['coef'] = weights[0]

eval_dates=chang_pts.loc[:,['patient_id','MRN_FULL','tx_date']]
eval_dates.rename(columns={'tx_date':'eval_date'},inplace=True)
eval_dates.loc[:,'date_int'] = ((eval_dates.eval_date-pd.Timestamp(2000,1,1)) / np.timedelta64(1, 'D')).astype('int32')
eval_dates.sort_values(by=['patient_id','date_int'], inplace=True)
eval_dates.reset_index(inplace=True,drop=True)
landmark_data_chang, _ = make_feat_array(pt_data, config, model_config, eval_dates, config_dir, data_dir, output_dir)
landmark_data_chang_scaled = standardize_data_and_impute_mean(landmark_data_chang,feature_means,feature_std,feature_include)
y_pred=model_notchang.predict_proba(landmark_data_chang_scaled,verbose=0)
pred_surv=np.cumprod(y_pred,axis=1)
eval_dates['median_pred_surv']=-1
eval_dates['pred_6mo_surv']=-1
eval_dates['pred_1yr_surv']=-1
for i in range(len(eval_dates)):
	eval_dates.loc[i,'median_pred_surv']=calc_median_surv(y_pred[i,:],model_config['breaks'])
	eval_dates.loc[i,'pred_6mo_surv']=calc_timepoint_surv(y_pred[i,:],model_config['breaks'],182)
	eval_dates.loc[i,'pred_1yr_surv']=calc_timepoint_surv(y_pred[i,:],model_config['breaks'],365)

eval_dates.to_parquet(output_dir+'models/'+model_config['model_name']+'/chang_model_results.parquet')
np.save(output_dir+'models/'+model_config['model_name']+'/chang_model_predsurv',pred_surv)
temp = pd.DataFrame(pred_surv)
temp.columns=[str(i) for i in model_config['breaks'][1:]]
temp.to_parquet(output_dir+'models/'+model_config['model_name']+'/chang_model_predsurv.parquet')

eval_dates_2wkshift=chang_pts.loc[:,['patient_id','MRN_FULL','tx_date']]
eval_dates_2wkshift.rename(columns={'tx_date':'eval_date'},inplace=True)
eval_dates_2wkshift.eval_date = eval_dates_2wkshift.eval_date - pd.DateOffset(days=14)
eval_dates_2wkshift.loc[:,'date_int'] = ((eval_dates_2wkshift.eval_date-pd.Timestamp(2000,1,1)) / np.timedelta64(1, 'D')).astype('int32')
eval_dates_2wkshift.sort_values(by=['patient_id','date_int'], inplace=True)
eval_dates_2wkshift.reset_index(inplace=True,drop=True)
landmark_data_chang_2wkshift, _ = make_feat_array(pt_data, config, model_config, eval_dates_2wkshift, config_dir, data_dir, output_dir)
landmark_data_chang_2wkshift_scaled = standardize_data_and_impute_mean(landmark_data_chang_2wkshift,feature_means,feature_std,feature_include)
y_pred_2wkshift=model_notchang.predict_proba(landmark_data_chang_2wkshift_scaled,verbose=0)
pred_surv_2wkshift=np.cumprod(y_pred,axis=1)
eval_dates['median_pred_surv_2wkshift']=-1
for i in range(len(eval_dates)):
	eval_dates.loc[i,'median_pred_surv_2wkshift']=calc_median_surv(y_pred_2wkshift[i,:],model_config['breaks'])

n_feats_report=5
sign_dict={True:' +',False:' -'}
eval_dates['improve_surv_feats'] = ''
eval_dates['worsen_surv_feats'] = ''
result=landmark_data_chang_scaled*np.transpose(np.tile(weights[0],landmark_data_chang_scaled.shape[0]))
sorted_feats=np.argsort(result,axis=1)
for i in range(len(eval_dates)):
	improve_surv_feats=feature_names.feature_name.iloc[sorted_feats[i,:-(n_feats_report+1):-1]]
	improve_surv_positive_sign=landmark_data_chang_scaled[i,sorted_feats[i,:-(n_feats_report+1):-1]]>0
	worsen_surv_feats=feature_names.feature_name.iloc[sorted_feats[i,0:n_feats_report]]
	worsen_surv_positive_sign=landmark_data_chang_scaled[i,sorted_feats[i,0:n_feats_report]]>0
	eval_dates['improve_surv_feats'].iloc[i]="".join(i + j for i, j in zip([sign_dict[x] for x in improve_surv_positive_sign],improve_surv_feats))
	eval_dates['worsen_surv_feats'].iloc[i]="".join(i + j for i, j in zip([sign_dict[x] for x in worsen_surv_positive_sign],worsen_surv_feats))

eval_dates.to_parquet(output_dir+'models/'+model_config['model_name']+'/chang_model_results.parquet')