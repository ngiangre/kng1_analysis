import numpy as np
import pandas as pd
import pickle
import time
import random
import os

from sklearn import linear_model, model_selection, ensemble
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import clone
from sklearn import metrics
from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold
import sklearn.metrics as m
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.utils import shuffle, resample

type_='marker'
basename = type_+'_features_expired_prediction_'
dir_ = '../../data/'

t0_all=time.time()

seed = 42
np.random.seed(seed)
max_depth = 1
C=1
tol=1e-3
min_samples_leaf=2
min_samples_split=2
n_estimators=100

models = {
		  "Logistic Regression" : linear_model.LogisticRegression(
			  C=C,
			  penalty='l1',
			  solver="liblinear",
			  tol=tol,
			  random_state=seed)
		 }

classification_metrics = ['roc_auc']
cv_split = 10
test_size = 0.15
n_jobs = 25
nboot=200

X_all_proteins = pd.read_csv(dir_+'integrated_X_raw_all_proteins.csv',index_col=0)
proteins_no_immunoglobulins = pickle.load(open(dir_+'proteins_no_immunoglobulins.pkl','rb'))
X_all_proteins = X_all_proteins.loc[:,proteins_no_immunoglobulins]

joined = pd.read_csv(dir_+'mortality_X_y.csv',index_col=0)
X_all_clinical = pd.read_csv(dir_+'integrated_X_clinical_and_cohort_covariates.csv',index_col=0)
Y_pgd = pd.read_csv(dir_+'integrated_pgd_y.csv',index_col=0,header=None)
Y_pgd.columns = ['PGD']
X_all_clinical = X_all_clinical.join(Y_pgd)
Y_mortality = joined[['expired']]
Y_mortality.index.name=''
X_all_clinical = X_all_clinical.join(Y_mortality)
Y_lvad = joined[['Mechanical_Support_Y']]
Y_lvad.index.name=''

idmap_sub = pd.read_csv(dir_+'protein_gene_map_full.csv')[['Protein','Gene_name']].dropna()

cov_df = X_all_clinical.loc[:,['Cohort_Columbia','Cohort_Cedar']].copy().astype(int)
all_cov_df = cov_df.copy()
all_cov_df.loc[:,'Cohort_Paris'] = (
    (all_cov_df['Cohort_Columbia'] + 
     all_cov_df['Cohort_Cedar'])==0).astype(int)


params = {'Y' : Y_pgd, 'cv_split' : cv_split, 
		  'metrics' : classification_metrics, 'n_jobs' : 1, 
		  'test_size' : test_size,
		  'retrained_models' : True, 'patient_level_predictions' : True}

def permute(Y,seed=42):
	"""
	shuffle sample values

	Parameters:
	----------

	Y : pandas series
		Index of samples and values are their class labels
	seed : int
		Random seed for shuffling

	Returns:
	------

	arr_shuffle: pandas series
		A shuffled Y
	"""
	arr = shuffle(Y.values,random_state=seed)
	arr_shuffle = (pd.Series(arr.reshape(1,-1)[0],index=Y.index))
	return arr_shuffle

def observed_val(X,Y,models,metrics=['roc_auc'],cv_split=10,seed=42,test_size=0.15,return_train_score=False,n_jobs=1,retrained_models=False,patient_level_predictions=False,return_estimator=True):
	# make sure given metrics are in list and not one metric is given as a string
	if type(metrics)!=list:
		metrics = [metrics]
	# 1/ reindex
	X = X.loc[Y.index]
	dfs = []
	model_retrained_fits = {}
	model_confs = []
	#iterate through model dictionary
	for name,mod in models.items():
		# /2 generate model parameters and fold scores with cv splitter
		fit = clone(mod).fit(X,Y.values.reshape(1,-1)[0])
		conf = pd.DataFrame({'y_true' : Y.values.reshape(1,-1)[0],'y_pred' : fit.predict(X),'bootstrap' : 'observed','model' : np.repeat(name,len(Y.index))},index=Y.index)
		model_confs.append(conf)
		#do prediction for each metric
		tmp = pd.DataFrame({'model' : name,'bootstrap' : 'observed'},index=[0])
		for metric in metrics:
			tmp[metric] = m.SCORERS[metric](fit,X,Y)
		model_retrained_fits[name] = fit
		dfs.append(tmp)
	return pd.concat(dfs).reset_index(drop=True), model_retrained_fits, pd.concat(model_confs)

def resample_observed_val(X,Y,models,metrics=['roc_auc'],cv_split=10,seed=42,test_size=0.15,return_train_score=False,n_jobs=1,retrained_models=False,patient_level_predictions=False,return_estimator=True):
	# make sure given metrics are in list and not one metric is given as a string
	if type(metrics)!=list:
		metrics = [metrics]
	# 1/ reindex
	X = X.loc[Y.index]
	Y_resample = resample(Y,random_state=seed)
	X = X.loc[Y_resample.index]
	Y = Y_resample.copy()
	dfs = []
	model_retrained_fits = {}
	model_confs = []
	#iterate through model dictionary
	for name,mod in models.items():
		# /2 generate model parameters and fold scores with cv splitter
		fit = clone(mod).fit(X,Y.values.reshape(1,-1)[0])
		conf = pd.DataFrame({'y_true' : Y.values.reshape(1,-1)[0],'y_pred' : fit.predict(X),'bootstrap' : 'observed','model' : np.repeat(name,len(Y.index))},index=Y.index)
		model_confs.append(conf)
		#do prediction for each metric
		tmp = pd.DataFrame({'model' : name,'bootstrap' : 'observed'},index=[0])
		for metric in metrics:
			tmp[metric] = m.SCORERS[metric](fit,X,Y)
		model_retrained_fits[name] = fit
		dfs.append(tmp)
	return pd.concat(dfs).reset_index(drop=True), model_retrained_fits, pd.concat(model_confs)

def permuted_observed_val(X,Y,models,metrics=['roc_auc'],cv_split=10,seed=42,test_size=0.15,return_train_score=False,n_jobs=1,retrained_models=False,patient_level_predictions=False,return_estimator=True):
	# make sure given metrics are in list and not one metric is given as a string
	if type(metrics)!=list:
		metrics = [metrics]
	# 1/ reindex
	X = X.loc[Y.index]
	Y_shuffle = permute(Y,seed=seed)
	X = X.loc[Y_shuffle.index]
	Y = Y_shuffle.copy()
	dfs = []
	model_retrained_fits = {}
	model_confs = []
	#iterate through model dictionary
	for name,mod in models.items():
		# /2 generate model parameters and fold scores with cv splitter
		fit = clone(mod).fit(X,Y.values.reshape(1,-1)[0])
		conf = pd.DataFrame({'y_true' : Y.values.reshape(1,-1)[0],'y_pred' : fit.predict(X),'bootstrap' : 'observed','model' : np.repeat(name,len(Y.index))},index=Y.index)
		model_confs.append(conf)
		#do prediction for each metric
		tmp = pd.DataFrame({'model' : name,'bootstrap' : 'observed'},index=[0])
		for metric in metrics:
			tmp[metric] = m.SCORERS[metric](fit,X,Y)
		model_retrained_fits[name] = fit
		dfs.append(tmp)
	return pd.concat(dfs).reset_index(drop=True), model_retrained_fits, pd.concat(model_confs)

def train_test_val_top_fold_01_within(X,Y,models,metrics=['roc_auc'],cv_split=10,seed=42,test_size=0.15,return_train_score=True,n_jobs=1,retrained_models=False,patient_level_predictions=False,return_estimator=True):
	# make sure given metrics are in list and not one metric given as a string
	if type(metrics)!=list:
		metrics = [metrics]
	# 1/ train and test split
	X = X.loc[Y.index]
	X_train, X_test, y_train, y_test = train_test_split(X,Y,
                                                            test_size=test_size,
                                                            random_state=seed,
                                                            shuffle=True)
	X_train = X_train.apply(lambda x : (x - min(x))/(max(x) - min(x)),axis=0)
	X_test = X_test.apply(lambda x : (x - min(x))/(max(x) - min(x)),axis=0)
	X_train[X_train.isna()]=0
	X_test[X_test.isna()]=0
	#define K fold splitter
	cv = StratifiedKFold(n_splits=cv_split,random_state=seed,shuffle=True)
	#Instantiate lists to collect prediction and model results
	dfs = []
	model_retrained_fits = {}
	model_confs = []
	#iterate through model dictionary
	for name,mod in models.items():
		# /2 generate model parameters and fold scores with cv splitter
		fit = cross_validate(clone(mod),X_train,y_train.values.reshape(1,-1)[0],cv=cv,scoring=metrics,
								n_jobs=n_jobs,return_train_score=return_train_score,
								return_estimator=return_estimator)
		tmp = pd.DataFrame({'fold' : range(cv_split),
                                    'model' : name},
                                   index=range(cv_split))
		#populate scores in dataframe
		cols = [k for k in fit.keys() if (k.find('test')+k.find('train'))==-1]
		for col in cols:
			tmp[col] = fit[col]
		# /3 Identify best performing model
		top_fold = np.where(fit['test_roc_auc']==fit['test_roc_auc'].max())[0][0]
		keys = [x for x in  fit.keys()]
		vals = [fit[x][top_fold] for x in keys]
		top_model_key_vals = {}
		for i in range(len(vals)):
			top_model_key_vals[keys[i]] = vals[i]
		#4/ train models on training set 
		# also get sample level predictions
		f = top_model_key_vals['estimator']
		fitted = clone(f).fit(X_train,y_train.values.reshape(1,-1)[0])
		conf = pd.DataFrame({'y_true' : y_test.values.reshape(1,-1)[0],
                                     'y_pred' : fitted.predict(X_test),
                                     'y_proba' : fitted.predict_proba(X_test)[:,1],
                                     'bootstrap' : np.repeat(seed,len(y_test.index)),
                                     'model' : np.repeat(name,len(y_test.index))},
                                    index=y_test.index)
		model_confs.append(conf)
		#do prediction for each metric
		for metric in metrics:
			tmp['validation_'+metric] = m.SCORERS[metric](fitted,X_test,y_test)
		model_retrained_fits[name] = fitted
		dfs.append(tmp.query('fold==@top_fold').drop('fold',1))
	return pd.concat(dfs,sort=True).reset_index(drop=True), model_retrained_fits, pd.concat(model_confs,sort=True)

def permuted_train_test_val_top_fold_01_within(X,Y,models,metrics=['roc_auc'],cv_split=10,seed=42,test_size=0.15,return_train_score=True,n_jobs=1,retrained_models=False,patient_level_predictions=False,return_estimator=True):
	X = X.loc[Y.index]
	Y_shuffle = permute(Y,seed=seed)
	X_shuffle = X.loc[Y_shuffle.index]
	# make sure given metrics are in list and not one metric given as a string
	if type(metrics)!=list:
		metrics = [metrics]
	# 1/ train and test split
	X_train, X_test, y_train, y_test = train_test_split(X_shuffle,Y_shuffle,
                                                            test_size=test_size,
                                                            random_state=seed,
                                                            shuffle=True)
	X_train = X_train.apply(lambda x : (x - min(x))/(max(x) - min(x)),axis=0)
	X_test = X_test.apply(lambda x : (x - min(x))/(max(x) - min(x)),axis=0)
	X_train[X_train.isna()]=0
	X_test[X_test.isna()]=0
	#define K fold splitter
	cv = StratifiedKFold(n_splits=cv_split,random_state=seed,shuffle=True)
	#Instantiate lists to collect prediction and model results
	dfs = []
	model_retrained_fits = {}
	model_confs = []
	#iterate through model dictionary
	for name,mod in models.items():
		# /2 generate model parameters and fold scores with cv splitter
		fit = cross_validate(clone(mod),X_train,y_train.values.reshape(1,-1)[0],cv=cv,scoring=metrics,
								n_jobs=n_jobs,return_train_score=return_train_score,
								return_estimator=return_estimator)
		tmp = pd.DataFrame({'fold' : range(cv_split),
                                    'model' : name},
                                   index=range(cv_split))
		#populate scores in dataframe
		cols = [k for k in fit.keys() if (k.find('test')+k.find('train'))==-1]
		for col in cols:
			tmp[col] = fit[col]
		# /3 Identify best performing model
		top_fold = np.where(fit['test_roc_auc']==fit['test_roc_auc'].max())[0][0]
		keys = [x for x in  fit.keys()]
		vals = [fit[x][top_fold] for x in keys]
		top_model_key_vals = {}
		for i in range(len(vals)):
			top_model_key_vals[keys[i]] = vals[i]
		#4/ train models on training set 
		# also get sample level predictions
		f = top_model_key_vals['estimator']
		fitted = clone(f).fit(X_train,y_train.values.reshape(1,-1)[0])
		conf = pd.DataFrame({'y_true' : y_test.values.reshape(1,-1)[0],
                                     'y_pred' : fitted.predict(X_test),
                                     'y_proba' : fitted.predict_proba(X_test)[:,1],
                                     'bootstrap' : np.repeat(seed,len(y_test.index)),
                                     'model' : np.repeat(name,len(y_test.index))},
                                    index=y_test.index)
		model_confs.append(conf)
		#do prediction for each metric
		for metric in metrics:
			tmp['validation_'+metric] = m.SCORERS[metric](fitted,X_test,y_test)
		model_retrained_fits[name] = fitted
		dfs.append(tmp.query('fold==@top_fold').drop('fold',1))
	return pd.concat(dfs,sort=True).reset_index(drop=True), model_retrained_fits, pd.concat(model_confs,sort=True)

def bootstrap_of_fcn(func=None,params={},n_jobs=4,nboot=2):
	if func==None:
		return "Need fcn to bootstrap"
	parallel = Parallel(n_jobs=n_jobs)
	return parallel(
		delayed(func)(
			seed=k,**params)
		for k in range(nboot))

def get_performance(lst):
    perf = (pd.
            concat(lst,keys=range(len(lst))).
            reset_index(level=1,drop=True).
            rename_axis('bootstrap').
            reset_index()
           )
    return perf

def model_feature_importances(boot_mods):
    dfs = []
    X = params['X'].copy()
    X.loc[:,'Intercept'] = 0
    for i in range(len(boot_mods)):
        for j in boot_mods[i].keys():
            mod = boot_mods[i][j]
            coef = []
            try:
                coef.extend([i for i in mod.feature_importances_])
            except:
                coef.extend([i for i in mod.coef_[0]])
            coef.extend(mod.intercept_)
            fs = []
            fs.extend(X.columns.values)
            df = pd.DataFrame({
                'Feature' : fs,
                'Gene_name' : (X.T.
                               join(idmap_sub.
                                    set_index('Protein'),how='left').
                               Gene_name.values),
                'Importance' : coef,
                'Model' : j,
                'Bootstrap' : i
            })
            dfs.append(df)
    return pd.concat(dfs,sort=True)

def patient_predictions(lst):
        col = pd.concat(lst).index.name
        dat = \
        (pd.
         concat(
             lst
         ).
         reset_index().
         rename(columns={col : 'Sample'}).
         set_index('Sample').
         join(all_cov_df).
         reset_index().
         melt(id_vars=['Sample','bootstrap','model','y_true','y_pred','y_proba'],
              var_name='cohort',value_name='mem')
        )
        dat.cohort = dat.cohort.str.split('_').apply(lambda x : x[1])
        dat = dat[dat.mem==1].drop('mem',1).reset_index(drop=True)
        return dat

def get_performance(lst):
    perf = (pd.
            concat(lst,keys=range(len(lst))).
            reset_index(level=1,drop=True).
            rename_axis('bootstrap').
            reset_index()
           )
    return perf

def model_feature_importances(boot_mods):
    dfs = []
    X = params['X'].copy()
    X.loc[:,'Intercept'] = 0
    for i in range(len(boot_mods)):
        for j in boot_mods[i].keys():
            mod = boot_mods[i][j]
            coef = []
            try:
                coef.extend([i for i in mod.feature_importances_])
            except:
                coef.extend([i for i in mod.coef_[0]])
            coef.extend(mod.intercept_)
            fs = []
            fs.extend(X.columns.values)
            df = pd.DataFrame({
                'Feature' : fs,
                'Gene_name' : (X.T.
                               join(idmap_sub.
                                    set_index('Protein'),how='left').
                               Gene_name.values),
                'Importance' : coef,
                'Model' : j,
                'Bootstrap' : i
            })
            dfs.append(df)
    return pd.concat(dfs,sort=True)

def patient_predictions(lst):
        col = pd.concat(lst).index.name
        dat = \
        (pd.
         concat(
             lst
         ).
         reset_index().
         rename(columns={col : 'Sample'}).
         set_index('Sample').
         join(all_cov_df).
         reset_index().
         melt(id_vars=['Sample','bootstrap','model','y_true','y_pred','y_proba'],
              var_name='cohort',value_name='mem')
        )
        dat.cohort = dat.cohort.str.split('_').apply(lambda x : x[1])
        dat = dat[dat.mem==1].drop('mem',1).reset_index(drop=True)
        return dat

import itertools
clin_combos = [[list(i) for i in itertools.combinations(
    np.intersect1d(
        X_all_clinical.columns.values,
        X_all_clinical.columns.values),r)
               ] for r in np.arange(1,2)]
prot_combos = [[list(i) for i in itertools.combinations(
    np.intersect1d(
        X_all_proteins.columns.values,
        X_all_proteins.columns.values),r)
               ] for r in np.arange(1,2)]

all_clin_1 = list(np.concatenate(list(itertools.chain(*clin_combos))))
print(len(all_clin_1))

all_prot_1 = list(np.concatenate(list(itertools.chain(*prot_combos))))
print(len(all_prot_1))

all_clin_1_and_prot_1 = list(
    itertools.chain(*[all_clin_1,all_prot_1])
)
print(len(all_clin_1_and_prot_1))

all_clin_1_prot_1 = list(
    itertools.chain(*
                    [[list(itertools.chain(*[[x],[y]])) for x in all_prot_1] for y in all_clin_1]
                   )
)
print(len(all_clin_1_prot_1))

all_clin_1_prot_1_and_clin_1_and_prot_1 = list(
    itertools.chain(*[all_clin_1,all_prot_1,all_clin_1_prot_1])
)
print(len(all_clin_1_prot_1_and_clin_1_and_prot_1))

all_clin_2 = [list(i) for i in itertools.combinations(all_clin_1,2)]
print(len(all_clin_2))

all_prot_2 = [list(i) for i in itertools.combinations(all_prot_1,2)]
print(len(all_prot_2))

all_clin_1_prot_1_and_prot_2 = list(
    itertools.chain(*[all_clin_1_prot_1,all_prot_2])
)
len(all_clin_1_prot_1_and_prot_2)

all_clin_2_and_clin_1_prot_1_and_prot_2 = list(
    itertools.chain(*[all_clin_2,all_clin_1_prot_1,all_prot_2])
)
len(all_clin_2_and_clin_1_prot_1_and_prot_2)

all_clin_2_and_clin_1_prot_1_and_prot_2_and_clin_1_and_prot_1 = list(
    itertools.chain(*[all_clin_2,all_clin_1_prot_1,all_prot_2,all_clin_1,all_prot_1])
)
print(len(all_clin_2_and_clin_1_prot_1_and_prot_2_and_clin_1_and_prot_1))

t0 = time.time()
fimps_dfs = []
perf_dfs = []
ppreds_dfs = []
perm_fimps_dfs = []
perm_perf_dfs = []
perm_ppreds_dfs = []
feature_set = {}

for i,features in enumerate(all_clin_1_and_prot_1):
        print(features)
        print(i)
        X_all = X_all_proteins.join(X_all_clinical)
        if type(features)==np.str_:
            X = X_all[[features]]
        if type(features)==list:
            X = X_all[features]
        feature_set[str(i)] = X.columns.tolist()
        params.update({'X' : X,'models' : models.copy()})
        lst = bootstrap_of_fcn(func=train_test_val_top_fold_01_within,
                       params=params,n_jobs=n_jobs,nboot=nboot)
        perf = get_performance([lst[i][0] for i in range(len(lst))])
        perf['set'] = str(i)
        perf_dfs.append(perf)
        fimps = model_feature_importances([lst[i][1] for i in range(len(lst))])
        fimps['set'] = str(i)
        fimps_dfs.append(fimps)
        ppreds = patient_predictions([lst[i][2] for i in range(len(lst))])
        ppreds['set'] = str(i)
        ppreds_dfs.append(ppreds)
        
        lst = bootstrap_of_fcn(func=permuted_train_test_val_top_fold_01_within,
               params=params,n_jobs=n_jobs,nboot=nboot)
        perm_perf = get_performance([lst[i][0] for i in range(len(lst))])
        perm_perf['set'] = str(i)
        perm_perf_dfs.append(perm_perf)
        perm_fimps = model_feature_importances([lst[i][1] for i in range(len(lst))])
        perm_fimps['set'] = str(i)
        perm_fimps_dfs.append(perm_fimps)
        perm_ppreds = patient_predictions([lst[i][2] for i in range(len(lst))])
        perm_ppreds['set'] = str(i)
        perm_ppreds_dfs.append(perm_ppreds)
        
perf_df = (pd.concat(perf_dfs).
           groupby(['set'])['validation_roc_auc'].
           describe(percentiles=[0.025,0.975]).
           loc[:,['2.5%','mean','97.5%']].
           sort_values('2.5%',ascending=False).
           reset_index()
          )
fimps_df = (pd.concat(fimps_dfs).
            groupby(['set','Feature'])['Importance'].
            describe(percentiles=[0.025,0.975]).
            loc[:,['2.5%','mean','97.5%']].
            sort_values('2.5%',ascending=False).
            reset_index()
          )
ppreds_df = (pd.concat(ppreds_dfs))

perm_perf_df = (pd.concat(perm_perf_dfs).
           groupby(['set'])['validation_roc_auc'].
           describe(percentiles=[0.025,0.975]).
           loc[:,['2.5%','mean','97.5%']].
           sort_values('2.5%',ascending=False).
           reset_index()
          )
perm_fimps_df = (pd.concat(perm_fimps_dfs).
            groupby(['set','Feature'])['Importance'].
            describe(percentiles=[0.025,0.975]).
            loc[:,['2.5%','mean','97.5%']].
            sort_values('2.5%',ascending=False).
            reset_index()
          )
perm_ppreds_df = (pd.concat(perm_ppreds_dfs))

t1_all = time.time()
print(np.round( (t1_all - t0_all) / 60, 2 ) )


perf_df = (
    perf_df.
    set_index('set').
    join(
        pd.DataFrame(
            feature_set.items(),columns=['set','set_features']
        ).
        set_index('set')
    ).
    sort_values('2.5%')
)
perf_df.to_csv(dir_+'mortality_predictions_'+type_+'_performance_pgd.csv')

fimps_df = (
    fimps_df.
    set_index('set').
    join(
        pd.DataFrame(
            feature_set.items(),columns=['set','set_features']
        ).
        set_index('set')
    ).
    sort_values('2.5%')
)
fimps_df.to_csv(dir_+'mortality_predictions_'+type_+'_feature_importance_pgd.csv')

ppreds_df.to_csv(dir_+'mortality_predictions_'+type_+'_patient_predictions_pgd.csv')


pd.concat(fimps_dfs).to_csv(dir_+'mortality_predictions_'+type_+'_full_feature_importance_pgd.csv')
pd.concat(perm_fimps_dfs).to_csv(dir_+'mortality_predictions_'+type_+'_full_permuted_feature_importance_pgd.csv')
