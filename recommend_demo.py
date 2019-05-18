#from flask import Flask,render_template,url_for,request
#import pickle
from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise import KNNBaseline
from surprise import SVD
from surprise import NMF
from surprise import CoClustering
#from collections import defaultdict
import pandas as pd
import numpy as np
import os

bs = pd.read_csv("selected_business_all_info.csv")
df = pd.read_csv("all_20.csv")

user_new = input("Enter your username: ")

item_1 = input("Rate 3 restaurants you've visited before: Restaurant 1 name: ")
print("You've visited the restaurant 1: ", bs[bs['business_id'] == item_1]['name'].values)
rating_1 = input("Rate Restaurant 1 (1 - 5): ")
item_2 = input("Restaurant 2 name: ")
print("You've visited the restaurant 2: ", bs[bs['business_id'] == item_2]['name'].values)
rating_2 = input("Rate Restaurant 2 (1 - 5): ")
item_3 = input("Restaurant 3 name: ")
print("You've visited the restaurant 2: ", bs[bs['business_id'] == item_3]['name'].values)
rating_3 = input("Rate Restaurant 3 (1 - 5): ")
top_n = input("How many restaurants you want: ")

item_1 = str(item_1)
item_2 = str(item_2)
item_3 = str(item_3)
stars_1 = float(rating_1)
stars_2 = float(rating_2)
stars_3 = float(rating_3)
rated_item = [item_1, item_2, item_3]
top_n = int(top_n)

df = df[['user_id','business_id','stars']]
df = df.dropna()
		
df_add = df.append({'user_id': user_new, 'business_id': item_1, 'stars': stars_1}, ignore_index=True)
df_add = df_add.append({'user_id': user_new, 'business_id': item_2, 'stars': stars_2}, ignore_index=True)
df_add = df_add.append({'user_id': user_new, 'business_id': item_3, 'stars': stars_3}, ignore_index=True)
		#df_add = df_add.append({'user_id': user_new, 'business_id': item_4, 'stars': rating_4}, ignore_index=True)
		#df_add = df_add.append({'user_id': user_new, 'business_id': item_5, 'stars': rating_5}, ignore_index=True)


df_pred = pd.DataFrame(set(df['business_id']) - set(rated_item)).rename(columns={0:"iid"})
df_pred['uid'] = user_new

reader = Reader(rating_scale=(1, 5))
data_all = Dataset.load_from_df(df_add, reader)
df_all = data_all.build_full_trainset()

		#Baseline
bsl_options = {'method': 'sgd',
           'reg': 0.001,
           'learning_rate': .005,
           }

algo_base = BaselineOnly(bsl_options=bsl_options)
algo_base.fit(df_all)

def baseline(x):
	x['est_base']=algo_base.predict(uid=x['uid'], iid=x['iid'])[3]
	return x

df_pred = df_pred.apply(baseline,axis=1)

		# knnBaseline
algo_knn = KNNBaseline(k=30, user_based=True)
algo_knn.fit(df_all)

def knn(x):
	x['est_knn']=algo_knn.predict(uid=x['uid'], iid=x['iid'])[3]
	return x

df_pred = df_pred.apply(knn,axis=1)

		# svd
algo_svd = SVD(n_factors=20, n_epochs=15)
algo_svd.fit(df_all)

def svd(x):
	x['est_svd']=algo_svd.predict(uid=x['uid'], iid=x['iid'])[3]
	return x

df_pred = df_pred.apply(svd,axis=1)

		# nmf
algo_nmf = NMF(n_factors=3,n_epochs=5,biased=True)
algo_nmf.fit(df_all)

def nmf(x):
	x['est_nmf']=algo_nmf.predict(uid=x['uid'], iid=x['iid'])[3]
	return x

df_pred = df_pred.apply(nmf,axis=1)

		# cocluster
algo_clu = CoClustering(n_cltr_u=2, n_cltr_i=3, n_epochs=10)
algo_clu.fit(df_all)

def clu(x):
	x['est_clu']=algo_clu.predict(uid=x['uid'], iid=x['iid'])[3]
	return x

df_pred = df_pred.apply(clu,axis=1)

		# get average
df_pred['est_mean'] = df_pred[['est_base','est_clu','est_svd','est_knn','est_nmf']].mean(axis=1)

top_n = df_pred.groupby('uid').apply(lambda x: x.sort_values(['est_mean'],ascending=False).head(top_n))
top_n_reset = top_n.reset_index(drop=True)
top = top_n_reset[['iid']]
df_final = pd.merge(top,
           bs[['name','address','categories','business_id']],
           left_on='iid',
           right_on='business_id',
           how='left')
out_put = df_final[['name','address']]
print(out_put)