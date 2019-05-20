from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import accuracy
from surprise.model_selection import PredefinedKFold
from collections import defaultdict
from surprise.model_selection.split import LeaveOneOut
import pandas as pd
import numpy as np
import os
import pickle


df = pd.read_csv("all_10.csv")
df = df[['user_id','business_id','stars']]
df = df.dropna()

df_train = pd.read_csv("train_33_selected.csv")
df_test = pd.read_csv("test_33_selected.csv")

reader = Reader(rating_scale=(1, 5))
data_train = Dataset.load_from_df(df_train, reader)
data_test = Dataset.load_from_df(df_test, reader)
data_all = Dataset.load_from_df(df, reader)

df_trainset_train = data_train.build_full_trainset()
df_trainset_test = df_trainset_train.build_testset()
df_testset = data_test.build_full_trainset().build_testset()

df_all = data_all.build_full_trainset()
newset = df_all.build_anti_testset()


def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

'''
top_n = defaultdict(list)
for uid, user_ratings in top_n.items():
    user_ratings.sort(key=lambda x: x[1], reverse=True)
    top_n[uid] = user_ratings[:10]
'''

#### BaselineOnly
# als
bsl_options = {'method': 'als',
               'n_epochs': 5,
               'reg_u': 8,
               'reg_i': 15
               }

algo_als = BaselineOnly(bsl_options=bsl_options)
algo_als.fit(df_trainset_train)

train_pred = algo_als.test(df_trainset_test)
print("BaselineOnly train biased RMSE", accuracy.rmse(train_pred))
test_pred = algo_als.test(df_testset)
print("BaselineOnly test unbiased RMSE", accuracy.rmse(test_pred))
# cross_validate(algo_als, data_train, measures=['RMSE', 'MAE'], cv=5, verbose=True)
# pickle.dump(algo_als, open('algo_als.sav', 'wb'))

# sgd
bsl_options = {'method': 'sgd',
               'reg': 0.001,
               'learning_rate': .005,
               }

algo_sgd = BaselineOnly(bsl_options=bsl_options)
algo_sgd.fit(df_trainset_train)

train_pred = algo_sgd.test(df_trainset_test)
print("BaselineOnly train biased RMSE", accuracy.rmse(train_pred))
test_pred = algo_sgd.test(df_testset)
print("BaselineOnly test unbiased RMSE", accuracy.rmse(test_pred))
# cross_validate(algo_sgd, data_train, measures=['RMSE', 'MAE'], cv=5, verbose=True)
# pickle.dump(algo_sgd, open('algo_als.sav', 'wb'))

# pred all
algo_sgd = BaselineOnly(bsl_options=bsl_options)
algo_sgd.fit(df_all)
# pickle.dump(algo_sgd, open('algo_sgd_all.sav', 'wb'))
predictions = algo_sgd.test(newset)

pred_one = algo.predict(uid= 'AWt-4_RGNqz9u2T8XfOy9g', iid= '2hSIeCX2cV-chFcBBXrZZA')
print(pred_one)

pred_base = pd.DataFrame(predictions)
pred_base.to_csv("pred_base_10.csv", index=False)


#### knn
algo_knn = KNNBaseline(k=30, user_based=True)
algo_knn.fit(df_trainset_train)

train_pred_knn = algo_knn.test(df_trainset_test)
print("KNN train biased RMSE", accuracy.rmse(train_pred_knn))
test_pred_knn = algo_knn.test(df_testset)
print("KNN test unbiased RMSE", accuracy.rmse(test_pred_knn))
# cross_validate(algo_knn, data_train, measures=['RMSE', 'MAE'], cv=5, verbose=True)
# pickle.dump(algo_knn, open('algo_knn.sav', 'wb'))

algo_knn = KNNBaseline(k=30, user_based=True)
algo_knn.fit(df_all)
predictions_knn = algo_knn.test(newset)
# pickle.dump(algo_sgd, open('algo_knn_all.sav', 'wb'))
pred_knn = pd.DataFrame(predictions_knn)
pred_knn.to_csv("pred_knn.csv", index=False)

def knn(x):
    x['est_knn']=algo_knn.predict(uid=x['uid'], iid=x['iid'])[3]
    return x

df_pred = df_pred.apply(knn,axis=1)
df_pred.head()

#### PMF
from surprise import SVD
algo_pmf = SVD(n_factors=12, n_epochs=20, biased=False)
algo_pmf.fit(df_trainset_train)

train_pred_pmf = algo_pmf.test(df_trainset_test)
print("PMF train biased RMSE", accuracy.rmse(train_pred_pmf))
test_pred_pmf = algo_pmf.test(df_testset)
print("PMF test unbiased RMSE", accuracy.rmse(test_pred_pmf))
#cross_validate(algo_pmf, data_train, measures=['RMSE', 'MAE'], cv=5, verbose=True)

algo_pmf = SVD(biased=False)

for trainset, testset in loo.split(data_all):
    
    algo_pmf.fit(trainset)
    predictions = algo_pmf.test(testset)

    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)


#### SVD
from surprise import SVD
algo_svd = SVD(n_factors=20, n_epochs=15)
algo_svd.fit(df_all)

def svd(x):
    x['est_']=algo_svd.predict(uid=x['uid'], iid=x['iid'])[3]
    return x

df_pred = df_pred.apply(svd,axis=1)
df_pred.head()


algo_svd_1 = SVD(n_factors=20, n_epochs=15)
algo_svd_1.fit(df_all)

train_pred_svd_1 = algo_svd_1.test(df_trainset_test)
print("SVD train biased RMSE", accuracy.rmse(train_pred_svd_1))
test_pred_svd_1 = algo_svd_1.test(df_testset)
print("SVD test unbiased RMSE", accuracy.rmse(test_pred_svd_1))
#cross_validate(algo_svd_1, data_train, measures=['RMSE', 'MAE'], cv=5, verbose=True)


algo_svd_2 = SVD(n_factors=50, n_epochs=10)
algo_svd_2.fit(df_trainset_train)

train_pred_svd_2 = algo_svd_2.test(df_trainset_test)
print("SVD train biased RMSE", accuracy.rmse(train_pred_svd_2))
test_pred_svd_2 = algo_svd_2.test(df_testset)
print("SVD test unbiased RMSE", accuracy.rmse(test_pred_svd_2))


algo_svd_3 = SVD(n_factors=30, n_epochs=10)
algo_svd_3.fit(df_trainset_train)

train_pred_svd_3 = algo_svd_3.test(df_trainset_test)
print("SVD train biased RMSE", accuracy.rmse(train_pred_svd_3))
test_pred_svd_3 = algo_svd_3.test(df_testset)
print("SVD test unbiased RMSE", accuracy.rmse(test_pred_svd_3))


algo_svd_4 = SVD(n_factors=30, n_epochs=15)
algo_svd_4.fit(df_trainset_train)

train_pred_svd_4 = algo_svd_4.test(df_trainset_test)
print("SVD train biased RMSE", accuracy.rmse(train_pred_svd_4))
test_pred_svd_4 = algo_svd_4.test(df_testset)
print("SVD test unbiased RMSE", accuracy.rmse(test_pred_svd_4))

algo_svd = SVD()

for trainset, testset in loo.split(data_all):
    
    algo_svd.fit(trainset)
    predictions = algo_svd.test(testset)

    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)

predictions_svd_1 = algo_svd_1.test(newset)


#### SVDpp
from surprise import SVDpp
algo_svdpp_1 = SVDpp(n_factors=10, n_epochs=10)
algo_svdpp_1.fit(df_trainset_train)

train_pred_svdpp_1 = algo_svdpp_1.test(df_trainset_test)
print("SVDpp train biased RMSE", accuracy.rmse(train_pred_svdpp_1))
test_pred_svdpp_1 = algo_svdpp_1.test(df_testset)
print("SVDpp test unbiased RMSE", accuracy.rmse(test_pred_svdpp_1))


algo_svdpp_2 = SVDpp(n_factors=5, n_epochs=5)
algo_svdpp_2.fit(df_trainset_train)

train_pred_svdpp_2 = algo_svdpp_2.test(df_trainset_test)
print("SVDpp train biased RMSE", accuracy.rmse(train_pred_svdpp_2))
test_pred_svdpp_2 = algo_svdpp_2.test(df_testset)
print("SVDpp test unbiased RMSE", accuracy.rmse(test_pred_svdpp_2))


algo_svdpp = SVDpp()
algo_svdpp.fit(df_trainset_train)
train_pred_svdpp = algo_svdpp.test(df_trainset_test)
print("SVDpp train biased RMSE", accuracy.rmse(train_pred_svdpp))
test_pred_svdpp = algo_svdpp.test(df_testset)
print("SVDpp test unbiased RMSE", accuracy.rmse(test_pred_svdpp))
#cross_validate(algo_svdpp_1, data_train, measures=['RMSE', 'MAE'], cv=5, verbose=True)


#### NMF
from surprise import NMF

algo_nmf_1 = NMF(n_factors=500,n_epochs=10,biased=True)
algo_nmf_1.fit(df_trainset_train)

train_pred_nmf_1 = algo_nmf_1.test(df_trainset_test)
print("NMF train biased RMSE", accuracy.rmse(train_pred_nmf_1))
test_pred_nmf_1 = algo_nmf_1.test(df_testset)
print("NMF test unbiased RMSE", accuracy.rmse(test_pred_nmf_1))


algo_nmf_2 = NMF(n_factors=3,n_epochs=5,biased=True)
algo_nmf_2.fit(df_trainset_train)

train_pred_nmf_2 = algo_nmf_2.test(df_trainset_test)
print("NMF train biased RMSE", accuracy.rmse(train_pred_nmf_2))
test_pred_nmf_2 = algo_nmf_2.test(df_testset)
print("NMF test unbiased RMSE", accuracy.rmse(test_pred_nmf_2))
#cross_validate(algo_nmf_2, data_train, measures=['RMSE', 'MAE'], cv=5, verbose=True)


algo_nmf_3 = NMF(n_factors=3,n_epochs=5,biased=True)
algo_nmf_3.fit(df_all)

predictions_nmf_3 = algo_nmf_3.test(newset)
pred_nmf_3 = pd.DataFrame(predictions_nmf_3)
pred_nmf_3.to_csv("pred_nmf_3_10.csv",index=False)


algo_nmf = NMF()
algo_nmf.fit(df_trainset_train)

train_pred_nmf = algo_nmf.test(df_trainset_test)
print("NMF train biased RMSE", accuracy.rmse(train_pred_nmf))
test_pred_nmf = algo_nmf.test(df_testset)
print("NMF test unbiased RMSE", accuracy.rmse(test_pred_nmf))


algo_nmf = NMF()

for trainset, testset in loo.split(data_all):
    
    algo_nmf.fit(trainset)
    predictions = algo_nmf.test(testset)

    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)


#### CoClustering
from surprise import CoClustering

algo_clu_2 = CoClustering(n_cltr_u=3, n_cltr_i=2, n_epochs=5)
algo_clu_2.fit(df_trainset_train)

train_pred_clu_2 = algo_clu_2.test(df_trainset_test)
print("CoClustering train biased RMSE", accuracy.rmse(train_pred_clu_2))
test_pred_clu_2 = algo_clu_2.test(df_testset)
print("CoClustering test unbiased RMSE", accuracy.rmse(test_pred_clu_2))


algo_clu_1 = CoClustering(n_cltr_u=2, n_cltr_i=2, n_epochs=5)
algo_clu_1.fit(df_trainset_train)

train_pred_clu_1 = algo_clu_1.test(df_trainset_test)
print("CoClustering train biased RMSE", accuracy.rmse(train_pred_clu_1))
test_pred_clu_1 = algo_clu_1.test(df_testset)
print("CoClustering test unbiased RMSE", accuracy.rmse(test_pred_clu_1))


algo_clu_3 = CoClustering(n_cltr_u=2, n_cltr_i=3, n_epochs=10)
algo_clu_3.fit(df_all)

train_pred_clu_3 = algo_clu_3.test(df_trainset_test)
print("CoClustering train biased RMSE", accuracy.rmse(train_pred_clu_3))
test_pred_clu_3 = algo_clu_3.test(df_testset)
print("CoClustering test unbiased RMSE", accuracy.rmse(test_pred_clu_3))
predictions_clu_3 = algo_clu_3.test(newset)

pred_clu_3 = pd.DataFrame(predictions_clu_3)
pred_clu_3.to_csv("pred_clu_3_10.csv",index=False)


from surprise import CoClustering

algo_clu = CoClustering()
algo_clu.fit(df_trainset_train)

train_pred_clu = algo_clu.test(df_trainset_test)
print("CoClustering train biased RMSE", accuracy.rmse(train_pred_clu))
test_pred_clu = algo_clu.test(df_testset)
print("CoClustering test unbiased RMSE", accuracy.rmse(test_pred_clu))
#cross_validate(algo_clu_3, data_train, measures=['RMSE', 'MAE'], cv=5, verbose=True)

from surprise import CoClustering

algo_clu = CoClustering()

for trainset, testset in loo.split(data_all):
    
    algo_clu.fit(trainset)
    predictions = algo_clu.test(testset)

    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)


#### Ensemble
pred_base = pd.read_csv("pred_base_10.csv")
pred_knn = pd.read_csv("pred_knn_10.csv")
pred_nmf = pd.read_csv("pred_nmf_3_10.csv")
pred_svd = pd.read_csv("pred_svd_1_10.csv")

test = pred_base

test['est_cluster'] = pd.DataFrame(pred_clu_3)['est']
test['est_svd'] = pd.DataFrame(pred_svd)['est']
#test['est_svdpp'] = pd.DataFrame(pred_svdpp_1)['est']
test['est_nmf'] = pd.DataFrame(pred_nmf)['est']
test['est_knn'] = pd.DataFrame(pred_knn)['est']

#averaging
test['est_mean'] = test[['est','est_cluster','est_svd','est_knn','est_nmf']].mean(axis=1)

test.to_csv("pred_10_est.csv",index=False)

from sklearn.metrics import mean_squared_error
from math import sqrt

train_rmse = sqrt(mean_squared_error(train['r_ui'], train['est_mean']))
test_rmse = sqrt(mean_squared_error(test['r_ui'], test['est_mean']))

print("ensemble train rmse", train_rmse)
print("ensemble test rmse", test_rmse)


#Blender
from sklearn.metrics import mean_squared_error
from math import sqrt

def train_test_model_performance(clf, X_train, y_train, X_test, y_test):
    # Fit a model by providing X and y from training set
    clf.fit(X_train, y_train)

    # Make prediction on the training data
    y_train_pred = clf.predict(X_train)

    # Make predictions on test data
    y_test_pred = clf.predict(X_test)
    
    train_rmse = sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))

    return train_rmse, test_rmse

#X_train = train[['est','est_cluster','est_svd','est_svdpp','est_nmf']]
#y_train = train['r_ui']
from sklearn.model_selection import train_test_split

X = test[['est','est_cluster','est_svd','est_svdpp','est_nmf']]
y = test['r_ui']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)


from sklearn.linear_model import LinearRegression

reg = LinearRegression()

train_rmse, test_rmse = train_test_model_performance(reg, X_train, y_train, X_test, y_test)

print('train rmse', train_rmse)
print('test rmse', test_rmse)


from sklearn.linear_model import Ridge

rig = Ridge(alpha=10)

train_rmse, test_rmse = train_test_model_performance(rig, X_train, y_train, X_test, y_test)

print('train rmse', train_rmse)
print('test rmse', test_rmse)


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=20, max_depth=7, min_samples_leaf=10, max_features=3)

train_rmse, test_rmse = train_test_model_performance(rf, X_train, y_train, X_test, y_test)

print('train rmse', train_rmse)
print('test rmse', test_rmse)

