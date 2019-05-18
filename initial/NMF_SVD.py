import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from time import time
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

plt.style.use('seaborn')


# NMF
from sklearn.decomposition import NMF

def fit_nmf(M,k):
    nmf = NMF(n_components=k)
    nmf.fit(M)
    W = nmf.transform(M);
    H = nmf.components_;
    err = nmf.reconstruction_err_
    return W,H,err

# decompose
W,H,err = fit_nmf(ratings_mat,200)
print(err)
print(W.shape,H.shape)

ratings_mat_fitted = W.dot(H)
errs = np.array((ratings_mat-ratings_mat_fitted).flatten()).squeeze()
mask = np.array((ratings_mat.todense()).flatten()).squeeze()>0

mse = np.mean(errs[mask]**2)
average_abs_err = abs(errs[mask]).mean()
print(mse)
print(average_abs_err)

# get recommendations for one user
user_id = 2
n = 10

pred_ratings = ratings_mat_fitted[user_id,:]
item_index_sorted_by_pred_rating = list(np.argsort(pred_ratings))[::-1]
items_rated_by_this_user = ratings_mat[user_id].nonzero()[1]
unrated_items_by_pred_rating = [item for item in item_index_sorted_by_pred_rating
                                if item not in items_rated_by_this_user]

print(unrated_items_by_pred_rating[:n])
business = unrated_items_by_pred_rating[:n]
print(df_selected.loc[df_selected['business_unique'].isin(business)].name.unique())

ratings_true = ratings_mat[user_id, items_rated_by_this_user].todense()
ratings_pred = pred_ratings[items_rated_by_this_user]
err_one_user = ratings_true-ratings_pred
print('Training MAE:', abs(err_one_user).mean())


### check errors
# truth
ratings_true = ratings_mat[user_id, items_rated_by_this_user].todense()
# prediction
ratings_pred = pred_ratings[items_rated_by_this_user]
print(list(zip(np.array(ratings_true).squeeze(),ratings_pred)))
err_one_user = ratings_true-ratings_pred
print(err_one_user)
print(abs(err_one_user).mean())


# SVD
from sklearn.decomposition import TruncatedSVD

def fit_uvd(M,k):
    # use TruncatedSVD to realize UVD
    svd = TruncatedSVD(n_components=k, n_iter=7, random_state=0)
    svd.fit(M)

    V = svd.components_
    U = svd.transform(M)
    
    return U,V, svd

# decompose
U,V,svd = fit_uvd(ratings_mat,200)

# reconstruct
ratings_mat_fitted = U.dot(V) # U*V

# calculate errs
errs = np.array((ratings_mat-ratings_mat_fitted).flatten()).squeeze()
mask = np.array((ratings_mat.todense()).flatten()).squeeze()>0

mse = np.mean(errs[mask]**2)
average_abs_err = abs(errs[mask]).mean()
print(mse)
print(average_abs_err)

# reconstruct M with inverse_transform
ratings_mat_fitted_2 = svd.inverse_transform(svd.transform(ratings_mat))
ratings_mat_fitted = U.dot(V)
print(sum(sum(ratings_mat_fitted - ratings_mat_fitted_2)))
# they are just equivalent!!

# get recommendations for one user
user_id = 2
n = 10

pred_ratings = ratings_mat_fitted[user_id,:]
item_index_sorted_by_pred_rating = list(np.argsort(pred_ratings))[::-1]
items_rated_by_this_user = ratings_mat[user_id].nonzero()[1]
unrated_items_by_pred_rating = [item for item in item_index_sorted_by_pred_rating
                                if item not in items_rated_by_this_user]

print(unrated_items_by_pred_rating[:n])
business = unrated_items_by_pred_rating[:n]
print(df_selected.loc[df_selected['business_unique'].isin(business)].name.unique())

ratings_true = ratings_mat[user_id, items_rated_by_this_user].todense()
ratings_pred = pred_ratings[items_rated_by_this_user]
err_one_user = ratings_true-ratings_pred
print('Training MAE:', abs(err_one_user).mean())

### check errors
# truth
ratings_true = ratings_mat[user_id, items_rated_by_this_user].todense()
# prediction
ratings_pred = pred_ratings[items_rated_by_this_user]
err_one_user = ratings_true-ratings_pred
print(abs(err_one_user).mean())

### check errors
# truth
ratings_true = ratings_mat[user_id, items_rated_by_this_user].todense()
# prediction
ratings_pred = pred_ratings[items_rated_by_this_user]
print(list(zip(np.array(ratings_true).squeeze(),ratings_pred)))
err_one_user = ratings_true-ratings_pred
print(err_one_user)
print(abs(err_one_user).mean())
