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

df = pd.read_csv('all_over_100.csv')

# Get business_id, user_id, stars for recommender
df_selected = df[['business_id', 'name', 'user_id', 'stars']]

# assign a unique index for each unique business_id and user_id for easy counting
business_unique = np.unique(df_selected['business_id'], return_inverse = True)
user_unique = np.unique(df_selected['user_id'], return_inverse = True)
df_selected['business_unique'] = business_unique[1]+1
df_selected['user_unique'] = user_unique[1]+1

# num_user = df_rec.user_id.value_counts().count()
highest_user_id = df_selected.user_unique.max()
highest_business_id = df_selected.business_unique.max()
ratings_mat = sparse.lil_matrix((highest_user_id, highest_business_id))
ratings_mat

for _, row in df_selected.iterrows():
    # subtract 1 from id's due to match 0 indexing
    ratings_mat[row.user_unique-1, row.business_unique-1] = row.stars

ratings_mat

# item-item
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from time import time


class ItemItemRecommender(object):

    def __init__(self, neighborhood_size):
        self.neighborhood_size = neighborhood_size

    def fit(self, ratings_mat):
        self.ratings_mat = ratings_mat
        self.n_users = ratings_mat.shape[0]
        self.n_items = ratings_mat.shape[1]
        self.item_sim_mat = cosine_similarity(self.ratings_mat.T)
        self._set_neighborhoods()

    def _set_neighborhoods(self):
        least_to_most_sim_indexes = np.argsort(self.item_sim_mat, 1)
        self.neighborhoods = least_to_most_sim_indexes[:, -self.neighborhood_size:]

    def pred_one_user(self, user_id, report_run_time=False):
        start_time = time()
        items_rated_by_this_user = self.ratings_mat[user_id].nonzero()[1]
        # Just initializing so we have somewhere to put rating preds
        out = np.zeros(self.n_items)
        for item_to_rate in range(self.n_items):
            relevant_items = np.intersect1d(self.neighborhoods[item_to_rate],
                                            items_rated_by_this_user,
                                            assume_unique=True)  # assume_unique speeds up intersection op
            out[item_to_rate] = self.ratings_mat[user_id, relevant_items] * \
                self.item_sim_mat[item_to_rate, relevant_items] / \
                self.item_sim_mat[item_to_rate, relevant_items].sum()
        if report_run_time:
            print("Execution time: %f seconds" % (time()-start_time))
        cleaned_out = np.nan_to_num(out)
        return cleaned_out

    def pred_all_users(self, report_run_time=False):
        start_time = time()
        all_ratings = [
            self.pred_one_user(user_id) for user_id in range(self.n_users)]
        if report_run_time:
            print("Execution time: %f seconds" % (time()-start_time))
        return np.array(all_ratings)

    def top_n_recs(self, user_id, n):
        pred_ratings = self.pred_one_user(user_id)
        item_index_sorted_by_pred_rating = list(np.argsort(pred_ratings))
        items_rated_by_this_user = self.ratings_mat[user_id].nonzero()[1]
        unrated_items_by_pred_rating = [item for item in item_index_sorted_by_pred_rating
                                        if item not in items_rated_by_this_user]
        return unrated_items_by_pred_rating[-n:]


# make prediction
my_rec_engine = ItemItemRecommender(neighborhood_size=75)
my_rec_engine.fit(ratings_mat)
user_1_preds = my_rec_engine.pred_one_user(user_id=2, report_run_time=True)
# Show predicted ratings for user #2 on first 100 items
print(user_1_preds[:100])
print(my_rec_engine.top_n_recs(2, 10))
business = my_rec_engine.top_n_recs(2, 10)
print(df_selected.loc[df_selected['business_unique'].isin(business)].name.unique())