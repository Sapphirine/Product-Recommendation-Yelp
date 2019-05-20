import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# from main import method0

def data_clean(df, feature, m):
    count = df[feature].value_counts()
    df = df[df[feature].isin(count[count > m].index)]
    return df
def data_clean_sum(df,features,m):
    fil = df.business_id.value_counts()
    fil2 = df.user_id.value_counts()
    
    df['#restaurant'] = df.business_id.apply(lambda x: fil[x])
    df['#Users'] = df.user_id.apply(lambda x: fil2[x])

    while (df.business_id.value_counts(ascending=True)[0]) < m or  (df.user_id.value_counts(ascending=True)[0] < m):

        df = data_clean(df,features[0],m)
        df = data_clean(df,features[1],m)
        
    return df

# check if it is correct


def data():
    df = pd.read_csv('all_over_100.csv')
    raw_data = data_clean_sum(df, ['business_id', 'user_id'], 10)
    # find X,and y
    raw_data['uid'] = pd.factorize(raw_data['user_id'])[0]
    raw_data['pid'] = pd.factorize(raw_data['business_id'])[0]
    sc = MinMaxScaler()

    raw_data['nuser']=sc.fit_transform(raw_data['#Users'].values.reshape(-1,1))
    raw_data['nproduct']=sc.fit_transform(raw_data['#restaurant'].values.reshape(-1,1))
  
    X1 = raw_data.loc[:,['uid','pid']]
    y = raw_data.stars
    # train_test split
    X1_train,X1_test,y_train,y_test = train_test_split(X1,y,test_size=0.3,random_state=2006)

    train = np.array(X1_train.join(y_train))
    test = np.array(X1_test.join(y_test))
    # got the productId to pid index
    pid2PID = raw_data.business_id.unique()
    uid2UID = raw_data.user_id.unique()

    data_mixed = X1.join(y)
    total_p = data_mixed['pid'].unique().shape[0]
    total_u = data_mixed['uid'].unique().shape[0]
    # make the user-item table
    table = np.zeros([total_u,total_p])
    z = np.array(data_mixed)
    return z, total_u,total_p,pid2PID,train,test,table,raw_data
    
    print(table.shape)
    print(total_u)
z, total_u,total_p,pid2PID,train,test,table,raw_data = data()

def rec(result, uid,n,rawId= False):
    if uid in range(total_u):
        #a=[]
        top_N = np.argpartition(result[uid],-n)[-n:]

           # a=data_100.loc[data_100['business_id'] == pid2PID[top_N], 'name'].iloc[0]
        print('the top{} recommanded restaurants for user {} are {}'.format(n,uid,top_N))
        print('the real ids are {}'.format(pid2PID[top_N]))
        print("the real names of those restaurants are:")
        for i in pid2PID[top_N]:    
            print(data_100.loc[data_100['business_id'] == i, 'name'].iloc[0])
        if rawId == True:
            print('the real ID is {}'.format(pid2PID[top_N]))
    else:
        print('this user has not bought anything, plz use other methods')
    return top_N

from sklearn.metrics.pairwise import pairwise_distances
def cf(table = table,distance = 'cosine'):
    user_similarity = pairwise_distances(table, metric=distance)
    item_similarity = pairwise_distances(table.T, metric=distance)
    sc = MinMaxScaler(feature_range=(1,5))
    a = sc.fit_transform(np.dot(user_similarity,table).dot(item_similarity))
    return a
result =cf()

rec(result, 300,10,rawId= False)