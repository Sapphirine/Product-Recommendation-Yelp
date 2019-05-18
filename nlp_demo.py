import csv
import pandas as pd
import numpy as np
df = pd.read_csv('all_33_date.csv')
df.head()

#clean data 
import re  
import nltk  
nltk.download('stopwords') 
# to remove stopword 
from nltk.corpus import stopwords 
#from nltk.stem.porter import PorterStemmer 
# to append clean text  
corpus = []   
for i in range(0, 20000):  
    review = re.sub('[^a-zA-Z]', ' ', df['text'][i])  
    review = review.lower()  
    review = review.split()   
    ps = PorterStemmer()      
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]  
    review = ' '.join(review)   
    corpus.append(review)
print(corpus)

# Creating the Bag of Words model 
from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer()
X = cv.fit_transform(corpus)

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(X, X)
cosine_sim

indices = pd.Series(df.index, index=df['name']).drop_duplicates()

def get_recommendations(name, cosine_sim=cosine_sim):

    idx = indices[name]
    for idx in range(0, 20000):
        cos_scores = list(enumerate(cosine_sim[idx]))
        cos_scores = sorted(cos_scores, key=lambda x: x[1], reverse=True)
        cos_scores = cos_scores[1:11]
        restaurant_indices = [i[0] for i in cos_scores]
    return df['name'].iloc[restaurant_indices]