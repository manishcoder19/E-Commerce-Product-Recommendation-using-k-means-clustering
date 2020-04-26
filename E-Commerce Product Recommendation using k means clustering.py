
# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline
plt.style.use("ggplot")

import sklearn
from sklearn.decomposition import TruncatedSVD


# In[2]:


amazon_ratings = pd.read_csv(r'C:\Users\hp\Desktop\ratings_Beauty.csv')
amazon_ratings = amazon_ratings.dropna()
amazon_ratings.head()


# In[3]:


amazon_ratings.shape


# In[4]:


popular_products = pd.DataFrame(amazon_ratings.groupby('ProductId')['Rating'].count())
most_popular = popular_products.sort_values('Rating', ascending=False)
most_popular.head(10)


# In[5]:


most_popular.head(30).plot(kind = "bar")


# In[6]:


amazon_ratings1 = amazon_ratings.head(10000)


# In[7]:


ratings_utility_matrix = amazon_ratings1.pivot_table(values='Rating', index='UserId', columns='ProductId', fill_value=0)
ratings_utility_matrix.head()


# In[8]:


ratings_utility_matrix.shape


# In[9]:


X = ratings_utility_matrix.T
X.head()


# In[10]:


X.shape


# In[11]:


X1 = X


# In[12]:


SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)
decomposed_matrix.shape


# In[13]:


correlation_matrix = np.corrcoef(decomposed_matrix)
correlation_matrix.shape


# In[14]:


X.index[99]


# In[15]:


i = "6117036094"

product_names = list(X.index)
product_ID = product_names.index(i)
product_ID


# In[16]:


correlation_product_ID = correlation_matrix[product_ID]
correlation_product_ID.shape


# In[17]:


Recommend = list(X.index[correlation_product_ID > 0.90])

# Removes the item already bought by the customer
Recommend.remove(i) 

Recommend[0:9]


# In[18]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


# In[19]:


product_descriptions = pd.read_csv(r'C:\Users\hp\Desktop\descriptions.csv')
product_descriptions.shape


# In[20]:


product_descriptions = product_descriptions.dropna()
product_descriptions.shape
product_descriptions.head()


# In[21]:


product_descriptions1 = product_descriptions.head(500)
# product_descriptions1.iloc[:,1]

product_descriptions1["product_description"].head(10)


# In[22]:


vectorizer = TfidfVectorizer(stop_words='english')
X1 = vectorizer.fit_transform(product_descriptions1["product_description"])
X1


# In[24]:


X=X1

kmeans = KMeans(n_clusters = 10, init = 'k-means++')
y_kmeans = kmeans.fit_predict(X)
plt.plot(y_kmeans, ".")
plt.show()


# In[26]:


def print_cluster(i):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print


# In[27]:



true_k = 10

model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X1)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print_cluster(i)


# In[28]:


def show_recommendations(product):
    #print("Cluster ID:")
    Y = vectorizer.transform([product])
    prediction = model.predict(Y)
    #print(prediction)
    print_cluster(prediction[0])


# In[29]:


show_recommendations("cutting tool")


# In[30]:


show_recommendations("spray paint")


# In[31]:


show_recommendations("steel drill")


# In[32]:


show_recommendations("water")


# In[ ]:




