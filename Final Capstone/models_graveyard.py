
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import ensemble
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA


# # Set up Test/Train for Clustering

# In[2]:

## IMPORT latest dataset:

data = pd.read_csv('all_features4.csv',index_col = None)
data = data.drop('Unnamed: 0',axis = 1)
data.shape


# In[3]:

data_clean = data.dropna(axis=0, how='any')
#data_clean = data

data_clean.shape


# In[4]:

X = data_clean.iloc[:,145:171]
X = X.drop(['filenum','filename','classified_shape'] , axis = 1)
X_norm = normalize(X)
Y = data_clean['classified_shape']


# # Supervised Learning

# ## Set up Test/Train for supervised learning

# In[7]:

# Split the data 
 
X_train, X_test, Y_train, Y_test = train_test_split(
    X,Y,
    test_size=0.25,
    random_state=None)

scaler = StandardScaler()  

scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 


# ### Add PCA

# In[13]:


n_components = 10
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# turn off PCA
X_train_pca = X_train
X_test_pca = X_test


# ## Neural Network (MLP)

# In[25]:

mlp = MLPClassifier(hidden_layer_sizes=(60,10,60))
mlp.fit(X_train_pca, Y_train)
mlp.score(X_train_pca, Y_train)



# In[102]:

#print(mlp.score(X_train_pca,Y_train))
mlp_score = mlp.score(X_test_pca,Y_test)
#print(mlp_score)

y_pred = mlp.predict(X_test_pca)
 
mlp_crosstab = pd.crosstab(Y_test, y_pred, margins=True)
#mlp_crosstab


# In[170]:

correct_list =[]
shape_list = []
for i in mlp_crosstab.index[0:5]:
    correct = (mlp_crosstab.at[i,i]/mlp_crosstab.at[i,'All'])
    correct = round(correct,2)* 100
    shape_list.append(i)
    correct_list.append(correct)

shape_list.append('Overall')
correct_list.append(round(mlp_score,2)*100)
results_df = pd.DataFrame()
results_df['shape']= shape_list
results_df['MLP']=correct_list


# ## KNN Classifier

# In[80]:

neigh = KNeighborsClassifier(n_neighbors=9)   #determined 9 was best through experimentation
neigh.fit(X_train_pca, Y_train) 


# In[96]:

#print(neigh.score(X_train_pca,Y_train))
#print(neigh.score(X_test_pca,Y_test))
y_pred = neigh.predict(X_test_pca)

KNN_crosstab = pd.crosstab(Y_test, y_pred,margins = True) 
KNN_crosstab


# In[171]:

correct_list =[]
for i in KNN_crosstab.index[0:5]:
    correct = (KNN_crosstab.at[i,i]/KNN_crosstab.at[i,'All'])
    correct = round(correct,2)* 100
    correct_list.append(correct)

correct_list.append(round(neigh.score(X_test_pca,Y_test),2)*100)
results_df['KNN']=correct_list


# ### Random Forest Classifier

# In[22]:


clf = RandomForestClassifier(max_depth=6, random_state=0)
clf.fit(X_train_pca, Y_train)


# In[109]:

#print(clf.score(X_train_pca,Y_train))
#print(clf.score(X_test_pca,Y_test))
y_pred = clf.predict(X_test_pca)

rfc_crosstab = pd.crosstab(Y_test, y_pred,margins = True) 
#rfc_crosstab


# In[172]:

correct_list =[]
for i in rfc_crosstab.index[0:5]:
    correct = (rfc_crosstab.at[i,i]/rfc_crosstab.at[i,'All'])
    correct = round(correct,2)* 100
    correct_list.append(correct)

correct_list.append(round(clf.score(X_test_pca,Y_test),2)*100)
results_df['Random_Forest']=correct_list


# ### Gradient Boosting

# In[111]:

# We'll make 500 iterations, use 2-deep trees, and set our loss function.
params = {'n_estimators': 500,
          'max_depth': 2,
          'loss': 'deviance'}

# Initialize and fit the model.
gb = ensemble.GradientBoostingClassifier(**params)
gb.fit(X_train_pca, Y_train)

predict_train = gb.predict(X_train_pca)
predict_test = gb.predict(X_test_pca)


# In[112]:

# Accuracy tables.
table_train = pd.crosstab(Y_train, predict_train, margins=True)
table_test = pd.crosstab(Y_test, predict_test, margins=True)

#print(gb.score(X_train_pca,Y_train))
#print(gb.score(X_test_pca,Y_test))
#table_test


# In[173]:

correct_list =[]
for i in table_test.index[0:5]:
    correct = (table_test.at[i,i]/table_test.at[i,'All'])
    correct = round(correct,2)* 100
    correct_list.append(correct)

correct_list.append(round(gb.score(X_test_pca,Y_test),2)*100)
results_df['Gradient_Boosting']=correct_list


# ## Linear Discriminant Analysis

# In[150]:


lda = LinearDiscriminantAnalysis()
lda.fit(X_train_pca, Y_train)


# In[151]:

#print(lda.score(X_train_pca, Y_train))
#print(lda.score(X_test_pca, Y_test))

predict_test = lda.predict(X_test_pca)
table_test = pd.crosstab(Y_test, predict_test, margins=True)
#table_test


# In[174]:

correct_list =[]
for i in table_test.index[0:5]:
    correct = (table_test.at[i,i]/table_test.at[i,'All'])
    correct = round(correct,2)* 100
    correct_list.append(correct)

correct_list.append(round(lda.score(X_test_pca,Y_test),2)*100)
results_df['LDA']=correct_list
results_df


# In[186]:

import matplotlib.pyplot as plt

def model_graph():
    ind = np.arange(6)  # the x locations for the groups
    width = 0.15       # the width of the bars

    fig, ax = plt.subplots(figsize=(15, 7))
    rects1 = ax.bar(ind, results_df['MLP'], width, color='g',tick_label = results_df['shape'])
    rects2 = ax.bar(ind + width, results_df['KNN'], width, color='y')
    rects3 = ax.bar(ind + width*2, results_df['Random_Forest'], width, color='b')
    rects4 = ax.bar(ind + width*3, results_df['Gradient_Boosting'], width, color='r')
    rects5 = ax.bar(ind + width*4, results_df['LDA'], width, color='purple')

    ax.legend(results_df.iloc[0:0,1:7],loc=0)
    plt.show()
    
model_graph()


# The neural network outperformed the other models for overall performance and for four out of the five shapes.
