
# coding: utf-8

# In[7]:

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.decomposition import PCA


# # Set up Test/Train for Clustering

# In[8]:

## IMPORT latest dataset:

data = pd.read_csv('all_features5.csv',index_col = None)
data = data.drop('Unnamed: 0',axis = 1)
data.shape


# In[9]:

data_clean = data.dropna(axis=0, how='any')
#data_clean = data

data_clean.shape


# In[10]:

#X = data_clean.iloc[:,145:171]
X = data_clean

X = X.drop(['filenum','filename','classified_shape'] , axis = 1)

Y = data_clean['classified_shape']
X.shape


# In[117]:

SMALL_X = data_clean.drop(['0','1','2','3','4','5','6','7','8','9','10','11',	'12',	'13',	'14',	'15',	'16','17',
                             '18',	'19',	'20',	'21',	'22',	'23',	'24','25',	'26',	'27',	'28',	'29',
                             '30',	'31',	'32',	'33',	'34',	'35',	'36',	'37',	'38',	'39',	'40',	'41',
                             '42',	'43',	'44',	'45',	'46',	'47',	'48',	'49',	'50',	'51',	'52',	'53',
                             '54',	'55',	'56',	'57',	'58',	'59',	'60',	'61',	'62',	'63',	'64',	'65',
                             '66',	'67',	'68',	'69',	'70',	'71',	'72',	'73',	'74',	'75',	'76',	'77',
                             '78',	'79',	'80',	'81',	'82',	'83',	'84',	'85',	'86',	'87',	'88',	'89',
                             '90',	'91',	'92',	'93',	'94',	'95',	'96',	'97',	'98',	'99',	'100',	'101',
                             '102',	'103',	'104',	'105',	'106',	'107',	'108',	'109',	'110',	'111',	'112',	'113',
                             '114',	'115',	'116',	'117',	'118',	'119',	'120',	'121',	'122',	'123',	'124',	'125',
                             '126',	'127',	'128',	'129',	'130',	'131',	'132',	'133',	'134',	'135',	'136',	'137',
                             '138',	'139',	'140',	'141',	'142',	'143'
                             ,'A1','A2','A3','A4','A5','A6','A7','A8'
                             ,'A9','A10','A11','A12','A13','A14','A15'
                            ,'Height','Width'
                            ,'MJ_width','Jaw_width'
                            #'H_W_Ratio','J_F_Ratio','MJ_J_width'
                               ],axis = 1)
#corrmat = SMALL_X.corr()

# Set up the matplotlib figure.
#f, ax = plt.subplots(figsize=(10, 10))

# Draw the heatmap using seaborn
#sns.heatmap(corrmat,vmin= -1, vmax=1, square=True)
#plt.show()


# # Supervised Learning

# In[40]:

# Standardize features by removing the mean and scaling to unit variance

scaler = StandardScaler()  
scaler.fit(X)  

X = scaler.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X,Y,
    test_size=0.25,
    random_state=None)


# ### Use PCA for dimension reduction

# In[79]:

n_components = 18
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X)


#print(
#    'The percentage of total variance in the dataset explained by each',
#    'component from Sklearn PCA.\n',
#    pca.explained_variance_ratio_
#)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


# In[80]:

# #Remove PCA 

X_train_pca = X_train
X_test_pca = X_test


# ## Neural Network (MLP)

# In[81]:

mlp = MLPClassifier(hidden_layer_sizes=(60,10,60,30), solver='sgd',shuffle  = True, 
                    learning_rate_init=0.01, max_iter = 1000,warm_start  = False)
#reducing the learning rate init allowed the MLP to converge 
mlp.fit(X_train_pca, Y_train)
mlp.score(X_train_pca, Y_train)


# In[82]:

#print(mlp.score(X_train_pca,Y_train))
mlp_score = mlp.score(X_test_pca,Y_test)
#print(mlp_score)

y_pred = mlp.predict(X_test_pca)
 
mlp_crosstab = pd.crosstab(Y_test, y_pred, margins=True)
mlp_crosstab


# In[83]:

from sklearn.model_selection import cross_val_score
cross_val_score(mlp, X, Y, cv=5)


# In[84]:

#print(classification_report(Y_test,y_pred))


# In[85]:

# Get the RECALL for each shape and overall
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

# In[86]:

#neigh = KNeighborsClassifier(n_neighbors=9,weights='distance')

#determined 9 was best through experimentation, wighting by distance led to overfitting

neigh = KNeighborsClassifier(n_neighbors=9) 
neigh.fit(X_train_pca, Y_train) 


# In[87]:

#print(neigh.score(X_train_pca,Y_train))
#print(neigh.score(X_test_pca,Y_test))
y_pred = neigh.predict(X_test_pca)

KNN_crosstab = pd.crosstab(Y_test, y_pred,margins = True) 
KNN_crosstab


# In[88]:

#print(cross_val_score(neigh, X, Y, cv=5))
#print(classification_report(Y_test,y_pred))


# In[89]:

correct_list =[]
for i in KNN_crosstab.index[0:5]:
    correct = (KNN_crosstab.at[i,i]/KNN_crosstab.at[i,'All'])
    correct = round(correct,2)* 100
    correct_list.append(correct)

correct_list.append(round(neigh.score(X_test_pca,Y_test),2)*100)
results_df['KNN']=correct_list


# ### Random Forest Classifier

# In[90]:


clf = RandomForestClassifier(max_depth=18, random_state=0,n_estimators=20)
clf.fit(X_train_pca, Y_train)


# In[91]:

#print(clf.score(X_train_pca,Y_train))
#print(clf.score(X_test_pca,Y_test))
y_pred = clf.predict(X_test_pca)

rfc_crosstab = pd.crosstab(Y_test, y_pred,margins = True) 
rfc_crosstab


# In[92]:

#print(cross_val_score(clf, X, Y, cv=5))
#print(classification_report(Y_test,y_pred))


# In[93]:

correct_list =[]
for i in rfc_crosstab.index[0:5]:
    correct = (rfc_crosstab.at[i,i]/rfc_crosstab.at[i,'All'])
    correct = round(correct,2)* 100
    correct_list.append(correct)

correct_list.append(round(clf.score(X_test_pca,Y_test),2)*100)
results_df['Random_Forest']=correct_list


# ### Gradient Boosting

# In[94]:

# GB is by far the slowest model to run


# In[95]:

# We'll make 500 iterations, use 2-deep trees, and set our loss function.
params = {'n_estimators': 500,
          'max_depth': 2,
          'loss': 'deviance'}

# Initialize and fit the model.
gb = ensemble.GradientBoostingClassifier(**params)
gb.fit(X_train_pca, Y_train)

predict_train = gb.predict(X_train_pca)
predict_test = gb.predict(X_test_pca)


# In[96]:

# Accuracy tables.
table_train = pd.crosstab(Y_train, predict_train, margins=True)
table_test = pd.crosstab(Y_test, predict_test, margins=True)

#print(gb.score(X_train_pca,Y_train))
#print(gb.score(X_test_pca,Y_test))
table_test


# In[97]:

#print(cross_val_score(gb, X, Y, cv=5))
#print(classification_report(Y_test,predict_test))


# In[98]:

correct_list =[]
for i in table_test.index[0:5]:
    correct = (table_test.at[i,i]/table_test.at[i,'All'])
    correct = round(correct,2)* 100
    correct_list.append(correct)

correct_list.append(round(gb.score(X_test_pca,Y_test),2)*100)
results_df['Gradient_Boosting']=correct_list


# ## Linear Discriminant Analysis

# In[99]:


lda = LinearDiscriminantAnalysis(n_components = 10)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_pca, Y_train)


# In[100]:

#print(lda.score(X_train_pca, Y_train))
#print(lda.score(X_test_pca, Y_test))

predict_test = lda.predict(X_test_pca)
table_test = pd.crosstab(Y_test, predict_test, margins=True)
table_test


# In[101]:

#print(cross_val_score(lda, X, Y, cv=5))
#print(classification_report(Y_test,predict_test))


# In[102]:

correct_list =[]
for i in table_test.index[0:5]:
    correct = (table_test.at[i,i]/table_test.at[i,'All'])
    correct = round(correct,2)* 100
    correct_list.append(correct)

correct_list.append(round(lda.score(X_test_pca,Y_test),2)*100)
results_df['LDA']=correct_list
results_df


# In[103]:

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

# In[104]:

results_df




