#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import silhouette_score, silhouette_samples
import sklearn.metrics

import itertools
import scipy

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))


# In[4]:


import os
os.getcwd()


# In[5]:


df = pd.read_csv("OJ.csv")

df.rename( columns={'Unnamed: 0':'ID'}, inplace=True )


# In[6]:


Id_col = 'ID'
target_col = 'Purchase'
df.info()
df.head()
    
#print(soda)


# In[7]:


#check for imbalance in target variable

df['Purchase'].value_counts()

#CH    653
#MM    417
#Name: Purchase, dtype: int64


# In[8]:


## descriptive analysis
df.describe()


# In[9]:


cleanup_nums = {"Purchase":     {"CH": 0, "MM": 1},
                "Store7": {"Yes": 1, "six": 6, "No": 0}}


# In[10]:


df.replace(cleanup_nums, inplace=True)
df.head()


# In[11]:


#check for imbalance in target variable after converting to binary
#CH = 0, MM = 1

df['Purchase'].value_counts()


#0    653
#1    417
#Name: Purchase, dtype: int64
#There are 61% CH Citrus Hill purchasers and 39% MM Minute Maid purchasers
#We do not need to balance our data in this case as we have good representation from both classes


# In[13]:


## correlation
df.corr()


# In[14]:


corr = df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,

)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'

);

#size(c1, c2) ~ abs(corr(c1, c2))


# In[15]:



X = df.iloc[:,0:18]  #independent columns
y = df.iloc[:,-1]    #target column i.e price range
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(18,18))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[23]:


#Feature Importance
#Identifying most important features using ExtraTrees classifier

X = df.iloc[:,2:17]  #independent columns
y = df.iloc[:,1]    #target column i.e price range
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(12).plot(kind='barh')
plt.show()


# In[21]:


y.head()


# In[22]:


# printing out all column names
data_top = df.head()

data_top


# In[14]:


from sklearn.model_selection import train_test_split

# droping some features
X = df.drop([Id_col, target_col,"PctDiscCH","PctDiscMM","PriceCH","PriceMM"], axis=1)
#X = df.drop([Id_col, target_col,"PctDiscCH","ListPriceDiff","PctDiscMM","STORE"], axis=1)
y = df[target_col]

#splitting data into train and test (80% train and 20% test)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)




# In[15]:


##check the shape of the training set 
X_train.shape


# In[16]:


## shape of the test size
X_val.shape


# In[17]:


y_val.shape


# In[18]:


X.info()
X.shape
X.head()

X_train.info()
X_train.shape
X_train.head()


# # Build Model 1 - DecisionTree classifier
# 

# In[29]:


# tuning our Decision Tree  classifier

from sklearn.tree import DecisionTreeClassifier 

clf2 = DecisionTreeClassifier(random_state=42, criterion="gini",min_samples_split= 4, min_samples_leaf=5, max_depth=50, max_leaf_nodes=10)
#this gave an accuracy of 0.8 which was the best I could get after trying different hyper parameters

clf2.fit(X_train, y_train)

pred_val2 = clf2.predict(X_val)
pred_val2

confusion_matrix(y_val, pred_val2)


# # Estimate Model Performance

# In[33]:


# performance metrics for tuned Decision Tree model


from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, log_loss

print("Accuracy = {:.2f}".format(accuracy_score(y_val, pred_val2)))
print("Kappa = {:.2f}".format(cohen_kappa_score(y_val, pred_val2)))
print("F1 Score = {:.2f}".format(f1_score(y_val, pred_val2)))
print("Log Loss = {:.2f}".format(log_loss(y_val, pred_val2)))


# In[32]:


from sklearn.metrics import classification_report, f1_score

print("Accuracy = {:.2f}".format(accuracy_score(y_val, pred_val2)))
print()
print(classification_report(y_val, pred_val2))


# In[20]:


# ROC CURVE

from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, auc

# Adopted from: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html


def plot_boundaries(X_train, X_test, y_train, y_test, clf, clf_name, ax, hide_ticks=True, show_train=True, show_test=True):
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02));
    
    
    score = clf.score(X_test, y_test);

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]);
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1];

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8);

    if show_train:
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap=cm_bright, edgecolors='k', alpha=0.6);
        
    if show_test:
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap=cm_bright, edgecolors='k', alpha=0.6);

    ax.set_xlim(xx.min(), xx.max());
    ax.set_ylim(yy.min(), yy.max());
    if hide_ticks:
        ax.set_xticks(());
        ax.set_yticks(());
    else:
        ax.tick_params(axis='both', which='major', labelsize=18)
        #ax.yticks(fontsize=18);
        
    ax.set_title(clf_name, fontsize=28);
    
    if show_test:
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=35, horizontalalignment='right');
    ax.grid();
    
    

def plot_roc(clf, X_test, y_test, name, ax, show_thresholds=True):
    y_pred_rf = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thr = roc_curve(y_test, y_pred_rf)

    ax.plot([0, 1], [0, 1], 'k--');
    ax.plot(fpr, tpr, label='{}, AUC={:.2f}'.format(name, auc(fpr, tpr)));
    ax.scatter(fpr, tpr);

    if show_thresholds:
        for i, th in enumerate(thr):
            ax.text(x=fpr[i], y=tpr[i], s="{:.2f}".format(th), fontsize=14, 
                     horizontalalignment='left', verticalalignment='top', color='black',
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1', alpha=0.1));
        
    ax.set_xlabel('False positive rate', fontsize=18);
    ax.set_ylabel('True positive rate', fontsize=18);
    ax.tick_params(axis='both', which='major', labelsize=18);
    ax.grid(True);
    ax.set_title('ROC Curve', fontsize=18)


# In[36]:



plt.style.use('default');
figure = plt.figure(figsize=(10, 6));    
ax = plt.subplot(1, 1, 1);
plot_roc(clf2, X_val, y_val, "Decision Tree", ax)
plt.legend(loc='lower right', fontsize=18);
plt.tight_layout();


# # Model 2 - XGBOOST
# 
# 

# In[21]:


import xgboost as xgb


# In[22]:


from sklearn.model_selection import train_test_split


# droping some features
X = df.drop([Id_col, target_col,"PctDiscCH","ListPriceDiff","PctDiscMM","PriceMM","PriceCH"], axis=1)
y = df[target_col]

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


#Train the XGboost Model for Classification
model1 = xgb.XGBClassifier()
model2 = xgb.XGBClassifier(n_estimators=30, max_depth=30, learning_rate=0.02, subsample=0.3)

#model2 = xgb.XGBClassifier(n_estimators=30, max_depth=30, learning_rate=0.02, subsample=0.3)
#Accuracy for model 2: 84.11 ----use this hyperparameter

train_model1 = model1.fit(X_train, y_train)
train_model2 = model2.fit(X_train, y_train)


# In[24]:


#prediction and Classification Report
from sklearn.metrics import classification_report

pred1 = train_model1.predict(X_val)
pred2 = train_model2.predict(X_val)

print('Model 1 XGboost Report %r' % (classification_report(y_val, pred1)))
print('Model 2 XGboost Report %r' % (classification_report(y_val, pred2)))


# In[25]:


#Let's use accuracy score
from sklearn.metrics import accuracy_score, f1_score

print("Accuracy for model 1: %.2f" % (accuracy_score(y_val, pred1) * 100))
print("Accuracy for model 2: %.2f" % (accuracy_score(y_val, pred2) * 100))

print("F1 Score for model 1: %.2f" % (f1_score(y_val, pred1) * 100))
print("F1 Score for model 2: %.2f" % (f1_score(y_val, pred2) * 100))



# In[26]:


# CONFUSION MATRIX FOR MODEL 2

confusion_matrix(y_val, pred2)


# In[27]:


# CONFUSION MATRIX FOR MODEL 1

confusion_matrix(y_val, pred1)


# # Model 3 - Random Forest

# In[39]:


df = pd.read_csv("OJ.csv")

df.rename( columns={'Unnamed: 0':'ID'}, inplace=True )
Id_col = 'ID'
target_col = 'Purchase'
df.info()
df.head()
    


# In[40]:


cleanup_nums = {"Purchase":     {"CH": 0, "MM": 1},
                "Store7": {"Yes": 1, "six": 6, "No": 0}}


# In[41]:


df.replace(cleanup_nums, inplace=True)
df.head()


# In[42]:


from sklearn.model_selection import train_test_split

# droping some features
X = df.drop([Id_col, target_col,"PctDiscCH","PriceCH","PctDiscMM","PriceMM"], axis=1)
y = df[target_col]



# In[43]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

##check the shape of the training set 
X_train.shape


# In[44]:


## shape of the test size
X_val.shape


y_val.shape


# In[45]:


X.info()
X.shape
X.head()

X_train.info()
X_train.shape
X_train.head()


# In[67]:


# Random Forest

model_rf1 = RandomForestClassifier(n_estimators = 10, min_samples_split= 4, min_samples_leaf=5, 
                                   max_depth=50, max_features=10)

#model,Summary_RF = PrintResults(model, X_train,y_train, 'Random Forest')
#y_train_RF = pd.Series(model.predict(X_train), name = "RF")
#y_test_RF = pd.Series(model.predict(X_test), name = "RF")


# In[68]:


model_rf1.fit(X_train, y_train)


# In[69]:


from sklearn.metrics import confusion_matrix

pred_val = model_rf1.predict(X_val)
pred_val


confusion_matrix(y_val, pred_val)


# In[70]:


#Let's use accuracy score
from sklearn.metrics import accuracy_score, f1_score

print("Accuracy for model rf1: %.2f" % (accuracy_score(y_val, pred_val) * 100))
#print("Accuracy for model 2: %.2f" % (accuracy_score(y_val, pred2) * 100))

print("F1 Score for model rf1: %.2f" % (f1_score(y_val, pred_val) * 100))
#print("F1 Score for model 2: %.2f" % (f1_score(y_val, pred2) * 100))



# In[ ]:




