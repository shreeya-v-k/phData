#!/usr/bin/env python
# coding: utf-8

# In[1]:


#data validation and quality
# inspect dataset, check for quality, make assumptions
#1. request meeting with business, verify assumptions
#2. revise data file

#data modeling

#presentation


# In[2]:


#Ingestion
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
#to ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[3]:


#import machine learning packages and libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score


# In[4]:


#Read in data
df = pd.read_csv("project_data.csv")
df


# ## Descriptive Statistics

# In[5]:


df.shape


# In[6]:


df.describe(include = 'all').T


# In[7]:


df.info()


# Missing value counts and percentages

# In[8]:


df.isnull().sum()


# In[9]:


(df.isnull().sum()/(len(df)))*100


# In[10]:


df.dtypes


# In[11]:


df.nunique()


# In[12]:


# data reduction 
#remove c8 because mostly blank
df = df.drop(['c8'], axis=1)


# In[13]:


#missing values replaced with most frequent
# for column in ['b2', 'school']:
#     df[column].fillna(df[column].mode()[0], inplace=True)


# In[14]:


#rows with missing values in columns 'marriage-status', 'b2','school' removed
df = df[df['marriage-status'].notna()]
df = df[df['b2'].notna()]
df = df[df['school'].notna()]


# In[15]:


#Recheck for null values
df.isnull().sum()


# ### check for class imbalance

# In[16]:


df['successful_sell'].value_counts()


# In[17]:


print("percentage of NO and YES \n", df['successful_sell'].value_counts()*100/len(df))


# the dataset is imbalanced

# # Exploratory Data Analysis (EDA)

# In[18]:


# Separate the features into categorical and numeric
cat_cols=df.select_dtypes(include=['object']).columns
num_cols = df.select_dtypes(include=np.number).columns.tolist()
print("Categorical Variables:")
print(cat_cols)
print("Numeric Variables:")
print(num_cols)


# ### EDA: Univariate analysis

# Categorical Variables

# In[19]:


# plotting bar chart for each categorical variable
plt.style.use("ggplot")

for column in cat_cols:
    plt.figure(figsize=(15,4))
    plt.subplot(121)
    df[column].value_counts().plot(kind="bar")
    plt.title(column)


# In[20]:


#observations
#The plot for the target variable shows heavy imbalance in the target variable.
# most of the people are married, followed by single, then divorced
# most are of the profession "Assistant", followed by laborer then engineer
# dow = day of week, all values are weekdays, no Sat/Sun
# c10, successful_sell. Is this causation?
# n3 is credit score? 
# more sales in may. What is tax season? US tax season is till April. Seems to be a post tax season software if this is US data
# c4 is account status?
#The missing values in some columns have been represented as unknown. unknown represents missing data.


# b1

# In[21]:


#Remove rows with b1 = -1 since that seems to be a faulty value
df = df[df['b1']!= -1]


# c3

# In[22]:


#c3: Dealing with "unknown" values
print("percentage of each value in c3 \n", df['c3'].value_counts()*100/len(df))


# In[23]:


# data reduction (remove c3 because mostly FALSE or blank. Only 3 rows have c3 = TRUE)
df = df.drop(['c3'], axis=1)
cat_cols=df.select_dtypes(include=['object']).columns


# In[24]:


cat_cols


# Numerical Variables

# In[25]:


# plotting histogram for each numerical variable
plt.style.use("ggplot")
for column in num_cols:
    
    plt.figure(figsize=(20,4))
    plt.subplot(121)
    sns.histplot(df[column], kde=True)
    plt.title(column)
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[column])


# In[26]:


# n4, n6, n2: high skewness, needs outlier removal, need log transformation??


# n4

# In[27]:


print("percentage of each value in n4 \n", df['n4'].value_counts()*100/len(df))


# In[28]:


#n4 is a constant value. 96% values are 999. should be removed 


# n6

# In[29]:


print("percentage of each value in n6 \n", df['n6'].value_counts()*100/len(df))


# In[30]:


# mostly 0s. should be removed for the sake of model simplicity.


# In[31]:


df = df.drop(['n4','n6'], axis=1)
num_cols = df.select_dtypes(include=np.number).columns.tolist()


# ### EDA: Bivariate analysis

# In[32]:


plt.style.use("ggplot")
for column in cat_cols:
    plt.figure(figsize=(20,4))
    plt.subplot(121)
    sns.countplot(x = df[column], hue=df["successful_sell"])
    plt.title(column)    
    plt.xticks(rotation=90)


# In[33]:


#observations:
# People who have purchased the software are mostly those that have school = 5 (a lot)
# Purchased more in the May month
# Purchased on a Thursday
# They are married
# Belong to the profession "Assistant"
# Have new accuounts with the company/ They are new buyers), c4
# have b2 = no, b1 = yes
# c10 is same as target var



# In[34]:


#Chi-Square Test: This test is used to derive the statistical significance of relationship between the variables. Also, it tests whether the evidence in the sample is strong enough to generalize that the relationship for a larger population as well. Chi-square is based on the difference between the expected and observed frequencies in one or more categories in the two-way table. It returns probability for the computed chi-square distribution with the degree of freedom.
#https://medium.com/@ritesh.110587/correlation-between-categorical-variables-63f6bd9bf2f7


# In[35]:


# cat_cols


# In[36]:


## Let us split this list into two parts
# cat_var1 = ('b1', 'b2', 'c10', 'c4', 'dow', 'employment', 'marriage-status',
#        'month', 'school', 'successful_sell')
# cat_var2 = ('b1', 'b2', 'c10', 'c4', 'dow', 'employment', 'marriage-status',
#        'month', 'school', 'successful_sell')
# cat_var_prod = list(product(cat_var1,cat_var2, repeat = 1))


# In[37]:


# cat_var_prod


# In[38]:


# df_cat = df.select_dtypes(include=['object'])

# import scipy.stats as ss
# result = []
# for i in cat_var_prod:
#     if i[0] != i[1]:
#         result.append((i[0],i[1],list(ss.chi2_contingency(pd.crosstab(
#                             df_cat[i[0]], df_cat[i[1]])))[1]))



# In[39]:


# result


# In[40]:


# chi_test_output = pd.DataFrame(result, columns = ['var1', 'var2', 
#                                                        'coeff'])
# ## Using pivot function to convert the above DataFrame into a crosstab
# chi_test_output.pivot(index='var1', columns='var2', values='coeff')


# In[41]:


#There exists a relationship between two variables if p value ≤ 0.05. 

#FRom the above table we can say that thre is definitely some association between SS & (b1,c10,c4,dow,employment, marriage status, month and school)


# ## Remove outliers
# 

# In[42]:


df[num_cols].describe()


# In[43]:


# n2 is skewed to the right so remove outiers
#compute interquantile range to calculate the boundaries
col = "n2"
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
# identify outliers
outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
# drop rows containing outliers
df = df.drop(outliers.index)


# In[44]:


df[num_cols].describe()


# ## Encoding categorical variables

# In[45]:


#Machine learning algorithm can only read numerical values. It is therefore essential to encode categorical features into numerical values


# In[46]:


# check categorical class
for i in cat_cols:
    print(i, ":", df[i].unique())


# In[47]:


#we will Label Encode them as One Hot Encoding would create so many columns


# In[48]:


# initializing label encoder
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()

# iterating through each categorical feature and label encoding them
for feature in cat_cols:
    df[feature]= le.fit_transform(df[feature])


# In[49]:


df.head()


# ## Checking Correlation of feature variables¶
# 

# In[50]:


plt.figure(figsize=(20,10))
sns.heatmap(round(df.corr(),2), annot=True)
#sns.heatmap(df.corr())


# In[51]:


#High corelation between i1,i4,i5 (close to 1) and poor corelation with successful_sell
#c10 is identical to the target var successful_Sell, highly corelated
#these features can be removed for the sake of model simplicity and speed.keeping these will just addd to complexity of model


# In[52]:


df = df.drop(['i4','i1','i5','c10'], axis=1)


# ## Separating independent and dependent variables

# In[53]:


# feature variables
x= df.iloc[:, :-1]

# target variable
y= df.iloc[:, -1]


# ### Handling imbalanced dataset
# 

# In[54]:


#Since the class distribution in the target variable is ~89:11 indicating an imbalance dataset, we need to resample it.


# In[55]:


#initialising oversampling
from imblearn.over_sampling import SMOTE
smote= SMOTE()

#implementing oversampling to training data
x_sm, y_sm= smote.fit_resample(x,y)

# x_sm and y_sm are the resampled data

# target class count of resampled dataset
y_sm.value_counts()


# ### Separating into test and train dataset

# In[56]:


x_train, x_test, y_train, y_test= train_test_split(x_sm, y_sm, test_size=0.2, random_state=42)


# In[57]:


print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)


# ## Model fitting and hyperparameter tuning

# ### Random Forest model

# In[ ]:


rf = RandomForestClassifier() 

# Define the parameter grid for RandomizedSearchCV
rf_param= { 
           "n_estimators": [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
           "max_features": ["auto", "sqrt", "log2"],
           "max_depth": [int(x) for x in np.linspace(start=5, stop=30, num=6)],
           "min_samples_split": [5,10,15,100],
           "min_samples_leaf": [1,2,5,10],
           "criterion":['gini', 'entropy'] 
          }

randomsearch_rf= RandomizedSearchCV(rf, param_distributions=rf_param, n_iter=10,
                                   cv=5, scoring='accuracy', random_state=42, n_jobs=-1, verbose = 2)

randomsearch_rf.fit(x_train, y_train)


print("best score is:", randomsearch_rf.best_score_)
print("best parameters are:", randomsearch_rf.best_params_)


# Check model performance

# In[ ]:


y_predicted_rf= randomsearch_rf.predict(x_test)

# Get predicted class probabilities for the test set 
y_predicted_rf_prob = randomsearch_rf.predict_proba(x_test)[:, 1] 


# In[ ]:


# Compute the false positive rate (FPR)  
# and true positive rate (TPR) for different classification thresholds 
fpr, tpr, thresholds = roc_curve(y_test, y_predicted_rf_prob, pos_label=1)
# Compute the ROC AUC score 
roc_auc = roc_auc_score(y_test, y_predicted_rf_prob) 
roc_auc


# In[ ]:


# Plot the ROC curve 
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc) 
# roc curve for tpr = fpr  
#plt.plot([0, 1], [0, 1], 'k--', label='Random classifier') 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve') 
plt.legend(loc="lower right") 
plt.show()


# In[ ]:


print(confusion_matrix(y_test, y_predicted_rf))
sns.heatmap(confusion_matrix(y_test, y_predicted_rf), annot=True)
print(accuracy_score(y_test, y_predicted_rf))
print(classification_report(y_test, y_predicted_rf))


# In[ ]:


print (randomsearch_rf.score(x_train, y_train))
print(randomsearch_rf.score(x_test, y_test))


# ## Gradient Boost model

# In[ ]:


# Import the model we are using
from sklearn.ensemble import GradientBoostingClassifier

# Define the parameter grid for RandomizedSearchCV
gb_param = {
    'n_estimators': np.arange(50, 251, 50),
    'learning_rate': np.linspace(0.01, 0.2, 10),
    'max_depth': np.arange(3, 8),
}
# Initialize the Gradient Boosting model
gb = GradientBoostingClassifier()
 
# Initialize RandomizedSearchCV
randomsearch_gb = RandomizedSearchCV(estimator=gb, param_distributions=gb_param, n_iter=10,
                                   cv=5, scoring='accuracy', random_state=42, n_jobs=-1, verbose = 2)


randomsearch_gb.fit(x_train, y_train)


print("best score is:", randomsearch_gb.best_score_)
print("best parameters are:", randomsearch_gb.best_params_)


# In[ ]:


# checking model performance
y_predicted_gb= randomsearch_gb.predict(x_test)


# Get predicted class probabilities for the test set 
y_predicted_gb_prob = randomsearch_gb.predict_proba(x_test)[:, 1] 

# Compute the false positive rate (FPR)  
# and true positive rate (TPR) for different classification thresholds 
fpr, tpr, thresholds = roc_curve(y_test, y_predicted_gb_prob, pos_label=1)
# Compute the ROC AUC score 
roc_auc = roc_auc_score(y_test, y_predicted_gb_prob) 
roc_auc


# In[ ]:


# Plot the ROC curve 
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc) 
# roc curve for tpr = fpr  
#plt.plot([0, 1], [0, 1], 'k--', label='Random classifier') 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve') 
plt.legend(loc="lower right") 
plt.show()


# In[ ]:


print(confusion_matrix(y_test, y_predicted_gb))
sns.heatmap(confusion_matrix(y_test, y_predicted_gb), annot=True)
print(accuracy_score(y_test, y_predicted_gb))
print(classification_report(y_test, y_predicted_gb))


# In[ ]:


print (randomsearch_gb.score(x_train, y_train))
print(randomsearch_gb.score(x_test, y_test))


# In[ ]:




