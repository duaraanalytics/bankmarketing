
# load pandas for 
##   1. reading various files into the dataframe
##   2. to performa various data manipulation tasks

import pandas as pd

# load numpy
import numpy as np
# for preprocessing
from sklearn import preprocessing

# for custom transformer
from sklearn.base import BaseEstimator, TransformerMixin

# for creating pipeline
from sklearn.pipeline import Pipeline, FeatureUnion

# for cross validation
#from sklearn import cross_validation
from sklearn import model_selection
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_predict, cross_val_score

# for various metrics and reporting
from sklearn import metrics 
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# feature selection
from sklearn.feature_selection import SelectFromModel

# xgboost library
from xgboost import XGBClassifier

# plot feature importance
from xgboost import plot_importance, plot_tree


# load matplot lib for various plotting
import matplotlib.pyplot as plt
plt.rc("font", size=14)

# we will use the seaborn for visually appealing plots
import seaborn as sns
sns.set() # set the seaborn stylesheet
#sns.set(style="white")
#sns.set(style="whitegrid")



# load csv file
# if you are using notebook on your laptop. use following to load
data_raw_all = pd.read_csv("/Users/dixon.maina/Downloads/bank/bank.csv", header=0, sep=";")


# preview 
print("(num_rows, num_cols) =", data_raw_all.shape)
print("attributes =", list(data_raw_all.columns))

print ("\n\n'Bank data set preview'")
data_raw_all.head()

# schema
print("'Data set schema'")
data_raw_all.dtypes

# clean data
## cleaning routines
def clean_data(df):
    """
     Clean the data for the exploratory analysis
     
     arguements:
     df -- pandas dataframe.
     
    """
    # drop the missing data row
    data = df.dropna()
    
    # first convert the day type to object as day is not of int64 type but a categorical type
    data['day'] = df.astype('object')

   
    return data

# 
data_ex = clean_data(data_raw_all.sample(frac=1.0))

#DO EDA on our data
#1.Count plot
sns.countplot(x='y', data=data_ex, palette='husl')
plt.show()

#2.Statistical description of the dataset
data_ex.describe()

# Includes categorical variable 
data_ex.describe(include='all')



col_names = ['age','duration', 'campaign', 'pdays', 'previous', 'balance']

fig, ax = plt.subplots(len(col_names), figsize=(16,12))

for i, col_val in enumerate(col_names):
        
    sns.distplot(data_ex[col_val], hist=True, ax=ax[i])
    ax[i].set_title('Freq dist '+col_val, fontsize=10)
    ax[i].set_xlabel(col_val, fontsize=8)
    ax[i].set_ylabel('Count', fontsize=8)
    
plt.show()

#outlier detection (boxplots)
col_names = ['age','duration', 'campaign', 'pdays', 'previous', 'balance']

fig, ax = plt.subplots(len(col_names), figsize=(8,40))

for i, col_val in enumerate(col_names):
        
    sns.boxplot(y=data_ex[col_val], ax=ax[i])
    ax[i].set_title('Box plot - '+col_val, fontsize=10)
    ax[i].set_xlabel(col_val, fontsize=8)
    
plt.show()


#data_ex = sales_data_hist.drop(['Order', 'File_Type','SKU_number','SoldFlag','MarketingType','ReleaseNumber','New_Release_Flag'], axis=1)
sns.pairplot(data_ex)



# Percentile based outlier removal 
def percentile_based_outlier(data, threshold=95):
    diff = (100 - threshold) / 2.0
    minval, maxval = np.percentile(data, [diff, 100 - diff])
    return (data < minval) | (data > maxval)

col_names = ['age','duration', 'campaign', 'pdays', 'previous', 'balance']

fig, ax = plt.subplots(len(col_names), figsize=(8,40))

for i, col_val in enumerate(col_names):
    x = data_ex[col_val][:1000]
    sns.distplot(x, ax=ax[i], rug=True, hist=False)
    outliers = x[percentile_based_outlier(x)]
    ax[i].plot(outliers, np.zeros_like(outliers), 'ro', clip_on=False)

    ax[i].set_title('Outlier detection - '+col_val, fontsize=10)
    ax[i].set_xlabel(col_val, fontsize=8)
    
plt.show()


# mean of the numeric features and how it effects output

data_ex.groupby('y').mean()

#Deduced:
#pdays feature indicates that usually the conversion happens if the client are contacted not frequently
#previous feature indicates that more the number of times client are contacted more likely he will be a positive sample.


# find the corelation between inputs
corr = data_ex[num_cols].corr()

# plot heatmap
sns.heatmap(corr, 
            xticklabels=corr.columns.values, yticklabels=corr.columns.values,
            cmap=sns.light_palette("navy"),
           )
plt.show()



#We note that:

#pdays and previous features are correlated heavily and we should use one of them


#convert to categorical

data_raw_all['marital'] = data_raw_all['marital'].astype('category')
data_raw_all['marital'] = data_raw_all['marital'].cat.codes


data_raw_all['education'] = data_raw_all['education'].astype('category')
data_raw_all['education'] = data_raw_all['education'].cat.codes


data_raw_all['default'] = data_raw_all['default'].astype('category')
data_raw_all['default'] = data_raw_all['default'].cat.codes


data_raw_all['housing'] = data_raw_all['housing'].astype('category')
data_raw_all['housing'] = data_raw_all['housing'].cat.codes

data_raw_all['loan'] = data_raw_all['loan'].astype('category')
data_raw_all['loan'] = data_raw_all['loan'].cat.codes

data_raw_all['y'] = data_raw_all['y'].astype('category')
data_raw_all['y'] = data_raw_all['y'].cat.codes


# Columns to remove 
remove_col_val = ['job', 'contact', 'day', 'month', 'previous', 'poutcome']


data_raw_all = data_raw_all.drop(remove_col_val, axis=1)


# create training and testing vars

y = data_raw_all['y']
training_features, testing_features, training_target, testing_target = train_test_split(data_raw_all, y, test_size=0.2)
print(training_features.shape, training_target.shape)
print(testing_features.shape, testing_target.shape)

x_train, x_val, y_train, y_val = train_test_split(training_features, training_target,
                                                  test_size = .1,
                                                  random_state=12)
                                                  
                                                  
data_raw_all.head()


#PREDICTIVE MODELLING USING LOGISTIC REGRESSION

#logistic regression

inputData=data_raw_all.iloc[:,:10]
outputData=data_raw_all.iloc[:,10]


from sklearn.linear_model import LogisticRegression
logit1=LogisticRegression()
logit1.fit(inputData,outputData)

logit1.score(inputData,outputData)



####Model performance
####Classification rate 'by hand'
##Correctly classified
np.mean(logit1.predict(inputData)==outputData)
##True positive
trueInput=data_raw_all.loc[data_raw_all['y']==1].iloc[:,:10]
trueOutput=data_raw_all.loc[data_raw_all['y']==1].iloc[:,10]
##True positive rate
np.mean(logit1.predict(trueInput)==trueOutput)
##True negative
falseInput=data_raw_all.loc[data_raw_all['y']==0].iloc[:,:10]
falseOutput=data_raw_all.loc[data_raw_all['y']==0].iloc[:,10]
##True negative rate
np.mean(logit1.predict(falseInput)==falseOutput)




###Confusion matrix with sklearn
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
confusion_matrix(logit1.predict(inputData),outputData)

###Roc curve
fpr, tpr,_=roc_curve(logit1.predict(inputData),outputData,drop_intermediate=False)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr, tpr, color='red',
         lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
plt.xlabel('False Positive ')
plt.ylabel('True Positive ')
plt.title('ROC curve')
plt.show()


roc_auc_score(logit1.predict(inputData),outputData)

###Coefficient value
coef_DF=pd.DataFrame(data={'Variable':list(inputData),
'value':(logit1.coef_[0])})

coef_DF_standardised=pd.DataFrame(data={'Variable':list(inputData),
'value':(logit1.coef_[0])*np.std(inputData,axis=0)/np.std(outputData)})




