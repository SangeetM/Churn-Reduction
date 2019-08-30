
# coding: utf-8

# # PROJECT NAME - CHURN REDUCTION
# 
# # Project Description:-
# 
# * Churn (loss of customers to competition) is a problem for companies because it is more expensive to acquire a new customer than to keep your existing one from leaving. This problem statement is targeted at enabling churn reduction using analytics concepts.

# # The Business Pain:
# 
# * Churn rate has strong impact on life time value of the customer because it affects the length of service and the future revenue of the company.
# 
# * It is estimated that 70 percent of the subscribers signing up with a new wireless carrier every year are coming from another service provider, which means they are supposed to be churners.
# 
# * Telecom Companies spend hundreds of crores of rupess to acquire a new customer and when that customer leaves, the company not only losses their revenue from that customer but also the resources spend to acquire a new customer.
# 

#  
# # The Problem Category Falls Under:
# 
# * Here the problem statement addresses that the problem category belongs to Supervised Learning and it belongs to Classification. 
# 
# * Because we are trying to classify a customer or any dataset and here the given dataset has a Target Variable: 
# 
# * That is if the customer has moved then <b>{1='Yes'/'True',0='No'/False}</b>, here <b>1=True/Yes</b>:- States that the <b>Customer has Moved</b> and <b>0=No/False</b>:- States that the <b>Customer Has not Moved</b> so whenever you are trying to "Predict the Categories" or you are trying to classify into the categories that "Use-Case" comes under the "Classification Problem".

# # Importing & Loading Standard Libraries

# In[1]:


#Load the Standard Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import chi2_contingency
import seaborn as sns
from fancyimpute import KNN

# Importing the sci-kit learn package modules for model development, evaluation & also optimization
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


import warnings
warnings.filterwarnings("ignore")


# In[2]:


#Checking the current working directory
os.getcwd()


# In[3]:


# Loading the dataset which is in '.CSV' format i.e; (Comma-Seperated-Values)
train_actual = pd.read_csv("Train_data.csv")
test_actual = pd.read_csv("Test_data.csv")


# In[4]:


# Before we make any manipulation let's create an alternate copy of our actual train and test data sets
train_data = train_actual.copy()
test_data = test_actual.copy()


# In[5]:


# lets Exploring few observations of the Train-dataset
train_data.head()


# In[6]:


# Lets also Exploring few observations of the Test-dataset
test_data.head()


# # Exploratory Data Analysis

# In[7]:


# Checking the Dimensions of the dataset
print("Dimensions of the training Dataset", train_data.shape)

print("\n******************************************************************************")

print("Dimensions of the test Dataset", test_data.shape)

print("\n******************************************************************************")

# Checking the total observations by combining both train & test data

data = train_data.append(test_data)

print("Total No. Observations of the combine Dataset", data.shape)

print("\n******************************************************************************")


# In[8]:


# Checking the Information about the dataframe

print("Information about  Training DataFrame including the Index data-type and Column data-types")

print(train_data.info())


# In[9]:


# Checking the descriptive statistics of the dataset

print("Generates descriptive statistics for each feature")

train_data.describe()


# In[10]:


# Column names of the dataset
train_data.columns


# In[11]:


#Extracting Unique values and Count using a for loop on the whole data

for i in train_data.columns:
    print(i, '***************', len(train_data[i].value_counts()))


# In[12]:


# Now lest check the churn count and precentage rate in out dataset
print(train_data['Churn'].value_counts())

print('\n***************************************')

print(train_data['Churn'].value_counts(normalize=True))

print('\n***************************************')


# # Missing Values Data Check in train & test and its Percentage

# In[13]:


# Missing value analysis check for Train Dataset
total = train_data.isnull().sum().sort_values(ascending=True)
percnt = (train_data.isnull().sum()/train_data.isnull().count()*100).sort_values(ascending=False)
miss_train_data = pd.concat([total,percnt], axis = 1, keys=['Total_miss_val_train','Percentage_train'])
miss_train_data


# In[14]:


# Missing value analysis check for Test Dataset
total = test_data.isnull().sum().sort_values(ascending=True)
percnt = (test_data.isnull().sum()/test_data.isnull().count()*100).sort_values(ascending=False)
miss_test_data = pd.concat([total,percnt], axis = 1, keys=['Total_miss_val_test','Percentage_test'])
miss_test_data


# In[15]:


# Before we proceed furthur let's extract categorical variables 
cat_names = train_data.select_dtypes(exclude=np.number).columns.tolist()
cat_names.append('area code')
cat_names


# In[16]:


# Lets Change the train and test columns to Categorical data types
train_data[cat_names] = train_data[cat_names].apply(pd.Categorical)
test_data[cat_names] = test_data[cat_names].apply(pd.Categorical)


# # Anlyzing Data Through Visualization

# In[17]:


# Lets analyze the target variable Churn
plt.figure(figsize=(10,8))
sns.countplot(x = train_data.Churn, palette='Set2')
plt.title('Customers Churning VS Not Churning', fontsize=22)
plt.xlabel('Customer Churn',fontsize=16)
plt.ylabel('Count',fontsize=16) 


# In[18]:


# Lets analyze the Churn of Customer's on basis of customer service calls
plt.figure(figsize=(10,8))
plt.title('Customer Churning on Basis of Customer Service calss', fontsize=20)
sns.countplot(x='number customer service calls', hue='Churn', data=train_data, palette="Set2")


# In[19]:


#Lets also analyze through area-code wise the customer's Churn
plt.figure(figsize=(10,8))
plt.title('Customer Churning on Basis of Area Code', fontsize=22)
sns.countplot(x='area code', hue='Churn', data=train_data, palette="Set2")


# In[20]:


# Lets analyze the customer Churn on basis of International Plan & Voice mail Plan by BAR PLot Analysis
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
sns.countplot(x='international plan', hue='Churn', data=train_data, ax=ax[0], palette="Set2")
sns.countplot(x='voice mail plan', hue='Churn', data=train_data, ax=ax[1], palette="Set2")


# In[21]:


# Lets analyze the Cutomer Churn by State here we are using Groupby function to unstack the Categories
train_data.groupby(['state','Churn']).size().unstack().plot(kind='bar', stacked=True, figsize=(25,10))


# In[22]:


#Lets analyze the linearity between Total-International-Charge & Minutes 
sns.lmplot(x='total intl charge',y='total intl minutes', data=train_data, 
          scatter_kws={'marker':'o','color':'indianred'},
          line_kws={'linewidth':1,'color':'blue'})


# In[23]:


#Lets analyze the linearity between the Total-Day-Charge & Minutes
sns.lmplot(x='total day charge',y='total day minutes', data=train_data, 
          scatter_kws={'marker':'o','color':'indianred'},
          line_kws={'linewidth':1,'color':'blue'})


# In[24]:


#Lets analyze the linearity between the Total-Evening-Charge & Minutes
sns.lmplot(x='total eve charge',y='total eve minutes', data=train_data, 
          scatter_kws={'marker':'o','color':'indianred'},
          line_kws={'linewidth':1,'color':'blue'})


# In[25]:


#Lets analyze the linearity between the Total-Night-Charge & Minutes
sns.lmplot(x='total night charge',y='total night minutes', data=train_data, 
          scatter_kws={'marker':'o','color':'indianred'},
          line_kws={'linewidth':1,'color':'blue'})


# # Boxplot Analysis for Outliers check

# In[26]:


# Boxplot for checking Outliers in our dataset
f, axes = plt.subplots(5, 3, figsize=(20, 20))
# total_day_calls
sns.boxplot(x='Churn', y='total day calls', data=train_data, hue='Churn', palette="Set2", ax=axes[0, 0])
# total_day_minutes
sns.boxplot(x='Churn', y='total day minutes', data=train_data, hue='Churn', palette="Set2", ax=axes[0, 1])
# total_day_charge
sns.boxplot(x='Churn', y='total day charge', data=train_data, hue='Churn', palette="Set2", ax=axes[0, 2])
# total_night_calls
sns.boxplot(x='Churn', y='total night calls', data=train_data, hue='Churn', palette="Set2", ax=axes[1, 0])
# total_night_minutes
sns.boxplot(x='Churn', y='total night minutes', data=train_data, hue='Churn', palette="Set2", ax=axes[1, 1])
# total_night_charge
sns.boxplot(x='Churn', y='total night charge', data=train_data, hue='Churn', palette="Set2", ax=axes[1, 2])
# total_eve_calls
sns.boxplot(x='Churn', y='total eve calls', data=train_data, hue='Churn', palette="Set2", ax=axes[2, 0])
# total_eve_minutes
sns.boxplot(x='Churn', y='total eve minutes', data=train_data, hue='Churn', palette="Set2", ax=axes[2, 1])
# total_eve_charge
sns.boxplot(x='Churn', y='total eve charge', data=train_data, hue='Churn', palette="Set2", ax=axes[2, 2])
# total_intl_calls
sns.boxplot(x='Churn', y='total intl calls', data=train_data, hue='Churn', palette="Set2", ax=axes[3, 0])
# total_intl_minutes
sns.boxplot(x='Churn', y='total intl minutes', data=train_data, hue='Churn', palette="Set2", ax=axes[3, 1])
# total_intl_charge
sns.boxplot(x='Churn', y='total intl charge', data=train_data, hue='Churn', palette="Set2", ax=axes[3, 2])
# account_length
sns.boxplot(x='Churn', y='account length', data=train_data, hue='Churn', palette="Set2", ax=axes[4, 0])
# number_vmail_messages
sns.boxplot(x='Churn', y='number vmail messages', data=train_data, hue='Churn', palette="Set2", ax=axes[4, 1])
# number_customer_service_calls
sns.boxplot(x='Churn', y='number customer service calls', data=train_data, hue='Churn', palette="Set2", ax=axes[4, 2])


# As we can see that almost all features/predictors contains outliers and we will try to remove the outliers from the dataset.
# but here we will remove the outliers from the combined dataset that is <b>'data'</b> dataset which we have combined both train
# test dataset so that if we remove the outliers and then do Knn imputation still the data will be reduced and already we have only 5000 records data for analysis which is small so here we will just apply Outlier Analysis but will not use this dataset for further predictions.
# 

# # Outlier Analysis
# 
# * An outlier is nothing but which is inconsistent with the rest of the dataset. In simple terms
# * We can say that any value which is falling away from the bunch of values is nothing but an <b>'Outlier'</b>

# In[27]:


# Lets seperate the numeric values becoz outlier analysis is applicable only on 'Numeric/Continous Values'
# Lets exclude the category vriables and only numeric columns will be selected her

cnames = train_data.columns[(train_data.dtypes=="float64")|(train_data.dtypes=="int64")].tolist()
print(cnames)


# In[28]:


# Lets detect and delete outliers from data-set 
# Outliers which fall above the upper fence which is 1.5*IQR and below fence 1.5*IQR will be dropped
for i in cnames:
    print(i)
    q75, q25 = np.percentile(data.loc[:,i], [75, 25])
    
    # iqr-Inter Quartile Range
    iqr = q75 - q25
    min = q25 - (iqr * 1.5)
    max = q75 + (iqr * 1.5)
    
    print(iqr)
    print(min)
    print(max)
    
# Replace the values with np.nan    
    data = data.drop(data[data.loc[:,i] < min].index)
    data = data.drop(data[data.loc[:,i] > max].index)


# # Lets do Boxplot Analysis as we have dropped Outliers from our Data Dataset

# In[29]:


# Boxplot for checking Outliers in our dataset
f, axes = plt.subplots(5, 3, figsize=(20, 20))
# total_day_calls
sns.boxplot(x='Churn', y='total day calls', data=data, hue='Churn', palette="Set2", ax=axes[0, 0])
# total_day_minutes
sns.boxplot(x='Churn', y='total day minutes', data=data, hue='Churn', palette="Set2", ax=axes[0, 1])
# total_day_charge
sns.boxplot(x='Churn', y='total day charge', data=data, hue='Churn', palette="Set2", ax=axes[0, 2])
# total_night_calls
sns.boxplot(x='Churn', y='total night calls', data=data, hue='Churn', palette="Set2", ax=axes[1, 0])
# total_night_minutes
sns.boxplot(x='Churn', y='total night minutes', data=data, hue='Churn', palette="Set2", ax=axes[1, 1])
# total_night_charge
sns.boxplot(x='Churn', y='total night charge', data=data, hue='Churn', palette="Set2", ax=axes[1, 2])
# total_eve_calls
sns.boxplot(x='Churn', y='total eve calls', data=data, hue='Churn', palette="Set2", ax=axes[2, 0])
# total_eve_minutes
sns.boxplot(x='Churn', y='total eve minutes', data=data, hue='Churn', palette="Set2", ax=axes[2, 1])
# total_eve_charge
sns.boxplot(x='Churn', y='total eve charge', data=data, hue='Churn', palette="Set2", ax=axes[2, 2])
# total_intl_calls
sns.boxplot(x='Churn', y='total intl calls', data=data, hue='Churn', palette="Set2", ax=axes[3, 0])
# total_intl_minutes
sns.boxplot(x='Churn', y='total intl minutes', data=data, hue='Churn', palette="Set2", ax=axes[3, 1])
# total_intl_charge
sns.boxplot(x='Churn', y='total intl charge', data=data, hue='Churn', palette="Set2", ax=axes[3, 2])
# account_length
sns.boxplot(x='Churn', y='account length', data=data, hue='Churn', palette="Set2", ax=axes[4, 0])
# number_vmail_messages
sns.boxplot(x='Churn', y='number vmail messages', data=data, hue='Churn', palette="Set2", ax=axes[4, 1])
# number_customer_service_calls
sns.boxplot(x='Churn', y='number customer service calls', data=data, hue='Churn', palette="Set2", ax=axes[4, 2])


# Now we can see that now most of the outliers have been dropped. Lets Check the dimension of the dataset.

# In[30]:


# Lets check the dimension of the data dataset

print("Total No. Observations of the  Dataset", data.shape)

print("\n******************************************************************************")

# As we can see that before we applied the Outlier Analysis the dimension of the dataset was
# Toatal No. Observations of the combine Dataset (5000, 21)
# After outlier Analysis the dimension of the dataset
# Total No. Observations of the  Dataset (3666, 21)
# As we can see that data is gradually reducing so here we are going to skip outlier analysis on our train_data & test_data 
# dataset which we are going to use for model development & predictions.


# Here We are skipping outlier analysis as their is already target class imbalance problem in our dataset
# and we may also loose so much of important information and already data is not too big if run outliers analysis our data is gradually reducing.

# # Feature Selection 

# In[31]:


# Generate the Correlation matrix 
corr = train_data[cnames].corr()

# Ploting using seaborn library
sns.heatmap(train_data.corr(), annot=True, cmap='RdYlGn', linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(20,16)
plt.title("Heatmap analysis between numeric columns")
plt.show()


# There is Multi-Collinearity between total_day_minutes & total_day_charge, total_eve_minutes & total_eve_charge,
# total_intl_minutes & total_intl_charge & total_night_minutes & total_night_charge

# In[32]:


# Lets analyze through pair plot 
sns.pairplot(train_data,hue='Churn',palette ='Set2',size=2.7,diag_kind='kde',diag_kws=dict(shade=True),plot_kws=dict(s=10))
plt.tight_layout()
plt.show()


# # Lets Analyze Chi2-Square Test of Independence for Categorical Variables

# In[33]:


# Chi2 square test of independence for checking relation between Categorical variables and target varoable
# Lets save all categorical column names
cat_names = ['state', 'area code', 'international plan', 'voice mail plan']

print("Chi2-Square Test of Independence")
print("\n*******************************")

# loop for chi2 square test of independence
for i in cat_names:
    print(i)
    # here Chi2-Square test compares two variables in contigency table
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(train_data['Churn'], train_data[i]))
    print(p)
    print("-----------------------------") 

# Here Chi2 - is the observed/actual value
# Here P -  is the pearson correlation value i.e: (p<0.05)
# Here DOF( Degrees of Freedom) = (no. of rows-1) - (no. of columns-1)
# Here EX - is the expected value
# Here Conclusion is that if the P-Value is less than (p<0.05) we reject Null Hypothesis saying that 2 values depend
# on each other else we accept alternate hypothesis saying that these 2 values are independent of each other


# From Chi2-Square Test of Independence we can analyze that the value of area code is grater than p-value which is 0.05 nad here
# area code value is (0.915055696024 > 0.05), so in dimension reduction we are going to drop the variable.

# # Dimension Reduction

# In[34]:


# Dropping the correlated variables total-day, evening, night and international charge, as well state area code & phone number
# which are not carrying usefull information to explain the variance of  target variable lets drop from both train&test dataset
train_data = train_data.drop(columns=['state','phone number','area code','total day charge', 
                                      'total eve charge', 'total night charge', 'total intl charge'])
test_data = test_data.drop(columns=['state','phone number','area code','total day charge', 
                                    'total eve charge', 'total night charge', 'total intl charge'])


# # Feature Selection and confirming our final Continous&Categorical Features

# In[35]:


# Lets re-check the numeric column variables and ctegorical columns variables
# Lets update them as we have dropped some of the features
cnames = ['account length', 'number vmail messages', 'total day minutes', 'total day calls', 'total eve minutes', 
          'total eve calls', 'total night minutes', 'total night calls', 'total intl minutes', 
          'total intl calls','number customer service calls']
print(cnames)
print('********************************************************')
# Lets recheck the categorical names now
cat_names = ['international plan', 'voice mail plan','Churn']

print(cat_names)


# # Assigning levels to Categorical Columns

# In[36]:


#Lets Handle Categorical Columns now by assigning levels (0 & 1) 
cat_names = train_data.columns[train_data.dtypes == 'category']
for i in cat_names:
    #print(i)
    train_data[i] = train_data[i].cat.codes
    test_data[i] = test_data[i].cat.codes


# # Lets Do Some More Exploration of our data  

# In[37]:


# Lets apply groupby with International-Plan
intl_plan=train_data.groupby("international plan").size()
intl_plan


# In[38]:


# Lets now analyze how many customers have subscribed and not subscribed to International plan in percent
print("Subscribed to International-Plan in percent:\t{}".format((intl_plan[0]/3333)*100))
print("Not Subscribed to International-Plan in percent:\t{}".format((intl_plan[1]/3333)*100))


# In[39]:


# Lets apply the groupby with Voice mail plan now
vmail_plan = train_data.groupby('voice mail plan').size()
vmail_plan


# In[40]:


# Lets now analyze how many customers have subscribed and not subscribed to Voice Mail Plan in percent
print("Subscribed to International-Plan in percent:\t{}".format((vmail_plan[1]/3333)*100))
print("Not Subscribed to International-Plan in percent:\t{}".format((vmail_plan[0]/3333)*100))


# In[41]:


# Lets groupby customer service calls now
custmr_calls = data.groupby('number customer service calls').size()
custmr_calls


# In[42]:


# Lets analyze the count of customer service calls
train_data['number customer service calls'].hist(bins=500,figsize=(10,8))
plt.title("Number of Customer Service Calls")
plt.xlabel("Customer Service Calls")
plt.ylabel("Count of Customer Service Calls")
plt.tight_layout()
plt.show()


# In[43]:


# Lets see Account Length of customers here
Account_Length = train_data.groupby(['account length']).size()
#Account_Length # has length of 212


# In[44]:


# Lets plot Histogram to analyze the account length
train_data['account length'].hist(bins=500,figsize=(10,8))
plt.title("Account length of customers ")
plt.xlabel("Customer Account length")
plt.ylabel("Count of Customer Account Length")
plt.tight_layout()
plt.show()


# In[45]:


# Now Lets check the CHURN of Customers in Percent
Churn=train_data.groupby(['Churn']).size()
Churn # Here 0- Means 'False.'- > 'No', & Here 1- Means 'True.'- > 'Yes'


# In[46]:


# Lets see the Percentage of Churn
print (" Negative Chrun in percent:{}".format((Churn[0]/3333)*100))
print (" Positive Chrun in percent:{}".format((Churn[1]/3333)*100))


# In[47]:


# Lets see churn by international plan
Intl_Churn = train_data.groupby(['international plan','Churn']).size()
Intl_Churn


# In[48]:


# Lets see churn by voice mail plan
Vmail_Churn = train_data.groupby(['voice mail plan', 'Churn']).size()
Vmail_Churn


# In[49]:


# Lets see churn by customer service calls
Custserv_Chrun=data.groupby(['number customer service calls','Churn']).size()
Custserv_Chrun


# In[50]:


# Lets Plot to analyze the customer service calls as per Churn
Custserv_Chrun.plot(kind= 'bar', figsize=(10,8))
plt.title('Churn By Customer Service calls Breakdown')
plt.xlabel('Customer Serv calls wise Churn', fontsize=18)
plt.ylabel('Count of Customer Serv calls wise Churn', fontsize=18)
plt.show()


# # Checking Distribution of Variables/ Normality Check

# In[51]:


# Lets Check the Data Distribution of Continous variables  
train_data[cnames].hist(figsize=(20,20), alpha=0.7)
plt.show()


# In[52]:


# Lets analyze better by also plotting density plot over histogram plot
# histogram and Density Plot togeather for checking distribution of our variables
f, axes = plt.subplots(4, 3, figsize=(20, 20))
#account length
sns.distplot(train_data['account length'], hist=True, bins='auto',kde=True,color = 'darkblue', ax=axes[0, 0],
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
#number vmail messages
sns.distplot(train_data['number vmail messages'], hist=True, bins='auto',ax=axes[0, 1],
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
#total day minutes
sns.distplot(train_data['total day minutes'], hist=True, bins='auto',ax=axes[0, 2],
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
#total day calls
sns.distplot(train_data['total day calls'], hist=True, bins='auto',ax=axes[1, 0],
            hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
#total eve minutes
sns.distplot(train_data['total eve minutes'], hist=True, bins='auto',ax=axes[1, 1],
            hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
#total eve calls
sns.distplot(train_data['total eve calls'], hist=True, bins='auto',ax=axes[1, 2],
            hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
#total night minutes
sns.distplot(train_data['total night minutes'], hist=True, bins='auto',ax=axes[2, 0],
            hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
#total night calls
sns.distplot(train_data['total night calls'], hist=True, bins='auto',ax=axes[2, 1],
            hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
#total international minutes
sns.distplot(train_data['total intl minutes'], hist=True, bins='auto',ax=axes[2, 2],
            hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
#total international calls
sns.distplot(train_data['total intl calls'], hist=True, bins='auto',ax=axes[3, 0],
            hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
#number customer service calls
sns.distplot(train_data['number customer service calls'], hist=True, bins='auto',ax=axes[3, 1],
            hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})


# As we can see that most of the variables are Uniformly Distributed.
# So it is better to go for Standardization Method itself for Feature Scaling

# # Feature Scaling and applying Standardization/Z-Score Method

# In[53]:


# Standardization/ Z-Score Method
stand_zs = ['account length', 'number vmail messages', 'total day minutes', 'total day calls', 
               'total eve minutes', 'total eve calls', 'total night minutes', 'total night calls', 
               'total intl minutes', 'total intl calls', 'number customer service calls']

for i in stand_zs:
    print(i)
    train_data[i] = (train_data[i] - train_data[i].mean())/train_data[i].std()        


# In[54]:


# Standardization/ Z-Score Method
stand_zs = ['account length', 'number vmail messages', 'total day minutes', 'total day calls', 
               'total eve minutes', 'total eve calls', 'total night minutes', 'total night calls', 
               'total intl minutes', 'total intl calls', 'number customer service calls']

for i in stand_zs:
    print(i)
    test_data[i] = (test_data[i] - test_data[i].mean())/test_data[i].std()


# In[55]:


# Splitting the data into Train and Test Sets
X_train = train_data.drop('Churn', axis = 1)
y_train = train_data['Churn']
X_test = test_data.drop('Churn', axis = 1)
y_test = test_data['Churn']


# In[56]:


# Lets print out the X_train, y_train, X_test and y_test
print('X_train values------>', X_train.shape)
print('y_train values------>', y_train.shape)
print('X_test values------->', X_test.shape)
print('y_test values------->', y_test.shape)


# In[57]:


# As we can see there is target class imbalance problem, training values before applying Smote 
y_train.value_counts()


# # SMOTE (Synthetic Minority Over-Sampling Technique)
# 
# we do synthetic data only on the train: SMOTE creates synthetic observations of the minority class (churn) by:
# 
# Finding the k-nearest-neighbors for minority class observations (finding similar observations)
# Randomly choosing one of the k-nearest-neighbors and using it to create a similar, but randomly tweaked, new observation.
# 
# we use smote only on the training data set

# In[58]:


# Using SMOTE
from imblearn.over_sampling import SMOTE
SMT = SMOTE()
X_train_balanced ,y_train_balanced = SMT.fit_sample(X_train, y_train)


# In[59]:


print('X_training data set after SMOTE--->', X_train_balanced.shape)
print('y_training data set after SMOTE--->', y_train_balanced.shape)


# # MODEL DEVELOPMENT

# In[60]:


# Lets define our  Prediction Function to fit and make predictions
def predct_func(classification_models, features, comparison):
    ''' Here we are going to fit our train data by passing 
        to the funtion predct_func and predict our X_test 
        data which is our new test cases here and it will result 
        out classification-report and all the error metrics of the
        particular model used here.
    ''' 
    # Lets fit the model first
    classification_models.fit(features, comparison)
    # Predict new test cases
    predicted_vals = classification_models.predict(X_test)
    # Lets apply K-Fold Croos Validatiion CV=10
    KVC = cross_val_score(estimator=classification_models, X=features, y=comparison,cv=10)
    KFoldCross_Accuracies = KVC.mean()
    print('K Fold Crossvalidation Accuracy------->', KFoldCross_Accuracies)
    print()
    # Generates the classification report of the model
    print("************Classification Report*************")
    print()
    class_report = classification_report(y_test,predicted_vals)
    print(class_report)
    # Generate the Confusion Matrix of the Model
    print("************Confusion Matrix*******************")
    print()
    CM = confusion_matrix(y_test, predicted_vals)
    print(CM)


# In[61]:


# Lets Define another function for Evaluation of out models
def eval_model(actual_vals, prediction_vals):
    ''' Function for evaluation of error metrics
        generates confusion matrix and results out
        False Positive Rate, False Negative Rate, 
        Sensitivity/TruePositiveRate/Recall & 
        specificity/TrueNegativeRate of models
    '''
    
    CM = pd.crosstab(actual_vals, prediction_vals)
    TN = CM.iloc[0,0]
    FN = CM.iloc[1,0]
    TP = CM.iloc[1,1]
    FP = CM.iloc[0,1]
    print()
    
    # Lets evaluate Error Metrics of the model algorithms
    print("<---------------ERROR METRICS-------------->")
    print()
    # False Negative Rate
    print("False Negative Rate-------------->",  (FN*100)/(FN+TP))
    print()
    # False Positive Rate
    print("False Positive Rate-------------->",  (FP*100)/(FP+TN))
    print()
    # Sensitivity
    print("Sensitivity/TPR/Recall----------->",  (TP*100)/(TP+FN))
    print()
    # Specificity
    print("Specificity/TNR------------------>",  (TN*100)/(TN+FP))


# In[62]:


# Lets Develop Decision Tree Model 
DT_Model = DecisionTreeClassifier(criterion='entropy',random_state=100)
predct_func(DT_Model, X_train_balanced, y_train_balanced)


# In[63]:


# Lets predict new test cases
DT_Predictions = DT_Model.predict(X_test)


# In[64]:


# Now Lets evaluate Error Metrics for Decision Tree model
eval_model(y_test, DT_Predictions)


# In[65]:


#ROC curve for false positive rate-fpr, true positive rate-tpr
# The ROC-Curve is the plot between sensitivity and (1-specificity) is also known as False positive rate
# and Sensitivity is also known as True Positive rate
# ROC-AUC Curve for Decision Tree MOdel
fpr, tpr, thresholds_DT = roc_curve(y_test, DT_Predictions)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC curve (area = %0.2f)'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[66]:


# Lets Develop Random Forest Model 
RF_Model = RandomForestClassifier(n_estimators=20, criterion='entropy',random_state=100)
predct_func(RF_Model, X_train_balanced, y_train_balanced)


# In[67]:


# Lets Predict new test cases
RF_Predictions = RF_Model.predict(X_test)


# In[68]:


# Lets Evaluate Error Metrics for Random Forest Model
eval_model(y_test, RF_Predictions)


# In[69]:


# ROC-AUC Curve for Random Forest Model
fpr, tpr, thresholds_RF = roc_curve(y_test, RF_Predictions)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC curve (area = %0.2f)'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[70]:


# Lets Develop Logistic Regression Model
LR_Model = LogisticRegression()
predct_func(LR_Model, X_train_balanced, y_train_balanced)


# In[71]:


# Lets Predict New Test Cases
LR_Predictions = LR_Model.predict(X_test)


# In[72]:


# Now lets evaluate Error Metrics For Logistic Regression Model
eval_model(y_test, LR_Predictions)


# In[73]:


# ROC-AUC Curve for Logistic Regression Model
fpr, tpr, thresholds_RF = roc_curve(y_test, LR_Predictions)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC curve (area = %0.2f)'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[74]:


# Lets Develop KNN Model Now
KNN_Model = KNeighborsClassifier(n_neighbors=5)
predct_func(KNN_Model, X_train_balanced, y_train_balanced)


# In[75]:


# Lets Predict New Test Cases
KNN_Predictions = KNN_Model.predict(X_test)


# In[76]:


# Now lets evaluate Error Metrics For K-Nearest-Neighbor Model
eval_model(y_test, KNN_Predictions)


# In[77]:


# ROC-AUC Curve for K-Nearest-Neighbor Model 
fpr, tpr, thresholds_RF = roc_curve(y_test, KNN_Predictions)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC curve (area = %0.2f)'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[78]:


# Lets Develop Naive Bayes Model
NB_Model  = GaussianNB()
predct_func(NB_Model, X_train_balanced, y_train_balanced)


# In[79]:


# Lets Predict new test cases
NB_Predictions = NB_Model.predict(X_test)


# In[80]:


# Now lets evaluate Error Metrics For Naive Bayes Model
eval_model(y_test, NB_Predictions)


# In[81]:


# ROC-AUC Curve for Naive Bayes Model
fpr, tpr, thresholds_RF = roc_curve(y_test, NB_Predictions)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC curve (area = %0.2f)'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# # Hyperparameter tuning using GridSearchCV
# 
# ~We will use <b>GridSearchCV</b> for tuning the parameters any two  algorithms, to achieve optimum performance.
# 
# ~ Here i am going for Decision Tree & Random Forest Model for Tuning as both have performed well but out of these two Random forest is doing well.
# 
# ~ We will be running the Hyper-Parameter Code for only once and save the results of our 
# Final Model based Results for <b> Decision Tree<b/> and <b>Random Forest</b> we will finalize our final model which is having optimum performance after hyper parameter tuning. and here i am commenting the below code as it will take lot of time for execution.

# In[94]:


#Hyper Parameter Tuning for Decision Tree Model
#from sklearn.model_selection import GridSearchCV

# Finding Best Parameters for Random Forest Model 
#dt_model = DecisionTreeClassifier()

#params_grid = [{'criterion': ['entropy','gini'],
#                'max_depth':[50,100,150,200,250,300,350]}] 

#grid_search = GridSearchCV(estimator=dt_model, param_grid=params_grid,
#                          scoring='recall', cv=10, n_jobs=-1)

#grid_search = grid_search.fit(X_train_balanced, y_train_balanced)
#best_accuracy = grid_search.best_score_
#print(best_accuracy)
#best_parameters = grid_search.best_params_
#print(best_parameters)

#accuracy_score = 0.944912280702
#best_parameters = {'criterion': 'gini', 'max_depth': 100}


# In[95]:


# Now lets apply the tuned parameters
#dt_model = DecisionTreeClassifier(criterion='gini',max_depth=100,random_state=100)
#predct_func(dt_model, X_train_balanced, y_train_balanced)


# In[96]:


# lets predict predict new test cases on tuned decision tree model
#dt_Predictions  = dt_model.predict(X_test)


# In[97]:


# Lets evaluate error metrics on our tuned decision tree model
#eval_model(y_test, dt_Predictions)


#<---------------ERROR METRICS-------------->

#False Negative Rate--------------> 20.9821428571

#False Positive Rate--------------> 13.86001386

#Sensitivity/TPR/Recall-----------> 79.0178571429

#Specificity/TNR------------------> 86.13998614


# In[98]:


# ROC-AUC Curve for tuned - decision tree
#fpr, tpr, thresholds_RF = roc_curve(y_test, dt_Predictions)
#roc_auc = auc(fpr, tpr)
#plt.title('Receiver Operating Characteristic')
#plt.plot(fpr, tpr, 'b',label='AUC curve (area = %0.2f)'% roc_auc)
#plt.legend(loc='lower right')
#plt.plot([0,1],[0,1],'r--')
#plt.xlim([0.0,1.0])
#plt.ylim([0.0,1.0])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()


# ~ As we can observe that False neagtive rate has reduced after applying hyper-parameter tuning on decision tree model and accuracy and aur-roc curve has also improved when compared.

# In[99]:


#Hyper Parameter Tuning for Random Forest Model
#from sklearn.model_selection import GridSearchCV

# Finding Best Parameters for Random Forest Model 
#rf_model = RandomForestClassifier()

#params_grid = [{'n_estimators':[100,200,300,400,500,800,1000],
#                'criterion': ['entropy','gini'],
#                'max_depth':[50,100,150,200,250,300,350]}] 

#grid_search = GridSearchCV(estimator=rf_model, param_grid=params_grid,
#                          scoring='recall', cv=10, n_jobs=-1)

#grid_search = grid_search.fit(X_train_balanced, y_train_balanced)
#best_accuracy = grid_search.best_score_
#print(best_accuracy)
#best_parameters = grid_search.best_params_
#print(best_parameters)

#accuracy_score = 0.964912280702
#best_parameters = {'criterion': 'entropy', 'max_depth': 50, 'n_estimators': 300}


# In[100]:


# lets run Tuned Random Forest Model on our prediction function 
#rf_model = RandomForestClassifier(criterion='entropy', n_estimators=300, max_depth =500, random_state=100)

#predct_func(rf_model, X_train_balanced, y_train_balanced)

## K Fold Crossvalidation Accuracy-------> 0.966666666667


# In[101]:


# Now lets Predict new test cases for Tuned Random Forest Final  Predictions
#rf_Predictions = rf_model.predict(X_test)


# In[102]:


# Now Lets Evaluate Error Metrics for Tuned Randoom Forest Model
#eval_model(y_test, rf_Predictions)

# Final Model Error metrics of FINAL MODEL Random Forest

#<---------------ERROR METRICS-------------->

#False Negative Rate--------------> 15.625

#False Positive Rate--------------> 9.21690921691

#Sensitivity/TPR/Recall-----------> 84.375

#Specificity/TNR------------------> 90.7830907831

# we can observe that the False Negative Rate has Reduced compare to previous RF_Model and Tuned rf_model is working well.


# In[103]:


# ROC-AUC Curve for Tuned - Random Forest Model
#fpr, tpr, thresholds_RF = roc_curve(y_test, rf_Predictions)
#roc_auc = auc(fpr, tpr)
#plt.title('Receiver Operating Characteristic')
#plt.plot(fpr, tpr, 'b',label='AUC curve (area = %0.2f)'% roc_auc)
#plt.legend(loc='lower right')
#plt.plot([0,1],[0,1],'r--')
#plt.xlim([0.0,1.0])
#plt.ylim([0.0,1.0])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()


# ~ So here we Conclude that Our <b>FINAL MODEL IS TUNED RANDOM FOREST MODEL</b> and it is giving best results with 
# optimum performance when compared with other models.

# # Feature Importances 
# 
# lets see which features/predictors carry much information to explain out Target Variable

# In[104]:


# Lets Plot Feature importances
#features = train_data.drop(["Churn"], axis=1).columns
#fig = plt.figure(figsize=(20, 18))
#ax = fig.add_subplot(111)
#FI = pd.DataFrame(rf_model.feature_importances_, columns=["importance"])
#FI["labels"] = features
#FI.sort_values("importance", inplace=True, ascending=False)
#display(FI.head(5))
#index = np.arange(len(rf_model.feature_importances_))
#bar_width = 0.5
#rects = plt.barh(index , FI["importance"], bar_width, alpha=0.4, color='b', label='Main')
#plt.yticks(index, FI["labels"])
#plt.show()


# ~ The above code section has been commented out because it takes more time consumption while running so if there is a need 
# to run i suggest to comment it out.

# In[105]:

# Lets Save the final results back to hard disk
# Writing a csv (output) Training & Test Data-set

train_data.to_csv("train_df_final_data.csv", index = False)
test_data.to_csv("test_df_final_data.csv", index = False)

