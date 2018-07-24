import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
from scipy import stats
from scipy.stats import norm, skew #for some statistics

import warnings
warnings.filterwarnings('ignore')

#We use a Random Forest on a select number of features that we find has the most 
#predictive power for the sale price. After transforming the numerical input to 
#log units and using the random Forest, we obtained a mean square log error of 
#3.03*10^(-5) and a R^2 of 0.9665. We are pleased with the low error and the 
#high explainability of our model.


pd.set_option('display.max_columns', None)  

test = pd.read_csv('/Users/quantumpc/Desktop/houseprices/test.csv')
train = pd.read_csv('/Users/quantumpc/Desktop/houseprices/train.csv')

print(train.describe())
print(train.head(3))
print(train.shape, test.shape)

#%%

correlationmatrix = train.corr(method ='pearson') #compures pairwise pearson correlation of columns, excluding NA/null values
f, ax = plt.subplots(figsize = (20, 9))
sns.heatmap(correlationmatrix, vmax = .8, annot=True);

#%%

# we look for most correlated features most correlated features 
correlationmatrix = train.corr(method='pearson')
top_corr_features = correlationmatrix.index[abs(correlationmatrix["SalePrice"])>0.5]
plt.figure(figsize = (10,10))
g = sns.heatmap(train[top_corr_features].corr(), annot=True, cmap="RdYlGn")

#%%
# let's look at overallQual vs saleprice bar graph
sns.barplot(train.OverallQual, train.SalePrice)

#%%

# let's continue our EDA
sns.set() #sets default asthetic parameters
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show();

#%%
sns.distplot(train['SalePrice'], fit=norm);

# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu,sigma)) 
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu,sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

#%%
train.SalePrice = np.log1p(train.SalePrice) #fcn applies log(1+x) to all elements of the column]
y = train.SalePrice


#%%
plt.scatter(y=train.SalePrice, x = train.GrLivArea, c='black')
#plt.show()

#%%

print("Find most important feature relative to target")
corr = train.corr()
corr.sort_values(['SalePrice'], ascending = False, inplace=True)
print(corr.SalePrice)

#%%

categorical_features = train.select_dtypes(include=["object"]).columns
numerical_features = train.select_dtypes(exclude=["object"]).columns
numerical_features = numerical_features.drop("SalePrice")
print("Numerical features: " + str(len(numerical_features)))
print("categorical features: " + str(len(categorical_features)))
train_num = train[numerical_features]
train_cat = train[categorical_features]

# fc applies log(1+x) to all elements
train_num = np.log1p(train_num)

#%%
# Handle remaining missing values for numerical features by using median as replacement
print("NAs for numerical features in train: " + str(train_num.isnull().values.sum()))
train_num = train_num.fillna(train_num.median())
print("Remaining NAs for numerical features in train : " + str(train_num.isnull().values.sum()))

#%%
from scipy.stats import skew
skewness = train_num.apply(lambda x: skew(x))
skewness.sort_values(ascending=False)

#%%
skewness = skewness[abs(skewness)>0.5]
skewness.index
skew_features = train[skewness.index]
skew_features.columns

#we can treat skewness of a feature with the help of log transformation
skew_features = np.log1p(skew_features)

#%%
# create dummy features for categorical values via one hot encoding
#train_cat = pd.get_dummies(train_cat)
#train_cat.shape
#str(train_cat.isnull().values.sum())

# create dummy features for numerical values via one hot encoding
train_num = pd.get_dummies(train_num)

#%%

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

#%%

#now after transformation(preprocessing) we'll join them to get the whole train set back

train = pd.concat([train_cat, train_num], axis=1)
#train.shape
#
## split the data to train the model
X_train, X_test, y_train, y_test = train_test_split(train_num, y, test_size = 0.3, random_state = 0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
#
#X_train.head(3)

#%%

from sklearn.metrics import mean_squared_error, mean_squared_log_error, make_scorer

#actually my decision tree is slightly better but whatever, this one avoids overfitting
## random forests uses many trees, and it makes a prediction by averaging the 
## predictions of each component tree

from sklearn.ensemble import RandomForestRegressor

def msleAndR2(forest_model, X_test, y_test):
    msleSum = 0
    R2Sum = 0
    for i in range(300):
        forest_model = RandomForestRegressor()
        forest_model.fit(X_test, y_test) #we train the model
        yforestpredicted = forest_model.predict(X_test) #now we use it to predict
        msle = mean_squared_log_error(y_test, yforestpredicted)
        R2 = forest_model.score(X_test, y_test)
        msleSum += msle
        R2Sum += R2
    msleAverage = msleSum/300
    R2Average = R2Sum/300
    print("msle Average is : ", msleAverage)
    print("R^2 average is : ", R2Average)
    return (msleAverage, R2Average)
    
msleAndR2(forest_model, X_test, y_test)


#%%

#Predict
yforestpredicted = forest_model.predict(X_test)

#plot the results
plt.figure()
plt.plot(X_test['LotArea'], ypredicted, '.')
plt.show()


#%%

# now, we will work on the actual test data,
# variables corresponding to test data will have 'Final' appended

numerical_featuresFinal = test.select_dtypes(exclude=["object"]).columns
test_numFinal = test[numerical_features]

# fcn applies log(1+x) to all elements
test_numFinal = np.log1p(test_numFinal)

# Handle remaining missing values for numerical features by using median as replacement
test_numFinal = test_numFinal.fillna(test_numFinal.median())

yforestpredictedFinal = forest_model.predict(test_numFinal) #now we use it to predict

# transform back from log units
yforestpredictedFinal = np.expm1(yforestpredictedFinal)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': yforestpredictedFinal})
my_submission.to_csv('submission.csv', index=False)
































