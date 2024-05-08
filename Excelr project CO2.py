#!/usr/bin/env python
# coding: utf-8

# ## Project:
# ### Business Objective:
# ### The fundamental goal here is to model the CO2 emissions as a function of several car engine features.
# 
# Data Set Details: 
# The file contains the data for this example. Here the number of variables (columns) is 12, and the number of instances (rows) is 7385. 
# 
# In that way, this problem has the 12 following variables:
# 
# make, car brand under study.
# model, the specific model of the car.
# vehicle_class, car body type of the car.
# engine_size, size of the car engine, in Liters.
# cylinders, number of cylinders.
# transmission, "A" for Automatic', "AM" for Automated manual', "AS" for 'Automatic with select shift', "AV" for 'Continuously variable', "M" for 'Manual'.
# fuel_type, "X" for 'Regular gasoline', "Z" for 'Premium gasoline', "D" for 'Diesel', "E" for 'Ethanol (E85)', "N" for 'Natural gas'.
# fuel_consumption_city, City fuel consumption ratings, in liters per 100 kilometers.
# fuel_consumption_hwy, Highway fuel consumption ratings, in liters per 100 kilometers.
# fuel_consumption_comb(l/100km), the combined fuel consumption rating (55% city, 45% highway), in L/100 km.
# fuel_consumption_comb(mpg), the combined fuel consumption rating (55% city, 45% highway), in miles per gallon (mpg).
# co2_emissions, the tailpipe emissions of carbon dioxide for combined city and highway driving, in grams per kilometer.
# Acceptance Criterion: Need to deploy the end results using Flask /Streamlit etc
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Load the dataset
df = pd.read_csv("co2_emissions.csv")
df


# ## EDA

# In[3]:


# 1. Summary Statistics
summary_stats = df.describe()
print(summary_stats)


# In[4]:


# 2. Data Visualization
# Histograms
df.hist(figsize=(10, 8))
plt.show()


# In[5]:


# Bar plots for categorical variables
plt.figure(figsize=(10, 6))
sns.countplot(x='vehicle_class', data=df)
plt.xticks(rotation=90)
plt.show()


# In[6]:


# Box plots
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['engine_size', 'fuel_consumption_comb(l/100km)', 'co2_emissions']])
plt.show()


# In[7]:


# Scatter plots
plt.figure(figsize=(10, 6))
sns.pairplot(df[['engine_size', 'fuel_consumption_comb(l/100km)', 'co2_emissions']])
plt.show()


# In[8]:


# Heatmap for correlation
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()


# In[9]:


# 3. Missing Values Analysis
missing_values = df.isnull().sum()
print(missing_values)


# In[10]:


# 4. Feature Relationships
# Scatter plot of engine_size vs. co2_emissions
plt.figure(figsize=(10, 6))
sns.scatterplot(x='engine_size', y='co2_emissions', data=df)
plt.show()


# In[11]:


# Categorical plot of transmission vs. co2_emissions
plt.figure(figsize=(10, 6))
sns.boxplot(x='transmission', y='co2_emissions', data=df)
plt.xticks(rotation=0)
plt.show()


# In[12]:


# 5. Feature Engineering Insights: This would involve creating new features or exploring interactions,
# Create interaction features
df['engine_size_cylinders_interaction'] = df['engine_size'] * df['cylinders']
df['fuel_consumption_comb_per_cylinder'] = df['fuel_consumption_comb(l/100km)'] / df['cylinders']
print(df['engine_size_cylinders_interaction'])
print(df['fuel_consumption_comb_per_cylinder'])


# In[13]:


# 6. Outlier Detection: This can be done visually using box plots or programmatically using z-scores.
# Visual outlier detection using box plots
sns.boxplot(x='fuel_type', y='co2_emissions', data=df)
plt.title('CO2 Emissions by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('CO2 Emissions (g/km)')
plt.show()


# In[14]:


from scipy import stats

z_scores = stats.zscore(df['co2_emissions'])
outliers = df[(z_scores > 3) | (z_scores < -3)]
print(outliers)


# In[15]:


# 7. Correlation Analysis: Already done using heatmap.


# In[16]:


# 8. Data Distribution: Check normality assumption using Q-Q plots or statistical tests.
# Q-Q plot for CO2 emissions
import scipy.stats as stats

stats.probplot(df['co2_emissions'], dist="norm", plot=plt)
plt.title('Q-Q Plot of CO2 Emissions')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Ordered Values')
plt.show()


# In[17]:


# 9. Model Assumptions: For linear regression, check for linearity, homoscedasticity, and normality of residuals.
# Check linearity assumption with scatter plots
sns.scatterplot(x='engine_size', y='co2_emissions', data=df)
plt.title('Engine Size vs CO2 Emissions')
plt.xlabel('Engine Size (liters)')
plt.ylabel('CO2 Emissions (g/km)')
plt.show()


# In[18]:


# Check homoscedasticity assumption with residuals vs. fitted plot
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import het_goldfeldquandt

import statsmodels.api as sm

# Assuming X contains your independent variables and y contains your dependent variable
X = df[['engine_size', 'cylinders']]
y = df['co2_emissions']

# Add constant to the independent variables
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Get residuals and fitted values
residuals = model.resid
fitted_values = model.fittedvalues

# Check homoscedasticity assumption with residuals vs. fitted plot
plt.scatter(fitted_values, residuals)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()


# In[19]:


# Check normality of residuals with Q-Q plot
qqplot(residuals, line='s')
plt.title('Q-Q Plot of Residuals')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Ordered Values')
plt.show()


# ### Multicollinearity

# In[20]:


# Multicollinearity Detection using VIF:
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Compute VIF for each feature
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data

# Assuming X contains your independent variables
X = df[['engine_size', 'cylinders', 'fuel_consumption_comb(l/100km)']]

vif_scores = calculate_vif(X)
print(vif_scores)


# ### Feature Selection

# In[21]:


from sklearn.linear_model import LassoCV

# Assuming X contains your independent variables and y contains your dependent variable
X = df[['engine_size', 'cylinders', 'fuel_consumption_comb(l/100km)']]
y = df['co2_emissions']

# Initialize LassoCV model
lasso = LassoCV(cv=5)

# Fit the model
lasso.fit(X, y)

# Get selected features
selected_features = X.columns[lasso.coef_ != 0]
print(selected_features)


# In[22]:


# Feature Selection using Lasso Regression:
from sklearn.linear_model import LassoCV

# Assuming X contains your independent variables and y contains your dependent variable
X = df[['engine_size', 'cylinders', 'fuel_consumption_comb(l/100km)']]
y = df['co2_emissions']

# Initialize LassoCV model
lasso = LassoCV(cv=5)

# Fit the model
lasso.fit(X, y)

# Get selected features
selected_features = X.columns[lasso.coef_ != 0]
print(selected_features)


# ## LinearRegression

# In[23]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Assuming X contains your independent variables and y contains your dependent variable
X = df[['engine_size', 'cylinders', 'fuel_consumption_comb(l/100km)']]
y = df['co2_emissions']

# Initialize linear regression model
model = LinearRegression()

# Initialize RFE
rfe = RFE(model, n_features_to_select=1)

# Fit RFE
rfe.fit(X, y)

# Get selected features
selected_features = X.columns[rfe.support_]
print(selected_features)


# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Assuming X contains your independent variables and y contains your dependent variable
X = df[['engine_size', 'cylinders', 'fuel_consumption_comb(l/100km)']]
y = df['co2_emissions']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)


# ## RandomForestRegresso

# In[29]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialize Random Forest regressor
rf = RandomForestRegressor()

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)

# Perform hyperparameter tuning
random_search.fit(X_train, y_train)

# Get the best parameters
print("Best Parameters:", random_search.best_params_)

# Evaluate the best model
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)


# ## GradientBoostingRegressor

# In[ ]:


# from sklearn.ensemble import GradientBoostingRegressor

# Initialize Gradient Boosting regressor
gbm = GradientBoostingRegressor()

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize RandomizedSearchCV
random_search_gbm = RandomizedSearchCV(estimator=gbm, param_distributions=param_grid, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)

# Perform hyperparameter tuning
random_search_gbm.fit(X_train, y_train)

# Get the best parameters
print("Best Parameters for Gradient Boosting:", random_search_gbm.best_params_)

# Evaluate the best model
best_model_gbm = random_search_gbm.best_estimator_
y_pred_gbm = best_model_gbm.predict(X_test)
mse_gbm = mean_squared_error(y_test, y_pred_gbm)
rmse_gbm = mse_gbm ** 0.5
r2_gbm = r2_score(y_test, y_pred_gbm)

print("Mean Squared Error for Gradient Boosting:", mse_gbm)
print("Root Mean Squared Error for Gradient Boosting:", rmse_gbm)
print("R-squared for Gradient Boosting:", r2_gbm)


# In[27]:


from sklearn.svm import SVR

# Initialize Support Vector Regression
svr = SVR()

# Define the parameter grid
param_grid_svr = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']
}

# Initialize RandomizedSearchCV
random_search_svr = RandomizedSearchCV(estimator=svr, param_distributions=param_grid_svr, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)

# Perform hyperparameter tuning
random_search_svr.fit(X_train, y_train)

# Get the best parameters
print("Best Parameters for SVR:", random_search_svr.best_params_)

# Evaluate the best model
best_model_svr = random_search_svr.best_estimator_
y_pred_svr = best_model_svr.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = mse_svr ** 0.5
r2_svr = r2_score(y_test, y_pred_svr)

print("Mean Squared Error for SVR:", mse_svr)
print("Root Mean Squared Error for SVR:", rmse_svr)
print("R-squared for SVR:", r2_svr)


# ##  neural_network

# In[28]:


from sklearn.neural_network import MLPRegressor

# Initialize MLPRegressor
mlp = MLPRegressor()

# Define the parameter grid
param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}

# Initialize RandomizedSearchCV
random_search_mlp = RandomizedSearchCV(estimator=mlp, param_distributions=param_grid_mlp, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)

# Perform hyperparameter tuning
random_search_mlp.fit(X_train, y_train)

# Get the best parameters
print("Best Parameters for MLPRegressor:", random_search_mlp.best_params_)

# Evaluate the best model
best_model_mlp = random_search_mlp.best_estimator_
y_pred_mlp = best_model_mlp.predict(X_test)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
rmse_mlp = mse_mlp ** 0.5
r2_mlp = r2_score(y_test, y_pred_mlp)

print("Mean Squared Error for MLPRegressor:", mse_mlp)
print("Root Mean Squared Error for MLPRegressor:", rmse_mlp)
print("R-squared for MLPRegressor:", r2_mlp)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




