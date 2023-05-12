# Final Project
# Class 3303
# Eulalia Rivera 

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as sm_api
import matplotlib.pylab as plt
import numpy as np
from sklearn.impute import SimpleImputer
import seaborn as sns
from statsmodels.formula.api import ols

# Create a data frame
filename = 'final_project_3.csv'
df = pd.read_csv('final_project_3.csv', na_values=["-"])
print('\n')

print(df.describe())
print('\n')

print(df['Accrued interest'].describe())
print('_____')

print(f"The data has {df.shape[0]} rows and {df.shape[1]} columns.")
cols = [ 'Accumulated deferred income taxes',
       'Other noncurrent liabilities & deferred credits', 'Accrued interest',
       'Restricted cash.1', 'Total liabilities',
       'Commodity & other derivative contractual liabilities.1',
       'Commodity & other derivative contractual liabilities',
       'Total current assets', 'Common stock', 'Short-term borrowings',
       'Materials & supplies', 'Total current liabilities',
       'Uncertain tax positions, including accrued interest',
       'Long-term debt, less amounts due currently',
       'Commodity & other derivative contractual assets.1', 'Total assets',
       'Accrued taxes', 'Trade accounts payable',
       'Allowance for uncollectible accounts', 'Long-term debt due currently',
       'Fuel stock', 'Accumulated other comprehensive income (loss)',
       'Natural gas in storage', 'Restricted cash', 'Inventories',
       'Other noncurrent assets', 'Less accumulated depreciation',
       'Construction work in progress', 'Accumulated deferred income taxes.1',
       'Electric', 'Investments', 'Goodwill', 'Other current assets',
       'Asset retirement obligations', 'Trade accounts receivable - net',
       'Cash & cash equivalents', 'Other current liabilities',
       'Commodity & other derivative contractual assets', 'Land']

#Converting the columns from objects to float types
for col in cols:
    df[col] = df[col].astype(float)
print(df.dtypes)
print("______ ")
df.dtypes.to_csv('C:/Users/erivera17/Desktop/df.csv',index=False)

## Counting the number of missing values in each column
print(df.isna().sum())
print("______ ")

## Handling missing values
##  Mean Imputation
df_mean = df.copy()
for col in cols:
    mean_imputer = SimpleImputer(strategy='mean')
    df_mean[col] = mean_imputer.fit_transform(df_mean[col].values.reshape(-1,1))


# Plot to show the scatter plot of the imputed values
#fig = plt.Figure()
null_values = df["Long-term debt, less amounts due currently"].isnull()
#fig = df_mean.plot(x='Report Date',y='Long-term debt, less amounts due currently', kind= "scatter", c=null_values, cmap = "winter", title= "Mean imputation", colorbar=False)
df_mean.plot(x='Report Date',y='Long-term debt, less amounts due currently', kind= "scatter", c=null_values, cmap = "winter", title= "Mean imputation", colorbar=False)
plt.xticks(rotation=90, ha='right')  # rotate the x-axis labels by 90 degrees
plt.show()

##  Mode Imputation
df_mode = df.copy(deep=True)
for col in cols:
    mode_imputer = SimpleImputer(strategy='most_frequent')
    df_mode[col] = mode_imputer.fit_transform(df_mode[col].values.reshape(-1,1))

# Plot to show the scatter plot of the imputed values
#fig = plt.Figure()
null_values = df["Long-term debt, less amounts due currently"].isnull()
#fig = df_mean.plot(x='Report Date',y='Long-term debt, less amounts due currently', kind= "scatter", c=null_values, cmap = "winter", title= "Mean imputation", colorbar=False)
df_mode.plot(x='Report Date',y='Long-term debt, less amounts due currently', kind= "scatter", c=null_values, cmap = "winter", title= "Mode imputation", colorbar=False)
plt.xticks(rotation=90, ha='right')  # rotate the x-axis labels by 90 degrees
plt.show()

rs = np.random.RandomState(0)
df11 = df_mean.drop("Report Date", axis=1)
corr = df11.corr()

# Compute the absolute correlations and make the heatmap symmetric
corr_abs = np.abs(corr)
corr_abs *= np.tri(*corr_abs.shape[::-1])  # zero out the upper triangular part
corr_abs += corr_abs.T  # add the transpose to make the matrix symmetric

plt.figure(figsize=(8, 8))
sns.heatmap(corr_abs, annot=False, cmap='coolwarm', linewidths=0.5, cbar=True, vmin=-1, vmax=1)
plt.title('Absolute Correlation Matrix')
plt.show()

# Define three predictors (x) variables and response variable
y = df['Inventories']
x = df[['Electric','Investments', 'Accrued taxes']]
# Add constant to predictor
x = sm.add_constant(x)

#Fit the ANOVA Model
formula = 'Q("Total current assets") ~  Q("Accrued interest")'
model = ols(formula, data=df).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
#Print the ANOVA table 
print(aov_table)

#for i in range(2,len(cols)+1):
#   print(i)

    
    
    

