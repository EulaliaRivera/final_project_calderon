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

# Create a data frame
filename = 'final_project_3.csv'
df = pd.read_csv('final_project_3.csv', na_values=["-"])
print('\n')
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

## Counting the number of missing values in each column
print(df.isna().sum())

## Handling missing values
##  Mean Imputation
df_mean = df.copy()
for col in cols:
    mean_imputer = SimpleImputer(strategy='mean')
    df_mean[col] = mean_imputer.fit_transform(df_mean[col].values.reshape(-1,1))

#df.plot.scatter(x='Report Date',y='Long-term debt, less amounts due currently')
#plt.show()

# Plot to show the scatter plot of the imputed values
#fig = plt.Figure()
null_values = df["Long-term debt, less amounts due currently"].isnull()
#fig = df_mean.plot(x='Report Date',y='Long-term debt, less amounts due currently', kind= "scatter", c=null_values, cmap = "winter", title= "Mean imputation", colorbar=False)
df_mean.plot(x='Report Date',y='Long-term debt, less amounts due currently', kind= "scatter", c=null_values, cmap = "winter", title= "Mean imputation", colorbar=False)
plt.show()


##  Mean Imputation
df_mode = df.copy(deep=True)
for col in cols:
    mode_imputer = SimpleImputer(strategy='most_frequent')
    df_mode[col] = mode_imputer.fit_transform(df_mode[col].values.reshape(-1,1))

#df.plot.scatter(x='Report Date',y='Long-term debt, less amounts due currently')
#plt.show()

# Plot to show the scatter plot of the imputed values
#fig = plt.Figure()
null_values = df["Long-term debt, less amounts due currently"].isnull()
#fig = df_mean.plot(x='Report Date',y='Long-term debt, less amounts due currently', kind= "scatter", c=null_values, cmap = "winter", title= "Mean imputation", colorbar=False)
df_mode.plot(x='Report Date',y='Long-term debt, less amounts due currently', kind= "scatter", c=null_values, cmap = "winter", title= "Mode imputation", colorbar=False)
plt.show()

rs = np.random.RandomState(0)
df11 = df_mean.drop("Report Date", axis=1)
corr = df11.corr()
#corr.style.backgroud_gradient(cmap = 'coolwarm')
plt.show()

model = sm_api.ols('Total current assets ~ C(Accrued interest)', data=df).fit()
anova_table = sm.ststs.anova_ln(model, type=2)
print(anova_table)
                              
corr_matrix = df.corr().round(2)
sns.heatmap(corr_matrix,annot=True)
plt.show()

print('\n')
print(df['Accrued interest'].describe())

