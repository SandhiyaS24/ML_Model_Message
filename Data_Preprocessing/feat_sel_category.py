import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

'''
(10/8 | Jenni)
This file aims to do feature selecting by finding features with the lowest variance for each y value.

Preliminary Features Chosen: X values
  COUNTRY, GENDER, AGE, MARITAL, EMPLOYMENT, INCOME, AFFILIATION
  Justification of choices: I chose features that could be easily asked questions of person (new object in dataset), and not require extensive questioning.
  I also removed AFF_NOW and decided to keep AFFILIATION instead. This means a user would have to enter a general rather than specific religion. E.g. Muslim instead of Sunni.

Preliminary Labels (classes) Chosen: y_vars

Key note: Correlation does not guarantee causation. But, correlation can help in feature selection.
'''

y_vars = ['RELG_NOW', 'SPIRIT_NOW', 'SBS', 'EDAS', 'DAQ', 
          'LE_Sum', 'DASS_Sum', 'EX_Sum', 'RELBEH_Sum', 'LOCUS_Sum', 
          'SWLS_Sum', 'GOD_pSum', 'GOD_nSum']

# ---------------------------------
# COUNTRY - categorical
# ---------------------------------

df = pd.read_csv('CLEAN_DATA.csv')
countries = df['COUNTRY'].unique()
var_country = {country: {} for country in countries}

for country in countries:
    country_data = df[df['COUNTRY'] == country]
    for y_var in y_vars:
        variance = np.var(country_data[y_var].dropna())
        var_country[country][y_var] = variance
results_df = pd.DataFrame(var_country).T
results_df.to_csv('var_country.csv')
df = pd.read_csv('var_country.csv', index_col=0)
average_variance_country = df.mean()
print("Country Variance")
print(average_variance_country)
print("\n")

# ---------------------------------
# GENDER - continuous (binary)
# ---------------------------------

df = pd.read_csv('CLEAN_DATA.csv')
genders = df['GENDER'].unique()
var_gender = {gender: {} for gender in genders}

for gender in genders:
    gender_data = df[df['GENDER'] == gender]
    for y_var in y_vars:
        variance = np.var(gender_data[y_var].dropna())
        var_gender[gender][y_var] = variance

results_df = pd.DataFrame(var_gender).T
results_df.to_csv('var_gender.csv')
average_variance_gender = results_df.mean()
print("Gender Variance")
print(average_variance_gender)
print("\n")

# ---------------------------------
# AGE - continuous
# ---------------------------------

df = pd.read_csv('CLEAN_DATA.csv')
df['AGE_GROUP'] = pd.cut(df['AGE'], bins=[0, 19, 29, 39, 49, 59, 69, 79,], 
                         labels=['Teens', '20s', '30s', '40s', '50s', '60s', '70s'])
age_groups = df['AGE_GROUP'].unique()
var_age_group = {age_group: {} for age_group in age_groups}

for age_group in age_groups:
    if pd.notna(age_group):
        age_data = df[df['AGE_GROUP'] == age_group]
        for y_var in y_vars:
            variance = np.var(age_data[y_var].dropna())
            var_age_group[age_group][y_var] = variance

results_df = pd.DataFrame(var_age_group).T
results_df.to_csv('var_age_group.csv')
average_variance_age = results_df.mean()
print("Age Variance")
print(average_variance_age)
print("\n")

# ---------------------------------
# MARITAL - continuous
# ---------------------------------

df = pd.read_csv('CLEAN_DATA.csv')
statuses = df['MARITAL'].unique()
var_marital = {marital: {} for marital in statuses}

for marital in statuses:
    marital_data = df[df['MARITAL'] == marital]
    for y_var in y_vars:
        variance = np.var(marital_data[y_var].dropna())
        var_marital[marital][y_var] = variance

results_df = pd.DataFrame(var_marital).T
results_df.to_csv('var_marital.csv')
average_variance_marital = results_df.mean()
print("Marital Variance")
print(average_variance_marital)
print("\n")

# ---------------------------------
# EMPLOYMENT - continuous
# ---------------------------------

df = pd.read_csv('CLEAN_DATA.csv')
employment_statuses = df['EMPLOYMENT'].unique()
var_employment = {status: {} for status in employment_statuses}

for status in employment_statuses:
    status_data = df[df['EMPLOYMENT'] == status]
    for y_var in y_vars:
        variance = np.var(status_data[y_var].dropna())
        var_employment[status][y_var] = variance

results_df = pd.DataFrame(var_employment).T
results_df.to_csv('var_employment.csv')
average_variance_employment = results_df.mean()
print("Employment Status Variance")
print(average_variance_employment)
print("\n")

# ---------------------------------
# INCOME - continuous
# ---------------------------------

df = pd.read_csv('CLEAN_DATA.csv')
income_levels = df['INCOME'].unique()
var_income = {income: {} for income in income_levels}

for income in income_levels:
    income_data = df[df['INCOME'] == income]
    for y_var in y_vars:
        variance = np.var(income_data[y_var].dropna())
        var_income[income][y_var] = variance

results_df = pd.DataFrame(var_income).T
results_df.to_csv('var_income.csv')
average_variance_income = results_df.mean()
print("Income Variance")
print(average_variance_income)
print("\n")

# ---------------------------------
# AFFILIATION - categorical, but pretty sure the original data was one hot encoded
# ---------------------------------

df = pd.read_csv('CLEAN_DATA.csv')
affiliation_groups = df['AFFILIATION'].unique()
var_affiliation = {affiliation: {} for affiliation in affiliation_groups}

for affiliation in affiliation_groups:
    aff_data = df[df['AFFILIATION'] == affiliation]
    for y_var in y_vars:
        variance = np.var(aff_data[y_var].dropna())
        var_affiliation[affiliation][y_var] = variance

results_df = pd.DataFrame(var_affiliation).T
results_df.to_csv('var_affiliation.csv')
average_variance_affiliation = results_df.mean()
print("Affiliation Variance")
print(average_variance_affiliation)
print("\n")

# Does the data have a normal distribution

df = pd.read_csv('CLEAN_DATA.csv')
y_vars = ['RELG_NOW', 'SPIRIT_NOW', 'SBS', 'EDAS', 'DAQ', 
          'LE_Sum', 'DASS_Sum', 'EX_Sum', 'RELBEH_Sum', 'LOCUS_Sum', 
          'SWLS_Sum', 'GOD_pSum', 'GOD_nSum']

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 20))
axes = axes.flatten()

for i, var in enumerate(y_vars):
    df[var].hist(ax=axes[i], bins=30)
    axes[i].set_title(var)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('y_variables_histograms.png')
plt.close()

# ---------------------------------
#  For each of the scale labels, what are in order of lowest to highest variance
# ---------------------------------

feature_variances = {
    'COUNTRY': average_variance_country,
    'GENDER': average_variance_gender,
    'AGE': average_variance_age,
    'MARITAL': average_variance_marital,
    'EMPLOYMENT': average_variance_employment,
    'INCOME': average_variance_income,
    'AFFILIATION': average_variance_affiliation
}

# Find the feature with the lowest variance for each y variable
lowest_variance_features = {}

for y_var in y_vars:
    lowest_variance = float('inf')
    lowest_feature = None
    
    for feature, variances in feature_variances.items():
        if y_var in variances and variances[y_var] < lowest_variance:
            lowest_variance = variances[y_var]
            lowest_feature = feature
    
    lowest_variance_features[y_var] = lowest_feature

# Create a DataFrame with y variables as columns and lowest variance features as the single row
lowest_variance_df = pd.DataFrame([lowest_variance_features])

# Transpose the DataFrame to have y variables as index and 'Lowest Variance Feature' as the column
lowest_variance_df = lowest_variance_df.T
lowest_variance_df.columns = ['Lowest Variance Feature']

# Save the result to a CSV file
lowest_variance_df.to_csv('feature_selection.csv')

