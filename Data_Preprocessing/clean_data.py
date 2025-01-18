import pandas as pd
import numpy as np
from googletrans import Translator
import matplotlib.pyplot as plt
import collections

'''
NOTES

(10/6 | Bryan):
If we get the ChatGPT API working, we may be able to use it to process the ethnicities.
There are many different variations of responses because the question was structured as 
free response and thus I'm not sure of a brute force script that will clean it up. We also could
ignore it but I think it may be interesting to see if ethnicity is more or less correlated to quality
of life than country of residence. What do yall think?

(10/7 | Jenni):
I tried translating the ETHNICITY column by importing Google, and it didn't work, so let's not do it.
'''

# import dataset
main_df = pd.read_csv('NegativeLifeExp_Data.csv')
print(main_df)

# Removing significantly underrepresented religions
rel_aff = main_df['AFFILIATION']
counter = collections.Counter(rel_aff)
clean_df = main_df
for key, values in counter.items():
    if values < 20:
        clean_df = clean_df[clean_df['AFFILIATION'] != key]
clean_df = clean_df.reset_index(drop=True)
print(clean_df)

# Removing signficantly underrepresneted marital statuses
marital = clean_df['MARITAL']
counter = collections.Counter(marital)
print(counter)
rows_to_drop = []
for i in range(len(clean_df['MARITAL'])):
    if '3' in str(clean_df['MARITAL'][i]): # if you are married
        clean_df['MARITAL'][i] = 3
    if '2' in str(clean_df['MARITAL'][i]): # if you are in a committed relationship
        clean_df['MARITAL'][i] = 2
    if '5' in str(clean_df['MARITAL'][i]): # if you are divorced
        clean_df['MARITAL'][i] = 5
    if '4' in str(clean_df['MARITAL'][i]): # if you are separated than you are single
        clean_df['MARITAL'][i] = 1
    if '1' in str(clean_df['MARITAL'][i]): # if you are single, convert string 1s to int 1s
        clean_df['MARITAL'][i] = 1
    
    if str(clean_df['MARITAL'][i]) not in ['1', '2', '3', '5']:
        rows_to_drop.append(i)
clean_df = clean_df.drop(rows_to_drop)
clean_df = clean_df.reset_index(drop=True)

# Removing significantly underrepresented genders
clean_df = clean_df[clean_df['GENDER'] != 3]
clean_df.reset_index(drop=True)
# plt.hist(clean_df['GENDER'])
# plt.show()

# Removing all unnecessary features
columns_to_drop = ['ID', 'GOODCOMPLETE', 'CONSENT', 'ETHNICITY', 'ABROAD', 'ABROAD_YEAR', 'MARITAL_TEXT', 
                   'EMPLOYMENT_TEXT', 'ATTENTION1', 'ATTENTION2', 'RELG_UP', 'AFF_UP', 'AFF_NOW', 'HONEST', 
                   'COMMENTS', 'DROP1', 'DROP2', 'CHECK']
clean_df = clean_df.drop(columns=columns_to_drop)

# For aggregate scores, I didn't use GOD1-14 since the options are a mix of positive and negative.

# Creating aggregate LE column. We may have to weight the events rather than assign them all a value of 1. 
  # E.g. being unemployed for a long time isn't necessarily equal to being physically abused by a romantic partner.
    # Score per item: 1-2
    # Total score range: 29-58
le_columns = [f'LE{i}' for i in range(1, 30)]
clean_df['LE_Sum'] = clean_df[le_columns].sum(axis=1)
clean_df = clean_df.drop(columns=le_columns)

# Creating aggregate DASS column
    # Score per item: 0-3
    # Total score range: 0-63
dass_columns = [f'DASS{i}' for i in range(1, 22)]
clean_df['DASS_Sum'] = clean_df[dass_columns].sum(axis=1)
clean_df = clean_df.drop(columns=dass_columns)

# Creating aggregate EX column
    # Score per item: 1 to 5
    # Total score range: 5 to 25
ex_columns = [f'EX{i}' for i in range(1, 6)]
clean_df['EX_Sum'] = clean_df[ex_columns].sum(axis=1)
clean_df = clean_df.drop(columns=ex_columns)

# Creating aggregate RELBEH column
    # Score per item: 0 to 8
    # Total score range: 0 to 40
relbeh_columns = [f'RELBEH{i}' for i in range(1, 6)]
clean_df['RELBEH_Sum'] = clean_df[relbeh_columns].sum(axis=1)
clean_df = clean_df.drop(columns=relbeh_columns)

# Creating aggregate LOCUS column
    # Score per item: -4, 0, 4
    # Total score range: -24 to 24
locus_columns = [f'LOCUS{i}' for i in range(1, 7)]
clean_df['LOCUS_Sum'] = clean_df[locus_columns].sum(axis=1)
clean_df = clean_df.drop(columns=locus_columns)

# Creating aggregate SWLS column
    # Score per item: -3, 0, 3
    # Total score range: -15 to 15
swls_columns = [f'SWLS{i}' for i in range(1, 6)]
clean_df['SWLS_Sum'] = clean_df[swls_columns].sum(axis=1)
clean_df = clean_df.drop(columns=swls_columns)

# Creating aggregate GOD column for attributes with a positive connotation.
    # Score per item: -4, 0, 4
    # Total score range: -24 to 24
God_pcolumns = [f'GOD{i}' for i in range(1, 8)]
clean_df['GOD_pSum'] = clean_df[God_pcolumns].sum(axis=1)
clean_df = clean_df.drop(columns=God_pcolumns)

# Creating aggregate GOD column for attributes with a negative connotation.
    # Score per item: -4, 0, 4
    # Total score range: -24 to 24
God_ncolumns = [f'GOD{i}' for i in range(8, 15)]
clean_df['GOD_nSum'] = clean_df[God_ncolumns].sum(axis=1)
clean_df = clean_df.drop(columns=God_ncolumns)

# Removing SBS columns 1-6, since the final SBS column is an average value for each person 
sbs_columns = [f'SBS{i}' for i in range(1, 7)]
clean_df = clean_df.drop(columns=sbs_columns)

# Removing DAQ columns 1-15, since the final DAQ column is an average value for each person 
daq_columns = [f'DAQ{i}' for i in range(1, 16)]
clean_df = clean_df.drop(columns=daq_columns)

# Removing JONG columns 1-12, since the final EDAS column is an average value for each person 
jong_columns = [f'JONG{i}' for i in range(1, 13)]
clean_df = clean_df.drop(columns=jong_columns)

clean_df.to_csv('CLEAN_DATA.csv', index=False)



    