import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

'''
(10/8 | Jenni)
This file aims to do feature selecting by finding the correlation between continuous features and labels.
'''

variables = [
    'RELG_NOW', 'SPIRIT_NOW', 
    'SBS', 'EDAS', 'DAQ', 'LE_Sum', 
    'DASS_Sum', 'EX_Sum', 'RELBEH_Sum', 
    'LOCUS_Sum', 'SWLS_Sum', 'GOD_pSum', 'GOD_nSum'
]

df = pd.read_csv('CLEAN_DATA.csv')
correlation_matrix = df[variables].corr()


plt.figure(figsize=(12, 10))
cmap = sns.diverging_palette(10, 130, as_cmap=True)
sns.heatmap(correlation_matrix, 
            annot=True,  # Show the correlation values
            cmap=cmap,   # Use the custom colormap
            vmin=-1, vmax=1,  # Set the range of values
            center=0,    # Center the colormap at 0
            square=True, # Make sure the cells are square
            linewidths=.5,  # Add lines between cells
            cbar_kws={"shrink": .8})  # Adjust the size of the colorbar

plt.title('Correlation of Continuous Variables', fontsize=16)
plt.tight_layout()
#plt.show()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')