#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 22:40:33 2021

@author: carolyndavis
"""

# =============================================================================
# Ace Realty wants to determine whether the average time it takes to sell homes 
# is different for its two offices. 
# A sample of 40 sales from office #1 revealed a mean of 90 days and a standard deviation of 15 days. 
# A sample of 50 sales from office #2 revealed a mean of 100 days and a standard deviation of 20 days. 
# Use a .05 level of significance.
# =============================================================================
# t = observed difference between sample means / standard error of the difference between the means





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from math import sqrt
from scipy import stats
from pydataset import data




# p-values with puppy vid


# pvalue - null hypothesis is true

#null: Fido is innocent
#describe the world where Fido is innocent 
# alt: fido is bad dog

# lower p value - more ridiculous null hypothesis is

#tells how riic our null hypothesis might look



#URL for exam data 
url = "https://gist.githubusercontent.com/ryanorsinger/2c13a71421037af127e9fa7fa1463cad/raw/3eb443414078b51af33fdb2d211159e5f3e220ab/exam_scores.csv"


df = pd.read_csv(url)  

df.study_strategy.fillna('None', inplace=True)


sns.pairplot(df, corner=True)
plt.suptitle("sns.pairplot visualizes continuous variable relationships")
plt.show()
