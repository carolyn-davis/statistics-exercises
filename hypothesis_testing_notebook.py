#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 22:40:33 2021

@author: carolyndavis
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from math import sqrt
from pydataset import data

# =============================================================================
#                             T-TEST EXERCISES
# =============================================================================
# =============================================================================
# 
# =============================================================================
# =============================================================================
# 1.)Ace Realty wants to determine whether the average time it takes to sell homes is
#  different for its two offices. A sample of 40 sales from office #1 revealed
#  a mean of 90 days and a standard deviation of 15 days. A sample of 50 sales
#  from office #2 revealed a mean of 100 days and a standard deviation of 20 days.
#  Use a .05 level of significance.
# =============================================================================
x = np.arange(50,150)
#range of 40
y1 = stats.norm(90,15).pdf(x)   #because time is continuous variable assume normal distribution
y2 = stats.norm(100,20).pdf(x)
                #(mean pop sample 1, std sample 1)

plt.plot(x, y1, label = 'office 1')
plt.plot(x, y2, label = 'office 2')
plt.axvline(90, ls = ':')  #pop mean of samp 1
plt.axvline(100, ls = ':', color = 'orange')   #pop mean if samp 2


plt.legend()

# Null: There us no difference in average sale time between the two offices
#Alt: There is a difference in average sale tume between the two offices 

alpha = 0.05 #prob threshold for rejecting the null hypothesis


t, p = stats.ttest_ind_from_stats(100,20,50, 90,15,40)
                            #stats(mean_sp2, std_samp2, sales, "sample 1 data")
#t=2.6252287036468456
#p=0.01020985244923939
#Because the p-value is less than the alpha value, we reject the null hypothesis.
                            #T Distribution
#T distribution can be can be performed with a two sample t-test 

alpha = 0.05   #significance level

xbar1 = 90 #sample mean 1
xbar2 = 100 #sample mean 2

samp_size1 = 40
samp_size2 = 50

std1 = 15
std2 = 20

degf = samp_size1 + samp_size2 - 2 #add the two samp pops and subract 1 from each pop

#pooled std's

std_pool = sqrt(((samp_size1 - 1) * std1**2 + (samp_size2 - 1) * std2**2) / 
                (samp_size1 + samp_size2 - 2))
#std_pool = 17.956702977389302

#calculating the t-statistic value

test = (xbar2 - xbar1) / (std_pool * sqrt(1 / samp_size1 + 1 / samp_size2 ))
#test statistic/t = 2.6252287036468456

#Now we can get the p-value
p = stats.t(degf).sf(test) * 2
#p-value/p = 0.01020985244923939


# =============================================================================
# 2.) Load the mpg dataset and use it to answer the following questions:
# =============================================================================
# a.) Is there a difference in fuel-efficiency in cars from 2008 vs 1999?
mpg_data = data('mpg')
#Null Hyp









mac_df = pd.DataFrame({'codeup_student':[49], 'not_codeup_student':[20]})


dont_df = pd.DataFrame({'codeup_student':[1], 'not_codeup_student':[30]})


mac_observed = pd.crosstab(mac_df.codeup_student, dont_df.not_codeup_student)


chi2, p, degf, expected = stats.chi2_contingency(mac_observed)


#p=1 yes it is contingent of being a codeip student
# =============================================================================
# 
# 2.)Choose another 2 categorical variables from the mpg dataset and perform a 
# c
# h
# i
# 2
#  contingency table test with them. Be sure to state your null and alternative hypotheses.
mpg_df = data('mpg')
# =============================================================================

mpg_observed = pd.crosstab(mpg_df.manufacturer, mpg_df.trans)

chi2, p, degf, expected = stats.chi2_contingency(mpg_observed)


model_observed = pd.crosstab(mpg_df.model, mpg_df.year)
chi2, p, degf, expected = stats.chi2_contingency(model_observed)

# =============================================================================
# 3.) Use the data from the employees database to answer these questions:

# Is an employee's gender independent of whether an employee works in sales or marketing? 
# (only look at current employees)
# Is an employee's gender independent of whether or not they are or have been a manager?
# =============================================================================

from env import host, user, password


def get_db_url(user, host, password, database, query):
    
    url = f'mysql+pymysql://{user}:{password}@{host}/employees'
    
    return pd.read_sql(query, url)


database = 'employees'

query = """

SELECT employees, dept_emp, department
FROM employees



"""


emp_df = get_db_url(user, host, password, database, query)
emp_df.head()