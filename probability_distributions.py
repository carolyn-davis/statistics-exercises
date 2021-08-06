#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 09:11:48 2021

@author: carolyndavis
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import viz

# =============================================================================
# 1.)A bank found that the average number of cars waiting during the noon hour at
#  a drive-up window follows a Poisson distribution with a mean of 2 cars. Make a 
#  chart of this distribution and answer these questions concerning the probability 
#  of cars waiting at the drive-up window.
# =============================================================================
#poson disst
#mean of 2 cars waiting during noon hour 
from scipy.stats import poisson
import pandas as pd

x_rvs = poisson.rvs(2, size=100000, random_state=2)
x_rvs.mean()   #checks to see if the mean is similar to the stated average


#So we can graph it and see it, run it as a series, turns array into Series

x_rvs = pd.Series(poisson.rvs(2, size=100000, random_state=2))

data = x_rvs.value_counts().sort_index().to_dict()

data

fig, ax = plt.subplots(figsize=(16, 6))
ax.bar(range(len(data)), list(data.values()), align='center')
plt.xticks(range(len(data)), list(data.keys))
plt.show()

#Poison: liklihood of an event happening a certain number of times an event will occur within a
#predertmined amount of time

# What is the probability that no cars drive up in the noon hour?
def poisson_pmf(k, lam):
    return (lam ** k * np.exp(-lam)) / np.math.factorial(k)

poisson_pmf(0, 2)
#Ans: 0.135   #can also do poisson.pmf()

# What is the probability that 3 or more cars come through the drive through?
poisson.sf(3, 2)     #sf tells us the probability of a variable falling above a certain value
#Ans: 0.1428
# How likely is it that the drive through gets at least 1 car?
poisson.sf(1, 2)
#Ans: 0.539

# =============================================================================
# 2.)Grades of State University graduates are normally distributed with a mean of 
# 3.0 and a standard deviation of .3. Calculate the following:
# =============================================================================
mean_avg = 3.0
std_dev = 0.3


# What grade point average is required to be in the top 5% of the graduating class?

x = stats.norm(mean_avg, std_dev)
top_five = x.ppf(.95)
x.plot()
# Ans: 3.49

data = np.random.normal(loc=mean_avg, scale=std_dev, size=1000)
data = pd.DataFrame(data)
data.plot.scatter(x=data.index,y=data[0])
# What GPA constitutes the bottom 15% of the class?
#ppf accapets a dprobability, gives value associated with probability

bottom_15_percent = stats.norm.ppf(.15, loc=mean_avg, scale=std_dev)
#Ans:2.68

# An eccentric alumnus left scholarship money for students in the third decile from
#  the bottom of their class. Determine the range of the third decile. Would a student 
#  with a 2.8 grade point average qualify for this scholarship?




# If I have a GPA of 3.5, what percentile am I in?




# =============================================================================
# 3.)A marketing website has an average click-through rate of 2%. One day they observ
#     4326 visitors and 97 click-throughs. How likely is it that this many people or more 
#     click through?
# =============================================================================





# =============================================================================
# 4.)You are working on some statistics homework consisting of 100 questions where all
#  of the answers are a probability rounded to the hundreths place. Looking to save time, 
#  you put down random probabilities as the answer to each question.
# =============================================================================




# What is the probability that at least one of your first 60 answers is correct?






# =============================================================================
# 5.)The codeup staff tends to get upset when the student break area is not cleaned up.
#  Suppose that there's a 3% chance that any one student cleans the break area when 
#  they visit it, and, on any given day, about 90% of the 3 active cohorts of 22 
#  students visit the break area. How likely is it that the break area gets cleaned 
#  up each day? How likely is it that it goes two days without getting cleaned up? All week?
# =============================================================================






# =============================================================================
# 6.)You want to get lunch at La Panaderia, but notice that the line is usually very 
# long at lunchtime. After several weeks of careful observation, you notice that 
# the average number of people in line when your lunch break starts is normally 
# distributed with a mean of 15 and standard deviation of 3. If it takes 2 minutes 
# for each person to order, and 10 minutes from ordering to getting your food, what 
# is the likelihood that you have at least 15 minutes left to eat your food before 
# you have to go back to class? Assume you have one hour for lunch, and ignore travel 
# time to and from La Panaderia.
# =============================================================================






# =============================================================================
# 7.)Connect to the employees database and find the average salary of current employees,
#     along with the standard deviation. For the following questions, calculate the answer
#     based on modeling the employees salaries with a normal distribution defined by the
#     calculated mean and standard deviation then compare this answer to the actual values 
#     present in the salaries dataset.
# =============================================================================


# a.)What percent of employees earn less than 60,000?


# b.)What percent of employees earn more than 95,000?



c.)What percent of employees earn between 65,000 and 80,000?
What do the top 5% of employees make?