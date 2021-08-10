#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 12:08:04 2021

@author: carolyndavis
"""
import numpy as np
from fractions import Fraction
from collections import Counter
import itertools as it


import pandas as pd
# =============================================================================
# 1.)How likely is it that you roll doubles when rolling two dice?
# =============================================================================
outcomes = [1,2,3,4,5,6]
n_rows = 10_000    #total number of trials 
n_cols = 2          # number of times the dice is being rolled

np.random.choice(outcomes, 4)


np.random.choice(outcomes, n_rows * n_cols)

rolls = np.random.choice(outcomes, n_rows * n_cols).reshape(n_rows, n_cols) #gives simultions in organized format

rolls_df = pd.DataFrame(rolls)

rolls_df['doubles'] = rolls_df[0] == rolls_df[1]
rolls_df.doubles.mean()
#Ans: 0.1692


# =============================================================================
# 2.)If you flip 8 coins, what is the probability of getting exactly 3 heads? What is the 
# probability of getting more than 3 heads?
# =============================================================================
outcomes = [1, 0]
flips = np.random.choice(outcomes, size=(100_000, 8))
num_of_heads = flips.sum(axis=1)
(num_of_heads == 3).mean()
#Ans: .21654
outcomes = [1, 0]
flips = np.random.choice(outcomes, size=(100_000, 8))
num_of_heads = flips.sum(axis=1)
(num_of_heads >= 3).mean()
#Ans: 3 or more heads is 0.85405




# =============================================================================
# 3.)There are approximitely 3 web development cohorts for every 1 data science cohort at Codeup. 
# Assuming that Codeup randomly selects an alumni to put on a billboard, what are the odds 
# that the two billboards I drive past both have data science students on them?
# =============================================================================


outcomes = ["Web Dev", "Data Science"]
poss = np.random.choice(outcomes, size=(10_000, 2), p=[.75, .25])
# (poss == "Data Science","Data Science").mean()
double = pd.DataFrame(poss)
double = double[(double[0] == "Data Science") & (double[1] == "Data Science")]
ans = len(double) / len(poss)
#Ans: .0613

# =============================================================================
# 4.)Codeup students buy, on average, 3 poptart packages (+- 1.5) a day from the snack 
# vending machine. If on monday the machine is restocked with 17 poptart packages, 
# how likely is it that I will be able to buy some poptarts on Friday afternoon?
# =============================================================================

n_rows = 10_000   # number of trials to be ran
n_cols = 5    #number of days to be simulated

rolls = np.random.normal(3, 1.5, n_rows * n_cols).astype(int).reshape(n_rows, n_cols)   #3 poptart packagesbought/on avg 1.5x a day

poptart_df = pd.DataFrame(rolls)

poptart_df['total_bought'] = poptart_df.sum(axis=1)
poptart_df['remain_poptarts'] = poptart_df['total_bought'] < 17

tarts_prob = int(round(poptart_df['remain_poptarts'].mean(), 2) * 100)
#Ans: odds of being able to buy poptart on friday afternoon = 88%


# =============================================================================
# 5.)Compare Heights
# 
# Men have an average height of 178 cm and standard deviation of 8cm.
# Women have a mean of 170, sd = 6cm.
# Since you have means and standard deviations, you can use np.random.normal to generate observations.
# If a man and woman are chosen at random, P(woman taller than man)?
# =============================================================================


mens_obs = np.random.normal(loc=178, scale=8, size=1_000_000)
woms_obs = np.random.normal(loc=170, scale=6, size=1_000_000)

#comparing the arrays and setting the indexes earlier 
height_df = pd.DataFrame({"mens_obs" : mens_obs,
                          "woms_obs" : woms_obs})

height_df['taller_females'] = height_df.woms_obs > height_df.mens_obs

height_df['taller_females'].mean()   #mean of female is taller 0.17

female_taller = int(round(height_df['taller_females'].mean(),2) * 100)
#Ans: 21% chance of female taller than man
# =============================================================================
# # 6.) When installing anaconda on a student's computer, there's a 1 in 250 chance that the 
# # download is corrupted and the installation fails. 
# =============================================================================
corrupt = 1/250
success = 1 - corrupt
# =============================================================================
# 6b.)What are the odds that after having 50 
# students download anaconda, no one has an installation issue? 100 students?
# =============================================================================


n_rows = 100_000
n_cols = 50  #num of students in each sim



install_range = [False for i in range(1,250)]
install_range.append(True)

installs = np.random.choice(install_range, n_rows * n_cols).reshape(n_rows, n_cols)
#^^creates row for each simulation, column for each student

installs = pd.DataFrame(installs)   #convert to df for col manipulation


no_corruption = installs['Not Corrupted'] = installs.sum(axis=1) == 0 #adding calculation columns

#gets mean calcs for True False values in columns
issues = int(round(installs['Not Corrupted'].mean(),2) * 100)

#Ans: 67 % no issues for 100 students/// 82% for 50 student total

# =============================================================================
# 6c.)What is the probability that we observe an installation issue within the first 
# 150 students that download anaconda?
# =============================================================================

n_rows = 100_000
n_cols = 150  #num of students in each sim

install_range = [False for i in range(1,250)]
install_range.append(True)

installs = np.random.choice(install_range, n_rows * n_cols).reshape(n_rows, n_cols)


installs = pd.DataFrame(installs) 

installs['Corrupted'] = installs.sum(axis=1) > 0

issues = int(round(installs['Corrupted'].mean(),2) * 100)
#Ans: odds of corruption in first 150 students = 45%

# =============================================================================
# 
# 6d.)How likely is it that 450 students all download anaconda without an issue?
# =============================================================================
n_rows = 100_000
n_cols = 450


install_range = [False for i in range(1,250)]
install_range.append(True)

installs = np.random.choice(install_range, n_rows * n_cols).reshape(n_rows, n_cols)

installs = pd.DataFrame(installs) 

no_corruption = installs['Not Corrupted'] = installs.sum(axis=1) == 0 

issues = int(round(installs['Not Corrupted'].mean(),2) * 100)

#Ans: odds of no corruption for 450 students is 17%
# =============================================================================
#     
# 7.)There's a 70% chance on any given day that there will be at least one food 
# truck at Travis Park. However, you haven't seen a food truck there in 3 days. 
# How unlikely is this?
# =============================================================================

n_rows = 10_000   #trials

n_cols = 3 #days of simulation

ft_sighting = np.random.choice([True, False], n_rows * n_cols, p=[.7,.3]).reshape(n_rows, n_cols)

ft_sighting = pd.DataFrame(ft_sighting)



ft_sighting['times_seen'] = ft_sighting.sum(axis=1)
ft_sighting['not_seen_3_days'] = ft_sighting.times_seen == 0


ft_sighting['not_seen_3_days'].mean()
truck_probability = int(round(ft_sighting['not_seen_3_days'].mean(),2) * 100)
#Ans: 3% liklihood of not seeing a foodtruck in Travis for 3 days
# =============================================================================
# How likely is it that a food truck will show up sometime this week?
# =============================================================================
n_rows = 10_000   #trials

n_cols = 2 #days of simulation

ft_sighting = np.random.choice([True, False], n_rows * n_cols, p=[.7,.3]).reshape(n_rows, n_cols)

ft_sighting = pd.DataFrame(ft_sighting)



ft_sighting['not_seen_2_days'] = ft_sighting.sum(axis=1) == 0


ft_sighting['not_seen_2_days'].mean()
truck_probability = int(round(ft_sighting['not_seen_2_days'].mean(),4) * 100)

#Ans: 8% liklihood of not seeing a foodtrycj in Travis remaining 2 days, 92% that you will see one



# =============================================================================
# 8.)If 23 people are in the same room, what are the odds that two of them share a birthday?
#  What if it's 20 people? 40?
# =============================================================================
n_rows = 100_000
n_cols = 23

#creates array of birthdays for each person in simulation
birthdays = np.random.choice([i for i in range(366)], n_rows * n_cols).reshape(n_rows, n_cols)
#^^value for each studetn in 1 calendar year

birth_df = pd.DataFrame(birthdays) #convert to data frame


#add calculation columns to df for unique bdays
birth_df['twoshared_bday'] = birth_df.nunique(axis=1) < n_cols
birth_df['exactly_2shared_bdays'] = birth_df.nunique(axis=1) == n_cols-1

#produces mean of calculation True/False
twoshared_bday = int(round(birth_df['twoshared_bday'].mean(),2) * 100)
exactly_2shared_birthdays = int(round(birth_df['exactly_2shared_bdays'].mean(),2) * 100)
#exactly two shared bdays = 13%
#two shared bdays = 51%


# =============================================================================
#                         20People
# =============================================================================
n_rows = 100_000
n_cols = 20


birthdays = np.random.choice([i for i in range(366)], n_rows * n_cols).reshape(n_rows, n_cols)

birth_df = pd.DataFrame(birthdays)

birth_df['twoshared_bday'] = birth_df.nunique(axis=1) < n_cols
birth_df['exactly_2shared_bdays'] = birth_df.nunique(axis=1) == n_cols-1

twoshared_bday = int(round(birth_df['twoshared_bday'].mean(),2) * 100)
exactly_2shared_birthdays = int(round(birth_df['exactly_2shared_bdays'].mean(),2) * 100)

#exactly two shared bdays = 9%
#two shared bdays = 41%


# =============================================================================
#                         40 PEOPLE
# =============================================================================
n_rows = 100_000
n_cols = 40


birthdays = np.random.choice([i for i in range(366)], n_rows * n_cols).reshape(n_rows, n_cols)

birth_df = pd.DataFrame(birthdays)

birth_df['twoshared_bday'] = birth_df.nunique(axis=1) < n_cols
birth_df['exactly_2shared_bdays'] = birth_df.nunique(axis=1) == n_cols-1

twoshared_bday = int(round(birth_df['twoshared_bday'].mean(),2) * 100)
exactly_2shared_birthdays = int(round(birth_df['exactly_2shared_bdays'].mean(),2) * 100)

#exactly two shared bdays = 28%
#two shared bdays = 89%