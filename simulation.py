#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 12:08:04 2021

@author: carolyndavis
"""
import numpy as np
# =============================================================================
# 1.)How likely is it that you roll doubles when rolling two dice?
# =============================================================================
outcomes = [1,2,3,4,5,6]
n_simulations = 10_000

die1 = np.random.choice(outcomes, size=n_simulations)
die2 = np.random.choice(outcomes, size=n_simulations)


#Each array here shares and dindex where index = simulation_number
die1[0:3], die2[0:3]

(die1 == 2)[0:10]

(die2 == 2)[0:10]

((die1 == 2) & (die2 == 2))[0:10]
d1_rolled_2 = die1 == 2
d2_rolled_2 = die2 == 2
d1_rolled_2[0:5], d2_rolled_2[0:5]

both_rolled_2 = d1_rolled_2 & d2_rolled_2
both_rolled_2.mean()
1/6 * 1/6
#Ans: 0.027/0.0261



# =============================================================================
# np.random.choice([1,2,3,4, 5, 6], 4)   #generates a list of four random numbers in this list
# rolls = np.random.choice([1,2,3,4,5,6], nrows * ncols).reshape(nrows, ncols)
# #convert to data frame
# df = pd.DataFrame(rolls)
# 
# #Add calculation column 
# df['doubles'] = df[0] == df[1]
# df.head()
# =============================================================================



# =============================================================================
# 2.)If you flip 8 coins, what is the probability of getting exactly 3 heads? What is the 
# probability of getting more than 3 heads?
# =============================================================================
outcomes = [1, 0]
flips = np.random.choice(outcomes, size=(100_000, 8))
num_of_heads = flips.sum(axis=1)
(num_of_heads == 3).mean()
#Ans: .21654

# =============================================================================
# 3.)There are approximitely 3 web development cohorts for every 1 data science cohort at Codeup. 
# Assuming that Codeup randomly selects an alumni to put on a billboard, what are the odds 
# that the two billboards I drive past both have data science students on them?
# =============================================================================
import pandas as pd

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
# =============================================================================
# 5.)Compare Heights
# 
# Men have an average height of 178 cm and standard deviation of 8cm.
# Women have a mean of 170, sd = 6cm.
# Since you have means and standard deviations, you can use np.random.normal to generate observations.
# If a man and woman are chosen at random, P(woman taller than man)?
# =============================================================================


mens_obs = np.random.normal(loc=178, scale=6, size=1_000_000)
woms_obs = np.random.normal(loc=170, scale=6, size=1_000_000)

#comparing the arrays and setting the indexes earlier 
(woms_obs > mens_obs).mean()
#Ans: 0.172602

#20% of the time 

# =============================================================================
# # 6.) When installing anaconda on a student's computer, there's a 1 in 250 chance that the 
# # download is corrupted and the installation fails. 
# =============================================================================

# =============================================================================
# 6b.)What are the odds that after having 50 
# students download anaconda, no one has an installation issue? 100 students?
# =============================================================================

outcomes = [1,2]
# 249/250
# 1/250
chance = np.random.choice(outcomes, size=(10_000, 50), p=[.996,.004])
actual = pd.DataFrame(chance)
actual = actual[(actual[0] == 2)]
ans = len(actual) / len(chance)
# =============================================================================
# 6c.)What is the probability that we observe an installation issue within the first 
# 150 students that download anaconda?
# =============================================================================



# =============================================================================
# 
# 6d.)How likely is it that 450 students all download anaconda without an issue?
# =============================================================================



    

