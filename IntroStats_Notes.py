#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 09:47:51 2021

@author: carolyndavis
"""

# =============================================================================
#                         Introdiuction to Stats
# =============================================================================
x = [1,2,3]   #This is an example of sigma notation 
sum(x)              #functions are already defined in the library


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 


# =============================================================================
#         Simulations == Monte Carlo method 
# =============================================================================
#Simulation means to run an experiment
#Trial is the number of items in each experiment 
# =============================================================================
#                  FOUR STEPS TO NEVER FORGET... 
# =============================================================================

#1.) Figure out a way to represent data

#2.) Create a matrix of random data, rows = simulations, columns = trial
    #for example rolling dice 10, 000 times means rows= 10,000 and columns = 2 because we roll  2 dice each time

#3.) Apply an aggreate function row wise to get the results of the simulation
#4.) Apply a final aggregate to get our probability 

#Let's flip a coin 100,000 times and figure out the probability of flipping Heads


#Representing the data
outcomes = ["Heads", "Tails"]
n_simulations = 1_000_000   #represents 1 million

flips = np.random.choice(outcomes, size=n_simulations)
flips[0:5]    #numpy doesnt have head() function
#After flipping 1 million coins our experimental probability of flipping heads 
(flips == "Heads").mean()   #Generates new random values everytime close to .50

#.mean gives us the proportion of times 

#Step 1 rep our outcomes 


outcomes = [1,2,3,4,5,6]

#Step 2 Create the data

n_simulations = 10_000
rolls = np.random.choice(outcomes, size=n_simulations)
#What are the chances we roll exactly a five?
(rolls == 5).mean()\
    


#What is the probability we will roll a five or a six?
(rolls >= 5).mean()

#What is the probability of rolling less than a 3 but not including a 3?
(rolls < 3).mean()

#What is the chances we roll something other than a threee?
(rolls != 3).mean()


#Lets Roll Two dice At Once

#What are the odds of rollijng Snake Euyes on two dice?

#Step 1
outcomes = [1,2,3,4,5,6]

#Simulation = the number if times we run the experiment
#Trials = the number of things in each experiment 
n_simulations = 1_000_000 
n_trials = 2

#Size argument can set our simulation and trial size
rolls = np.random.choice(outcomes, size=n_simulations)

rolls[0:5]

#Step 34 apply Agg tow wise
#axis=1 mean sum across the the rows
sum_of_rolls = rolls.sum(axis=1)
sum_of_rolls[0:5]

#Axis=0 means sum ip the entire column
#If you dont put an axis the default is  
rolls.sum(axis=0)

#Step 4
#Add up all the the tumes that an experiment produces the su, of 2



#What is the probability oof rolling a 7 on two dice
# 1+6, 2+5, 3+4....

#Step1
outcomes = [1,2,3,4,5,6]
rolls = np.random.choice(outcomes, size=(10_000, 2))
rolls[0:4]

#Step 3 apply rowise aggregate
#axis=1 to apply sum to rows 

sum_of_rolls = rolls.sum(axis=1)
sum_of_rolls[0:4]

p = (sum_of_rolls == 7).mean()
print(f"The experimental probability of rolling a sum of 7 on two dice at once is {p}")


# =============================================================================
# 
# #Whats the probability of rolling 2 pips on two dice?
# #Since we are not doing an aggregate across the row 
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

#What are the experimental probabilities of rolling each possible sum?
df = pd.DataFrame()

#possible sum outcomes from 2 dice
df["outcome"] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def get_prob(n):
    return (sum_of_rolls == n).mean()

get_prob(2)


#Set the proba to its own column
#since we have a df
df.probability.hist()


#Set our OWN probabilities 
#Whats the probability of flipping "Heads" on a coin?
#Let's flip a coin 100, 000 times and figure out the prob opf flipping Heads


outcomes = ["Heads", "Tails"]

ps = np.random.choice(outcomes, size=(10_000, 1), p=[0.55, .45])
(flips == "Heads").mean()

#What are the chances of flipping two heads in a row?
ps = np.random.choice(outcomes, size=(10_000, 2), p=[0.55, .45])   #55% and 45%
flips[0:5]

#Easier to check with 1, 0/ What if this is a fair coin?

outcomes = [1,0]
flips = np.random.choice(outcomes, size=(100_000, 2), p=[0.55, .45]) 
flips[0:4]

num_of_heads = flips.sum(axis=1)

(num_of_heads == 2).mean()






#theoretical probability of flipping two unfair coin heads in a row
0.55 * 0.55