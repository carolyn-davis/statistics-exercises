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
#                             SIMULATION EXERCISES
# =============================================================================


# =============================================================================
# 1.)How likely is it that you roll doubles when rolling two dice?
# =============================================================================
outcomes = [1,2,3,4,5,6]
n_rows = 10_000    #total number of trials 
n_cols = 2          # number of times the dice is being rolled

np.random.choice(outcomes, 4) #random choice select random options from list, samp of 4
                                #makes array^^
                                
np.random.choice(outcomes, n_rows * n_cols)  #all random poss given list

rolls = np.random.choice(outcomes, n_rows * n_cols).reshape(n_rows, n_cols) 
#^^^^gives simultions in organized format


rolls_df = pd.DataFrame(rolls)

rolls_df['doubles'] = rolls_df[0] == rolls_df[1]
rolls_df.doubles.mean()
#Ans: 0.1692 odds of rolling doubles on two fair 6 sided dice


# =============================================================================
# 2.)If you flip 8 coins, what is the probability of getting exactly 3 heads? 
#What is the probability of getting more than 3 heads?
# =============================================================================
outcomes = [1, 0]
flips = np.random.choice(outcomes, size=(100_000, 8))
num_of_heads = flips.sum(axis=1)
(num_of_heads == 3).mean()
#Ans: .21654 probab of getting exactly 3 heads 
outcomes = [1, 0]
flips = np.random.choice(outcomes, size=(100_000, 8))
num_of_heads = flips.sum(axis=1)
(num_of_heads >= 3).mean()
#Ans: 3 or more heads is 0.85405


# =============================================================================
# =============================================================================
# # 3.)There are approximitely 3 web development cohorts for every 1 data science cohort 
# at Codeup. Assuming that Codeup randomly selects an alumni to put on a billboard, 
# what are the odds that the two billboards I drive past both have data science students 
# on them?
# # =============================================================================
# =============================================================================

outcomes = ["Web Dev", "Data Science"]
poss = np.random.choice(outcomes, size=(10_000, 2), p=[.75, .25])
# (poss == "Data Science","Data Science").mean()
double = pd.DataFrame(poss)
double = double[(double[0] == "Data Science") & (double[1] == "Data Science")]
ans = len(double) / len(poss)
#Ans: .0613 #The odds of both billboards being data science


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
# # 6.) When installing anaconda on a student's computer, there's a 1 in 250 
# chance that the download is corrupted and the installation fails. 
# =============================================================================

n_rows = 100_000   #trials
n_cols = 50  #num of students in each sim



install_range = [False for i in range(1,250)]  #Treat all corrups i range as True
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
# 7.a)How likely is it that a food truck will show up sometime this week?
# =============================================================================
n_rows = 10_000   #trials

n_cols = 2 #days of simulation

ft_sighting = np.random.choice([True, False], n_rows * n_cols, p=[.7,.3]).reshape(n_rows, n_cols)

ft_sighting = pd.DataFrame(ft_sighting)



ft_sighting['not_seen_2_days'] = ft_sighting.sum(axis=1) == 0


ft_sighting['not_seen_2_days'].mean()
truck_probability = int(round(ft_sighting['not_seen_2_days'].mean(),4) * 100)

#Ans: 8% liklihood of not seeing a foodtrycj in Travis remaining 2 days, 
# 92% that you will see one



# =============================================================================
# 8.)If 23 people are in the same room, what are the odds that two of them share 
# a birthday?
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
# 20 People:
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
# 40 People:
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
                            #stats(mean_sp2, std_samp2, sample, "sample 1 data")
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
#Null Hyp: There is no difference in fuel efficiency from 99' to 08
#Alt Hyp: There is a difference in fuel eficiency from 99' to 08'

#Bc we are comparing fuel economy in two different subgroups, this is a 2 samp, 2 tailed test

#Should the arithmetic mean or harmonic mean be used?
#arithmetic mean: app if the value have the same units   fe_am = (cty + hwy)2
#geometric mean: app if the values have differing units
#Harmonic mean: app if data values are ratios of two variable with differing measures
                #^^^called rates   fe_hm = 2/(1/cty + 1/hwy)   OR 
mpg_data['avg_fuel_econ'] = stats.hmean(mpg_data[['cty', 'hwy']], axis =1)   #2 cols 2 bracks
#^^ adds new column to mpg_df for average fuel economy 

fuelecon_08 = mpg_data[mpg_data.year == 2008].avg_fuel_econ
fuelecon_99 = mpg_data[mpg_data.year == 1999].avg_fuel_econ

t, p = stats.ttest_ind(fuelecon_08, fuelecon_99)
#t, p = (-0.3011962975077886, 0.7635345888327115)
#Bc the pvalue is greater than the alpha (significance) (0.5),failed to reject the null hypothesis 
# that there is no difference in fuel efficiency in cars from 99'to 08'

fuelecon_08.mean(), fuelecon_99.mean()
#(19.488662986569054, 19.68238764869729)

plt.hist([fuelecon_99, fuelecon_08], label=["1999 cars", "2008 cars"])
plt.legend(loc="upper right")


# b.) Are compact cars more fuel-efficient than the average car?

#Bc we are comparing the fuel econ oftwo diff sub groups 1 samp, 1 tailed t test

#Null: There is no difference in fuel-efficiency btn compact cars and population avg efficiency
#Alt: Compact cars are more fuel efficient than population's avg car

#For the alternative hypothesis to be true the t stat must be greater than 0
#the p/2 value must be less than 5%

fuel_compact = mpg_data[mpg_data['class'] == 'compact'].avg_fuel_econ
pop_mean = mpg_data.avg_fuel_econ.mean()

t, p = stats.ttest_1samp(fuel_compact, pop_mean) 
#t,p = (test 7.512360093161354, p_value = 1.5617666348807727e-09 )

p_2 = p / 2

#bc the p/2 value is less than the alpha, null hypothesis is rejected, that there is 
#no diff in fuel economy btn normal and compact cars


# c.)
# Do manual cars get better gas mileage than automatic cars?
#Null: There is no difference in gas mileage between manual and automatic cars
#Alt: There is a difference in gas mileage between manuak and automatic cars

fuel_auto = mpg_data[mpg_data.trans.str.contains('auto')].avg_fuel_econ
#use contains because this column contans addition elements to str 'auto'
#same for manual

fuel_manual = mpg_data[mpg_data.trans.str.contains('manual')].avg_fuel_econ



t, p = stats.ttest_ind(fuel_manual, fuel_auto)  #samp 2 data first
#t, p = (t=4.652577547151351, p_value=5.510464610044005e-06 )

#p/2 = 2.7552323050220026e-06 which is less than the alpha
#b/c the p/2 is less than the alpha, the null hypothesis is rejected that there
#is no difference in full economy of automatic and manual cars

mpg_data['transmission_type'] = np.where(mpg_data.trans.str.contains('auto'), 'auto_transmission',
                                         'manual_transmission')
#^adds column of renamed values corresponding to tranmission type
mpg_data.groupby('transmission_type').avg_fuel_econ.mean().plot.bar()
#plots avg fuel econ by categ of transmision type and avg fuel eco
plt.xticks(rotation=0)
plt.xlabel('')
plt.ylabel('average mileage')
plt.title('Mileage Difference by Transmission Type')
# =============================================================================
# 
# =============================================================================
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