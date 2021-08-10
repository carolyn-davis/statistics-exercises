#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 09:11:48 2021

@author: carolyndavis
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


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

# =============================================================================
# # What is the probability that no cars drive up in the noon hour?
# =============================================================================
def poisson_pmf(k, lam):
    return (lam ** k * np.exp(-lam)) / np.math.factorial(k)

poisson_pmf(0, 2)
#Ans: 0.135 chance that no cars will drive up at noon  #can also do poisson.pmf()

# =============================================================================
# # What is the probability that 3 or more cars come through the drive through?
# =============================================================================
poisson(2).sf(2)     #sf tells us the probability of a variable falling above a certain value
#Ans: 0.32 chance 3 or more cars will pull through        #3 cars-1 
# =============================================================================
# # How likely is it that the drive through gets at least 1 car?
# =============================================================================
poisson(2).sf(0)
#Ans: 0.86466.. chance that atleast one car will come through

# =============================================================================
# 2.)Grades of State University graduates are normally distributed with a mean of 
# 3.0 and a standard deviation of .3. Calculate the following:
# =============================================================================
mean_avg = 3.0
std_dev = 0.3


# =============================================================================
# #a.) What grade point average is required to be in the top 5% of the graduating class?
# =============================================================================

x = stats.norm(mean_avg, std_dev)
top_five = x.ppf(.95)  #ppf gives the value that is assoc w/ probability
x.plot()
# Ans: 3.49 gpa to be in top 5%

data = np.random.normal(loc=mean_avg, scale=std_dev, size=1000)
data = pd.DataFrame(data)
data.plot.scatter(x=data.index,y=data[0])
# =============================================================================
# #b.) What GPA constitutes the bottom 15% of the class?
# #ppf accapets a dprobability, gives value associated with probability
# =============================================================================

bottom_15_percent = stats.norm.ppf(.15, loc=mean_avg, scale=std_dev)
#Ans:2.68906.. constitutes a GPA in the bottom 15%

# c.)An eccentric alumnus left scholarship money for students in the third decile from
#  the bottom of their class. Determine the range of the third decile. 
range_start = stats.norm(mean_avg, std_dev).ppf(.20) #second decile .20

range_end = x = stats.norm(mean_avg, std_dev).ppf(.30) #third decile .30

print(f"Range for Third Decile is : {round(range_start,2)} --> {round(range_end,2)}")
#Ans: 2.75 to 2.84


# =============================================================================
# Would a student  with a 2.8 grade point average qualify for this scholarship?
# =============================================================================
#Ans: Yes


# =============================================================================
# # If I have a GPA of 3.5, what percentile am I in?
# =============================================================================

stats.norm(mean_avg, std_dev).cdf(3.5) #CDF tells liklihood of single outcome 
#Ans: 0.9522.. With this GPA student is in 95th percentile


# =============================================================================
# 3.)A marketing website has an average click-through rate of 2%. One day they observ
#     4326 visitors and 97 click-throughs. How likely is it that this many people or more 
#     click through?
# =============================================================================
avg_clickrate = .02   #probability of success
n_visitors = 4326   #number of trials
num_clickthru = 97 #target successes

stats.binom(n_visitors, avg_clickrate).sf(num_clickthru-1)
#binome distrib used to model # of successes after # of trials
#def by # of trials and prob of success


#Ans: 0.13975 prob that all of these people will actually click thru



# =============================================================================
# 4.)You are working on some statistics homework consisting of 100 questions where all
#  of the answers are a probability rounded to the hundreths place. Looking to save time, 
#  you put down random probabilities as the answer to each question.
# =============================================================================


# What is the probability that at least one of your first 60 answers is correct?


prob_correct = .01
first_sixty_answers = 60
goal = 1

stats.binom(first_sixty_answers, prob_correct).sf(goal-1)
#Ans: .4528 prob you will get one question right in the first 60 questions



# =============================================================================
# 5.)The codeup staff tends to get upset when the student break area is not cleaned up.
#  Suppose that there's a 3% chance that any one student cleans the break area when 
#  they visit it, and, on any given day, about 90% of the 3 active cohorts of 22 
#  students visit the break area. How likely is it that the break area gets cleaned 
#  up each day? How likely is it that it goes two days without getting cleaned up? All week?
# =============================================================================
chance_success = .03
num_visitors = (22 * 3) * .9     #total num of trials 
clean_goal = 1 #desired success that one student cleans the break area


stats.binom(num_visitors, chance_success).sf(clean_goal-1)

#Ans: .83421.. % chance break room is cleaned each day




#prob break room is not cleaned one day of the week
1-stats.binom(num_visitors, chance_success).sf(clean_goal-1)

#prob break room is not cleaned two days of the week

(1-stats.binom(num_visitors, chance_success).sf(clean_goal-1))**2 # odds of not being cleaned for two days in a row
#Ans: .83421...

#prob break room is not cleaned all week
(1-stats.binom(num_visitors, chance_success).sf(clean_goal-1))**5 
#Ans: 0.000125...

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
#normal  dist

mean_inline_cust = 15 #avg num cust in line
std_cust_inline = 3  #std cus

pp_order_time = 2 #per person order time

time_tgf = 10 #time to get food
time_tef = 15 #time to eat food 

tot_time = 60 #60 min 1 hour

max_line_time = tot_time - (time_tgf + time_tef)
#35 total line time 

max_cust_inline = max_line_time / pp_order_time
#17.5
# odds of max_cust_inline or less customers
stats.norm(mean_inline_cust, std_cust_inline).cdf(max_cust_inline)
#norm dist for single likely outcome(cdf)
# =============================================================================
# 7.)Connect to the employees database and find the average salary of current employees,
#     along with the standard deviation. For the following questions, calculate the answer
#     based on modeling the employees salaries with a normal distribution defined by the
#     calculated mean and standard deviation then compare this answer to the actual values 
#     present in the salaries dataset.
# =============================================================================
from env import host, user, password


def get_db_url(user, host, password, database, query):
    
    url = f'mysql+pymysql://{user}:{password}@{host}/employees'
    
    return pd.read_sql(query, url)


database = 'employees'

query = """

SELECT salary
FROM salaries

where salaries.to_date = "9999-01-01"

"""


emp_df = get_db_url(user, host, password, database, query)
emp_df.head()

avg_salary = round(emp_df['salary'].mean(),0)   #avg_salary = 72012.0
std_salary = round(emp_df['salary'].std(),0)     #std_salary = 17310.0

emp_df.salary.hist(bins=20)
plt.title("The Distribution of Employee Salaries")
plt.show()


# =============================================================================
# # a.)What percent of employees earn less than 60,000?
# =============================================================================
stats.norm(avg_salary, std_salary).cdf(60_000) #cdf below certain mark
#Ans: 0.24386..

# =============================================================================
# # b.)What percent of employees earn more than 95,000?
# =============================================================================
stats.norm(avg_salary, std_salary).sf(95_000)   #sf prob rand var above a certain value

len(emp_df[emp_df.salary > 95_000]) / len(emp_df)  #divide emp with salary > 95000 by total num of emps

#Ans: 0.1086272

# =============================================================================
# c.)What percent of employees earn between 65,000 and 80,000?
# =============================================================================
# subtract the percent of employees making 65,000 from the percent of employees 
#making 80,000 or less to get the percent of people making between 65,000 and 80,000

stats.norm(avg_salary, std_salary).cdf(80_000) - stats.norm(avg_salary, std_salary).cdf(65_000)




# =============================================================================
# d.)What do the top 5% of employees make?
# =============================================================================
stats.norm(avg_salary, std_salary).isf(.05)  #isf - inverse survival function
#Ans:100484.41628253                         #like ppf gives value when given probability
