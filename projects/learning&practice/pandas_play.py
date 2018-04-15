# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pandas import DataFrame,Series
from matplotlib import pyplot as plt

series = pd.Series(["liuming", "gray", "Udacity", 42, -1789710578])
print(series)

series_1 = pd.Series(["liuming", "gray", "Udacity", 42, -1789710578],index=["a","b","c","d","e"])
print(series_1)

print(series_1['a'])
print("\n")
print(series_1[['a','b','e']])#only print selected rows

series_2 = pd.Series([1, 2, 3, 4, 5], index=['Cockroach', 'Fish', 'Mini Pig',
                                                 'Puppy', 'Kitten'])
print(series_2 > 3)
print("\n")
print(series_2[series_2 > 3])

series_3 = pd.Series(['a','b','c','d'], index=[1,2,3,4])
print(series_3.index > 3)
print(series_3[series_3.index > 2])


"""
create the dataframe
"""
data = {'year': [2010, 2011, 2012, 2011, 2012, 2010, 2011, 2012],
            'team': ['Bears', 'Bears', 'Bears', 'Packers', 'Packers', 'Lions',
                     'Lions', 'Lions'],
            'wins': [11, 8, 10, 15, 11, 6, 10, 4],
            'losses': [5, 8, 6, 1, 5, 10, 6, 12]}
football = pd.DataFrame(data)
print (football)

print(football['year'])
print(football[['year','wins','losses']])

#average wins in each team
football.groupby('team')['wins'].mean()

'''
Row selection can be done through multiple ways.

Some of the basic and common methods are:
   1) Slicing
   2) An individual index (through the functions iloc or loc)
   3) Boolean indexing

You can also combine multiple selection requirements through boolean
operators like & (and) or | (or)
'''

print(football.iloc[0])
print(football.loc[[0]])
print(football[3:5])
print(football[football.wins>10])
print(football[(football.wins > 10) & (football.team=='Packers')]) #no and

countries = ['Russian Fed.', 'Norway', 'Canada', 'United States',
                 'Netherlands', 'Germany', 'Switzerland', 'Belarus',
                 'Austria', 'France', 'Poland', 'China', 'Korea', 
                 'Sweden', 'Czech Republic', 'Slovenia', 'Japan',
                 'Finland', 'Great Britain', 'Ukraine', 'Slovakia',
                 'Italy', 'Latvia', 'Australia', 'Croatia', 'Kazakhstan']

gold = [13, 11, 10, 9, 8, 8, 6, 5, 4, 4, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
silver = [11, 5, 10, 7, 7, 6, 3, 0, 8, 4, 1, 4, 3, 7, 4, 2, 4, 3, 1, 0, 0, 2, 2, 2, 1, 0]
bronze = [9, 10, 5, 12, 9, 5, 2, 1, 5, 7, 1, 2, 2, 6, 2, 4, 3, 1, 2, 1, 0, 6, 2, 1, 0, 1]
    
olympic_medal_counts = {'country_name':countries,
                            'gold': Series(gold),
                            'silver': Series(silver),
                            'bronze': Series(bronze)}    
df = DataFrame(olympic_medal_counts)

print(df)
df.loc[[0]]['gold']

#these two approaches give same result
metals = df[['gold','silver','bronze']]
np.dot(metals,[4,2,1])

metals = np.array([gold,silver,bronze])
scores = np.array([4,2,1])
np.dot(scores,metals)

#visualize distribution of values using plot
df['gold'].plot(kind='line')
df['bronze'].plot(kind='box')




















