#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 08:40:02 2022

@author: robertmegnia
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df=pd.read_csv('/Users/robertmegnia/Downloads/draftkings-contest-entry-history.csv')
df=df[df.Entry_Fee!='Entry_Fee']
df['Entry_Fee']=df.Entry_Fee.apply(lambda x: float(x.split('$')[1]))
df['Winnings_Non_Ticket']=df.Winnings_Non_Ticket.apply(lambda x: float(x.split('$')[1]))
df['Winnings_Ticket']=df.Winnings_Ticket.apply(lambda x: float(x.split('$')[1]))
df['Net_Gain_Loss']=df.Winnings_Non_Ticket - df.Entry_Fee
df['Contest_Date_EST']=pd.to_datetime(df.Contest_Date_EST)
df=df[df.Contest_Date_EST>'2022-09-01']
df.sort_values(by='Contest_Date_EST',inplace=True)
df.reset_index(drop=True,inplace=True)
#%%
