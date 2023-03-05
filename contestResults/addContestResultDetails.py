#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 10:30:06 2022

@author: robertmegnia
"""
import requests
import pandas as pd
from pandas.errors import *
import os
import time
import sys
from os.path import exists
contest_url='https://api.draftkings.com/contests/v1/contests/'
contestResultsPath = '/Volumes/XDrive/DFS/contestResults'
#%%
def assignPrizes(df,minPosition,maxPosition,Prize):
    df.iloc[minPosition-1:maxPosition]['Prize']=Prize
    return df

for directory in ['NFL','MLB']:
    for contest_date in os.listdir(f'{contestResultsPath}/{directory}'):
        print(contest_date)
        for contest in os.listdir(f'{contestResultsPath}/{directory}/{contest_date}'):
            if 'Millionaire_Results' in contest:
                continue
            if 'contest-standings' in contest:
                contestKey=contest.split('-')[2][0:9]
            if 'details' in contest:
                continue
            elif exists(f'{contestResultsPath}/{directory}/{contest_date}/contest-standings-{contestKey}_details.zip'):
                continue
            print(contest)
            try:
                contest_df=pd.read_csv(f'{contestResultsPath}/{directory}/{contest_date}/{contest}') 
            except EmptyDataError:
                continue
            contest_df['Prize']=0
            if 'zip' in contest:
                contestKey=contest.split('-')[2].split('.zip')[0]
            else:
                contestKey=contest.split('-')[2].split('.csv')[0]

            url=f'{contest_url}/{contestKey}?format=json'
            contest_details=requests.get(url).json()['contestDetail']
            payouts=pd.DataFrame(contest_details['payoutSummary'])
            try:
                payouts['Prize']=payouts.payoutDescriptions.apply(lambda x: x[0]['value'])
            except:
                payouts['Prize']=payouts.tierPayoutDescriptions.apply(lambda x: x[0]['value'])

            sport=contest_details['sport']
            for row in payouts.iterrows():
                minPosition=row[1]['minPosition']
                maxPosition=row[1]['maxPosition']
                prize=row[1]['Prize']
                contest_df.loc[minPosition-1:maxPosition,'Prize']=prize
            contest_df['cashLine']=contest_df.loc[maxPosition].Points
            contest_df['contestName']=contest_details['name']
            contest_df['contestKey']=contestKey
            if contest_details['maximumEntriesPerUser']==1:
                contest_df['singleEntry']=True
            else:
                contest_df['singleEntry']=False
            contest_df.to_csv(f'{contestResultsPath}/{directory}/{contest_date}/contest-standings-{contestKey}_details.csv',index=False)
            
            
            
            