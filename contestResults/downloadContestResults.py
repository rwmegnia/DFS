#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:13:46 2022

@author: robertmegnia
"""
import pandas as pd
import os
import time
import sys
from os.path import exists
import requests
from pandas.errors import EmptyDataError
DOWNLOAD_PATH= '/Users/robertmegnia/Downloads'
contest_results_path='http://www.draftkings.com/contest/exportfullstandingscsv/'
contestDB='/Volumes/XDrive/DFS/contestResults'
contest_url='https://api.draftkings.com/contests/v1/contests/'
contestResultsPath = '/Volumes/XDrive/DFS/contestResults'
contestHistory=pd.read_csv(f'{DOWNLOAD_PATH}/draftkings-contest-entry-history.csv')
contestHistory['game_date']=contestHistory.Contest_Date_EST.apply(lambda x: x[0:10],)
contestHistory=contestHistory.groupby('Contest_Key',as_index=False).first()
contestHistory=contestHistory[contestHistory.Entry.str.contains('rwmegnia')==False]
def downloadContestResults(contestKey,entryKey,sport,game_date):
        print(game_date)
        if not exists(f'{contestDB}/{sport}/{game_date}'):
            os.mkdir(f'{contestDB}/{sport}/{game_date}')
        files=os.listdir(f'{contestDB}/{sport}/{game_date}')
        for file in files:
            if '.zip' in file:
                print(sport,file)
            if f'contest-standings-{contestKey}' in file:
                return
        os.system(f'open -a "Google Chrome" {contest_results_path}{contestKey}')
        time.sleep(2)
        os.system(f'chmod 755 /Users/robertmegnia/Downloads/contest-standings-{contestKey}*')
        os.system(f'mv {DOWNLOAD_PATH}/contest-standings-{contestKey}* {contestDB}/{sport}/{game_date}/')

def assignPrizes(df,minPosition,maxPosition,Prize):
    df.iloc[minPosition-1:maxPosition]['Prize']=Prize
    return df
Sports=['NHL','NBA','NFL']
contestHistory=contestHistory[contestHistory.Sport.isin(Sports)]
contestHistory.apply(lambda x: downloadContestResults(x.Contest_Key,x.Entry_Key,x.Sport,x.game_date),axis=1)
#%%
for directory in ['NHL','NFL','NBA']:
    for contest_date in os.listdir(f'{contestResultsPath}/{directory}'):
        print(contest_date)
        for contest in os.listdir(f'{contestResultsPath}/{directory}/{contest_date}'):
            if 'contest-standings' in contest:
                try:
                    contestKey=contest.split('-')[2].split('.')[0]
                    if contestKey in ['131504474','132053801','133524465','132053671','138068566']:
                        continue
                except:
                    continue
            if 'details' in contest:
                continue
            if 'Millionaire_Results' in contest:
                continue
            if exists(f'{contestResultsPath}/{directory}/{contest_date}/contest-standings-{contestKey}_details.zip'):
                continue
            if exists(f'{contestResultsPath}/{directory}/{contest_date}/contest-standings-{contestKey}_details.csv'):
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
                try:
                    contestKey=contest.split('-')[2].split('.csv')[0]
                except:
                    continue

            url=f'{contest_url}/{contestKey}?format=json'
            contest_details=requests.get(url).json()['contestDetail']
            payouts=pd.DataFrame(contest_details['payoutSummary'])
            draft_group=contest_details['draftGroupId']
            try:
                payouts['Prize']=payouts.payoutDescriptions.apply(lambda x: x[0]['value'])
                competitions= len(requests.get(
                            f"https://api.draftkings.com/draftgroups/v1/draftgroups/{draft_group}/draftables"
                        ).json()['competitions'])
            except:
                continue
            sport=contest_details['sport']
            for row in payouts.iterrows():
                minPosition=row[1]['minPosition']
                maxPosition=row[1]['maxPosition']
                prize=row[1]['Prize']
                contest_df.loc[minPosition-1:maxPosition,'Prize']=prize
            contest_df['cashLine']=contest_df.loc[maxPosition].Points
            contest_df['contestName']=contest_details['name']
            if 'Millionaire' in contest_df.contestName.unique():
                print(f'Found Milli Maker! {contestKey}')
            contest_df['contestKey']=contestKey
            contest_df['Entrants'] = contest_details['entries']
            contest_df['payouts']=payouts.maxPosition.max()
            contest_df['total_teams']=competitions*2
            if contest_details['maximumEntriesPerUser']==1:
                contest_df['singleEntry']=True
            else:
                contest_df['singleEntry']=False
            contest_df.to_csv(f'{contestResultsPath}/{directory}/{contest_date}/contest-standings-{contestKey}_details.csv',index=False)
            print('yerrrp')