2#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 21:14:54 2022

@author: robertmegnia
"""

# Build Salary Database
import requests
import pandas as pd
from datetime import datetime
from os.path import exists
import os
import numpy as np
basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f'{basedir}/../../etc'
datadir= f'{basedir}/../../data'
## 2223 2013 season starts
for draft_group in range(20055,63968):
    print(draft_group)
    try:
        response=requests.get(f'https://api.draftkings.com/draftgroups/v1/draftgroups/{draft_group}/draftables').json()
    except:
        continue
    if 'draftables' not in response.keys():
        continue
    if len(response['draftables'])==0:
        continue
    competitions=pd.DataFrame(response['competitions'])
    sport=competitions.sport.unique()[0]
    if sport=='NHL':
        print('derp')
        competitions.startTime=pd.to_datetime(competitions.startTime).dt.tz_convert('US/Eastern')
        competitions['Slate']=competitions.startTime.apply(lambda x: x.strftime('%Y-%m-%d_%I:%M%p'))
        Slate=competitions.Slate.unique()[0]
        GameDay=Slate[0:10]
        url=f'https://www.draftkings.com/lineup/getavailableplayerscsv?contestTypeId=125&draftGroupId={draft_group}'
        try:
            salaries=pd.read_csv(url)   
        except:
            salaries=[]
        if len(salaries)==0:
            salaries=pd.DataFrame(response['draftables'])
            salaries.rename({'displayName':'Name',
                                'position':'Position',
                                'playerId':'ID',
                                'salary':'Salary',
                                'teamAbbreviation':'TeamAbbrev',
                                },axis=1,inplace=True)
            salaries=salaries.groupby('Name',as_index=False).first()
            salaries=salaries[['Name','Position','ID','Salary','TeamAbbrev']]
            salaries['Name + ID']=salaries.Name + ' ('+salaries.ID.astype(str)+')'
            salaries['Game Info']=None
            salaries['AvgPointsPerGame']=np.nan
            salaries['Roster Position']=salaries.Position
            salaries=salaries[['Position',
                               'Name + ID',
                               'Name',
                               'ID',
                               'Roster Position',
                               'Salary',
                               'Game Info',
                               'TeamAbbrev',
                               'AvgPointsPerGame']]
        # Make sure salaries are not for Showdown or Tiers
        # If CPT in Roster Positions this is a showdown slate. No thanks!
        if 'CPT' in salaries['Roster Position'].unique():
            continue
        if 'Salary' not in salaries.columns:
            continue
        #If Max Salary is less than 1000, it's probably  a Tier contest
        if salaries['Salary'].max()<2000:
            continue
        if not exists(f'{datadir}/salaryData/{GameDay}'):
            os.mkdir(f'{datadir}/salaryData/{GameDay}')
        salaries.to_csv(f'{datadir}/salaryData/{GameDay}/{Slate}_salaries.csv',index=False)