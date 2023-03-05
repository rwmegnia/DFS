#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 08:59:54 2021
@author: robertmegnia
"""
import numpy as np
import pandas as pd
import nflfastpy as nfl
from scipy.stats import norm
import requests
import os
from ModelFunctions import PlayerSharesPrediction
basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f'{basedir}/../../etc'
datadir= f'{basedir}/../../data'
projdir= f'{basedir}/../../LiveProjections'
from config.ColumnMappings import *
from getDKPts import getDKPts
from RosterUtils import *
os.chdir(f'{basedir}/../')
from Models.ModelDict import *


def rolling_average(df,window=8):
    return df.rolling(min_periods=1, window=window).mean()

#%%
for season in range(2022,2023):              
    for week in range(1,10):
        if (season==2020)&(week==18):
            continue
        print(week)
        proj_frames=[]
        datadir= f'{basedir}/../../data'
        df=pd.read_csv(f'{datadir}/game_logs/Full/Offense_Database.csv')
        df.sort_values(by='DKPts',ascending=False)
        df["depth_team"] = df.groupby(
            ["team", "position", "week", "season"]
        ).salary.rank(ascending=False, method="first")
        df=df[df.position!='FB']
        df.game_date=pd.to_datetime(df.game_date)
        if (season!=2021)&(week==18):
            break
        game_date=df[(df.season==season)&(df.week==week)].game_date.min()
        if pd.isnull(game_date):
            game_date=getGameDate(week,season)
        stats_df=df[df.game_date<game_date]
        stats_df.sort_values(by='game_date',inplace=True)
        weekly_proj_df=df[(df.season==season)&(df.week==week)]
        proj_df=PlayerSharesPrediction(stats_df,weekly_proj_df)
        # proj_df.fillna(0,inplace=True)
        proj_df.to_csv(f'{datadir}/Projections/PlayerShareProjections/{season}_Week{week}_PlayerShareProjections2.csv',index=False)
        
        
        
        
        
        
