#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 08:56:43 2022

@author: robertmegnia
"""

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
from PMMPlayer import getPMM
basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f'{basedir}/../../etc'
datadir= f'{basedir}/../../data'
projdir= f'{datadir}/Projections'
from config.ColumnMappings import *
from ModelFunctions import rookiePrediction
from getDKPts import getDKPts
from RosterUtils import *
os.chdir(f'{basedir}/../')


def rolling_average(df,window=8):
    return df.rolling(min_periods=1, window=window).mean()

# Build Projections Database
datadir= f'{basedir}/../../data'
for season in range(2022,2023):  
    print(season)
    for week in range(4,10):
        print(week)
        proj_frames=[]
        db=pd.read_csv(f'{datadir}/game_logs/Full/Offense_Database.csv')
        db.game_date=pd.to_datetime(db.game_date)
        if (season!=2021)&(week==18):
            break
        game_date=db[(db.season==season)&(db.week==week)].game_date.min()
        if pd.isnull(game_date):
            game_date=getGameDate(week,season)
        stats_df=db[db.game_date<game_date]
        stats_df.sort_values(by='game_date',inplace=True)
        games_played=stats_df.groupby('gsis_id',as_index=False).size().rename({'size':'games_played'},axis=1)
        stats_df=stats_df.merge(games_played,on='gsis_id',how='left')
        db=db.merge(games_played,on='gsis_id',how='left')
        db.games_played.fillna(0,inplace=True)
        weekly_proj_df=db[(db.season==season)&(db.week==week)]
        weekly_proj_df=weekly_proj_df[(weekly_proj_df.position.isin(['QB','RB','WR','TE']))&(weekly_proj_df.injury_status=='Active')]
        # Some Instances where player appears twice on depth chart. Need to drop duplcates in this case
        weekly_proj_df=weekly_proj_df[(weekly_proj_df.games_played<3)&(weekly_proj_df.salary.isna()==False)&(weekly_proj_df.salary>0)]
        rookie_df=rookiePrediction(weekly_proj_df.drop('games_played',axis=1),stats_df) 
        rookie_df=getPMM(rookie_df,season,week)
        rookie_df.to_csv(f'{projdir}/{season}/rookies/{season}_Week{week}_RookieProjections.csv',index=False)