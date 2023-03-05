#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 01:19:50 2022

@author: robertmegnia
"""
import os
import warnings
import pandas as pd
import nfl_data_py as nfl
import numpy as np
from ModelFunctions import TeamDepthChartPrediction
from config.ColumnMappings import *
basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f'{basedir}/../../data/'
warnings.simplefilter(action='ignore', category=Warning)
os.chdir(f'{basedir}/../')
os.chdir(f'{basedir}/../')

from Models.ModelDict import TeamModelDict

def rolling_average(df,window=4):
    return df.rolling(min_periods=1, window=window).mean()

def getGameDate(week,season):
    schedule=nfl.import_schedules([season])
    schedule.gameday=pd.to_datetime(schedule.gameday)
    game_date=schedule[schedule.week==week].gameday.min()
    return game_date

def getSchedule(week,season):   
    # Download Schedule
    db=pd.read_csv(f'{datadir}/game_logs/{season}/{season}_Offense_GameLogs.csv')
    db['spread_line']=(db.proj_team_score+(db.proj_team_score-db.total_line))*-1
    db=db.groupby(['team','season','week'],as_index=False).mean()[['team','week','season','spread_line','total_line','proj_team_score']]
    try:
        sdf=pd.read_csv(f'{datadir}/scheduleData/{season}_ScheduleData.csv')
    except FileNotFoundError:
        sdf=nfl.import_schedules([season])
        pass
    sdf['game_date']=sdf.gameday+' '+sdf.gametime
    sdf.game_date=pd.to_datetime(sdf.game_date.values).strftime('%A %y-%m-%d %I:%M %p')
    sdf['game_day']=sdf.game_date.apply(lambda x: x.split(' ')[0])
    sdf.gametime=pd.to_datetime(sdf.gametime)
    sdf.loc[(sdf.game_day=='Sunday')&(sdf.gametime>='13:00:00')&(sdf.gametime<'17:00:00'),'Slate']='Main'
    sdf.loc[sdf.Slate!='Main','Slate']='Full'
    sdf=sdf[sdf.week==week]
    teams=pd.concat([sdf.home_team,sdf.away_team])
    opps=pd.concat([sdf.away_team,sdf.home_team])
    sdf=pd.concat([sdf,sdf])
    sdf['team']=teams
    sdf['opp']=opps

    sdf=sdf.merge(db[['team','proj_team_score','total_line','spread_line','week','season']],on=['team','season','week'],how='left')
    sdf.rename({'proj_team_score':'ImpliedPoints'},axis=1,inplace=True)
    return sdf


frames=[]
for season in range(2022,2023):
    print(season)
    if season>2020:
        end_week=18
    else:
        end_week=17
    for week in range(4,5):
        print(week)
        #schedule=getSchedule(week,season)
        game_date=getGameDate(week,season)
        off_db=pd.read_csv(f'{datadir}/TeamDatabase/TeamOffenseStats_DB.csv')
        schedule=off_db[(off_db.season==season)&(off_db.week==week)]
        off_db.game_date=pd.to_datetime(off_db.game_date)
        off_db=off_db[off_db.game_date<game_date]
        proj_df=TeamDepthChartPrediction(off_db,schedule)
        proj_df.reset_index(inplace=True)
        proj_df.to_csv(f'{datadir}/Projections/TeamDepthChartProjections/{season}_Week{week}_TeamDepthChartProjections.csv',index=False)
        frames.append(proj_df)
    
db=pd.concat(frames)
db.to_csv(f'{datadir}/Projections/TeamDepthChartProjections/2020_2022_TeamDepthChartProjections.csv',index=False)
