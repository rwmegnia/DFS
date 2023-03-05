#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 00:25:20 2022

@author: robertmegnia
"""
import pandas as pd
# import nflfastpy as nfl
import requests
import os
basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f'{basedir}/../../../etc'
datadir= f'{basedir}/../../../data'
os.chdir(f'{basedir}/../')


def getSchedule(week,season):   
    # Download Schedule
    try:
        sdf=pd.read_csv(f'{datadir}/scheduleData/{season}_ScheduleData.csv')
    except FileNotFoundError:
        sdf=nfl.load_schedule_data(season)
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
    return sdf

def filterRosterData(df,season,week):
    df=df[df.position.isin(['QB','RB','WR','TE','DST'])]
    # Get Rookie gsis_ids which are not up to date in API
    rookies=df[df.years_exp<=1]
    df.drop(df[df.years_exp<=1].index,inplace=True)
    rookies=rookies[~rookies.team.isin([None])].drop('gsis_id',axis=1)
    roster=nfl.load_roster_data(season)
    roster=roster[roster.position.isin(['QB','RB','WR','TE'])]
    roster=roster[roster.years_exp<=1]
    rookies=rookies.merge(roster[['full_name','gsis_id']],on='full_name',how='left')
    rookies=rookies[rookies.gsis_id.isna()==False]
    df=pd.concat([df,rookies])
    #
    df=df[~df.gsis_id.isin([None])]
    df.position.replace('DEF','DST',inplace=True)
    df.loc[df.position=='DST','full_name']=df.loc[df.position=='DST'].index
    df.loc[df.position=='DST','gsis_id']=df.loc[df.position=='DST'].index
    df.loc[df.position=='DST','gsis_id']=df.loc[df.position=='DST'].index
    df.loc[df.position=='DST','depth_chart_order']=1
    df.gsis_id=df.gsis_id.apply(lambda x: x.split(' ')[-1])
    # Export Roster to weekly roster database
    try:
        df.to_csv(f'{datadir}/rosterData/weekly_rosters/{season}/week{week}/Week{week}RosterData.csv',index=False)
    except FileNotFoundError:
        print('Roster directory does not exist. Creating directory...')
        os.mkdir(f'{datadir}/rosterData/weekly_rosters/{season}/Week{week}/')
        df.to_csv(f'{datadir}/rosterData/weekly_rosters/{season}/week{week}/Week{week}RosterData.csv',index=False)
    df=df[df.status=='Active']
    df=df[~df.injury_status.isin(['IR','PUP','OUT','Doubtful','Sus','COV'])]
    df=df[df.depth_chart_order<=3]
    df.drop(df[(df.position=='QB')&(df.depth_chart_order>1)].index,inplace=True)
    return df

def getWeeklyRosters(season,week):
    # Download Rosters from sleeper API
    roster_json = requests.get('https://api.sleeper.app/v1/players/nfl').json()
    df = pd.DataFrame.from_dict(roster_json).T    
    # Fix some minor formatting issues in data frame
    df.team.replace('OAK','LV',inplace=True)
    df.team.replace('LAR','LA',inplace=True)
    df.position.replace('FB','RB',inplace=True)
    # Remove spaces from gsis strings
    df=filterRosterData(df,season)
    
    df=df[(df.position.isin(['QB','RB','WR','TE']))&(df.team.isna()==False)&(df.gsis_id.isna()==False)]
    df['season']=season
    df=df[['season','team','position','full_name','gsis_id','depth_chart_order']]
    df.full_name=df.full_name.apply(lambda x: x.split(' ')[0]+' '+x.split(' ')[1])
    df.replace('Amon-Ra St.','Amon-Ra St. Brown',inplace=True)
    df.replace('Equanimeous St.','Equanimeous St. Brown',inplace=True)
    df.replace('Jason Vander','Jason Vander Laan',inplace=True)
    df['week']=week
    sdf=getSchedule(week,season)
    df=df.merge(sdf[['week','opp','team','Slate']],on=['week','team'],how='left')
    odds=ScrapeBettingOdds(week)
    df=df.merge(odds[['team','proj_team_score','total_line','spread_line']],on='team',how='left')
    
    # Get Opponent Ranks vs position
    offense_opp_Ranks=pd.read_csv(f'{datadir}/game_logs/Full/Offense_Latest_OppRanks.csv')
    dst_opp_Ranks=pd.read_csv(f'{datadir}/game_logs/Full/DST_Latest_OppRanks.csv')
    opp_Ranks=pd.concat([offense_opp_Ranks,dst_opp_Ranks])
    df=df.merge(opp_Ranks,on=['opp','position'],how='left')
    return df

def getGameDate(week,season):
    schedule=nfl.load_schedule_data(season)
    schedule.gameday=pd.to_datetime(schedule.gameday)
    game_date=schedule[schedule.week==week].gameday.min()
    return game_date