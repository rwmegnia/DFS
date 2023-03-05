#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 16:26:50 2022

@author: robertmegnia
"""
import pandas as pd
import os
from datetime import datetime
import warnings
import sys
sys.path.append('./Tools')
from team_db_config import *
from Tools.utils import *
from Tools.FantasyPointsFunctions import *
from Tools.PlayerStatFunctions import *
from Tools.team_mapper import *
import nfl_data_py as nfl
basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f'{basedir}/../../data/'
warnings.simplefilter(action='ignore', category=Warning)
os.chdir(f'{basedir}/..')
from PlayerDatabase.DatabaseUtils.LoadData import *
# pd.set_option('display.max_columns', 5)
# pd.set_option('display.max_rows',60)
#%%
def __init__(season,update=False):
        start_time=datetime.utcnow()
        timer_start=datetime.utcnow()
        print('Started script at {} '.format(start_time))
        stats=team_stats(season)
        print(f'{season} Season Database Update Complete!')
        timer_end=datetime.utcnow()
        script_time=(timer_end-timer_start).seconds
        minutes=script_time//60
        seconds=script_time-minutes*60
        print(f'Database update took {minutes} minutes {seconds} seconds')
        return stats
    
def team_stats(season):
    '''
    

    Parameters
    ----------
    team : str
        Team Abbreviation String (i.e. ARI, SF, NE)
    season : int


    Returns
    -------
    offense : DataFrame
    defense : DataFrame
    
        game log of offensive/defensive team rushing/passing statistics for the past 2 seasons

    '''
    # Load pbp data frames from previous two season and combine
    df1=load_pbp_data(season)
    # Use a for loop to get frames for offense and defense
    for formation in ['posteam','defteam']:
        full_df=df1
        # Determine big run/pass plays
        full_df.loc[full_df.passing_yards>=25,'big_pass_play']=1
        full_df.loc[full_df.rushing_yards>=10,'big_run']=1
        
        # Determine Early Down Stats
        full_df.loc[(full_df.pass_attempt==1)&(full_df.down<=2),'early_down_pass']=1
        full_df.loc[(full_df.pass_attempt==0)&(full_df.down<=2),'early_down_pass']=0

        full_df.loc[full_df.down<=2,'early_down']=1
        full_df.loc[(full_df.early_down==1)&((full_df.rushing_yards>=full_df.ydstogo/2)|(full_df.passing_yards>=full_df.ydstogo/2)),'early_down_success']=1
        
        # Determine 3rd/4th Down Stats
        full_df.loc[full_df.down==3,'3rdDown']=1
        full_df.loc[(full_df['3rdDown']==1)&((full_df.passing_yards>=full_df.ydstogo)|(full_df.rushing_yards>=full_df.ydstogo)),'3rdDown_success']=1
        
        full_df.loc[full_df.down==4,'4thDown']=1
        full_df.loc[(full_df.down==4)&((full_df.pass_attempt==1)|(full_df.rush_attempt==1)),'4thDown_attempt']=1
        full_df.loc[(full_df['4thDown_attempt']==1)&((full_df.passing_yards>=full_df.ydstogo)|(full_df.rushing_yards>=full_df.ydstogo)),'4thDown_success']=1
        
        full_df.loc[(full_df.down==4)&(full_df.game_half=='Half1'),'EarlyGame_4thDown']=1
        full_df.loc[(full_df.down==4)&(full_df.game_half=='Half1')&((full_df.pass_attempt==1)|(full_df.rush_attempt==1)),'EarlyGame_4thDown_attempt']=1
        full_df.loc[(full_df['EarlyGame_4thDown_attempt']==1)&((full_df.passing_yards>=full_df.ydstogo)|(full_df.rushing_yards>=full_df.ydstogo)),'EarlyGame_4thDown_success']=1
        # Group data by game/week
        df=full_df.groupby([formation,
                       'home_team',
                       'away_team',
                       'home_score',
                       'away_score',
                       'spread_line',
                       'total_line',
                       'week',
                       'season',
                       'game_date',
                       'game_id'],as_index=False).agg(TEAM_STATS)
        
        # Determine Success Rates
        df['early_down_pass_rate']=df.early_down_pass/df.early_down
        df['early_down_success_rate']=df.early_down_success/df.early_down
        df['3rdDown_success_rate']=df['3rdDown_success']/df['3rdDown']
        df['4thDown_attempt_rate']=df['4thDown_attempt']/df['4thDown']
        df['4thDown_success_rate']=df['4thDown_success']/df['4thDown_attempt']
        df['EarlyGame_4thDown_attempt_rate']=df.EarlyGame_4thDown_attempt/df.EarlyGame_4thDown
        df['EarlyGame_4thDown_success_rate']=df.EarlyGame_4thDown_success/df.EarlyGame_4thDown_attempt
        
        # Determine Points scored by the team, ImpliedPoints, TotalPoints, and Points Allowed, spreadline
        df[['Points','ImpliedPoints','spread_line']]=getTeamPointsStats(df,formation)
        df['TotalPoints']=df[['home_score','away_score']].sum(axis=1)
        df['OppPoints']=df.TotalPoints-df.Points
        # Compute Average Depth of Target (ADOT), passer rating, yards per rush attempt, and TotalPoints
        df['ADOT']=df.air_yards/df.pass_attempt
        df['passer_rating']=df.apply(lambda x: getPasserRating(x.complete_pass,x.pass_touchdown,x.interception,x.passing_yards,x.pass_attempt),axis=1)
        df['ypa']=df.rushing_yards/df.rush_attempt
        
        # Change posteam/defteam column to team
        df.rename({formation:'team'},axis=1,inplace=True)
        # Determine if the team was home or away
        df.loc[df.home_team==df.team,'Location']='Home'
        df.loc[df.away_team==df.team,'Location']='Away'
        # Make sure dataframe is sorted chronologically
        df.loc[df.team!=df.home_team,'opp']=df.loc[df.team!=df.home_team,'home_team']
        df.loc[df.team!=df.away_team,'opp']=df.loc[df.team!=df.away_team,'away_team']


        df.sort_values(by='game_date',inplace=True)
        # If we're constructing a defensive stats data frame, add the suffix _allowed" to appropriate columns
        if formation=='defteam':
            # columns to add suffix to
            columns=list(TEAM_STATS.keys())
            columns+=['ADOT','passer_rating','ypa','ImpliedPoints']
            
            # create new frame with a suffix for the selected columns
            df=pd.concat([df[['team',
                              'opp',
                              'home_team',
                              'Location',
                              'away_team',
                              'home_score',
                              'away_score',
                              'proj_home_score',
                              'proj_away_score',
                              'week',
                              'season',
                              'game_date',
                              'game_id',
                              'Points',
                              'ImpliedPoints',
                              'spread_line',
                              'total_line']],df[columns+['early_down_success_rate',
                                                         'early_down_pass_rate',
                                                         '3rdDown_success_rate',
                                                         '4thDown_success_rate',
                                                         '4thDown_attempt_rate',
                                                         'EarlyGame_4thDown_attempt_rate',
                                                         'EarlyGame_4thDown_success_rate']].add_suffix('_allowed')],axis=1)
            
            # rename some columns
            df.rename({
                        'fumble_allowed':'fumble_forced',
                        'fumble_lost_allowed':'fumble_recovery',
                        'interception_allowed':'interception',
                        'qb_hit_allowed':'qb_hits',
                        'sack_allowed':'sacks',
                        'tackled_for_loss_allowed':'tfl',},axis=1,inplace=True)
            
            df=getQBData(df,defense=True)
            for dc in range(1,4):
                print(dc)
                df=getRBData(df,dc,defense=True)
            for dc in range(1,6):
                df=getWRData(df,dc,defense=True)
            for dc in range(1,4):
                df=getTEData(df,dc,defense=True)
            df.reset_index(drop=True,inplace=True)
            defense=df[df.team.isna()==False]
        else:
            df=getQBData(df)
            for dc in range(1,4):
                df=getRBData(df,dc)
            for dc in range(1,6):
                df=getWRData(df,dc)
            for dc in range(1,4):
                df=getTEData(df,dc)
            df.reset_index(drop=True,inplace=True)
            offense=df
        #reset index and return
    offense=get_team_fantasy_points(offense)
    offense.drop(['home_team','away_team','home_score','away_score','proj_home_score',
                  'proj_away_score','early_down','early_down_success','early_down_pass',
                  '4thDown','4thDown_attempt','4thDown_success','EarlyGame_4thDown',
                  'EarlyGame_4thDown_attempt','EarlyGame_4thDown_success','3rdDown',
                  '3rdDown_success'],axis=1,inplace=True)
    defense=get_team_fantasy_points(defense,defense=True)
    defense.drop(['home_team','away_team','home_score','away_score','proj_home_score',
                  'proj_away_score','early_down_allowed','early_down_success_allowed',
                  'early_down_pass_allowed',
                  '4thDown_allowed','4thDown_attempt_allowed','4thDown_success_allowed',
                  'EarlyGame_4thDown_allowed',
                  'EarlyGame_4thDown_attempt_allowed','EarlyGame_4thDown_success_allowed',
                  '3rdDown_allowed',
                  '3rdDown_success_allowed'],axis=1,inplace=True)
    offense=offense[offense.team.isna()==False]
    defense=defense[defense.team.isna()==False]
    snaps=nfl.import_snap_counts([season])
    offense_snaps=snaps.groupby(['season','week','team'],as_index=False).offense_snaps.max()
    defense_snaps=snaps.groupby(['season','week','team'],as_index=False).defense_snaps.max()
    offense=offense.merge(offense_snaps,on=['season','week','team'],how='left')
    defense=defense.merge(defense_snaps,on=['season','week','team'],how='left')
    offense.to_csv(f'{datadir}/TeamDatabase/{season}_TeamOffenseStats.csv',index=False)
    defense.to_csv(f'{datadir}/TeamDatabase/{season}_TeamDefenseStats.csv',index=False)
    # NonFeatures= defense.columns.difference(offense.columns)
    # stats=offense.merge(defense[NonFeatures],left_index=True,right_index=True,how='outer')
    # stats=stats[stats.team.isna()==False]
    # stats.to_csv(f'{datadir}/{season}_TeamStats.csv',index=False)

def concatSeasons():
    for stat in ['Offense','Defense']:
        frames=[]
        for season in range(2014,2023):
            df=pd.read_csv(f'{datadir}/TeamDatabase/{season}_Team{stat}Stats.csv')
            frames.append(df)
        df=pd.concat(frames)
        df.to_csv(f'{datadir}/TeamDatabase/Team{stat}Stats_DB.csv',index=False)
# #%% Execution Line
current_season=2022
for season in range(2022,2023):
    print(season)
    if season!=current_season:
        __init__(season=season,update=False) 
    else:
        __init__(season=season,update=True)   
concatSeasons()