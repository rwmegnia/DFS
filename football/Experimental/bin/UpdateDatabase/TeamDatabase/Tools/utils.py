#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 19:55:05 2021

@author: robertmegnia
"""
import nflfastpy as nfl
from datetime import datetime
import pandas as pd
from bs4 import BeautifulSoup as BS
import requests
import numpy as np
import os
basedir = os.path.dirname(os.path.abspath(__file__))
os.chdir(f'{basedir}/../')
from Tools.ScrapeBettingOdds import ScrapeBettingOdds
from Tools.FantasyPointsFunctions import *
from Tools.PlayerStatFunctions import *
from team_db_config import TEAM_STATS
from nflfastpy.errors import SeasonNotFoundError
topdir='/Volumes/XDrive/fantasyfootball/TopDownModel'
def getGameSlates(season=None,week=None,slate=None):
    
    '''
    Parameters
    ----------
    date : datetime object, optional
        DESCRIPTION. The default is None.
        
    season: int, optional
    
    week: int, optional
    
    slate: str, optional

    Returns
    -------
    A DataFrame of Games for the immediate nfl week or user
    selected week if week and season parameters are assigned.
    
    Use slate to get games from a particular day that week (Thursday, Sunday, Monday)
    

    '''
    
    # If season and week are not supplied, determine the next week on the nfl schedule in real time
    if season==None:
        date=datetime.now()
        season=date.year
        month=date.month 
        if month in [1,2]:
            season-=1
        schedule=nfl.load_schedule_data(season)
        date=date.strftime('%Y-%m-%d')
        schedule=schedule[schedule.gameday>=date]
        week=schedule.week.min()
        schedule=schedule[schedule.week==week]
    # If season and week are supplied, get the corresponding slate of games
    else:
        if (season==None)|(week==None):
            raise Exception('If season is left as none, you must pass in values for season and week')
        schedule=nfl.load_schedule_data(season)
        schedule=schedule[schedule.week==week]
    if slate!=None:
        schedule=schedule[schedule.weekday==slate]
    return schedule

def team_schedule(season,abbrevs):
    '''
    Parameters
    ----------
    season : int
    team : str
        team abbreviation (i.e. NE, ARI, LAR etc)

    Returns
    -------
    schedule : pandas dataframe
        
    data frame with team schedule information for a season
    returned as attribute for Team instance
    bye_week : int

    '''
    
    # Retrieve schedule for season  and question and filter it down to the selected team
    schedule=nfl.load_schedule_data(season)
    schedule=schedule[(schedule.home_team.isin(abbrevs))|(schedule.away_team.isin(abbrevs))]
    
    # Determine the opponent
    schedule.loc[schedule.away_team.isin(abbrevs),'Opponent']=schedule.loc[schedule.away_team.isin(abbrevs),'home_team']
    schedule.loc[~schedule.away_team.isin(abbrevs),'Opponent']=schedule.loc[~schedule.away_team.isin(abbrevs),'away_team']
    
    # Determine the points scored by the opponent
    schedule.loc[~schedule.home_team.isin(abbrevs),'OpponentScore']=schedule.loc[~schedule.home_team.isin(abbrevs),'home_score']
    schedule.loc[~schedule.away_team.isin(abbrevs),'OpponentScore']=schedule.loc[~schedule.away_team.isin(abbrevs),'away_score']
    
    # Determine the points scored by the team
    schedule.loc[schedule.home_team.isin(abbrevs),'TeamScore']=schedule.loc[schedule.home_team.isin(abbrevs),'home_score']
    schedule.loc[schedule.away_team.isin(abbrevs),'TeamScore']=schedule.loc[schedule.away_team.isin(abbrevs),'away_score']
    
    # Determine if the team is home or away
    schedule.loc[schedule.home_team.isin(abbrevs),'Location']='Home'
    schedule.loc[schedule.away_team.isin(abbrevs),'Location']='Away'
    
    # Compute TotalPoints scored
    schedule['TotalPoints']=schedule.OpponentScore+schedule.TeamScore
    
    # Determine the teams bye week
    bye_week=[i for i in range(1,19) if i not in schedule.week.values][0]
    schedule=schedule[['game_id',
                       'season',
                       'week',
                       'gameday',
                       'weekday',
                       'gametime',
                       'Opponent',
                       'Location',
                       'TeamScore',
                       'OpponentScore',
                       'TotalPoints']]
    return schedule,bye_week

def team_roster(season,abbrevs):
    '''
    

    Parameters
    ----------
    season : int
    team : str
        team abbreviation (i.e. NE, ARI, LAR etc)

    Returns
    -------
    None.

    '''
    roster = nfl.load_roster_data(season)
    roster = roster[roster.team.isin(abbrevs)][['season',
                                        'team',
                                        'position',
                                        'depth_chart_position',
                                        'status',
                                        'full_name',
                                        'first_name',
                                        'last_name',
                                        'birth_date',
                                        'height',
                                        'weight',
                                        'college',
                                        'high_school',
                                        'gsis_id',
                                        'pff_id',
                                        'years_exp']]
    return roster



def getBettingLines(week=None,season=None):
    if (season==None):
        odds=ScrapeBettingOdds(week)
        return odds
    else:
        df=nfl.load_pbp_data(season)
        df=df[df.week==week]
        away_teams=df.groupby('away_team',as_index=False).last()[['away_team','spread_line','total_line']]
        home_teams=df.groupby('home_team',as_index=False).last()[['home_team','spread_line','total_line']]
        away_teams['proj_team_score']=(away_teams.total_line/2)-(away_teams.spread_line/2)
        home_teams['proj_team_score']=(home_teams.total_line/2)+(home_teams.spread_line/2)
        away_teams.rename({'away_team':'team'},axis=1,inplace=True)
        home_teams.rename({'home_team':'team'},axis=1,inplace=True)
        home_teams.spread_line*=-1
        df=pd.concat([home_teams,away_teams])
        df.reset_index(drop=True,inplace=True)
        df.rename({'total_line':'O/U',},axis=1,inplace=True)
        return df[['team','O/U','spread_line','proj_team_score']]
    
def getPasserRating(cmp,td,pass_int,yds,att):
    a=((cmp/att)-.3)*5
    b=((yds/att)-3)*.25
    c=((td/att)*20)
    d=2.375-((pass_int/att)*25)
    terms={'a':a,'b':b,'c':c,'d':d}
    top=0
    for term in terms.keys():
        if terms[term]>2.375:
            terms[term]=2.375
        elif terms[term]<0:
            terms[term]=0
        top+=terms[term]
    return (top/6)*100

def getTeamPointsStats(df,formation):
    '''

    Parameters
    ----------
    df : pandas DataFrame
        nfl play by play data frame
    team : str
        team 2-3 letter id (i.e. NE,ARI,MIA,NO etc.)

    Returns
    -------
    4 dataframe columns with Points/Points Allwed/Implied Points/Implied Points allowed for team
        
    '''
    ## Get Points
    home_scores=df.groupby(['home_team','week'],as_index=False).home_score.mean()
    home_scores.rename({'home_team':formation,'home_score':'Points'},axis=1,inplace=True)
    away_scores=df.groupby(['away_team','week'],as_index=False).away_score.mean()
    away_scores.rename({'away_team':formation,'away_score':'Points'},axis=1,inplace=True)
    Points=pd.concat([home_scores,away_scores])

    ## Get Implied Points
    df['proj_away_score']=(((df.total_line)/2)-(df.spread_line/2))
    df['proj_home_score']=(((df.total_line)/2)+(df.spread_line/2)) 
    home_implied_scores=df.groupby(['home_team','week'],as_index=False).proj_home_score.mean()
    home_implied_scores.rename({'home_team':formation,'proj_home_score':'ImpliedPoints'},axis=1,inplace=True)
    away_implied_scores=df.groupby(['away_team','week'],as_index=False).proj_away_score.mean()
    away_implied_scores.rename({'away_team':formation,'proj_away_score':'ImpliedPoints'},axis=1,inplace=True)
    ImpliedPoints=pd.concat([home_implied_scores,away_implied_scores])
    Points=Points.merge(ImpliedPoints,on=['week',formation],how='left')
    df=df.merge(Points,on=['week',formation],how='left')
    # Recompute spreadline so that it is not strictly relative to the away team
    df['spread_line']=((df.ImpliedPoints)-(df.total_line-df.ImpliedPoints))*-1
    return df[['Points','ImpliedPoints','spread_line']]

def get_team_fantasy_points(df,defense=False):
    if defense==True:
        df['passing_fpts_allowed']=df.apply(lambda x: get_team_passing_fpts(x.passing_yards_allowed,x.pass_touchdown_allowed),axis=1)
        df['rushing_fpts_allowed']=df.apply(lambda x: get_team_rushing_fpts(x.rushing_yards_allowed,x.rush_touchdown_allowed),axis=1)
        df['receiving_fpts_allowed']=df.apply(lambda x: get_team_receiving_fpts(x.passing_yards_allowed,x.complete_pass_allowed,x.pass_touchdown_allowed),axis=1)
        df['team_fpts_allowed']=df[['passing_fpts_allowed','rushing_fpts_allowed','receiving_fpts_allowed']].sum(axis=1)-df[['fumble_recovery','interception']].sum(axis=1)
    else:
        df['passing_fpts']=df.apply(lambda x: get_team_passing_fpts(x.passing_yards,x.pass_touchdown),axis=1)
        df['rushing_fpts']=df.apply(lambda x: get_team_rushing_fpts(x.rushing_yards,x.rush_touchdown),axis=1)
        df['receiving_fpts']=df.apply(lambda x: get_team_receiving_fpts(x.passing_yards,x.complete_pass,x.pass_touchdown),axis=1)
        df['team_fpts']=df[['passing_fpts','rushing_fpts','receiving_fpts']].sum(axis=1)-df[['fumble_lost','interception']].sum(axis=1)
    return df

def getLatestWeek(schedule):
    return schedule[schedule.TeamScore.isna()==False].week.values[-1]
    
    