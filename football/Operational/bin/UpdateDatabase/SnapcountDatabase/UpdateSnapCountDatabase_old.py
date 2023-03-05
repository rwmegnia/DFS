#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 19:34:44 2022

@author: robertmegnia
"""
import os
basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f'{basedir}/../../../data'
from bs4 import BeautifulSoup as bs
import requests
import nfl_data_py as nfl
import pandas as pd
club_ids={'ATL':'atl',
          'BUF':'buf',
          'CHI':'chi',
          'CIN':'cin',
          'BAL':'rav',
          'DAL':'dal',
          'DEN':'den',
          'DET':'det',
          'GB':'gnb',
          'TEN':'oti',
          'IND':'clt',
          'KC':'kan',
          'LV':'rai',
          'OAK':'rai',
          'LA':'ram',
          'STL':'ram',
          'MIA':'mia',
          'MIN':'min',
          'NE':'nwe',
          'NO':'nor',
          'NYG':'nyg',
          'NYJ':'nyj',
          'PHI':'phi',
          'ARI':'crd',
          'PIT':'pit',
          'LAC':'sdg',
          'SD':'sdg',
          'SF':'sfo',
          'SEA':'sea',
          'TB':'tam',
          'WAS':'was',
          'CAR':'car',
          'JAX':'jax',
          'CLE':'cle',
          'HOU':'htx',}
rename_players={'Beanie Wells':'Chris Wells',
                'Ben Watson':'Benjamin Watson',
                'Tim Wright':'Timothy Wright',
                'Danny Vitale':'Dan Vitale',
                'Chris Herndon':'Christopher Herndon',
                'Jeff Wilson':'Jeffery Wilson',
                'Jonathan Baldwin':'Jon Baldwin',
                'Matt Mcgloin':'Matthew Mcgloin',
                'Matthew Slater':'Matt Slater',
                'Michael Higgins':'Mike Higgins',
                'Josh Cribbs':'Joshua Cribbs',
                'Walt Powell':'Walter Powell',
                'Pj Walker':'Phillip Walker',
                'Robert Kelley':'Rob Kelley',
                'Clyde Gates':'Edmond Gates',
                'Drew Davis':'Dj Davis',
                'Gerrell Robinson':'Gerell Robinson'
                    }
def reformatNames(df):
        df['first_name']=df.full_name.apply(lambda x: x.split(' ')[0])
        df['last_name']=df.full_name.apply(lambda x: ' '.join(x.split(' ')[1::]))

        # Remove suffix from last name but keep prefix
        df['last_name']=df.last_name.apply(lambda x: x if x in ['St. Brown','Vander Laan'] else x.split(' ')[0])

        # Remove non-alpha numeric characters from first names.
        df['first_name']=df.first_name.apply(lambda x: ''.join(c for c in x if c.isalnum()))
        df['last_name']=df.last_name.apply(lambda x: ''.join(c for c in x if c.isalnum()))
        # Recreate full_name
        df['full_name']=df.apply(lambda x: x.first_name+' '+x.last_name,axis=1)
        df['full_name']=df.full_name.apply(lambda x: x.lower())
        df.drop(['first_name','last_name'],axis=1,inplace=True)
        df.full_name=df.full_name.apply(lambda x: x.split(' ')[0][0].upper()+x.split(' ')[0][1::]+' '+x.split(' ')[-1][0].upper()+x.split(' ')[-1][1::])
        df.full_name.replace(rename_players,inplace=True)
        return df.full_name
    
def get_game_snapcounts(gameday,home_team,away_team,week,season):
    '''
    
    Scrape offensive snapcount data from home and away teams from a selected game
    and a selected season
    '''
    print(season,week,gameday,home_team,away_team)
    frames=[]
    home=club_ids[home_team]
    gameday=''.join(gameday.split('-'))
    # Get home snap counts
    URL=f'https://widgets.sports-reference.com/wg.fcgi?css=1&site=pfr&url=%2Fboxscores%2F{gameday}0{home}.htm&div=div_home_snap_counts'
    response=requests.get(URL)
    df=pd.read_html(response.content)[0][[('Unnamed: 0_level_0', 'Player'),
                                          ('Unnamed: 1_level_0', 'Pos'),
                                          ('Off.', 'Num'),
                                          ('Off.', 'Pct')]].droplevel(0,axis=1)
    df['Pct']=df.Pct.str.rstrip('%').astype(float)/100
    df=df[df.Pos.isin(['QB','RB','FB','WR','TE'])]
    df['week']=week
    df['season']=season
    df['team']=home_team
    frames.append(df)
    # Get away snap counts
    URL=f'https://widgets.sports-reference.com/wg.fcgi?css=1&site=pfr&url=%2Fboxscores%2F{gameday}0{home}.htm&div=div_vis_snap_counts'
    response=requests.get(URL)
    df=pd.read_html(response.content)[0][[('Unnamed: 0_level_0', 'Player'),
                                          ('Unnamed: 1_level_0', 'Pos'),
                                          ('Off.', 'Num'),
                                          ('Off.', 'Pct')]].droplevel(0,axis=1)
    df['Pct']=df.Pct.str.rstrip('%').astype(float)/100
    df=df[df.Pos.isin(['QB','RB','FB','WR','TE'])]
    df['week']=week
    df['season']=season
    df['team']=away_team
    frames.append(df)  
    df=pd.concat(frames)
    # Rename columns
    df.rename({'Player':'full_name',
               'Pos':'position',
               'Num':'offensive_snapcounts',
               'Pct':'offensive_snapcount_percentage'},axis=1,inplace=True)
    df['full_name']=reformatNames(df)
    return df

def UpdateSnapCountDatabase(season,week):
    schedule=nfl.import_schedules([season])
    schedule=schedule[schedule.week==week]
    snapcounts=pd.concat([frame for frame in schedule.apply(lambda x: get_game_snapcounts(x.gameday,x.home_team,x.away_team,x.week,x.season),axis=1)])
    # Update Season Database
    try:
        season_snapcounts=pd.read_csv(f'{datadir}/SnapCountData/{season}_offensive_snapcounts.csv')
        season_snapcounts=pd.concat([season_snapcounts,snapcounts])
        season_snapcounts.to_csv(f'{datadir}/SnapCountData/{season}_offensive_snapcounts.csv',index=False)
    except FileNotFoundError:
        snapcounts.to_csv(f'{datadir}/SnapCountData/{season}_offensive_snapcounts.csv',index=False)
    # Update Master Database
    db=pd.read_csv(f'{datadir}/SnapCountData/2012_present_offensive_snapcounts.csv')
    db=pd.concat([db,snapcounts])
    db.to_csv(f'{datadir}/SnapCountData/2012_present_offensive_snapcounts.csv',index=False)
    
    snapcounts.to_csv(f'{datadir}/SnapCountData/{season}_offensive_snapcounts.csv',index=False)