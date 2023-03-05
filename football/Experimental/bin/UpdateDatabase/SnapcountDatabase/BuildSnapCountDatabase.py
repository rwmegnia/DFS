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
    
def get_game_snapcounts(season):
    '''
    
    Scrape offensive snapcount data from home and away teams from a selected game
    and a selected season
    '''
    # Get snap counts
    df=nfl.import_snap_counts([season])
    df=df[df.position.isin(['QB','RB','FB','WR','TE'])]
    df.team.replace({'SD':'LAC',
                     'STL':'LA',
                     'OAK':'LV'},inplace=True)

    # Rename columns
    df.rename({'player':'full_name',
               'offense_snaps':'offensive_snapcounts',
               'offense_pct':'offensive_snapcount_percentage',
               'pfr_player_id':'pfr_id'},axis=1,inplace=True)
    df=df.merge(ids[['pfr_id','gsis_id']],on='pfr_id',how='left')
    df['full_name']=reformatNames(df)
    df=df[df.season==season]
    return df

ids=nfl.import_ids()
master=[]
for season in range(2012,2022):
    season_snapcounts=get_game_snapcounts(season)
    season_snapcounts.to_csv(f'{datadir}/SnapCountData/{season}_offensive_snapcounts.csv',index=False)
    master.append(season_snapcounts)
master=pd.concat(master)
master.to_csv(f'{datadir}/SnapCountData/2012_present_offensive_snapcounts.csv',index=False)
    
    
    
    
    