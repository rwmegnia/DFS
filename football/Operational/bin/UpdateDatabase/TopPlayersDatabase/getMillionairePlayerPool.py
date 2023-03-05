# -*- coding: utf-8 -*-

import pandas as pd
import os
from parseContestFileLineup import *
basedir = os.path.dirname(os.path.abspath(__file__))
datadir= f'{basedir}/../../../data'
db=pd.read_csv(f'{datadir}/game_logs/Full/Offense_Database.csv')
dst_db=pd.read_csv(f'{datadir}/game_logs/Full/DST_Database.csv')
db=pd.concat([db[['full_name','position','week','season','salary','DKPts']],dst_db[['full_name','position','week','season','salary','DKPts']]])
frames=[]
def getMilliPlayers(week,season):
    # Read in contest results
    df=pd.read_csv(f'{datadir}/MillionaireMakerContestResults/{season}/Week{week}_Millionaire_Results.csv')
    # Cut Top 10 Contest Scores 
    df=df.head(100)
    # Parse players from the Top10 Lineups
    df=pd.concat([lineup for lineup in df.Lineup.apply(lambda x: parseContestFileLineup(x,week,season))])
    # Remove duplicates
    # remove alphanumerics
    df=reformatNames(df)
    df.full_name.replace({'Jeff Wilson':'Jeffery Wilson',
                              'OAK':'LV',
                              'Eli Mitchell':'Elijah Mitchell',
                              'Amonra St': 'Amonra Stbrown',
                              'Joshua Palmer':'Josh Palmer'},inplace=True)
        
    df=df.merge(db[['full_name','week','position','season','salary','DKPts']],on=['full_name','week','season'],how='left')
    frames.append(df)
df=pd.concat(frames)
df.to_csv(f'{datadir}/TopLineupPlayers/2020_present_TopLineupPlayers.csv',index=False)