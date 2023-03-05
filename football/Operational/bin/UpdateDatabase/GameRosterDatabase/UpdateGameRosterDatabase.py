#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 17:05:54 2022

@author: robertmegnia
"""

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import os
basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f'{basedir}/../../../data'
inactives_database=pd.read_csv('f{datadir}/injuryData/2001_present_inactives.csv')
# ClubIds

club_ids={'BLT':'BAL',
          'ARZ':'ARI',
          'CLV':'CLE',
          'HST':'HOU',}
seasontype='REG'  # or 'POST' 
def updateGameRosterDatabase(season,week,seasontype):
    weekly_frame=[]
    clubid=-1
    URL=f'https://www.nfl.info//nfldataexchange/dataexchange.asmx/getExpandedRosterByWeek?season={season}&week={week}&seasontype={seasontype}&clubid={clubid}'
    response = requests.get(URL,auth = HTTPBasicAuth('media', 'media')).json()
    df=pd.DataFrame.from_dict(response)
    df['full_name']=df.FootballName+' '+df.LastName
    df['team']=df.CurrentClub.apply(lambda x: club_ids[x] if x in club_ids.keys() else x)        
    df.rename({'Season':'season',
               'Week':'week',
               'GsisID':'gsis_id',
               'Position':'position',
               'RookieYear':'rookie_year',
               'StatusShortDescription':'injury_status',
                   },axis=1,inplace=True)
    df=df[['full_name','position','team','season','week','gsis_id','injury_status']]
    print(season,week,clubid,df.team.unique())
    weekly_frame.append(df)
    df=pd.concat(weekly_frame)
    df.to_csv(f'{datadir}/injuryData/{season}_Week{week}_InactiveLists.csv',index=False)
    df=pd.concat([inactives_database,df])
    df.to_csv('f{datadir}/injuryData/2001_present_inactives.csv')
