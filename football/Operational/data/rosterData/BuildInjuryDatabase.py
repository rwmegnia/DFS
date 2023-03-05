#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 22:43:15 2022

@author: robertmegnia

Build Injury Database
"""

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import os
basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f'{basedir}/../../../data'
# ClubIds

club_ids={1:'ATL',
          2:'BUF',
          3:'CHI',
          4:'CIN',
          5:'BAL',
          6:'DAL',
          7:'DEN',
          8:'DET',
          9:'GB',
          10:'TEN',
          11:'IND',
          12:'KC',
          13:'LV',
          14:'LA',
          15:'MIA',
          16:'MIN',
          17:'NE',
          18:'NO',
          19:'NYG',
          20:'NYJ',
          21:'PHI',
          22:'ARI',
          23:'PIT',
          24:'LAC',
          25:'SF',
          26:'SEA',
          27:'TB',
          28:'WAS',
          32:'CAR',
          30:'JAX',
          36:'CLE',
          39:'HOU',}
seasontype='REG'  # or 'POST'
for season in range(2001,2022):
    if season>=2021:
        end_week=18
    else:
        end_week=17
    for week in range(1,end_week+1):
        weekly_frames=[]
        for clubid in club_ids.keys():
            URL=f'https://www.nfl.info//nfldataexchange/dataexchange.asmx/getExpandedRosterByWeek?season={season}&week={week}&seasontype={seasontype}&clubid={clubid}'
            response = requests.get(URL,auth = HTTPBasicAuth('media', 'media')).json()
            df=pd.DataFrame.from_dict(response)
            if len(df)==0:
                continue
            df['full_name']=df.FootballName+' '+df.LastName
            df['team']=club_ids[clubid]
            df.rename({'Season':'season',
                       'Week':'week',
                       'GsisID':'gsis_id',
                       'Position':'position',
                       'RookieYear':'rookie_year',
                       'StatusShortDescription':'injury_status',
                       },axis=1,inplace=True)
            df=df[['full_name','position','team','season','week','gsis_id','injury_status']]
            print(season,week,clubid,df.team.unique())
            weekly_frames.append(df)
        df=pd.concat(weekly_frames)
        df.to_csv(f'{datadir}/injuryData/{season}_Week{week}_InactiveLists.csv',index=False)




                                