#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 07:00:23 2022

@author: robertmegnia


Run this script to build a database 


"""

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import os
basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f'{basedir}/../../../data'
# ClubIds

club_ids={'ATL':'ATL',
          'BUF':'BUF',
          'CHI':'CHI',
          'CIN':'CIN',
          'BAL':'BLT',
          'DAL':'DAL',
          'DEN':'DEN',
          'DET':'DET',
          'GB':'GB',
          'TEN':'TEN',
          'IND':'IND',
          'KC':'KC',
          'LV':'LV',
          'LA':'LA',
          'MIA':'MIA',
          'MIN':'MIN',
          'NE':'NE',
          'NO':'NO',
          'NYG':'NYG',
          'NYJ':'NYJ',
          'PHI':'PHI',
          'ARI':'ARZ',
          'PIT':'PIT',
          'LAC':'LAC',
          'SF':'SF',
          'SEA':'SEA',
          'TB':'TB',
          'WAS':'WAS',
          'CAR':'CAR',
          'JAX':'JAX',
          'CLE':'CLV',
          'HOU':'HTX',}
seasontype='REG'  # or 'POST'
for season in range(2003,2022):
    if season>=2021:
        end_week=18
    else:
        end_week=17
    for week in range(1,end_week+1):
        URL=f'https://www.nfl.info//nfldataexchange/dataexchange.asmx/getGameDepthChart?lseason={season}&lseasontype={seasontype}&lclub=ALL&lweek={week}'
        response = requests.get(URL,auth = HTTPBasicAuth('media', 'media')).content
        df=pd.read_xml(response)
        if len(df)==0:
            continue
        df['full_name']=df.FootballName+' '+df.LastName
        df.rename({'Season':'season',
                   'Week':'week',
                   'ClubCode':'team',
                   'GsisID':'gsis_id',
                   'Position':'position',
                   'DepthTeam':'depth_team',
                   'DepthPosition':'depth_position',
                       },axis=1,inplace=True)
        df=df[['full_name',
               'position',
               'depth_position',
               'team',
               'season',
               'week',
               'gsis_id',
               'depth_team']]
        df.to_csv(f'{datadir}/DepthChartData/{season}_Week{week}_DepthChart.csv',index=False)
