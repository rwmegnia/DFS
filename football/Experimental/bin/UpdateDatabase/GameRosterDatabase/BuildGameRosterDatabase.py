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

club_ids={'BLT':'BAL',
          'ARZ':'ARI',
          'CLV':'CLE',
          'HST':'HOU',}
seasontype='REG'  # or 'POST'
for season in range(2021,2022):
    if season>=2021:
        end_week=18
    else:
        end_week=17
    if season==2021:
        start_week=5
    else:
        start_week=1
    for week in range(start_week,end_week+1):
        weekly_frames=[]
        URL=f'https://www.nfl.info//nfldataexchange/dataexchange.asmx/getExpandedRosterByWeek?season={season}&week={week}&seasontype={seasontype}&clubid=-1'
        response = requests.get(URL,auth = HTTPBasicAuth('media', 'media')).json()
        df=pd.DataFrame.from_dict(response)
        if len(df)==0:
            continue
        df['full_name']=df.FootballName+' '+df.LastName
        df['team']=df.CurrentClub.apply(lambda x: club_ids[x] if x in club_ids.keys() else x)        
        df.rename({'Season':'season',
                   'Week':'week',
                   'GsisID':'gsis_id',
                   'Position':'position',
                   'RookieYear':'rookie_year',
                   'StatusShortDescription':'injury_status',
                   'Height':'height',
                   'Weight':'weight',
                   'DraftNumber':'draft_number',
                   'DraftRound':'draft_round',
                   'Draftround':'draft_round',
                   'College':'college',
                   'CollegeConference':'college_conference',
                   'Playerid':'player_id'
                   },axis=1,inplace=True)
        df=df[['full_name',
               'position',
               'team',
               'season',
               'week',
               'gsis_id',
               'injury_status',
               'rookie_year',
               'height',
               'weight',
               'draft_number',
               'draft_round',
               'college',
               'college_conference',
               'player_id']]
        weekly_frames.append(df)
        df=pd.concat(weekly_frames)
        df.to_csv(f'{datadir}/gameRosterData/{season}_Week{week}_InactiveLists.csv',index=False)




                                