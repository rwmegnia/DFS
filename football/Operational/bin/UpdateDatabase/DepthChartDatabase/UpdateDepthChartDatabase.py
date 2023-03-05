#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:48:43 2022

@author: robertmegnia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 07:00:23 2022

@author: robertmegnia
"""

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import os
basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f'{basedir}/../../../data'
full_df=pd.read_csv(f'{datadir}/DepthChartData/2001_present_game_depth_charts.csv')
seasontype='REG'  # or 'POST'
def updateDepthChartsDB(season,week):
    URL=f'https://www.nfl.info//nfldataexchange/dataexchange.asmx/getGameDepthChart?lseason={season}&lseasontype={seasontype}&lclub=ALL&lweek={week}'
    response = requests.get(URL,auth = HTTPBasicAuth('media', 'media')).content
    df=pd.read_xml(response)
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
    print(season,week)
    df.to_csv(f'{datadir}/DepthChartData/{season}_Week{week}_DepthChart.csv',index=False)
    df=pd.concat([full_df,df])
    df.team.replace({'OAK':'LV','SL':'LA','SD':'LAC','HST':'HOU','CLV':'CLE','ARZ':'ARI','BLT':'BAL'},inplace=True)
    df.to_csv(f'{datadir}/DepthChartData/2001_present_game_depth_charts.csv')
