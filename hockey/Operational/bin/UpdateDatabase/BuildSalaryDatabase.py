#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 13:46:56 2022

@author: robertmegnia
"""
from bs4 import BeautifulSoup as BS
import requests
import pandas as pd
import os
from datetime import datetime
basedir = os.path.dirname(os.path.abspath(__file__))
datadir= f'{basedir}/../../data'
frames=[]
game_date=datetime.utcnow().strftime('%Y-%m-%d')
print(game_date)
soup=BS(requests.get(f'https://rotogrinders.com/lineups/nhl?date={game_date}&site=draftkings').content,'html.parser')
players=[]
positions=[]
projections=[]
salaries=[]
ownerships=[]
teams=[]
lines=[]
for game in soup.find_all(attrs={'data-role':'lineup-card'}):
    away=game.get('data-away')
    home=game.get('data-home')
    counter=1
    line_counter=1
    players.append(game.find_all(attrs={'class':'nolink name'})[0].text)
    positions.append('G')
    projections.append(float(game.find_all(attrs={'class':'pitcher players'})[0].find_all(attrs={'class':'fpts'})[0].get('data-fpts')))
    salaries.append(float(game.find_all(attrs={'class':'pitcher players'})[0].find_all(attrs={'class':'salary'})[0].get('data-salary')))
    ownerships.append(float(game.find_all(attrs={'class':'pitcher players'})[0].find_all(attrs={'class':'pown'})[0].get('data-pown').split('%')[0]))
    teams.append(away)
    lines.append(1)
    for player in game.find_all(attrs={'class':'pname'}):
        if counter<=18:
            teams.append(away)
        else:
            teams.append(home)
        if line_counter<=3:
            line=1
        elif (line_counter>3)&(line_counter<=6):
            line=2
        elif (line_counter>6)&(line_counter<=9):
            line=3
        elif (line_counter>9)&(line_counter<=12):
            line=4
        elif (line_counter>12)&(line_counter<=14):
            line=1
        elif (line_counter>14)&(line_counter<=16):
            line=2
        elif (line_counter>16)&(line_counter<=18):
            line=3
        elif (line_counter>18)&(line_counter<=21):
            line=1
        elif (line_counter>21)&(line_counter<=24):
            line=2
        elif (line_counter>24)&(line_counter<=27):
            line=3
        elif (line_counter>27)&(line_counter<=30):
            line=4
        elif (line_counter>30)&(line_counter<=32):
            line=1
        elif (line_counter>32)&(line_counter<=34):
            line=2
        else:
            line=3
        name=' '.join(player.text.split('\n')[1].split(' ')[-2:])
        position=player.find_all(attrs={'class':'position'})[0].text
        projection=float(player.find_all(attrs={'class':'fpts'})[0].get('data-fpts'))
        salary=float(player.find_all(attrs={'class':'salary'})[0].get('data-salary'))
        ownership=float(player.find_all(attrs={'class':'pown'})[0].get('data-pown').split('%')[0])
        players.append(name)
        positions.append(position)
        projections.append(projection)
        salaries.append(salary)
        ownerships.append(ownership)
        lines.append(line)
        counter+=1
        line_counter+=1
    players.append(game.find_all(attrs={'class':'nolink name'})[1].text)
    positions.append('G')
    projections.append(float(game.find_all(attrs={'class':'pitcher players'})[1].find_all(attrs={'class':'fpts'})[0].get('data-fpts')))
    salaries.append(float(game.find_all(attrs={'class':'pitcher players'})[1].find_all(attrs={'class':'salary'})[0].get('data-salary')))
    ownerships.append(float(game.find_all(attrs={'class':'pitcher players'})[1].find_all(attrs={'class':'pown'})[0].get('data-pown').split('%')[0]))
    teams.append(home)
    lines.append(1)
            
df=pd.DataFrame({'full_name':players,
                     'position':positions,
                     'RG_projection':projections,
                     'Salary':salaries,
                     'ownership_proj':ownerships,
                     'team':teams,
                     'line':lines})
df['full_name']=df.full_name.apply(lambda x: x.split('...')[0])
df['game_date']=game_date
frames.append(df)
salaries=pd.concat(frames)
salaries.to_csv(f'{datadir}/SalaryDatabase/DKSalaries_DB.csv',index=False)