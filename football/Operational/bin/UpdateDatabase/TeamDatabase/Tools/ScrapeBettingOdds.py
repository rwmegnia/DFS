#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 13:34:18 2021

@author: robertmegnia

Scrape Rotogrinders for Spreads and Over/Under
"""
from bs4 import BeautifulSoup as BS
import requests
import pandas as pd
import numpy as np

def ScrapeBettingOdds(Week):
    teams=pd.DataFrame(columns=['team'])
    soup=BS(requests.get('https://rotogrinders.com/nfl/odds').content,'html.parser')
    # Scrape Teams
    for link in soup.find_all(attrs={'class':'row game'}):
        home=link.get('data-team-home')
        away=link.get('data-team-away')
        teams=teams.append(pd.DataFrame({'team':[away]}), ignore_index=True)
        teams=teams.append(pd.DataFrame({'team':[home]}), ignore_index=True)    
    
    # Scrape Totals
    i=0
    total=pd.DataFrame(columns=['total'])
    for link in soup.find_all(attrs={'class':'sb data card-data','data-type':'total'}):
        if 'o' in link.span.get_text():
            OU=float(link.span.get_text().split('o')[1].split('\n')[0])
            
        elif 'u' in link.span.get_text():
            OU=float(link.span.get_text().split('u')[1].split('\n')[0])
        else:
            continue
        total=total.append(pd.DataFrame({'total':[OU]}),ignore_index=True)
        total=total.append(pd.DataFrame({'total':[OU]}),ignore_index=True)
        i+=2
        if i==len(teams):
            break
    # Scrape Spread
    i=0
    spread=pd.DataFrame(columns=['spread'])
    for link in soup.find_all(attrs={'class':'sb data card-data','data-type':'spread'}):
        for string in link.strings:
            try:
                val=float(string)
            except:
                continue
            if np.abs(val)>=100:
                continue
            spread=spread.append(pd.DataFrame({'spread':[float(string)]}),ignore_index=True)
        i+=2
        if i==len(teams):
            break
        
    df=pd.concat([teams,total,spread],axis=1)
    df['proj_team_score']=(df.total/2)-(df.spread/2)
    df.rename({'total':'O/U'},axis=1,inplace=True)
    df.rename({'spread':'spread_line'},axis=1,inplace=True)
    df.replace('LAR','LA',inplace=True)
    return df