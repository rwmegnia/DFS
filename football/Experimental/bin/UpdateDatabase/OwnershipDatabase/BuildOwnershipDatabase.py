#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 19:00:08 2022

@author: robertmegnia

Build Ownership Database from contest results files
"""
import os
import unidecode
basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f'{basedir}/../../../data'
import pandas as pd

roster_percent_dst_names_dict={
    '49ers ':'SF',
    'Bears ':'CHI',
    'Bengals ':'CIN',
    'Bills ':'BUF',
    'Browns ':'CLE',
    'Buccaneers ':'TB',
    'Cardinals ':'ARI',
    'Chargers ':'LAC',
    'Colts ':'IND',
    'Dolphins ':'MIA',
    'Eagles ':'PHI',
    'Falcons ':'ATL',
    'Jaguars ':'JAX',
    'Jets ':'NYJ',
    'Lions ':'DET',
    'Packers ':'GB',
    'Panthers ':'CAR',
    'Patriots ':'NE',
    'Raiders ':'LV',
    'Ravens ':'BAL',
    'Saints ':'NO',
    'Seahawks ':'SEA',
    'Vikings ':'MIN',
    'WAS Football Team ':'WAS',
    'Broncos ':'DEN',
    'Chiefs ':'KC',
    'Cowboys ':'DAL',
    'Giants ':'NYG',
    'Rams ':'LAR',
    'Steelers ':'PIT',
    'Texans ':'HOU',
    'Titans ':'TEN'
    }
def reformatName(df):
    ## REFORMAT PLAYER NAMES BY REMOVING NON-ALPHA-NUMERICS
    df["first_name"] = df.full_name.apply(lambda x: x.split(" ")[0])
    df["last_name"] = df.full_name.apply(
        lambda x: " ".join(x.split(" ")[1::])
    )

    # Remove non-alpha numeric characters from first/last names.
    df["first_name"] = df.first_name.apply(
        lambda x: "".join(c for c in x if c.isalnum())
    )
    df["last_name"] = df.last_name.apply(
        lambda x: "".join(c for c in x if c.isalnum())
    )

    # Recreate full_name to fit format "Firstname Lastname" with no accents
    df["full_name"] = df.apply(
        lambda x: x.first_name + " " + x.last_name, axis=1
    )
    df["full_name"] = df.full_name.apply(lambda x: x.lower())
    df.drop(["first_name", "last_name"], axis=1, inplace=True)
    df.loc[df.position!='DST',"full_name"] = df.loc[df.position!='DST'].full_name.apply(
        lambda x: x.split(" ")[0][0].upper()
        + x.split(" ")[0][1::]
        + " "
        + x.split(" ")[-1][0].upper()
        + x.split(" ")[-1][1::]
    )
    df["full_name"] = df.full_name.apply(lambda x: unidecode.unidecode(x))

    # Create Column to match with RotoGrinders
    df["RotoName"] = df.full_name.apply(
        lambda x: x.lower().split(" ")[0][0:4] + x.lower().split(" ")[1][0:5]
    )
    df.loc[df.position=='DST','RotoName']=df.loc[df.position=='DST','full_name'].str.upper()
    return df 

frames=[]
for season in range(2020,2022):
    for week in range(1,19):
        if (season==2020)&(week==18):
            continue
        print(season,week)
        contest_file=f'/Volumes/XDrive/DFS/football/Experimental/data/MillionaireMakerContestResults/{season}/Week{week}_Millionaire_Results.csv'
        df=pd.read_csv(contest_file)
        df.reset_index(inplace=True,drop=True)
        df=df[df['Roster Position'].isna()==False]
        # Reformat DST names from mascot to identifier (i.e Seahawks to SEA) for merging with projections dataframe
    
        df.loc[df['Roster Position']=='DST','Player']=df.loc[df['Roster Position']=='DST','Player'].apply(lambda x: roster_percent_dst_names_dict[x])
        df.loc[df['Roster Position']!='DST','first_name']=df.loc[df['Roster Position']!='DST','Player'].apply(lambda x: x.split(' ')[0])
        df.loc[df['Roster Position']!='DST','last_name']=df.loc[df['Roster Position']!='DST','Player'].apply(lambda x: ' '.join(x.split(' ')[1::]))
        df.loc[df['Roster Position']!='DST','last_name']=df.loc[df['Roster Position']!='DST','last_name'].apply(lambda x: x if x=='St. Brown' else x.split(' ')[0])
        
    
        # Remove non alpha-numeric characters form player names
        df.loc[df['Roster Position']!='DST','first_name']=df.loc[df['Roster Position']!='DST','first_name'].apply(lambda x: ''.join(c for c in x if c.isalnum()))
        df.loc[df['Roster Position']!='DST','last_name']=df.loc[df['Roster Position']!='DST','last_name'].apply(lambda x: ''.join(c for c in x if c.isalnum()))
        df.loc[df['Roster Position']!='DST','full_name']=df.loc[df['Roster Position']!='DST'].apply(lambda x: ' '.join([x.first_name,x.last_name]),axis=1)
        
        
        # Recreate full_name
        df.loc[df['Roster Position']!='DST','full_name']=df.loc[df['Roster Position']!='DST'].full_name.apply(lambda x: x.lower())
        df.drop(['first_name','last_name'],axis=1,inplace=True)
        df.full_name.fillna('DST',inplace=True)
        df.full_name=df.full_name.apply(lambda x: x.split(' ')[0][0].upper()+x.split(' ')[0][1::]+' '+x.split(' ')[-1][0].upper()+x.split(' ')[-1][1::])
        df.loc[df['Roster Position']=='DST','full_name']=df.loc[df['Roster Position']=='DST','Player']
    
        df['RosterPercent']=df['%Drafted'].apply(lambda x: float(x.split('%')[0]))
        # Reformat first names that don't matchup
        if season ==2020:
            df.full_name.replace('Chris Herndon','Christopher Herndon',inplace=True)
            df.full_name.replace('Jeff Wilson','Jeffery Wilson',inplace=True)
        df.full_name.replace('Pj Walker','Phillip Walker',inplace=True)
        df.full_name.replace('Josh Dobbs','Joshua Dobbs',inplace=True)
        df.full_name.replace('Eli Mitchell','Elijah Mitchell',inplace=True)
        df.full_name.replace('Dee Eskridge','Dwayne Eskridge',inplace=True)
        df.full_name.replace('Joshua Palmer','Josh Palmer',inplace=True)
        df.full_name.replace('Scotty Miller','Scott Miller',inplace=True) 
        df['week']=week
        df['season']=season
        df.rename({'Roster Position':'position'},axis=1,inplace=True)
        df=reformatName(df)
        df.loc[df.position=='DST','RotoName']=df.loc[df.position=='DST','RotoName'].apply(lambda x: x.split(' ')[0])
        df=df[['RotoName','week','season','RosterPercent']]
        df.to_csv(f'{datadir}/Ownership/{season}_Week{week}_MainSlate_Ownership.csv',index=False)
        frames.append(df)
df=pd.concat(frames)
df.to_csv(f'{datadir}/Ownership/2020_present_MainSlate_Ownership.csv',index=False)
    