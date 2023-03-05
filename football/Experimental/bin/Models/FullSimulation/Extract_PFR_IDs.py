#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:36:03 2023

@author: robertmegnia
"""
import pandas as pd
import numpy as np
import os
import warnings
import unidecode
import nfl_data_py as nfl
import requests
from requests.auth import HTTPBasicAuth
from sqlalchemy import create_engine
import pymysql
import mysql.connector
def reformatName(df):
    ## REFORMAT PLAYER NAMES BY REMOVING NON-ALPHA-NUMERICS
    df["first_name"] = df.full_name.apply(lambda x: x.split(" ")[0])
    df["last_name"] = df.full_name.apply(lambda x: " ".join(x.split(" ")[1::]))

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
    df["full_name"] = df.full_name.apply(lambda x: unidecode.unidecode(x))

    # Create Column to match with RotoGrinders
    df["mergename"] = df.full_name.apply(
        lambda x: x.lower().split(" ")[0][0:4] + x.lower().split(" ")[1][0:5]
    )
    return df

mydb = mysql.connector.connect(
        host="footballai-db-prod.cxgq1kandeps.us-east-2.rds.amazonaws.com",
        user="gridironai",
        password="thenameofthewind",
        database="gridironai",
    )
sqlEngine = create_engine(
        "mysql+pymysql://gridironai:thenameofthewind@footballai-db-prod.cxgq1kandeps.us-east-2.rds.amazonaws.com/gridironai",
        pool_recycle=3600,
    )
df=pd.read_sql('stats_pfrplayer',con=sqlEngine)
df.drop('full_name',axis=1,inplace=True)
df.rename({'name':'full_name'},axis=1,inplace=True)
df=df[df.full_name.isnull()==False]
df=reformatName(df)
df.mergename.replace(
    {'benwatso':'benjwatso',
     'tedginnj':'tedginn',
     'evanhood':'zigghood',
     'jonweeks':'jonaweeks',
     'evandietr':'evansmith',
     'gabrdavis':'gabedavis',
     'jodyforts':'joeforts',
     'michperso':'mikeperso',
     'mikemorga':'michmorga',
     'jackjenki':'janojenki',
     'jakeschum':'jacoschum',
     'daxtswans':'daxswans',
     'jonabosti':'jonbosti',
     'tankcarra':'corncarra',
     'ayodolato':'dejiolato',
     'charlenoj':'charleno',
     'mikeliedt':'michliedt',
     'robekelle':'robkelle',
     'lacedwar':'lachedwar',
     'dannvital':'danvital',
     'pjwalke':'philwalke',
     'michtyson':'miketyson',
     'lanohill':'delahill',
     'deatwisej':'deatwise',
     'jakemarti':'jacomarti',
     'jimmmurra':'jamemurra',
     'zeketurne':'ezekturne',
     'kahlmcken':'regimcken',
     'samueguav':'sameguav',
     'oliudoh':'olisudoh',
     'cjgardn':'chaugardn',
     'shaqcalho':'deiocalho',
     'suaopeta':'iosuopeta',
     'jakedoleg':'jacodoleg',
     'marvtelli':'marvtell',
     'boogbasha':'carlbasha',
     'camsampl':'camesampl',
     'nathlandm':'natelandm',
     'daxthill':'daxhill',
     'deaualfor':'deealfor',
     'ugoamadi':'ugocamadi'},
    inplace=True
     )
#%%
dc=nfl.import_depth_charts(range(2016,2023))
dc=reformatName(dc)
dc = dc.groupby(['gsis_id','mergename'],as_index=False).first()
dc = dc.merge(df[['id','mergename']],on='mergename',how='left')
dc.rename({'id':'pfr_player_id'},axis=1,inplace=True)
dc = dc.groupby(['gsis_id','pfr_player_id'],as_index=False).first()
dc.to_csv('/Volumes/XDrive/DFS/football/Experimental/data/PlayerIDs/pfr_gsis_ids.csv',index=False)