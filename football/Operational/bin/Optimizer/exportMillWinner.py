#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:16:12 2022

@author: robertmegnia
"""

import pandas as pd
import numpy as np
import os
import warnings
from unidecode import unidecode
import nfl_data_py as nfl
import requests
from requests.auth import HTTPBasicAuth
from sqlalchemy import create_engine
import pymysql
import mysql.connector
import sys
warnings.simplefilter("ignore")
basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
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
    df.drop(["first_name", "last_name"], axis=1, inplace=True)
    df.loc[df.position != "DST", "full_name"] = df.loc[
        df.position != "DST"
    ].full_name.apply(
        lambda x: x.split(" ")[0][0].upper()
        + x.split(" ")[0][1::]
        + " "
        + x.split(" ")[-1][0].upper()
        + x.split(" ")[-1][1::]
    )
    df["full_name"] = df.full_name.apply(lambda x: unidecode(x))

    # Create Column to match with RotoGrinders
    df["RotoName"] = df.full_name.apply(
        lambda x: x.lower().split(" ")[0][0:4] + x.lower().split(" ")[1][0:5]
    )
    try:
        df.loc[df.position == "DST", "RotoName"] = df.loc[
            df.position == "DST", "team"
        ]
    except:
        pass
    # df['game_time']=df['Game Info'].apply(lambda x: x.split(' ')[2])
    # Replace misspelled names
    return df  # , slate.split("_df.csv")[0]

def getDKPlayerData(contest_id):
    print(contest_id)
    # Pull contest info from draftkings api with contest_id
    contest_url = f"https://api.draftkings.com/contests/v1/contests/{contest_id}?format=json"
    contestInfo = requests.get(contest_url).json()

    # get draftGroupId from contestInfo json
    draftGroup = contestInfo["contestDetail"]["draftGroupId"]

    # Use draftGroup Number to pull seach players DK Player ID from DK API
    draftGroup_url = f"https://api.draftkings.com/draftgroups/v1/draftgroups/{draftGroup}/draftables"
    draftGroupInfo = pd.DataFrame(
        requests.get(draftGroup_url).json()["draftables"]
    )[["displayName", "playerId", "teamAbbreviation", "salary"]]

    # Rename some columns for merging purposes
    draftGroupInfo.rename(
        {
            "displayName": "draftkings_name",
            "playerId": "draftkings_player_id",
            "teamAbbreviation": "team",
            "salary": "Salary",
        },
        axis=1,
        inplace=True,
    )

    # Pull Salary csv data so we have each players salary for FLEX/CPT positions
    csv_url = f"https://www.draftkings.com/lineup/getavailableplayerscsv?draftGroupId={draftGroup}"
    salaries = pd.read_csv(csv_url)

    # rename columns for merging purposes
    salaries.rename(
        {
            "Roster Position": "roster_position",
            "Name": "draftkings_name",
            "TeamAbbrev": "team",
        },
        axis=1,
        inplace=True,
    )

    # Filter salaries to columns that we will use
    salaries = salaries[
        [
            "draftkings_name",
            "team",
            "Position",
            "roster_position",
            "Salary",
            "ID",
        ]
    ]

    # Need to add a space at the end of the team names. i.e. convert "Cowboys " to "Cowboys"
    # In order to merge all draftkings_player_ids correctly
    draftGroupInfo["draftkings_name"] = draftGroupInfo["draftkings_name"].apply(
        lambda x: x + " " if len(x.split(" ")) == 1 else x
    )

    # drop suffix from name columns i.e. 'Ronald Jones II' becomes 'Ronald Jones'
    # for merging purpsoses
    salaries["draftkings_name"] = salaries.draftkings_name.apply(
        lambda x: " ".join(x.split(" ")[0:2]) if len(x.split(" ")) > 1 else x
    )
    draftGroupInfo["draftkings_name"] = draftGroupInfo.draftkings_name.apply(
        lambda x: " ".join(x.split(" ")[0:2]) if len(x.split(" ")) > 1 else x
    )
    draftGroupInfo["draftkings_name"] = draftGroupInfo.draftkings_name.apply(
        lambda x: unidecode(x)
    )
    # Have found that in some case the draftkings names in "draftGroupInfo" and salaries
    # don't matchup. Replace names as necessary such that they match the Player names in the
    # draftkings contest results file
    # salaries.draftkings_name.replace(
    #     {
    #         "Robbie Anderson": "Robby Anderson",
    #         "Mitch Trubisky": "Mitchell Trubisky",
    #         "Joshua Dobbs": "Josh Dobbs",
    #         "Deonte Harty": "Deonte Harris",
    #         "Dan Brown": "Daniel Brown",
    #         "J.J. Arcega-Whiteside":"JJ Arcega-Whiteside",
    #         "Commanders ":"WAS Football",
    #         "Van Jefferson Jr.": "Van Jefferson",
    #         "Gabe Davis":"Gabriel Davis",
    #         "Bisi Johnson":"Olabisi Johnson",
    #     },
    #     inplace=True,
    # )

    # merge salaries with draftGroupInfo
    salaries = salaries.merge(
        draftGroupInfo, on=["draftkings_name", "team", "Salary"], how="left"
    )
    salaries['contest_id']=contest_id
    # converting player_id to int may raise exception if there was a merging
    # mismatch and one of the player_ids were filled as NaN
    try:
        salaries["draftkings_player_id"] = salaries.draftkings_player_id.astype(
            int
        )
    except:
        mismatch = salaries[
            salaries.draftkings_player_id.isna() == True
        ].draftkings_name.unique()[0]
        print(f"Mismatch in playername {mismatch}")
        sys.exit()
    salaries.rename({'draftkings_name':'Player'},axis=1,inplace=True)
    return salaries
def parseNFLLineup(df):
    qb = pd.DataFrame(
        df[df.Lineup.isna() == False]
        .Lineup.str.findall("QB ([A-Za-z\.'-]+ [A-Za-z\.'-]+)")
        .to_list(),
        columns=["Player"],
    )
    rbs = pd.DataFrame(
        df[df.Lineup.isna() == False]
        .Lineup.str.findall("RB ([A-Za-z\.'-]+ [A-Za-z\.'-]+)")
        .to_list(),
        columns=["Player", "Player"],
    )
    rb1 = rbs.iloc[:, 0].to_frame()
    rb2 = rbs.iloc[:, 1].to_frame()

    wrs = pd.DataFrame(
        df[df.Lineup.isna() == False]
        .Lineup.str.findall("WR ([A-Za-z\.'-]+ [A-Za-z\.'-]+)")
        .to_list(),
        columns=["Player", "Player", "Player"],
    )
    wr1 = wrs.iloc[:, 0].to_frame()
    wr2 = wrs.iloc[:, 1].to_frame()
    wr3 = wrs.iloc[:, 2].to_frame()
    te = pd.DataFrame(
        df[df.Lineup.isna() == False]
        .Lineup.str.findall("TE ([A-Za-z\.'-]+ [A-Za-z\.'-]+)")
        .to_list(),
        columns=["Player"],
    )
    flex = pd.DataFrame(
        df[df.Lineup.isna() == False]
        .Lineup.str.findall("FLEX ([A-Za-z\.'-]+ [A-Za-z\.'-]+)")
        .to_list(),
        columns=["Player"],
    )
    dst = pd.DataFrame(
        df[df.Lineup.isna() == False]
        .Lineup.str.findall("DST (\w+ )")
        .to_list(),
        columns=["Player"],
    )
    lineups = pd.concat(
        [
            qb,
            rb1,
            rb2,
            wr1,
            wr2,
            wr3,
            te,
            flex,
            dst,
        ]
    )
    lineups["lineup_id"] = lineups.index
    df = lineups.merge(
        df[
            [
                "lineup_id",
                "Points",
                "Roster Position",
                "cashLine",
                "contestName",
                "contestKey",
                "singleEntry",
            ]
        ],
        on="lineup_id",
    )
    return df

def exportMilliWinner(season,week):
    df=pd.read_csv(f'{datadir}/MillionaireMakerContestResults/{season}/Week{week}_Millionaire_Results.csv')
    df['lineup_id']=df.index
    players=df[df.Player.isna()==False].groupby('Player').first()[['FPTS','%Drafted']]
    lineups=parseNFLLineup(df)
    winner=lineups[lineups.lineup_id==0]
    winner=winner.merge(players,on='Player',how='left')
    winner=winner[['Player','FPTS','%Drafted']]
    contest_id=df.contestKey.unique()[0]
    dk_data=getDKPlayerData(contest_id)
    dk_data.drop_duplicates(inplace=True)
    winner=winner.merge(dk_data[['Player','team','Position','Salary']],on='Player',how='left')
    headshots=pd.read_csv(f'{datadir}/Projections/{season}/megnia_projections.csv')
    headshots=reformatName(headshots)
    headshots=headshots.groupby('RotoName').first().headshot_url.to_frame()
    winner.rename({'Player':'full_name',
                   'Position':'position'},axis=1,inplace=True)
    winner=reformatName(winner)
    winner=winner.merge(headshots,on='RotoName',how='left')
    winner.rename({'full_name':'Player',
                   'position':'Position',
                   'team':'Team'},axis=1,inplace=True)
    winner.drop('RotoName',axis=1,inplace=True)
    winner.set_index('Player',inplace=True)
    winner['week']=week
    winner['season']=season
    # Export to Database
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
    last_export=pd.read_sql('Milli_Winners',con=sqlEngine)
    last_export=last_export[last_export.week!=week]
    winner=pd.concat([last_export,winner])
    winner.to_sql(
        con=sqlEngine, name="Milli_Winners", if_exists="replace",index=False
    )