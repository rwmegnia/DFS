#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:13:42 2023

@author: robertmegnia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:13:46 2022

@author: robertmegnia
"""
import pandas as pd
import os
import time
from os.path import exists
import numpy as np
import requests
import sys
from unidecode import unidecode


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
    salaries.draftkings_name.replace(
        {
            "Robbie Anderson": "Robby Anderson",
            "Mitch Trubisky": "Mitchell Trubisky",
            "Joshua Dobbs": "Josh Dobbs",
            "Deonte Harty": "Deonte Harris",
            "Dan Brown": "Daniel Brown",
            "J.J. Arcega-Whiteside":"JJ Arcega-Whiteside",
            "Commanders ":"WAS Football",
            "Van Jefferson Jr.": "Van Jefferson",
            "Gabe Davis":"Gabriel Davis",
            "Bisi Johnson":"Olabisi Johnson",
        },
        inplace=True,
    )

    # merge salaries with draftGroupInfo
    salaries = salaries.merge(
        draftGroupInfo, on=["draftkings_name", "team", "Salary"], how="left"
    )
    salaries['contest_id']=contest_id
    # converting player_id to int may raise exception if there was a merging
    # mismatch and one of the player_ids were filled as NaN
    # try:
    #     salaries["draftkings_player_id"] = salaries.draftkings_player_id.astype(
    #         int
    #     )
    # except:
    #     mismatch = salaries[
    #         salaries.draftkings_player_id.isna() == True
    #     ].draftkings_name.unique()[0]
    #     print(f"Mismatch in playername {mismatch}")
    #     sys.exit()

    return salaries


def parseMLBLineup(df):
    pitchers = pd.DataFrame(
        df[df.Lineup.isna() == False]
        .Lineup.str.findall("P ([A-Za-z\.?'-]+ [A-Za-z\.\?'-]+)")
        .to_list(),
        columns=["P", "P"],
    )
    p1 = pitchers.iloc[:, 0].to_frame()
    p1.rename({"P": "Player"}, axis=1, inplace=True)
    p2 = pitchers.iloc[:, 1].to_frame()
    p2.rename({"P": "Player"}, axis=1, inplace=True)
    catchers = pd.DataFrame(
        df[df.Lineup.isna() == False]
        .Lineup.str.findall("C ([A-Za-z\.\?'-]+ [A-Za-z\.\?'-]+)")
        .to_list(),
        columns=["C"],
    )
    catchers.rename({"C": "Player"}, axis=1, inplace=True)
    outfielders = pd.DataFrame(
        df[df.Lineup.isna() == False]
        .Lineup.str.findall("OF ([A-Za-z\.\?'-]+ [A-Za-z\.\?'-]+)")
        .to_list(),
        columns=["OF", "OF", "OF"],
    )
    of1 = outfielders.iloc[:, 0].to_frame()
    of2 = outfielders.iloc[:, 1].to_frame()
    of3 = outfielders.iloc[:, 2].to_frame()
    of1.rename({"OF": "Player"}, axis=1, inplace=True)
    of2.rename({"OF": "Player"}, axis=1, inplace=True)
    of3.rename({"OF": "Player"}, axis=1, inplace=True)
    first_base = pd.DataFrame(
        df[df.Lineup.isna() == False]
        .Lineup.str.findall("1B ([A-Za-z\.\?'-]+ [A-Za-z\.\?'-]+)")
        .to_list(),
        columns=["1B"],
    )
    first_base.rename({"1B": "Player"}, axis=1, inplace=True)
    second_base = pd.DataFrame(
        df[df.Lineup.isna() == False]
        .Lineup.str.findall("2B ([A-Za-z\.\?'-]+ [A-Za-z\.\?'-]+)")
        .to_list(),
        columns=["2B"],
    )
    second_base.rename({"2B": "Player"}, axis=1, inplace=True)
    third_base = pd.DataFrame(
        df[df.Lineup.isna() == False]
        .Lineup.str.findall("3B ([A-Za-z\.\?'-]+ [A-Za-z\.\?'-]+)")
        .to_list(),
        columns=["3B"],
    )
    third_base.rename({"3B": "Player"}, axis=1, inplace=True)
    short_stops = pd.DataFrame(
        df[df.Lineup.isna() == False]
        .Lineup.str.findall("SS ([A-Za-z\.\?'-]+ [A-Za-z\.\?'-]+)")
        .to_list(),
        columns=["SS"],
    )
    short_stops.rename({"SS": "Player"}, axis=1, inplace=True)
    lineups = pd.concat(
        [
            p1,
            p2,
            catchers,
            of1,
            of2,
            of3,
            first_base,
            second_base,
            third_base,
            short_stops,
        ]
    )
    lineups["lineup_id"] = lineups.index
    df = lineups.merge(
        df[
            [
                "lineup_id",
                "Points",
                "Roster Position",
                "%Drafted",
                "FPTS",
                "Prize",
                "cashLine",
                "contestName",
                "contestKey",
                "singleEntry",
            ]
        ],
        on="lineup_id",
    )
    return df


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
                "%Drafted",
                "FPTS",
                "Prize",
                "cashLine",
                "contestName",
                "contestKey",
                "singleEntry",
            ]
        ],
        on="lineup_id",
    )
    return df

def parseNHLLineup(df):
    cs = pd.DataFrame(
        df[df.Lineup.isna() == False]
        .Lineup.str.findall("C ([A-Za-z\.'-]+ [A-Za-z\.'-]+)")
        .to_list(),
        columns=["Player", "Player"],
    )
    c1 = cs.iloc[:, 0].to_frame()
    c2 = cs.iloc[:, 1].to_frame()
    c1['Roster Position']='C'
    c2['Roster Position']='C'
    ws = pd.DataFrame(
        df[df.Lineup.isna() == False]
        .Lineup.str.findall("W ([A-Za-z\.'-]+ [A-Za-z\.'-]+)")
        .to_list(),
        columns=["Player", "Player", "Player"],
    )
    w1 = ws.iloc[:, 0].to_frame()
    w2 = ws.iloc[:, 1].to_frame()
    w3 = ws.iloc[:, 2].to_frame()
    w1['Roster Position']='W'
    w2['Roster Position']='W'
    w3['Roster Position']='W'
    ds = pd.DataFrame(
        df[df.Lineup.isna() == False]
        .Lineup.str.findall("D ([A-Za-z\.'-]+ [A-Za-z\.'-]+)")
        .to_list(),
        columns=["Player", "Player"],
    )
    d1 = ds.iloc[:, 0].to_frame()
    d2 = ds.iloc[:, 1].to_frame()
    d1['Roster Position']='D'
    d2['Roster Position']='D'
    g = pd.DataFrame(
        df[df.Lineup.isna() == False]
        .Lineup.str.findall("G ([A-Za-z\.'-]+ [A-Za-z\.'-]+)")
        .to_list(),
        columns=["Player"],
    )
    g['Roster Position']='G'
    util = pd.DataFrame(
        df[df.Lineup.isna() == False]
        .Lineup.str.findall("UTIL ([A-Za-z\.'-]+ [A-Za-z\.'-]+)")
        .to_list(),
        columns=["Player"],
    )
    util['Roster Position']='Util'
    lineups = pd.concat(
        [
            c1,
            c2,
            w1,
            w2,
            w3,
            d1,
            d2,
            g,
            util,
        ]
    )
    lineups["lineup_id"] = lineups.index
    df = lineups.merge(
        df[
            [
                "lineup_id",
                "Points",
                "%Drafted",
                "FPTS",
                "Prize",
                "cashLine",
                "contestName",
                "contestKey",
                "singleEntry",
                "Entrants",
                "payouts",
                "total_teams"
            ]
        ],
        on="lineup_id",
    )
    return df

#%%
frames=[]
game_dates = os.listdir('/Volumes/XDrive/DFS/contestResults/NHL')
for game_date in game_dates:
    print(game_date)
    files=os.listdir(f'/Volumes/XDrive/DFS/contestResults/NHL/{game_date}')
    for file in files:
        if 'details' in file:
            df=pd.read_csv(f'/Volumes/XDrive/DFS/contestResults/NHL/{game_date}/{file}')
            if len(df[(df.contestName.str.contains('Double Up')==True)&(df.contestName.str.contains('Single Entry')==True)])>0:
                players = df[df.Player.isna() == False]
                players.loc[players["%Drafted"].isna() == True, "%Drafted"] = "0%"
                players["ownership"] = players["%Drafted"].apply(
                    lambda x: float(x.split("%")[0])
                )
                players = players[["Player", "contestKey", "ownership"]]
                players["Player"] = players.Player.apply(lambda x: " ".join(x.split(" ")[0:2]))
                df = df.groupby("Lineup", as_index=False).first()
                df.sort_values(by='Points',ascending=False,inplace=True)
                df.reset_index(drop=True, inplace=True)
                df["lineup_id"] = df.index
                df.loc[df.Points > df.cashLine, "winning"] = True
                df.loc[df.Points <= df.cashLine, "winning"] = False
                df = parseNHLLineup(df)
                df = df.merge(players, on=["Player", "contestKey"], how="left")
                playerPool = getDKPlayerData(df.contestKey.values[0])
                playerPool=playerPool.groupby('draftkings_player_id',as_index=False).first()
                playerPool.rename({'draftkings_name':'Player'},axis=1,inplace=True)
                playerPool.Player.replace({'Calvin de Haan':'Calvin de'},inplace=True)
                df = df.merge(playerPool[['Player','team','Position','Salary','draftkings_player_id']])
                df['game_date']=game_date
                frames.append(df)
            else:
                continue
lineups_df=pd.concat(frames)
allLineups=lineups_df.groupby(['lineup_id','contestKey']).size()
allLineups.name='playersInLineup'
allLineups=allLineups.to_frame()
allLineups.reset_index(inplace=True)
lineups_df=lineups_df.merge(allLineups,on=['lineup_id','contestKey'],how='left')
# lineups_df=lineups_df[lineups_df.playersInLineup==9]
#%%
frames=[]
game_dates = os.listdir('/Volumes/XDrive/DFS/hockey/Experimental/data/Projections/RealTime/2022/Classic')
for game_date in game_dates:
    files = os.listdir(f'/Volumes/XDrive/DFS/hockey/Experimental/data/Projections/RealTime/2022/Classic/{game_date}')
    for file in files:
        if '.csv' in file:
            df=pd.read_csv(f'/Volumes/XDrive/DFS/hockey/Experimental/data/Projections/RealTime/2022/Classic/{game_date}/{file}')
            frames.append(df)
