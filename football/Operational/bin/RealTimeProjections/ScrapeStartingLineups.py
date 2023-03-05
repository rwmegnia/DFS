#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 06:55:27 2022

@author: robertmegnia
"""
from bs4 import BeautifulSoup as BS
import requests
import pandas as pd
import os
from datetime import datetime
import unidecode
import numpy as np

basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f"{basedir}/../../data"

def reformatNames(df):
    df = df[df.RotoName != " "]
    #     # Remove ending dots and hyphens
    ## REFORMAT PLAYER NAMES BY REMOVING NON-ALPHA-NUMERICS
    df["first_name"] = df.RotoName.apply(lambda x: x.split(" ")[0])
    df["last_name"] = df.RotoName.apply(lambda x: " ".join(x.split(" ")[1::]))

    # Remove non-alpha numeric characters from first/last names
    df["first_name"] = df.first_name.apply(
        lambda x: "".join(c for c in x if c.isalnum())
    )
    df["last_name"] = df.last_name.apply(
        lambda x: "".join(c for c in x if c.isalnum())
    )

    # Recreate full_name to fit format "Firstname Lastname" with no accents
    df["full_name"] = df.apply(
        lambda x: " ".join([x.first_name, x.last_name]), axis=1
    )
    df["full_name"] = df.full_name.apply(lambda x: x.lower())
    df.drop(["first_name", "last_name"], axis=1, inplace=True)
    df["full_name"] = df.full_name.apply(
        lambda x: x.split(" ")[0][0].upper()
        + x.split(" ")[0][1::]
        + " "
        + x.split(" ")[-1][0].upper()
        + x.split(" ")[-1][1::] if len(x.split(" ")[0])>0 else np.nan
    )
    df=df[df.full_name.isna()==False]
    df["full_name"] = df.full_name.apply(lambda x: unidecode.unidecode(x))

    # Create Column to match with RotoGrinders
    df["RotoName"] = df.full_name.apply(
        lambda x: x.lower().split(" ")[0][0:4] + x.lower().split(" ")[1][0:5]
    )
    # If there are still discrepancies between name formats after reformatting
    # you can change them here
    # df.RotoName.replace(
    #     {"vinccapra": "vinncapra", "natelowe": "nathlowe"}, inplace=True
    # )
    return df

def scrapeStartingLineups(week,season):
    game_date = datetime.now().strftime("%Y-%m-%d")
    soup = BS(
        requests.get(f"https://rotogrinders.com/lineups/nfl").content,
        "html.parser",
    )
    players = []
    positions = []
    projections = []
    ownerships = []
    away_teams = []
    home_teams = []
    player_away_teams = []
    player_home_teams = []
    teams = []
    opps = []
    for game in soup.find_all(attrs={"data-role": "lineup-card"}):
        team_names = game.find_all(attrs={"class": "shrt"})
        away_team = (
            game.find_all(attrs={"class": "blk crd lineup"})[0]
            .find_all(attrs={"class": "shrt"})[0]
            .text
        )
        home_team = (
            game.find_all(attrs={"class": "blk crd lineup"})[0]
            .find_all(attrs={"class": "shrt"})[1]
            .text
        )
        away_teams.append(away_team)
        home_teams.append(home_team)

    for index, tm in enumerate(away_teams):
        away_team_players = soup.find_all(attrs={"blk away-team nfl"})[index]
        for player in away_team_players.find_all(attrs={"class": "player"}):
            teams.append(tm)
            opps.append(home_teams[index])
            name = player.find_all(attrs={"class": "pname"})[0].text.split(
                "\n"
            )[1]
            position = player.get("data-pos")
            try:
                ownership = float(
                    player.find_all(attrs={"class": "pown"})[0]
                    .get("data-pown")
                    .split("%")[0]
                )
            except:
                ownership=0
            try:
                projection = float(
                    player.find_all(attrs={"class": "fpts"})[0].get("data-fpts")
                )
            except:
                projection = np.nan

            players.append(name)
            positions.append(position)
            ownerships.append(ownership)
            projections.append(projection)
    for index, ht in enumerate(home_teams):
        home_team_players = soup.find_all(attrs={"blk home-team nfl"})[index]
        for player in home_team_players.find_all(attrs={"class": "player"}):
            teams.append(ht)
            opps.append(away_teams[index])
            name = player.find_all(attrs={"class": "pname"})[0].text.split(
                "\n"
            )[1]
            position = player.get("data-pos")
            try:
                ownership = float(
                    player.find_all(attrs={"class": "pown"})[0]
                    .get("data-pown")
                    .split("%")[0]
                )
            except:
                ownership = 0
            try:
                projection = float(
                    player.find_all(attrs={"class": "fpts"})[0].get("data-fpts")
                )
            except:
                projection = np.nan
            players.append(name)
            positions.append(position)
            ownerships.append(ownership)
            projections.append(projection)
    df = pd.DataFrame(
        {
            "RotoName": players,
            "position": positions,
            "ownership_proj": ownerships,
            "RG_projection": projections,
            "team": teams,
            "opp": opps,
        }
    )
    df["game_date"] = game_date
    df.team.replace(
        {
            "NOS": "NO",
            "NEP": "NE",
            "JAC": "JAX",
            "SFO": "SF",
            "LVR": "LV",
            "GBP": "GB",
            "KCC":"KC",
            "TBB":"TB",
            "LAR":"LA",
            
        },
        inplace=True,
    )
    df.opp.replace(
        {
            "NOS": "NO",
            "NEP": "NE",
            "JAC": "JAX",
            "SFO": "SF",
            "LVR": "LV",
            "GBP": "GB",
            "KCC":"KC",
            "TBB":"TB",
            "LAR":"LA",
        },
        inplace=True,
    )
    # If you want to exclude kickers
    # df=df[df.position!='K']
    # If you don't want to do any remformatting you can return the df here.
    # Continuing beyond here will createa  new column called RotoName that will
    # convert a players name to the first 4 letters of their first name and last 5
    # letters of their last name i.e. Patrick Mahomes - patrmahom. I use this column
    # for merging purposes with other dfs/projections where the names for some players 
    # might be formatted a little differently. 

    df = reformatNames(df)
    try:
        old_df=pd.read_csv(f"{datadir}/StartingLineupsRotoGrinders/{season}_Week{week}_StartingLineups.csv")
        df=pd.concat([old_df,df])
        df=df.groupby(['full_name','position','team'],as_index=False).last()
    except FileNotFoundError:
        pass
    df.to_csv(
        f"{datadir}/StartingLineupsRotoGrinders/{season}_Week{week}_StartingLineups.csv",
        index=False,
    )
    return df