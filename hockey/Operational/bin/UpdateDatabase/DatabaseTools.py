#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 15:50:18 2022

@author: robertmegnia
"""
import pandas as pd
from NHL_API_TOOLS import *
import numpy as np
import os
import unidecode

basedir = os.path.dirname(os.path.abspath(__file__))
DK_dir = f"{basedir}/../../../../DraftKingsMegaDatabase/NHL"


def getOT(home_score, away_score):
    if home_score == away_score:
        return True
    else:
        return False


def getPlayerStats(location, player_type, boxscore, OT):
    player_frames = []
    team_name = boxscore[location]["team"]["name"]
    team_abbrev = teamAbbrevsDict[team_name]
    if location == "home":
        opp_name = boxscore["away"]["team"]["name"]
        opp_abbrev = teamAbbrevsDict[opp_name]
    else:
        opp_name = boxscore["home"]["team"]["name"]
        opp_abbrev = teamAbbrevsDict[opp_name]
    player_IDs = boxscore[location][player_type]
    for player in player_IDs:
        if player in boxscore[location]["scratches"]:
            continue
        full_name = boxscore[location]["players"][f"ID{player}"]["person"]["fullName"]
        position = boxscore[location]["players"][f"ID{player}"]["position"][
            "abbreviation"
        ]
        position_type = boxscore[location]["players"][f"ID{player}"]["position"]["type"]
        # Only care about skaters, pass on goalies
        if (player_type == "skaters") & (position == "G"):
            continue
        stats = pd.DataFrame.from_dict(
            boxscore[location]["players"][f"ID{player}"]["stats"]
        ).T.reset_index(drop=True)
        stats["full_name"] = full_name
        stats["player_id"] = player
        stats["position"] = position
        stats["position_type"] = position_type
        stats["team"] = team_abbrev
        stats["game_location"] = location
        stats["opp"] = opp_abbrev
        # Convert Time on Ice columns to float type values
        stats["minutes"] = stats.timeOnIce.apply(lambda x: float(x.split(":")[0]))
        stats["seconds"] = stats.timeOnIce.apply(lambda x: float(x.split(":")[1]))
        stats["timeOnIce"] = stats.minutes + (stats.seconds / 60)
        if player_type == "skaters":
            stats["minutes"] = stats.powerPlayTimeOnIce.apply(
                lambda x: float(x.split(":")[0])
            )
            stats["seconds"] = stats.powerPlayTimeOnIce.apply(
                lambda x: float(x.split(":")[1])
            )
            stats["powerPlayTimeOnIce"] = stats.minutes + (stats.seconds / 60)
            stats["minutes"] = stats.shortHandedTimeOnIce.apply(
                lambda x: float(x.split(":")[0])
            )
            stats["seconds"] = stats.shortHandedTimeOnIce.apply(
                lambda x: float(x.split(":")[1])
            )
            stats["shortHandedTimeOnIce"] = stats.minutes + (stats.seconds / 60)
            stats.drop(["minutes", "seconds"], axis=1, inplace=True)
        #
        if player_type == "goalies":
            if not hasattr(stats, "decision"):
                continue
            if stats.decision.values[0] == "":
                stats.decision = 0
            if OT == True:
                if stats.decision.values[0] == "L":
                    stats.decision = "OTL"
            stats["even_goals_allowed"] = stats.evenShotsAgainst - stats.evenSaves
            stats["powerPlay_goals_allowed"] = (
                stats.powerPlayShotsAgainst - stats.powerPlaySaves
            )
            stats["shortHanded_goals_allowed"] = (
                stats.shortHandedShotsAgainst - stats.shortHandedSaves
            )
            stats["goals_allowed"] = stats[
                [
                    "even_goals_allowed",
                    "powerPlay_goals_allowed",
                    "shortHanded_goals_allowed",
                ]
            ].sum(axis=1)
            stats["saves"] = stats[
                ["evenSaves", "powerPlaySaves", "shortHandedSaves"]
            ].sum(axis=1)
            try:
                stats["savePercentage"] = stats.saves / stats.shots
            except ZeroDivisionError:
                stats["savePercentage"] = np.nan
        stats["DKPts"] = getDKPts(stats, player_type)
        player_frames.append(stats)
    df = pd.concat(player_frames)
    return df


def getPlayerSalaries(players, game_date):
    salaryFiles = os.listdir(f"{DK_dir}/{game_date}/Classic")
    frames = []
    for file in salaryFiles:
        df = pd.read_csv(f"{DK_dir}/{game_date}/{file}")
        frames.append(df)
    df = pd.concat(frames)
    df = df.groupby(["Name", "Position", "TeamAbbrev"], as_index=False).first()
    df.rename({"TeamAbbrev": "team", "Position": "position"}, axis=1, inplace=True)
    df.team.replace(
        {
            "NJ": "NJD",
            "MON": "MTL",
            "WAS": "WSH",
            "SJ": "SJS",
            "LA": "LAK",
            "TB": "TBL",
            "CLS": "CBJ",
            "ANH": "ANA",
        },
        inplace=True,
    )
    # Remove Alphanumerics and accents from Names so we can merge effectively
    df["Name"] = df.Name.apply(lambda x: x.lower())
    df["Name"] = df.Name.apply(lambda x: "".join([a for a in x if a.isalnum()]))
    df["Name"] = df.Name.apply(lambda x: unidecode.unidecode(x))

    #
    players["Name"] = players.full_name.apply(lambda x: x.lower())
    players["Name"] = players.Name.apply(
        lambda x: "".join([a for a in x if a.isalnum()])
    )
    players["Name"] = players.Name.apply(lambda x: unidecode.unidecode(x))

    players = players.drop("position", axis=1).merge(
        df[["Name", "team", "Salary", "position"]], on=["Name", "team"], how="left"
    )
    players.drop("Name", axis=1, inplace=True)
    return players
