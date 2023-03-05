#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 15:50:18 2022

@author: robertmegnia
"""
import pandas as pd
from MLB_API_TOOLS import *
import numpy as np
import os
import unidecode

basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f"{basedir}/../../data"
DK_dir = f"{basedir}/../../../../DraftKingsSalaryMegaDatabase2/MLB"
pitch_hand_db = pd.read_csv(f"{datadir}/game_logs/PitcherThrowingHands.csv")
bat_side_db = pd.read_csv(f"{datadir}/game_logs/BatterBatSide.csv")
player_URL = "https://statsapi.mlb.com/api/v1/people"


def getHand(player_id, stat_type):
    response = requests.get(f"{player_URL}/{player_id}").json()
    return response["people"][0][stat_type]["code"]


def getPlayerStats(location, player_type, boxscore):
    if player_type == "batters":
        stat_type = "batting"
    else:
        stat_type = "pitching"
    player_frames = []
    team_name = boxscore[location]["team"]["id"]
    team_abbrev = teamAbbrevsDict[team_name]
    if location == "home":
        opp_name = boxscore["away"]["team"]["id"]
        opp_abbrev = teamAbbrevsDict[opp_name]
    else:
        opp_name = boxscore["home"]["team"]["id"]
        opp_abbrev = teamAbbrevsDict[opp_name]
    player_IDs = boxscore[location][player_type]
    for player in player_IDs:
        full_name = boxscore[location]["players"][f"ID{player}"]["person"][
            "fullName"
        ]
        position = boxscore[location]["players"][f"ID{player}"]["position"][
            "abbreviation"
        ]
        position_type = boxscore[location]["players"][f"ID{player}"][
            "position"
        ]["type"]
        try:
            order = boxscore[location]["battingOrder"].index(player) + 1
        except ValueError:
            order = -1
        # Only care about batters, pass on pitchers
        if (player_type == "batters") & (position == "P") & (full_name!= 'Harold Castro'):
            continue
        stats = pd.DataFrame(
            [boxscore[location]["players"][f"ID{player}"]["stats"][stat_type]]
        )
        if player_type == "batters":
            stats["singles"] = stats.hits - stats[
                ["homeRuns", "doubles", "triples"]
            ].sum(axis=1)
        stats["full_name"] = full_name
        stats["player_id"] = player
        stats["position"] = position
        stats["position_type"] = position_type
        stats["team"] = team_abbrev
        stats["game_location"] = location
        stats["opp"] = opp_abbrev
        stats["order"] = order
        stats["DKPts"] = getDKPts(stats, player_type)
        if stat_type == "pitching":
            if int(player) not in pitch_hand_db.player_id.to_list():
                handedness = getHand(player, "pitchHand")
            else:
                handedness = pitch_hand_db[
                    pitch_hand_db.player_id == player
                ].throws.values[0]
            stats["throws"] = handedness
        else:
            stats.drop(
                [
                    "singles",
                    "doubles",
                    "triples",
                    "homeRuns",
                    "baseOnBalls",
                    "hitByPitch",
                    "strikeOuts",
                    "intentionalWalks",
                    "flyOuts",
                    "groundOuts",
                    "sacFlies",
                    "sacBunts",
                    "rbi",
                ],
                axis=1,
                inplace=True,
            )
        player_frames.append(stats)
    if len(player_frames) == 0:
        return None
    df = pd.concat(player_frames)
    return df


def getPlayerSalaries(players, game_date):
    salaryFiles = os.listdir(f"{DK_dir}/Classic/{game_date}")
    frames = []
    for file in salaryFiles:
        df = pd.read_csv(f"{DK_dir}/Classic/{game_date}/{file}")
        frames.append(df)
    df = pd.concat(frames)
    df = df.groupby(["full_name","team","position"], as_index=False).first()
    df.team.replace(
        {
            "WAS": "WSH",
        },
        inplace=True,
    )
    df['Name'] = df.full_name
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
        df[["Name", "team", "Salary", "position", "Roster Position"]],
        on=["Name", "team"],
        how="left",
    )
    players.drop("Name", axis=1, inplace=True)
    return players


def getWOBA(df, season):
    df["uBB"] = df.baseOnBalls - df.intentionalWalks
    woba_db = pd.read_csv(f"{datadir}/wOBA_weights_database.csv")
    wBB = woba_db[woba_db.Season == season]["wBB"].values[0]
    wHBP = woba_db[woba_db.Season == season]["wHBP"].values[0]
    w1B = woba_db[woba_db.Season == season]["w1B"].values[0]
    w2B = woba_db[woba_db.Season == season]["w2B"].values[0]
    w3B = woba_db[woba_db.Season == season]["w3B"].values[0]
    wHR = woba_db[woba_db.Season == season]["wHR"].values[0]
    df["wOBA"] = (
        (df.uBB * wBB)
        + (df.hitByPitch * wHBP)
        + (df.singles * w1B)
        + (df.doubles * w2B)
        + (df.triples * w3B)
        + (df.homeRuns * wHR)
    ) / (df.atBats + df.uBB + df.sacFlies + df.hitByPitch)
    df.drop("uBB", axis=1, inplace=True)
    return df


def getExWOBA(df, season):
    df["uBB"] = df.baseOnBalls - df.intentionalWalks
    woba_db = pd.read_csv(f"{datadir}/wOBA_weights_database.csv")
    wBB = woba_db[woba_db.Season == season]["wBB"].values[0]
    wHBP = woba_db[woba_db.Season == season]["wHBP"].values[0]
    w1B = woba_db[woba_db.Season == season]["w1B"].values[0]
    w2B = woba_db[woba_db.Season == season]["w2B"].values[0]
    w3B = woba_db[woba_db.Season == season]["w3B"].values[0]
    wHR = woba_db[woba_db.Season == season]["wHR"].values[0]
    df["exWOBA"] = (
        (df.uBB * wBB)
        + (df.hitByPitch * wHBP)
        + (df.probSingle * w1B)
        + (df.probDouble * w2B)
        + (df.probTriple * w3B)
        + (df.probHomeRun * wHR)
    ) / (df.plateAppearances)
    df.drop("uBB", axis=1, inplace=True)
    return df


def getAverages(df):
    # Compute Batting Average, OBP, OPS, ISO
    df["AVG"] = df.hits / df.atBats
    df["OBP"] = df[["hits", "baseOnBalls", "hitByPitch"]].sum(axis=1) / df[
        ["atBats", "baseOnBalls", "hitByPitch", "sacFlies"]
    ].sum(axis=1)
    df["SLG"] = (
        df["singles"]
        + (2 * df["doubles"])
        + (3 * df["triples"])
        + (4 * df.homeRuns)
    ) / df.atBats
    df["OPS"] = df.OBP + df.SLG
    df["ISO"] = df.SLG - df.AVG
    return df
