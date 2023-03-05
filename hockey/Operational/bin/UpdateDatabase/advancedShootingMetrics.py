#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 01:24:00 2022

@author: robertmegnia


Build a data frame of shots taken in a game

Frame must contain players name, Id, and shot result
"""


def getScoringChance(x, y):
    x = 89 - np.abs(x)
    y = np.abs(y)
    if (x <= 15) & (y <= 10):
        """
            See if coordinates fall in the high chance scoring zone
        """
        return 3
    elif ((x > 15) & (x <= 40)) & (y <= 10):
        return 2
    elif (y <= 20) & ((x > 15) & (x <= 35)):
        return 2
    elif isInside(0, 10, 15, 10, 15, 15, x, y):
        return 2
    else:
        return 1


# Medium chance Rectangle 2
# y<20 and y>-20 & x  & x>=20 & x<35
#
# Triangle coordinate (10,0),(10,20),(20,20)
def area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


def isInside(x1, y1, x2, y2, x3, y3, x, y):

    # Calculate area of triangle ABC
    A = area(x1, y1, x2, y2, x3, y3)

    # Calculate area of triangle PBC
    A1 = area(x, y, x2, y2, x3, y3)

    # Calculate area of triangle PAC
    A2 = area(x1, y1, x, y, x3, y3)

    # Calculate area of triangle PAB
    A3 = area(x1, y1, x2, y2, x, y)

    # Check if sum of A1, A2 and A3
    # is same as A
    if A == A1 + A2 + A3:
        return True
    else:
        return False


import os
from os.path import exists
from datetime import datetime
import pandas as pd
import requests
import numpy as np
from NHL_API_TOOLS import *
from DatabaseTools import *
import sys

basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f"{basedir}/../../data"
API_BASEURL = "https://statsapi.web.nhl.com"
URL = "https://statsapi.web.nhl.com/api/v1/teams"
response = requests.get(URL).json()
team_ids = pd.DataFrame.from_dict(response["teams"])["id"]
today = datetime.now().strftime("%Y-%m-%d")
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import requests


def advancedShootingMetrics(game_data, game_date):
    # Start by creating dataframe that will help determine which end ecah team is on  coordinate wise#
    home_team = game_data["boxscore"]["teams"]["home"]["team"]["triCode"]
    away_team = game_data["boxscore"]["teams"]["away"]["team"]["triCode"]
    plays = game_data["plays"]["allPlays"]
    shots = []
    x_coords = []
    period = []
    team = []
    neutral_zone = np.arange(-25, 26)
    for play in plays:
        if play["about"]["period"] > 4:
            continue
        event = play["result"]["event"]
        if (event in ["Shot", "Goal"]) & ("x" in play["coordinates"].keys()):
            shots.append(1)
            x_coords.append(play["coordinates"]["x"])
        else:
            continue
        period.append(play["about"]["period"])
        if play["team"]["triCode"] == home_team:
            team.append("Home")
        else:
            team.append("Away")
    end_df = pd.DataFrame(
        {"shot": shots, "x": x_coords, "period": period, "team": team}
    )
    end_df = end_df.groupby(["team", "period"], as_index=False).mean()
    frames = []
    player_name = []
    player_id = []
    Corsi = []
    Fenwick = []
    result = []
    rebound = []
    rush = []
    x_coords = []
    y_coords = []
    strength = []
    period_types = []
    shootout_goal = []
    teams = []
    #
    plays = game_data["plays"]["allPlays"]
    stoppage = False
    lastEventTime = None
    lastShotTime = None
    periodTimeLast = datetime.strptime("00:00", "%M:%S")
    for play in plays:
        event = play["result"]["event"]
        period_type = play["about"]["periodType"]
        period = play["about"]["period"]
        if period not in end_df.period.unique():
            continue
        ## Determine if There was a Time Stoppage or if the period is starting
        if event in ["Period Ready", "Stoppage"]:
            stoppage = True
            lastEventTime = None
            lastShotTime = None
            ## Define Defensive Zones if theres a period change
            if play["result"]["event"] == "Period Ready":
                try:
                    X = end_df[
                        (end_df.team == "Away") & (end_df.period == period)
                    ].x.values[0]
                    if X > 0:
                        away_zone = np.arange(-100, -25, 1)
                        home_zone = np.arange(26, 101)
                    else:
                        away_zone = np.arange(26, 101)
                        home_zone = np.arange(-100, -26, 1)
                except IndexError:
                    X = end_df[
                        (end_df.team == "Home") & (end_df.period == period)
                    ].x.values[0]
                    if X > 0:
                        away_zone = np.arange(26, 101)
                        home_zone = np.arange(-100, -26, 1)
                    else:
                        away_zone = np.arange(-100, -25, 1)
                        home_zone = np.arange(26, 101)
        else:
            stoppage = False
        # Determine Time and Location of Event
        eventTime = datetime.strptime(play["about"]["periodTime"], "%M:%S")
        if lastEventTime:
            timeSinceLastEvent = (eventTime - lastEventTime).seconds
        if ("x" in play["coordinates"].keys()) & ("y" in play["coordinates"].keys()):
            x = play["coordinates"]["x"]
            y = play["coordinates"]["y"]
            if x in neutral_zone:
                eventLocation = "NeutralZone"
                lastEventLocation = "NeutralZone"
                lastEventTime = datetime.strptime(play["about"]["periodTime"], "%M:%S")
                continue
            elif x in away_zone:
                eventLocation = "Away"
            else:
                eventLocation = "Home"
        else:
            continue
        if event in ["Shot", "Missed Shot", "Blocked Shot", "Goal"]:
            shotTime = datetime.strptime(play["about"]["periodTime"], "%M:%S")
            # Determine which team shot the puck
            team = play["team"]["triCode"]
            if event != "Blocked Shot":
                team = team
            else:
                if home_team == team:
                    team = away_team
                else:
                    team = home_team
            # If the puck was shot in the neutral zone or defensive zone continiue
            if (
                (eventLocation == "NeturalZone")
                | ((team == away_team) & (eventLocation == "Away"))
                | ((team == home_team) & (eventLocation == "Home"))
            ):
                lastEventLocation = eventLocation
                lastEventTime = eventTime
                continue
            teams.append(team)
            for player in play["players"]:
                if player["playerType"] in ["Shooter", "Scorer"]:
                    # Compute time that has passed since last shot
                    if lastShotTime:
                        timeSinceLastShot = (shotTime - lastShotTime).seconds
                        # If there hasn't been a stoppage and timeSinceLastShot <=3 seconds the shot was a rebound
                        if (
                            (stoppage == False)
                            & (timeSinceLastShot <= 3)
                            & (period_type != "SHOOTOUT")
                        ):
                            rebound.append(1)
                        else:
                            rebound.append(0)
                        # If there hasn't been a stoppage, elapsed_time <=4 and last event was in netural/defensive zone, shot was a rush
                        if (
                            (stoppage == False)
                            & (timeSinceLastEvent <= 4)
                            & (period_type != "SHOOTOUT")
                        ):
                            if lastEventLocation == "NeutralZone":
                                rush.append(1)
                            elif (
                                (team == home_team) & (lastEventLocation == "Home")
                            ) | ((team == away_team) & (lastEventLocation == "Away")):
                                rush.append(1)
                            else:
                                rush.append(0)
                        else:
                            rush.append(0)
                    else:
                        rush.append(0)
                        rebound.append(0)
                    player_name.append(player["player"]["fullName"])
                    player_id.append(player["player"]["id"])
                    result.append(play["result"]["eventTypeId"])
                    if event == "Goal":
                        strength.append(play["result"]["strength"]["name"])
                    else:
                        strength.append("Even")
                    period_types.append(period_type)
                    x_coords.append(play["coordinates"]["x"])
                    y_coords.append(play["coordinates"]["y"])

                    if (event != "Blocked Shot") & (period_type != "SHOOTOUT"):
                        Corsi.append(1)
                        Fenwick.append(1)
                        shootout_goal.append(0)
                    # Add if to Corsi fi not a Fenwick shot
                    elif period_type != "SHOOTOUT":
                        Corsi.append(1)
                        Fenwick.append(0)
                        shootout_goal.append(0)
                    # Add to shootout goals
                    elif (period_type == "SHOOTOUT") & (event == "Goal"):
                        Corsi.append(0)
                        Fenwick.append(0)
                        shootout_goal.append(1)
                    # Add Nothing
                    else:
                        Corsi.append(0)
                        Fenwick.append(0)
                        shootout_goal.append(0)
            lastShotTime = shotTime
        lastEventTime = eventTime
        lastEventLocation = eventLocation
        lastEvent = event

    df = pd.DataFrame(
        {
            "full_name": player_name,
            "player_id": player_id,
            "team": teams,
            "Corsi": Corsi,
            "Fenwick": Fenwick,
            "result": result,
            "rebound": rebound,
            "rush": rush,
            "shootout_goal": shootout_goal,
            "x": x_coords,
            "y": y_coords,
            "strength": strength,
            "period_type": period_types,
        }
    )
    df["game_date"] = game_date
    df["scoringChance"] = df.apply(lambda x: getScoringChance(x.x, x.y), axis=1)
    df["scoringChance"] = df[["scoringChance", "rush", "rebound"]].sum(axis=1)
    df.loc[(df.period_type != "SHOOTOUT") & (df.scoringChance >= 3), "HDSC"] = 1
    df.loc[(df.HDSC == 1) & (df.result == "GOAL"), "HD_goals"] = 1
    df.loc[(df.period_type != "SHOOTOUT") & (df.scoringChance == 2), "MDSC"] = 1
    df.loc[(df.MDSC == 1) & (df.result == "GOAL"), "MD_goals"] = 1
    df.loc[(df.period_type != "SHOOTOUT") & (df.scoringChance == 1), "LDSC"] = 1
    df.loc[(df.LDSC == 1) & (df.result == "GOAL"), "LD_goals"] = 1
    df = (
        df[
            [
                "player_id",
                "Corsi",
                "Fenwick",
                "shootout_goal",
                "scoringChance",
                "HDSC",
                "HD_goals",
                "MDSC",
                "MD_goals",
                "LDSC",
                "LD_goals",
            ]
        ]
        .groupby("player_id", as_index=False)
        .sum()
    )
    return df
