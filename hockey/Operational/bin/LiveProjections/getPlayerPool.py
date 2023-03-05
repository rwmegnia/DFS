#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 03:45:38 2022

@author: robertmegnia
"""
import requests
from NHL_API_TOOLS import *
import unidecode
import pandas as pd


def getPlayerPool(season):
    # Get next game on schedule for every team
    print("Retrieving Rosters...")
    url = "https://statsapi.web.nhl.com/api/v1/teams?expand=team.schedule.next"
    response = requests.get(url).json()
    player_df = []
    for team in response["teams"]:
        team_abbrev = team["abbreviation"]
        team_name = team["name"]
        team_id = team["id"]
        game_date = team["nextGameSchedule"]["dates"][0]["date"]
        game_id = team["nextGameSchedule"]["dates"][0]["games"][0]["gamePk"]
        if (
            team_name
            == team["nextGameSchedule"]["dates"][0]["games"][0]["teams"]["away"][
                "team"
            ]["name"]
        ):
            game_location = "away"
            opp = team["nextGameSchedule"]["dates"][0]["games"][0]["teams"]["home"][
                "team"
            ]["name"]
            opp = teamAbbrevsDict[opp]
        else:
            game_location = "home"
            opp = team["nextGameSchedule"]["dates"][0]["games"][0]["teams"]["away"][
                "team"
            ]["name"]
            opp = teamAbbrevsDict[opp]
        print(team_name, team_id)
        roster_url = (
            f"https://statsapi.web.nhl.com/api/v1/teams/{team_id}?expand=team.roster"
        )
        roster = requests.get(roster_url).json()["teams"][0]["roster"]["roster"]
        for player in roster:
            full_name = player["person"]["fullName"]
            player_id = player["person"]["id"]
            position = player["position"]["code"]
            position_type = player["position"]["type"]
            player_frame = pd.DataFrame(
                {
                    "full_name": [full_name],
                    "player_id": [player_id],
                    "position": [position],
                    "position_type": [position_type],
                    "team": [team_abbrev],
                    "game_location": [game_location],
                    "opp": [opp],
                    "game_date": [game_date],
                    "game_id": game_id,
                }
            )
            player_df.append(player_frame)
    players = pd.concat(player_df)
    players = players[players.game_date == players.game_date.min()]
    players.loc[players.position == "L", "position"] = "LW"
    players.loc[players.position == "R", "position"] = "RW"
    ## REFORMAT PLAYER NAMES BY REMOVING NON-ALPHA-NUMERICS
    players["first_name"] = players.full_name.apply(lambda x: x.split(" ")[0])
    players["last_name"] = players.full_name.apply(
        lambda x: " ".join(x.split(" ")[1::])
    )
    # Remove non-alpha numeric characters from first and last names.
    players["first_name"] = players.first_name.apply(
        lambda x: "".join(c for c in x if c.isalnum())
    )
    players["last_name"] = players.last_name.apply(
        lambda x: "".join(c for c in x if c.isalnum())
    )

    # Recreate full_name to fit format "Firstname Lastname"
    players["full_name"] = players.apply(
        lambda x: x.first_name + " " + x.last_name, axis=1
    )
    players["full_name"] = players.full_name.apply(lambda x: x.lower())
    players.drop(["first_name", "last_name"], axis=1, inplace=True)
    players["full_name"] = players.full_name.apply(
        lambda x: x.split(" ")[0][0].upper()
        + x.split(" ")[0][1::]
        + " "
        + x.split(" ")[-1][0].upper()
        + x.split(" ")[-1][1::]
    )
    players["full_name"] = players.full_name.apply(lambda x: unidecode.unidecode(x))
    players["RotoName"] = players.full_name.apply(
        lambda x: x.lower().split(" ")[0][0] + x.lower().split(" ")[1][0:7]
    )

    players["season"] = season
    return players
