#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 03:45:38 2022

@author: robertmegnia
"""
import requests
from MLB_API_TOOLS import *
import unidecode
import pandas as pd
import pytz
est = pytz.timezone('US/Eastern')
teamAbbrevsDict = {
    133: "OAK",
    134: "PIT",
    135: "SD",
    136: "SEA",
    137: "SF",
    138: "STL",
    139: "TB",
    140: "TEX",
    141: "TOR",
    142: "MIN",
    143: "PHI",
    144: "ATL",
    145: "CWS",
    146: "MIA",
    147: "NYY",
    158: "MIL",
    108: "LAA",
    109: "ARI",
    110: "BAL",
    111: "BOS",
    112: "CHC",
    113: "CIN",
    114: "CLE",
    115: "COL",
    116: "DET",
    117: "HOU",
    118: "KC",
    119: "LAD",
    120: "WSH",
    121: "NYM",
    159:"ALA",
    160:'NLA',
}


def getPlayerPool(game_date, season):
    # Get next game on schedule for every team
    print("Retrieving Rosters...")
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={game_date}"
    response = requests.get(url).json()
    player_df = []
    for game_day in response["dates"]:
        if game_date != game_day["date"]:
            continue
        for game in game_day["games"]:
            for TEAM in ["home", "away"]:
                team = game["teams"][TEAM]
                team_id = team["team"]["id"]
                team_name = team["team"]["name"]
                team_abbrev = teamAbbrevsDict[team_id]
                game_id = game["gamePk"]
                game_time=pd.to_datetime(game['gameDate'])
                game_time=game_time.astimezone(est).strftime('%I:%M%p')
                if team_name == game["teams"]["away"]["team"]["name"]:
                    game_location = "away"
                    opp = game["teams"]["home"]["team"]["name"]
                    opp_id = game["teams"]["home"]["team"]["id"]
                    opp = teamAbbrevsDict[opp_id]
                else:
                    game_location = "home"
                    opp_id = game["teams"]["away"]["team"]["id"]
                    opp = teamAbbrevsDict[opp_id]
                print(team_name, team_id)
                roster_url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=fullRoster"
                # roster_url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=40Man"

                roster = requests.get(roster_url).json()["roster"]
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
                            "game_time":game_time,
                        }
                    )
                    player_df.append(player_frame)
    players = pd.concat(player_df)
    players = players[players.game_date == players.game_date.min()]
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
    # Create Column to match with RotoGrinders
    players["RotoName"] = players.full_name.apply(
        lambda x: x.lower().split(" ")[0][0:4] + x.lower().split(" ")[1][0:5]
    )
    players.RotoName.replace({'bobbwittj':'bobbwitt',
                              'joshhsmit':'joshsmith',
                              'hoypark':'hoyjunpa',
                              'lamowadej':'lamowade',
                              'donowalto':'donnwalto',
                              'tjfried':'terrfried'
                              },inplace=True)
    players["season"] = season
    return players
