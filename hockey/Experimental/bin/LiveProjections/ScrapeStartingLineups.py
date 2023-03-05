#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 13:46:56 2022

@author: robertmegnia
"""
from bs4 import BeautifulSoup as BS
import requests
import pandas as pd
import os
from datetime import datetime
import unidecode
from os.path import exists
basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f"{basedir}/../../data"


def scrapeStartingLineups():
    game_date = datetime.utcnow().strftime("%Y-%m-%d")
    soup = BS(
        requests.get(
            f"https://rotogrinders.com/lineups/nhl?date={game_date}"
        ).content,
        "html.parser",
    )
    players = []
    positions = []
    projections = []
    salaries = []
    ownerships = []
    teams = []
    lines = []
    powerplay = []
    for game in soup.find_all(attrs={"class": "blk crd lineup"}):
        team_names = game.find_all(attrs={"class": "shrt"})
        away_team = team_names[0].text.upper()
        home_team = team_names[1].text.upper()
        away_team_players = game.find_all(attrs={"blk away-team"})[0]
        home_team_players = game.find_all(attrs={"blk home-team"})[0]
        counter = 1
        line_counter = 1
        try:
            players.append(
                away_team_players.find_all(attrs={"class": "nolink name"})[0].text
            )
        except IndexError:
            continue
        positions.append("G")
        projections.append(
            float(
                away_team_players.find_all(attrs={"class": "pitcher players"})[0]
                .find_all(attrs={"class": "fpts"})[0]
                .get("data-fpts")
            )
        )
        salaries.append(
            float(
                away_team_players.find_all(attrs={"class": "pitcher players"})[0]
                .find_all(attrs={"class": "salary"})[0]
                .get("data-salary")
            )
        )
        ownerships.append(
            float(
                away_team_players.find_all(attrs={"class": "pitcher players"})[0]
                .find_all(attrs={"class": "pown"})[0]
                .get("data-pown")
                .split("%")[0]
            )
        )
        teams.append(away_team)
        lines.append(1)
        powerplay.append(True)
        for player in away_team_players.find_all(attrs={"class": "pname"}):
            teams.append(away_team)
            if line_counter <= 3:
                line = 1
            elif (line_counter > 3) & (line_counter <= 6):
                line = 2
            elif (line_counter > 6) & (line_counter <= 9):
                line = 3
            elif (line_counter > 9) & (line_counter <= 12):
                line = 4
            elif (line_counter > 12) & (line_counter <= 14):
                line = 1
            elif (line_counter > 14) & (line_counter <= 16):
                line = 2
            elif (line_counter > 16) & (line_counter <= 18):
                line = 3
            if ("PP1" in player.text) | ("PP2" in player.text):
                powerplay.append(True)
            else:
                powerplay.append(False)
            name = player.text.split("\n")[1]
            position = player.find_all(attrs={"class": "position"})[0].text
            projection = float(
                player.find_all(attrs={"class": "fpts"})[0].get("data-fpts")
            )
            salary = float(
                player.find_all(attrs={"class": "salary"})[0].get("data-salary")
            )
            ownership = float(
                player.find_all(attrs={"class": "pown"})[0]
                .get("data-pown")
                .split("%")[0]
            )
            players.append(name)
            positions.append(position)
            projections.append(projection)
            salaries.append(salary)
            ownerships.append(ownership)
            lines.append(line)
            counter += 1
            line_counter += 1
        counter = 1
        line_counter = 1
        players.append(
            home_team_players.find_all(attrs={"class": "nolink name"})[0].text
        )
        positions.append("G")
        projections.append(
            float(
                home_team_players.find_all(attrs={"class": "pitcher players"})[0]
                .find_all(attrs={"class": "fpts"})[0]
                .get("data-fpts")
            )
        )
        salaries.append(
            float(
                home_team_players.find_all(attrs={"class": "pitcher players"})[0]
                .find_all(attrs={"class": "salary"})[0]
                .get("data-salary")
            )
        )
        ownerships.append(
            float(
                home_team_players.find_all(attrs={"class": "pitcher players"})[0]
                .find_all(attrs={"class": "pown"})[0]
                .get("data-pown")
                .split("%")[0]
            )
        )
        teams.append(home_team)
        lines.append(1)
        powerplay.append(True)
        for player in home_team_players.find_all(attrs={"class": "pname"}):
            teams.append(home_team)
            if line_counter <= 3:
                line = 1
            elif (line_counter > 3) & (line_counter <= 6):
                line = 2
            elif (line_counter > 6) & (line_counter <= 9):
                line = 3
            elif (line_counter > 9) & (line_counter <= 12):
                line = 4
            elif (line_counter > 12) & (line_counter <= 14):
                line = 1
            elif (line_counter > 14) & (line_counter <= 16):
                line = 2
            elif (line_counter > 16) & (line_counter <= 18):
                line = 3
            if ("PP1" in player.text) | ("PP2" in player.text):
                powerplay.append(True)
            else:
                powerplay.append(False)
            name = player.text.split("\n")[1]
            position = player.find_all(attrs={"class": "position"})[0].text
            projection = float(
                player.find_all(attrs={"class": "fpts"})[0].get("data-fpts")
            )
            salary = float(
                player.find_all(attrs={"class": "salary"})[0].get("data-salary")
            )
            ownership = float(
                player.find_all(attrs={"class": "pown"})[0]
                .get("data-pown")
                .split("%")[0]
            )
            players.append(name)
            positions.append(position)
            projections.append(projection)
            salaries.append(salary)
            ownerships.append(ownership)
            lines.append(line)
            counter += 1
            line_counter += 1
    df = pd.DataFrame(
        {
            "RotoName": players,
            "RotoPosition": positions,
            "RG_projection": projections,
            "Salary": salaries,
            "ownership_proj": ownerships,
            "team": teams,
            "line": lines,
            "powerplay": powerplay,
        }
    )
    # Remove ending dots and hyphens
    df["RotoName"] = df.RotoName.apply(lambda x: "".join([c for c in x if c.isalnum()]))
    df["RotoName"] = df.RotoName.apply(lambda x: x.lower())
    df["RotoName"] = df.RotoName.apply(lambda x: x[0:8])
    df["RotoName"] = df.RotoName.apply(lambda x: unidecode.unidecode(x))
    df.RotoName.replace(
        {
            "rnugent": "rnugenth",
            "oekmanl": "oekmanla",
            "jvanrie": "jvanriem",
            "tvanrie": "tvanriem",
            "naubeku": "naubekub",
            "tstuetzl": "tstutzle",
            "zastonr": "zastonre",
        },
        inplace=True,
    )
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
    df["game_date"] = game_date
    df.loc[(df.RotoPosition == "D") & (df.line == 4), "line"] = 1
    if not exists(f'{datadir}/startingLineups/{game_date}/{game_date}_Lineups.csv'):
        os.mkdir(f'{datadir}/startingLineups/{game_date}')
        df.to_csv(f'{datadir}/startingLineups/{game_date}/{game_date}_Lineups.csv',index=False)
    else:
        last_saved=pd.read_csv(f'{datadir}/startingLineups/{game_date}/{game_date}_Lineups.csv')
        last_saved=last_saved[~last_saved.team.isin(df.team)]
        df=pd.concat([df,last_saved])
        df.to_csv(f'{datadir}/startingLineups/{game_date}/{game_date}_Lineups.csv',index=False)
    return df
