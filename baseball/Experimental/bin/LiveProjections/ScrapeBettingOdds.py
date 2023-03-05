#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 13:34:18 2021

@author: robertmegnia

Scrape Rotogrinders for Spreads and Over/Under
"""
from bs4 import BeautifulSoup as BS
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import os

basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f"{basedir}/../../data"


def ScrapeBettingOdds():
    game_date = datetime.now().strftime("%Y-%m-%d")
    teams = pd.DataFrame(columns=["team"])
    times = pd.DataFrame(columns=['game_time'])
    soup = BS(requests.get("https://rotogrinders.com/mlb/odds").content, "html.parser")
    # Scrape Teams
    for link in soup.find_all(attrs={"class":'time'}):
        try:
            home_time=link.time.text+' '+link.strong.text
            home_time=''.join(home_time.split(' ')[0:2])
            away_time=link.time.text+' '+link.strong.text
            away_time=''.join(away_time.split(' ')[0:2])
            times=times.append(pd.DataFrame({'game_time':[away_time]}),ignore_index=True)
            times=times.append(pd.DataFrame({'game_time':[home_time]}),ignore_index=True)
        except:
            pass
    for link in soup.find_all(attrs={"class": "row game"}):
        home = link.get("data-team-home")
        away = link.get("data-team-away")
        teams = teams.append(pd.DataFrame({"team": [away]}), ignore_index=True)
        teams = teams.append(pd.DataFrame({"team": [home]}), ignore_index=True)
    team_frames = []
    books = len(
        soup.find_all(attrs={"class": "sb data card-data", "data-type": "total"})
    ) / (len(teams) / 2)
    team_frames = []
    for n in range(0, int(books)):
        team_frames.append(teams)
    teams = pd.concat(team_frames)
    teams.reset_index(drop=True, inplace=True)
    # Scrape Totals
    i = 0
    total = pd.DataFrame(columns=["total"])
    for link in soup.find_all(
        attrs={"class": "sb data card-data", "data-type": "total"}
    ):
        if "o" in link.span.get_text():
            OU = float(link.span.get_text().split("o")[1].split("\n")[0])

        elif "u" in link.span.get_text():
            OU = float(link.span.get_text().split("u")[1].split("\n")[0])
        else:
            OU = np.nan
            # continue
        total = total.append(pd.DataFrame({"total": [OU]}), ignore_index=True)
        total = total.append(pd.DataFrame({"total": [OU]}), ignore_index=True)
        i += 2
        print(OU)
        # if i == len(teams):
        #     break
    # Scrape Spread
    i = 0
    spread = pd.DataFrame(columns=["spread"])
    for link in soup.find_all(
        attrs={"class": "sb data card-data", "data-type": "spread"}
    ):
        val1 = link.find_all("span")[0].text
        val2 = link.find_all("span")[2].text
        if val1 == "n/a":
            val1 = np.nan
            val2 = np.nan
        else:
            val1 = float(
                "".join([c for c in val1 if (c.isalnum()) | (c in ["-", "."])])
            )
            val2 = float(
                "".join([c for c in val2 if (c.isalnum()) | (c in ["-", "."])])
            )

        spread = spread.append(pd.DataFrame({"spread": [val1]}), ignore_index=True)
        spread = spread.append(pd.DataFrame({"spread": [val2]}), ignore_index=True)
        # if i == len(teams):
        #     break

    # Scrape MoneyLine
    i = 0
    moneyline = pd.DataFrame(columns=["moneyline"])
    for link in soup.find_all(
        attrs={"class": "sb data card-data", "data-type": "moneyline"}
    ):
        val1 = link.find_all("span")[0].text
        val2 = link.find_all("span")[1].text
        if val1 == "n/a":
            val1 = np.nan
            val2 = np.nan
        else:
            val1 = float(
                "".join([c for c in val1 if (c.isalnum()) | (c in ["-", "."])])
            )
            val2 = float(
                "".join([c for c in val2 if (c.isalnum()) | (c in ["-", "."])])
            )

        moneyline = moneyline.append(
            pd.DataFrame({"moneyline": [val1]}), ignore_index=True
        )
        moneyline = moneyline.append(
            pd.DataFrame({"moneyline": [val2]}), ignore_index=True
        )
        # if i == len(teams):
        #     break
    df = pd.concat([teams, times, total, spread, moneyline], axis=1)
    df["proj_runs"] = (df.total / 2) - (df.spread / 2)
    df.rename({"total": "total_runs"}, axis=1, inplace=True)
    df = df[df.spread.isna() == False]
    df = df.groupby(["team","game_time"], as_index=False).mean()
    df.team.replace({'CHW':'CWS'},inplace=True)
    df.to_csv(f"{datadir}/BettingOdds/{game_date}_BettingOdds.csv", index=False)
    return df
