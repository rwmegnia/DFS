#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 07:15:46 2022

@author: robertmegnia
"""

"""
Quicky script to scrape projections from fantasypros.com
"""
import sys
import requests
from bs4 import BeautifulSoup
import pandas as pd
from getDKPts import getDKPts
from RosterUtils import reformatName

# set up some parameters for scrape
team_mapper = {
    "San Francisco 49ers": "SF",
    "New Orleans Saints": "NO",
    "Baltimore Ravens": "BAL",
    "Tennessee Titans": "TEN",
    "Indianapolis Colts": "IND",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Green Bay Packers": "GB",
    "Pittsburgh Steelers": "PIT",
    "Denver Broncos": "DEN",
    "Los Angeles Chargers": "LAC",
    "New York Jets": "NYJ",
    "Carolina Panthers": "CAR",
    "Philadelphia Eagles": "PHI",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "New England Patriots": "NE",
    "Jacksonville Jaguars": "JAX",
    "Washington Commanders": "WSH",
    "Chicago Bears": "CHI",
    "Seattle Seahawks": "SEA",
    "New York Giants": "NYG",
    "Kansas City Chiefs": "KC",
    "Buffalo Bills": "BUF",
    "Tampa Bay Buccaneers": "TB",
    "Detroit Lions": "DET",
    "Atlanta Falcons": "ATL",
    "Houston Texans": "HOU",
    "Las Vegas Raiders": "LV",
    "Arizona Cardinals": "ARI",
    "Dallas Cowboys": "DAL",
    "Minnesota Vikings": "MIN",
}


def scrapeFantasyPros(week):
    base_url = "http://www.fantasypros.com/nfl/projections"
    position_list = ["qb", "rb", "wr", "te", "dst"]
    experts = {
        "44": "Dave Richard, CBS Sports",
        "45": "Jamey Eisenberg, CBS Sports",
        "71": "ESPN",
        "73": "numberFire",
        "120": "Bloomberg Sports",
        "152": "FFToday",
        "469": "Pro Football Focus",
    }
    frames = []
    for position in position_list:
        # make the request, use trick of expert:expert to get the
        # results from just one source
        url = "%s/%s.php" % (base_url, position)
        params = {
            "week": week,
            "scoring": "PPR",
            "filters": "71:71",
            "filters": ":".join(list(experts.keys())),
        }
        #'filters': '%i:%i' % (expert_code, expert_code),

        response = requests.get(url, params=params)
        msg = f"getting projections for week {week}, postition {position}"

        # use expert:expert in request to get only one expert at a time
        # use pandas to parse the HTML table for us
        df = pd.io.html.read_html(response.text, attrs={"id": "data"})[0]
        if position == "qb":
            player = df["Unnamed: 0_level_0"]
            passing = df["PASSING"]
            passing.rename(
                {"YDS": "pass_yards", "TDS": "pass_td", "INTS": "int"},
                axis=1,
                inplace=True,
            )
            rushing = df["RUSHING"]
            rushing.rename(
                {
                    "YDS": "rush_yards",
                    "TDS": "rush_td",
                },
                axis=1,
                inplace=True,
            )
            misc = df["MISC"]
            misc.rename({"FL": "fumbles_lost"}, axis=1, inplace=True)
            df = pd.concat([player, passing, rushing, misc], axis=1)
            df[["rec", "rec_yards", "rec_td"]] = 0
            df["FP_Proj"] = getDKPts(df, "Offense")
        elif position in ["rb", "wr"]:
            player = df["Unnamed: 0_level_0"]
            rushing = df["RUSHING"]
            rushing.rename(
                {
                    "YDS": "rush_yards",
                    "TDS": "rush_td",
                },
                axis=1,
                inplace=True,
            )
            misc = df["MISC"]
            misc.rename({"FL": "fumbles_lost"}, axis=1, inplace=True)
            receiving = df["RECEIVING"]
            receiving.rename(
                {"YDS": "rec_yards", "TDS": "rec_td", "REC": "rec"},
                axis=1,
                inplace=True,
            )
            df = pd.concat([player, rushing, receiving, misc], axis=1)
            df[["pass_yards", "pass_td", "int"]] = 0
            df["FP_Proj"] = getDKPts(df, "Offense")
        elif position == "te":
            player = df["Unnamed: 0_level_0"]
            misc = df["MISC"]
            misc.rename({"FL": "fumbles_lost"}, axis=1, inplace=True)
            receiving = df["RECEIVING"]
            receiving.rename(
                {"YDS": "rec_yards", "TDS": "rec_td", "REC": "rec"},
                axis=1,
                inplace=True,
            )
            df = pd.concat([player, receiving, misc], axis=1)
            df[["pass_yards", "pass_td", "int", "rush_yards", "rush_td"]] = 0
            df["FP_Proj"] = getDKPts(df, "Offense")
        else:
            df.rename(
                {
                    "SACK": "sack",
                    "FR": "fumble_recoveries",
                    "INT": "interception",
                    "TD": "return_touchdown",
                    "SAFETY": "safety",
                    "PA": "points_allowed",
                },
                axis=1,
                inplace=True,
            )
            df["blocks"] = 0
            df["FP_Proj"] = getDKPts(df, "DST")
        df["week"] = week
        df["position"] = position.upper()
        df = df[["Player", "FP_Proj", "position", "week"]]
        frames.append(df)
    df = pd.concat(frames)
    df.loc[df.position != "DST", "full_name"] = df.loc[
        df.position != "DST", "Player"
    ].apply(lambda x: " ".join(x.split(" ")[0:2]))
    df.loc[df.position == "DST", "full_name"] = df.loc[
        df.position == "DST", "Player"
    ].apply(lambda x: team_mapper[x])
    df.loc[df.position != "DST", "team"] = df.loc[
        df.position != "DST", "Player"
    ].apply(lambda x: x.split(" ")[-1])
    df.loc[df.position == "DST", "team"] = df.loc[
        df.position == "DST", "full_name"
    ]
    df.full_name.replace({'Gabe Davis':'Gabriel Davis',
                          'Ken Walker':'Kenneth Walker'},inplace=True)
    df = reformatName(df)
    return df[["RotoName", "position", "team", "week", "FP_Proj"]]
