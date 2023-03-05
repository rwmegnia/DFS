#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 21:14:54 2022

@author: robertmegnia
"""

# Build Salary Database
import requests
import pandas as pd
from datetime import datetime
from os.path import exists
import os
import numpy as np
import unidecode
# Replace team abbreviations in DK files to match
# databases
new_teams={'NFL':
           {
               "LAR": "LA"
               },
           'NBA':{
               "WSH": "WAS", 
               "PHO": "PHX",
               },
           'NHL':{
               'TB':'TBL',
               'CLS':'CBJ',
               'ANH':'ANA',
               'LA':'LAK',
               'SJ':'SJS',
               'NJ':'NJD',
               'MON':'MTL',
               'WAS':'WSH'
               }
           }
# NFL
def reformatSalaries(salaries,Sport):
    # Reformat column names and team abbreviations to match database
    if Sport == 'NBA':
        new_name='player_name'
        salaries.Name.replace(
            {
                "Moe Harkless": "Maurice Harkless",
                "J.R. Smith": "JR Smith",
                "C.J. Miles": "CJ Miles",
                "Luigi Datome": "Gigi Datome",
                "Perry Jones": "Perry Jones III",
                "J.J. Hickson": "JJ Hickson",
                "A.J. Price": "AJ Price",
                "Louis Amundson": "Lou Amundson",
                "D.J. Stephens": "DJ Stephens",
                "P.J. Hairston": "PJ Hairston",
                "K.J. McDaniels": "KJ McDaniels",
                "Jakarr Sampson": "JaKarr Sampson",
                "CJ Wilcox": "C.J. Wilcox",
                "R.J. Hunter": "RJ Hunter",
                "TJ Leaf": "T.J. Leaf",
                "Robert Williams": "Robert Williams III",
                "B.J. Johnson": "BJ Johnson",
                "Jacob Evans III": "Jacob Evans",
                "PJ Dozier": "P.J. Dozier",
                "Mitch Creek": "Mitchell Creek",
                "Billy Garrett Jr.": "Billy Garrett",
                "PJ Washington": "P.J. Washington",
                "Nicolas Claxton": "Nic Claxton",
                "Michael Frazier": "Michael Frazier II",
                "Charlie Brown": "Charline Brown Jr.",
                "Kenyon Martin Jr.": "Kenyon Martin",
                "Cameron Thomas": "Cam Thomas",
                "RJ Nembhard": "RJ Nembhard Jr.",
                "Greg Brown": "Greg Brown III",
                "M.J. Walker": "MJ Walker",
                "Luc Richard Mbah a Moute": "Luc Mbah a Moute",
                "Nene Hilario": "Nene",
                "Glen Rice Jr.": "Glen Rice",
                "Wayne Selden Jr.": "Wayne Selden",
            },
            inplace=True,
        )
    else:
        new_name='full_name'
    salaries.rename(
        {
            "Name": new_name,
            "TeamAbbrev": "team",
            "Position": "position",
        },
        axis=1,
        inplace=True,
    )
    if Sport=='NFL':
        salaries.rename({'Salary':'salary'},axis=1,inplace=True)
    ## REFORMAT PLAYER NAMES BY REMOVING NON-ALPHA-NUMERICS
    salaries["first_name"] = salaries[new_name].apply(lambda x: x.split(" ")[0])
    salaries["last_name"] = salaries[new_name].apply(
        lambda x: " ".join(x.split(" ")[1::])
    )

    # Remove non-alpha numeric characters from first/last names.
    salaries["first_name"] = salaries.first_name.apply(
        lambda x: "".join(c for c in x if c.isalnum())
    )
    salaries["last_name"] = salaries.last_name.apply(
        lambda x: "".join(c for c in x if c.isalnum())
    )

    # Recreate full_name to fit format "Firstname Lastname" with no accents
    salaries[new_name] = salaries.apply(
        lambda x: x.first_name + " " + x.last_name, axis=1
    )
    salaries[new_name] = salaries[new_name].apply(lambda x: x.lower())
    salaries.drop(["first_name", "last_name"], axis=1, inplace=True)
    salaries.loc[salaries.position != "DST", new_name] = salaries.loc[
        salaries.position != "DST"
    ][new_name].apply(
        lambda x: x.split(" ")[0][0].upper()
        + x.split(" ")[0][1::]
        + " "
        + x.split(" ")[-1][0].upper()
        + x.split(" ")[-1][1::]
    )
    salaries[new_name] = salaries[new_name].apply(
        lambda x: unidecode.unidecode(x)
    )
    # Create Column to match with RotoGrinders
    salaries["RotoName"] = salaries[new_name].apply(
        lambda x: x.lower().split(" ")[0][0:4] + x.lower().split(" ")[1][0:5]
    )
    if Sport =='NFL':
        salaries.RotoName.replace(
            {
                "gabedavis": "gabrdavis",
                "deeeskri": "dwayeskri",
                "elimitch": "elijmitch",
                "michwoods": "mikewoods",
                "stevsimsj": "stevsims",
                "jodyforts": "joeforts",
                "pjwalke": "philwalke",
                
            },
            inplace=True,
        )
    elif Sport=='NHL':
        salaries[new_name].replace({"Tim Stuetzle": "Tim Stutzle"}, inplace=True)

    salaries.team.replace(new_teams[Sport], inplace=True)
    salaries["game_time"] = salaries["Game Info"].apply(
        lambda x: x.split(" ")[2] if len(x.split(" ")) > 1 else x
    )
    return salaries 


basedir = os.path.dirname(os.path.abspath(__file__))
database_name='DraftKingsSalaryMegaDatabase2'

# Create directory for database
if not exists(f"{basedir}/{database_name}"):
    os.mkdir(f"{database_name}")

sports = ["NFL", "MLB", "NBA", "NHL", "GOLF"]

leagues = {}
# Create directories for each sport if they don't already exists
# Include Classic and Showdown directories for each sport
for sport in sports:
    if not exists(f"{basedir}/{database_name}/{sport}"):
        os.mkdir(f"{basedir}/{database_name}/{sport}")
        os.mkdir(f"{basedir}/{database_name}/{sport}/Classic")
        os.mkdir(f"{basedir}/{database_name}/{sport}/Showdown")

# Loop through every Draft Group DraftKings has ever had. (78521 as of 11/28/2022)
# First draft group that will return data is 680.
for draft_group in range(79273,79606):
    print(draft_group)
    # Not every iteration will return a response. Continue if there is an exception or if
    # size of response['draftables']==0
    try:
        # Get group details first to ensure the group isn't for a simulated sport (i.e. Madden NFL)
        details = requests.get(
            f"https://api.draftkings.com/draftgroups/v1/{draft_group}"
        ).json()
        league = details["draftGroup"]["games"][0]["league"]
        print(league)
        if league not in ['NHL','NBA']:
            continue
        response = requests.get(
            f"https://api.draftkings.com/draftgroups/v1/draftgroups/{draft_group}/draftables"
        ).json()
    except Exception as e:
        print(e)
        continue
    if "draftables" not in response.keys():
        continue
    if len(response["draftables"]) == 0:
        continue
    # Get the competitions aka games on the slate for the retrieved draft group
    competitions = pd.DataFrame(response["competitions"])
    # ID the sport for the draftgroup
    sport = competitions.sport.unique()[0]

    # If the sport for this draft group is one you want data for, archive it.
    if sport in sports:
        print(f"Draft Group found for {sport}")
        competitions.startTime = pd.to_datetime(
            competitions.startTime
        ).dt.tz_convert("US/Eastern")
        # Create a column that will serve as a string to identify the start time of each game on the slate
        competitions["Slate"] = competitions.startTime.apply(
            lambda x: x.strftime("%Y-%m-%d_%I:%M%p")
        )
        # Get number of games on the slate
        n_games = len(competitions)
        # If only one game in competition it may be a showdown. Get home/away teams to use in filename
        if n_games == 1:
            home_team = competitions["homeTeam"][0]["abbreviation"]
            away_team = competitions["awayTeam"][0]["abbreviation"]
        # Define the slate as the earliest start time of the games on the slate + number of games on the slate
        # Ex 2021-02-25_7:00PM_5games
        Slate = competitions.Slate.min()
        GameDay = Slate[0:10]

        # First try and download an existing csv file of the players on the slate
        # If it returns empty we can build one using the data in the response variable
        url = f"https://www.draftkings.com/lineup/getavailableplayerscsv?draftGroupId={draft_group}"
        try:
            salaries = pd.read_csv(url)
            salaries["game_date"] = GameDay
        except:
            salaries = []
        # If we cant't pull the csv, build it.
        if len(salaries) == 0:
            salaries = pd.DataFrame(response["draftables"])
            salaries.rename(
                {
                    "displayName": "Name",
                    "position": "Position",
                    "playerId": "ID",
                    "salary": "Salary",
                    "teamAbbreviation": "TeamAbbrev",
                },
                axis=1,
                inplace=True,
            )
            salaries = salaries.groupby("Name", as_index=False).first()
            salaries = salaries[
                ["Name", "Position", "ID", "Salary", "TeamAbbrev"]
            ]
            salaries["Name + ID"] = (
                salaries.Name + " (" + salaries.ID.astype(str) + ")"
            )
            salaries["Game Info"] = None
            salaries["AvgPointsPerGame"] = np.nan
            salaries["Roster Position"] = salaries.Position
            salaries = salaries[
                [
                    "Position",
                    "Name + ID",
                    "Name",
                    "ID",
                    "Position",
                    "Roster Position",
                    "Salary",
                    "Game Info",
                    "TeamAbbrev",
                    "AvgPointsPerGame",
                ]
            ]
            salaries["game_date"] = GameDay
            salaries['Game Info']=pd.to_datetime(salaries.game_date)
        salaries=reformatSalaries(salaries, sport)
        # Determine if salaries are for showdown
        # If CPT in Roster Positions this is a showdown slate.
        if "CPT" in salaries["Roster Position"].unique():
            if not exists(
                f"{basedir}/{database_name}/{sport}/Showdown/{GameDay}"
            ):
                os.mkdir(
                    f"{basedir}/{database_name}/{sport}/Showdown/{GameDay}"
                )
            if not exists( f"{basedir}/{database_name}/{sport}/Showdown/{GameDay}/{Slate}_{away_team}_{home_team}_salaries.csv"):
                salaries.to_csv(
                    f"{basedir}/{database_name}/{sport}/Showdown/{GameDay}/{Slate}_{away_team}_{home_team}_salaries.csv",
                    index=False,
                )
                continue
            else:
                continue
            
        # If Max Salary is less than 1000 or no Salary column or string in column, it's probably a Tiers contest
        if ("Salary" not in salaries.columns)&('salary' not in salaries.columns):
            continue
        try:
            if salaries["Salary"].max() < 2000:
                continue
        except TypeError:
            continue
        # If we've made it this far, the draft group is for a Classic slate for the sport in the iteration
        if not exists(
            f"{basedir}/{database_name}/{sport}/Classic/{GameDay}"
        ):
            os.mkdir(
                f"{basedir}/{database_name}/{sport}/Classic/{GameDay}"
            )
        
            salaries.to_csv(
                f"{basedir}/{database_name}/{sport}/Classic/{GameDay}/{Slate}_salaries.csv",
                index=False,
            )
        else:
            salaries.to_csv(
                f"{basedir}/{database_name}/{sport}/Classic/{GameDay}/{Slate}_salaries.csv",
                index=False,
            )
