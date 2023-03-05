#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 03:50:59 2022

@author: robertmegnia
"""
import pandas as pd
import os
import requests
import unidecode
from os.path import exists
basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f"{basedir}/../../data"
salary_database = f"{basedir}/../../../../DraftKingsSalaryMegaDatabase2/NBA"
SALARY_URL = "https://www.draftkings.com/lineup"
import datetime
def reformatSalaries(salaries):
    # Reformat column names and team abbreviations to match database
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
    salaries.rename(
        {
            "Name": "player_name",
            "TeamAbbrev": "team",
            "Position": "position",
        },
        axis=1,
        inplace=True,
    )
    ## REFORMAT PLAYER NAMES BY REMOVING NON-ALPHA-NUMERICS
    salaries["first_name"] = salaries.player_name.apply(lambda x: x.split(" ")[0])
    salaries["last_name"] = salaries.player_name.apply(
        lambda x: " ".join(x.split(" ")[1::])
    )

    # Remove non-alpha numeric characters from first/last names.
    salaries["first_name"] = salaries.first_name.apply(
        lambda x: "".join(c for c in x if c.isalnum())
    )
    salaries["last_name"] = salaries.last_name.apply(
        lambda x: "".join(c for c in x if c.isalnum())
    )

    # Recreate player_name to fit format "Firstname Lastname" with no accents
    salaries["player_name"] = salaries.apply(
        lambda x: x.first_name + " " + x.last_name, axis=1
    )
    salaries["player_name"] = salaries.player_name.apply(lambda x: x.lower())
    salaries.drop(["first_name", "last_name"], axis=1, inplace=True)
    salaries.loc[salaries.position != "DST", "player_name"] = salaries.loc[
        salaries.position != "DST"
    ].player_name.apply(
        lambda x: x.split(" ")[0][0].upper()
        + x.split(" ")[0][1::]
        + " "
        + x.split(" ")[-1][0].upper()
        + x.split(" ")[-1][1::]
    )
    salaries["player_name"] = salaries.player_name.apply(
        lambda x: unidecode.unidecode(x)
    )
    # Create Column to match with RotoGrinders
    salaries["RotoName"] = salaries.player_name.apply(
        lambda x: x.lower().split(" ")[0][0:4] + x.lower().split(" ")[1][0:5]
    )
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
    salaries.team.replace({"WSH": "WAS", 
                           "PHO": "PHX",}, inplace=True)
    if salaries['Game Info'].unique()[0]=='Final':
        salaries['game_time']=datetime.strptime(f'{game_date} 07:30PM','%Y-%m-%d %H:%M%p')
    else:
        salaries["game_time"] = salaries["Game Info"].apply(
            lambda x: x.split(" ")[2] if len(x.split(" ")) > 1 else x
        )
    # Replace misspelled names
    return salaries 

def downloadDKSalaries(game_date, contest):
    """

    Downloads  Draft Kings Salaries for given game date and contest type

    Parameters
    ----------
    game_date : str
        date string of format %Y-%m-%d
    contest : str
        Draft Kings contest type "Classic" or "Showdown"

    Returns
    -------
    df : pandas.DataFrame
         Draft Kings exported salary csv

    """
    url = "https://www.draftkings.com/lobby/getcontests?sport=NBA"
    response = requests.get(url).json()
    DraftGroups = pd.DataFrame(response["DraftGroups"])
    # Filter contest types to classic  and showdown using IDs 125,127 for hockey
    DraftGroups = DraftGroups[DraftGroups.ContestTypeId.isin([70,81])]
    DraftGroups["StartDate"] = pd.to_datetime(
        DraftGroups.StartDateEst
    ).dt.tz_localize("US/Eastern")

    # Create column for slate starting time
    DraftGroups["Slate"] = DraftGroups.StartDate.apply(
        lambda x: x.strftime("%Y-%m-%d_%I:%M%p")
    )

    # Take first 10 characters of Slate to get %Y-%m-%d format game day
    DraftGroups["GameDay"] = DraftGroups.Slate.apply(lambda x: x[0:10])

    # Filter retrieved draftgroups to day in question
    DraftGroups = DraftGroups[DraftGroups.GameDay==game_date]
    daily_sal_frames = []
    daily_sal_frames_SD = []
    classic = None
    showdown = None
    # Download all slates for game_date
    for group, slate, suffix in zip(
        DraftGroups.DraftGroupId,
        DraftGroups.Slate,
        DraftGroups.ContestStartTimeSuffix,
    ):
        url = f"{SALARY_URL}/getavailableplayerscsv?contestTypeId=28&draftGroupId={group}"
        df = pd.read_csv(url)
        if len(df)==0:
            continue
        if "CPT" in df["Roster Position"].unique():
            showdown = True
            team1, team2 = df.TeamAbbrev.unique()
            df = reformatSalaries(df)
            daily_sal_frames_SD.append(df)
            # Try to export, if directory doesn't exist, create it.
            try:
                if not exists(
                    f"{salary_database}/Showdown/{game_date}/{slate}_{team1}_{team2}_salaries.csv"
                ):
                    df.to_csv(
                        f"{salary_database}/Showdown/{game_date}/{slate}_{team1}_{team2}_salaries.csv",
                        index=False,
                    )
            except OSError:
                os.mkdir(f"{salary_database}//Showdown/{game_date}/")
                if not exists(
                    f"{salary_database}/Showdown/{game_date}/{slate}_{team1}_{team2}_salaries.csv"
                ):
                    df.to_csv(
                        f"{salary_database}/Showdown/{game_date}/{slate}_{team1}_{team2}_salaries.csv",
                        index=False,
                    )
        else:
            # Try to export, if directory doesn't exist, create it.
            classic = True
            df = reformatSalaries(df)
            daily_sal_frames.append(df)
            if suffix is None:
                suffix = ""
            elif "-" in suffix:
                suffix = suffix.replace("-", "_")
                suffix = suffix.replace("(", "")
                suffix = suffix.replace(")", "")
                suffix = suffix.replace(" ", "")
                suffix = f"_{suffix}"
            elif "Only" in suffix:
                suffix = suffix.replace(" ", "_")
                suffix = suffix.replace("(", "")
                suffix = suffix.replace(")", "")
                suffix = f"{suffix}"
            try:
                if not exists(
                    f"{salary_database}/Classic/{game_date}/{slate}_salaries{suffix}.csv"
                ):
                    df.to_csv(
                        f"{salary_database}/Classic/{game_date}/{slate}_salaries{suffix}.csv",
                        index=False,
                    )
            except OSError:
                os.mkdir(f"{salary_database}/Classic/{game_date}/")
                if not exists(
                    f"{salary_database}/Classic/{game_date}/{slate}_salaries{suffix}.csv"
                ):
                    df.to_csv(
                        f"{salary_database}/Classic/{game_date}/{slate}_salaries{suffix}.csv",
                        index=False,
                    )
    if classic == True:
        daily_sal = pd.concat(daily_sal_frames)
        daily_sal = daily_sal.groupby(
            ["player_name", "team", "Roster Position", "position"], as_index=False
        ).first()
        if not exists(f"{salary_database}/Classic/{game_date}/{game_date}_salaries.csv"):
            try:
                daily_sal.to_csv(
                    f"{salary_database}/Classic/{game_date}/{game_date}_salaries.csv",
                    index=False,
                )
            except OSError:
                os.mkdir(f"{salary_database}/Classic/{game_date}/")
                daily_sal.to_csv(
                    f"{salary_database}/Classic/{game_date}/{game_date}_salaries.csv",
                    index=False,
                )
    if showdown == True:
        daily_sal_SD = pd.concat(daily_sal_frames_SD)
        if not exists(f"{salary_database}/Showdown/{game_date}/{game_date}_salaries.csv"):
            try:
                daily_sal_SD.to_csv(
                    f"{salary_database}/Showdown/{game_date}/{game_date}_salaries.csv",
                    index=False,
                )
            except OSError:
                os.mkdir(f"{salary_database}/Showdown/{game_date}/")
                daily_sal_SD.to_csv(
                    f"{salary_database}/Showdown/{game_date}/{game_date}_salaries.csv",
                    index=False,
                )
    if contest == "Classic":
        daily_sal = pd.read_csv(f"{salary_database}/Classic/{game_date}/{game_date}_salaries.csv")
        return daily_sal
    else:
        return daily_sal_SD

def getDKSalaries(game_date,contest="Classic"):
    """


    Parameters
    ----------
    game_date : str
        date string of format %Y-%m-%d
    contest : str
        Draft Kings contest type "Classic" or "Showdown"

    Returns
    -------
    salaries : pandas.DataFrame
    slate: str

    """

    salaries = downloadDKSalaries(game_date, contest)
    return salaries

