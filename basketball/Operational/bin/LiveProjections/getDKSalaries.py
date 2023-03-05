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

basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f"{basedir}/../../data"
salary_database = f"{basedir}/../../../../DraftKingsSalaryMegaDatabase/NBA"
SALARY_URL = "https://www.draftkings.com/lineup"


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
    # Filter contest types to classic  and showdown using IDs 70,127 for basketball
    DraftGroups = DraftGroups[DraftGroups.ContestTypeId.isin([70, 81])]
    DraftGroups["StartDate"] = pd.to_datetime(DraftGroups.StartDateEst).dt.tz_localize(
        "US/Eastern"
    )

    # Create column for slate starting time
    DraftGroups["Slate"] = DraftGroups.StartDate.apply(
        lambda x: x.strftime("%Y-%m-%d_%I:%M%p")
    )

    # Take first 10 characters of Slate to get %Y-%m-%d format game day
    DraftGroups["GameDay"] = DraftGroups.Slate.apply(lambda x: x[0:10])

    # Filter retrieved draftgroups to day in question
    DraftGroups = DraftGroups[DraftGroups.GameDay == game_date]

    # Download all slates for game_date
    for group, slate in zip(DraftGroups.DraftGroupId, DraftGroups.Slate):
        url = f"{SALARY_URL}/getavailableplayerscsv?contestTypeId=125&draftGroupId={group}"
        df = pd.read_csv(url)

        # Determine if slate is Classic or Showdown and export accordingly
        if "CPT" in df["Roster Position"].unique():
            team1, team2 = df.TeamAbbrev.unique()
            # Try to export, if directory doesn't exist, create it.
            try:
                df.to_csv(
                    f"{salary_database}/Showdown/{game_date}/{slate}_{team1}_{team2}_salaries.csv",
                    index=False,
                )
            except FileNotFoundError:
                os.mkdir(f"{salary_database}//Showdown/{game_date}/")
                df.to_csv(
                    f"{salary_database}/Showdown/{game_date}/{slate}_{team1}_{team2}_salaries.csv",
                    index=False,
                )
        else:
            # Try to export, if directory doesn't exist, create it.
            try:
                df.to_csv(
                    f"{salary_database}/Classic/{game_date}/{slate}_salaries.csv",
                    index=False,
                )
            except FileNotFoundError:
                os.mkdir(f"{salary_database}/Classic/{game_date}/")
                df.to_csv(
                    f"{salary_database}/Classic/{game_date}/{slate}_salaries.csv",
                    index=False,
                )
    return df


def getDKSalaries(game_date, contest="Classic"):
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

    # See if salary files already exist, if not, download them.
    try:
        salaryFiles = os.listdir(f"{salary_database}/{contest}/{game_date}")
    except FileNotFoundError:
        salaries = downloadDKSalaries(game_date, contest)
        salaryFiles = os.listdir(f"{salary_database}/{contest}/{game_date}")

    # List slate options to select from
    choices = {i + 1: salaryFiles[i] for i in range(len(salaryFiles))}
    print("Select a Slate")
    n = 1
    for file in salaryFiles:
        print(n, file)
        n += 1
    slate = input("Select Slate by number (1,2,3, etc...) ")
    slate = choices[int(slate)]
    salaries = pd.read_csv(f"{salary_database}/{contest}/{game_date}/{slate}")
    salaries.Name.replace(
        {
            "Moe Harkless": "Maurice Harkless",
            "Cameron Thomas": "Cam Thomas",
            "Bones Hyland": "Nah'Shon Hyland",
        },
        inplace=True,
    )
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
        {"Name": "player_name", "TeamAbbrev": "team", "Position": "position"},
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
    salaries["player_name"] = salaries.player_name.apply(
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
    salaries.team.replace({"WSH": "WAS", "PHO": "PHX",}, inplace=True)
    # Replace misspelled names
    return salaries, slate.split("_salaries.csv")[0]
