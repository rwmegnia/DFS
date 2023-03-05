#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 03:50:59 2022

@author: robertmegnia
"""
import pandas as pd
import os
from os.path import exists
import requests
import unidecode

basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f"{basedir}/../../data"
salary_database = f"{basedir}/../../../../DraftKingsSalaryMegaDatabase/NBA"


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
    if not exists(f"{salary_database}/{contest}/{game_date}"):
        return

    salaryFiles = os.listdir(f"{salary_database}/{contest}/{game_date}")
    sal_frames = []
    for slate in salaryFiles:
        sal_frames.append(
            pd.read_csv(f"{salary_database}/{contest}/{game_date}/{slate}")
        )
    salaries = pd.concat(sal_frames)
    salaries = salaries.groupby("ID", as_index=False).first()
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
    # Reformat column names and team abbreviations to match database
    salaries.rename(
        {"Name": "PLAYER_NAME", "TeamAbbrev": "TEAM_ABBREVIATION"},
        axis=1,
        inplace=True,
    )

    ## REFORMAT PLAYER NAMES BY REMOVING NON-ALPHA-NUMERICS
    salaries["first_name"] = salaries.PLAYER_NAME.apply(lambda x: x.split(" ")[0])
    salaries["last_name"] = salaries.PLAYER_NAME.apply(
        lambda x: " ".join(x.split(" ")[1::])
    )

    # Remove non-alpha numeric characters from first/last names.
    salaries["first_name"] = salaries.first_name.apply(
        lambda x: "".join(c for c in x if c.isalnum())
    )
    salaries["last_name"] = salaries.last_name.apply(
        lambda x: "".join(c for c in x if c.isalnum())
    )

    # Recreate PLAYER_NAME to fit format "Firstname Lastname" with no accents
    salaries["PLAYER_NAME"] = salaries.apply(
        lambda x: x.first_name + " " + x.last_name, axis=1
    )
    salaries["PLAYER_NAME"] = salaries.PLAYER_NAME.apply(lambda x: x.lower())
    salaries.drop(["first_name", "last_name"], axis=1, inplace=True)
    salaries["PLAYER_NAME"] = salaries.PLAYER_NAME.apply(
        lambda x: x.split(" ")[0][0].upper()
        + x.split(" ")[0][1::]
        + " "
        + x.split(" ")[-1][0].upper()
        + x.split(" ")[-1][1::]
    )
    salaries["PLAYER_NAME"] = salaries.PLAYER_NAME.apply(
        lambda x: unidecode.unidecode(x)
    )

    salaries.TEAM_ABBREVIATION.replace(
        {
            "WSH": "WAS",
            "PHO": "PHX",
            "NO": "NOP",
            "NY": "NYK",
            "GS": "GSW",
            "SA": "SAS",
        },
        inplace=True,
    )
    # Replace misspelled names


    return salaries
