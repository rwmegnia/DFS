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
salary_database = f"{basedir}/../../../../DraftKingsSalaryMegaDatabase2/NBA"


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
    # Reformat column names and team abbreviations to match database
    salaries.rename(
        {"player_name": "PLAYER_NAME", "team": "TEAM_ABBREVIATION"},
        axis=1,
        inplace=True,
    )

    return salaries
