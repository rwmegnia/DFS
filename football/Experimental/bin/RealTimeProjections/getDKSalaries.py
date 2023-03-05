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
salary_database = f"{basedir}/../../../../DraftKingsSalaryMegaDatabase2/NFL"
SALARY_URL = "https://www.draftkings.com/lineup"


def reformatSalaries(salaries):
    # Reformat column names and team abbreviations to match database
    salaries.rename(
        {
            "Name": "full_name",
            "TeamAbbrev": "team",
            "Position": "position",
            "Salary": "salary",
        },
        axis=1,
        inplace=True,
    )
    ## REFORMAT PLAYER NAMES BY REMOVING NON-ALPHA-NUMERICS
    salaries["first_name"] = salaries.full_name.apply(lambda x: x.split(" ")[0])
    salaries["last_name"] = salaries.full_name.apply(
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
    salaries["full_name"] = salaries.apply(
        lambda x: x.first_name + " " + x.last_name, axis=1
    )
    salaries["full_name"] = salaries.full_name.apply(lambda x: x.lower())
    salaries.drop(["first_name", "last_name"], axis=1, inplace=True)
    salaries.loc[salaries.position != "DST", "full_name"] = salaries.loc[
        salaries.position != "DST"
    ].full_name.apply(
        lambda x: x.split(" ")[0][0].upper()
        + x.split(" ")[0][1::]
        + " "
        + x.split(" ")[-1][0].upper()
        + x.split(" ")[-1][1::]
    )
    salaries["full_name"] = salaries.full_name.apply(
        lambda x: unidecode.unidecode(x)
    )
    # Create Column to match with RotoGrinders
    salaries["RotoName"] = salaries.full_name.apply(
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
    # salaries.loc[salaries.position=='DST','RotoName']=salaries.loc[salaries.position=='DST','team'].str.upper()
    salaries.team.replace({"LAR": "LA"}, inplace=True)
    salaries["game_time"] = salaries["Game Info"].apply(
        lambda x: x.split(" ")[2] if len(x.split(" ")) > 1 else x
    )
    # Replace misspelled names
    return salaries  # , slate.split("_salaries.csv")[0]


def downloadDKSalaries(game_date, game_dates, contest, week, season):
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
    url = "https://www.draftkings.com/lobby/getcontests?sport=NFL"
    response = requests.get(url).json()
    DraftGroups = pd.DataFrame(response["DraftGroups"])
    # Filter contest types to classic  and showdown using IDs 125,127 for hockey
    DraftGroups = DraftGroups[DraftGroups.ContestTypeId.isin([21, 96])]
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
    DraftGroups = DraftGroups[DraftGroups.GameDay.isin(game_dates)]
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
        # Use draftGroup Number to pull seach players DK Player ID from DK API
        # draftGroup_url = f"https://api.draftkings.com/draftgroups/v1/draftgroups/{group}/draftables"
        # draftGroupInfo = pd.DataFrame(requests.get(draftGroup_url).json()["draftables"])[["displayName", "playerId", "teamAbbreviation",'position']]
        # # Rename some columns for merging purposes
        # draftGroupInfo.rename(
        #     {
        #         "displayName": "Name",
        #         "playerId": "dk_player_id",
        #         "teamAbbreviation": "TeamAbbrev",
        #     },
        #     axis=1,
        #     inplace=True,
        # )
        # draftGroupInfo.loc[draftGroupInfo.position=='DST','Name']=draftGroupInfo.loc[draftGroupInfo.position=='DST','Name'].apply(lambda x: x+' ')
        # df=df.merge(draftGroupInfo,on=['Name','TeamAbbrev'],how='left')
        # missing=df[df.dk_player_id.isna()==True]
        # if len(missing)!=0:
        #     print(f'The following players draftkings_player_ids were not merged',missing)
        # Determine if slate is Classic or Showdown and export accordingly
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
            ["full_name", "team", "Roster Position", "position"], as_index=False
        ).first()
        if not exists(f"{salary_database}/Classic/{game_date}/{game_date}_salaries.csv"):
            try:
                daily_sal['week']=week
                daily_sal['season']=season
                daily_sal.to_csv(
                    f"{salary_database}/Classic/{game_date}/{game_date}_salaries.csv",
                    index=False,
                )
            except OSError:
                os.mkdir(f"{salary_database}/Classic/{game_date}/")
                daily_sal['week']=week
                daily_sal['season']=season
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


def getDKSalaries(game_date, game_dates, week, season, contest="Classic"):
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

    salaries = downloadDKSalaries(game_date, game_dates, contest, week, season)
    db = pd.read_csv(f'{datadir}/DKSalaries/{season}_salaries.csv')
    db=db[db.week!=week]
    db=pd.concat([salaries,db])
    db.to_csv(f'{datadir}/DKSalaries/{season}_salaries.csv',index=False)
    salaries.loc[salaries.position == "DST", "full_name"] = salaries.loc[
        salaries.position == "DST", "team"
    ]
    salaries.loc[salaries.position == "DST", "RotoName"] = salaries.loc[
        salaries.position == "DST", "team"
    ]
    return salaries
