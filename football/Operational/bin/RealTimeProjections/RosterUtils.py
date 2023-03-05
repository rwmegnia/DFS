#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 00:25:20 2022

@author: robertmegnia
"""
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import os
import unidecode
from datetime import datetime
basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../../etc"
datadir = f"{basedir}/../../data"


def reformatName(df):
    ## REFORMAT PLAYER NAMES BY REMOVING NON-ALPHA-NUMERICS
    df["first_name"] = df.full_name.apply(lambda x: x.split(" ")[0])
    df["last_name"] = df.full_name.apply(lambda x: " ".join(x.split(" ")[1::]))

    # Remove non-alpha numeric characters from first/last names.
    df["first_name"] = df.first_name.apply(
        lambda x: "".join(c for c in x if c.isalnum())
    )
    df["last_name"] = df.last_name.apply(
        lambda x: "".join(c for c in x if c.isalnum())
    )

    # Recreate full_name to fit format "Firstname Lastname" with no accents
    df["full_name"] = df.apply(
        lambda x: x.first_name + " " + x.last_name, axis=1
    )
    df["full_name"] = df.full_name.apply(lambda x: x.lower())
    df.drop(["first_name", "last_name"], axis=1, inplace=True)
    df.loc[df.position != "DST", "full_name"] = df.loc[
        df.position != "DST"
    ].full_name.apply(
        lambda x: x.split(" ")[0][0].upper()
        + x.split(" ")[0][1::]
        + " "
        + x.split(" ")[-1][0].upper()
        + x.split(" ")[-1][1::]
    )
    df["full_name"] = df.full_name.apply(lambda x: unidecode.unidecode(x))

    # Create Column to match with RotoGrinders
    df["RotoName"] = df.full_name.apply(
        lambda x: x.lower().split(" ")[0][0:4] + x.lower().split(" ")[1][0:5]
    )
    try:
        df.loc[df.position == "DST", "RotoName"] = df.loc[
            df.position == "DST", "team"
        ]
    except:
        pass
    # df['game_time']=df['Game Info'].apply(lambda x: x.split(' ')[2])
    # Replace misspelled names
    return df  # , slate.split("_df.csv")[0]


def getSchedule(week, season, seasontype="REG"):
    frames = []
    URL = f"https://www.nfl.info//nfldataexchange/dataexchange.asmx/getSchedule?lseason={season}&lseasontype={seasontype}&lclub=ALL&lweek={week}"
    response = requests.get(URL, auth=HTTPBasicAuth("media", "media")).content
    df = pd.read_xml(response)
    df["game_date"] = df.GameDate + " " + df.StartTime
    df.game_date = pd.to_datetime(df.game_date.values).strftime(
        "%A %y-%m-%d %I:%M %p"
    )
    df.rename(
        {
            "GameDay": "game_day",
            "StartTime": "game_time",
            "Season": "season",
            "Week": "week",
        },
        axis=1,
        inplace=True,
    )
    df.game_time = pd.to_datetime(df.game_time)
    df.loc[
        (df.game_day == "Sunday")
        & (df.game_time >= "12:00:00")
        & (df.game_time < "17:00:00"),
        "Slate",
    ] = "Main"
    df.loc[df.Slate != "Main", "Slate"] = "Full"
    df["team"] = df.HomeTeam
    df["opp"] = df.VisitTeam
    df["game_location"] = "home"
    frames.append(df)
    home = pd.concat(frames)
    df["team"] = df.VisitTeam
    df["opp"] = df.HomeTeam
    df["game_location"] = "away"
    away = pd.concat([df])
    df = pd.concat([home, away])
    df.team.replace(
        {
            "OAK": "LV",
            "LAR": "LA",
            "HST": "HOU",
            "ARZ": "ARI",
            "BLT": "BAL",
            "CLV": "CLE",
        },
        inplace=True,
    )
    df.opp.replace(
        {
            "OAK": "LV",
            "LAR": "LA",
            "HST": "HOU",
            "ARZ": "ARI",
            "BLT": "BAL",
            "CLV": "CLE",
        },
        inplace=True,
    )
    return df


def filterRosterData(df, season, week):
    df = df[df.position.isin(["QB", "RB", "WR", "TE", "DST"])]
    # Get Rookie gsis_ids which are not up to date in API
    #
    df = df[~df.gsis_id.isin([None])]
    df.position.replace("DEF", "DST", inplace=True)
    df.loc[df.position == "DST", "full_name"] = df.loc[
        df.position == "DST"
    ].index
    df.loc[df.position == "DST", "gsis_id"] = df.loc[df.position == "DST"].index
    df.loc[df.position == "DST", "gsis_id"] = df.loc[df.position == "DST"].index
    df = df[df.injury_status == "Active"]
    df = df[
        ~df.injury_status.isin(["IR", "PUP", "OUT", "Doubtful", "Sus", "COV"])
    ]
    return df


def getWeeklyRosters(season, week, schedule, odds, game_date):
    # Download Rosters from NFL API
    # URL=f'https://www.nfl.info//nfldataexchange/dataexchange.asmx/getRoster?lseason={season}&lseasontype=REG&lclub=ALL&lweek={week}'
    # response = requests.get(URL,auth = HTTPBasicAuth('media', 'media')).content
    # df=pd.read_xml(response)
    URL = f"https://www.nfl.info//nfldataexchange/dataexchange.asmx/getExpandedRosterByDate?date={game_date}&clubid=-1"
    response = requests.get(URL, auth=HTTPBasicAuth("media", "media")).json()
    df = pd.DataFrame(response)
    # Rename columns
    df.rename(
        {
            "Season": "season",
            "Week": "week",
            "GsisID": "gsis_id",
            "Position": "position",
            "CurrentClub": "team",
            "StatusShortDescription": "injury_status",
            "Playerid": "player_id",
            "Weight": "weight",
            "RookieYear": "rookie_year",
            "DraftNumber": "draft_number",
            "Draftround": "draft_round",
        },
        axis=1,
        inplace=True,
    )
    df["full_name"] = df.FootballName + " " + df.LastName
    df.full_name.replace({"Gabe Davis": "Gabriel Davis"}, inplace=True)
    # if week<7:
    #     df.loc[df.full_name=='Deshaun Watson','injury_status']='SUS'
    # Rename some team abbreviations
    df.team.replace(
        {"OAK": "LV", "HST": "HOU", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE"},
        inplace=True,
    )
    df.position.replace("FB", "RB", inplace=True)
    df.loc[df.full_name=='Cooper Rush','injury_status']='Active'
    df.loc[df.full_name=='DeAndre Hopkins','injury_status']='Active'
    df.loc[df.full_name=='Van Jefferson','injury_status']='Active'

    #### INSERT SALARY RETRIEVAL CODE HERE
    # Remove spaces from gsis strings
    df = filterRosterData(df, season, week)

    df = df[
        (df.position.isin(["QB", "RB", "WR", "TE"]))
        & (df.team.isna() == False)
        & (df.gsis_id.isna() == False)
    ]
    DST = df.groupby("team", as_index=False).first()
    DST["full_name"] = DST.team
    DST["gsis_id"] = DST.team
    DST["position"] = "DST"
    df = pd.concat([df, DST])
    df = df[
        [
            "season",
            "week",
            "team",
            "position",
            "full_name",
            "gsis_id",
            "player_id",
            "weight",
            "draft_number",
            "draft_round",
            "rookie_year",
            "injury_status",
        ]
    ]
    df = df.merge(
        schedule[
            [
                "week",
                "opp",
                "team",
                "Slate",
                "game_date",
                "game_day",
                "game_time",
                "game_location",
            ]
        ],
        on=["week", "team"],
        how="left",
    )
    df = df.merge(
        odds[["team", "proj_team_score", "total_line", "spread_line"]],
        on="team",
        how="left",
    )
    # Get Opponent Ranks vs position
    offense_opp_Ranks = pd.read_csv(
        f"{datadir}/game_logs/Full/Offense_Latest_OppRanks.csv"
    )
    dst_opp_Ranks = pd.read_csv(
        f"{datadir}/game_logs/Full/DST_Latest_OppRanks.csv"
    )
    opp_Ranks = pd.concat([offense_opp_Ranks, dst_opp_Ranks])
    df = df.merge(opp_Ranks, on=["opp", "position"], how="left")
    df = reformatName(df)
    df.loc[df.position == "DST", "RotoName"] = df.loc[
        df.position == "DST", "team"
    ]
    injuries=getWeeklyInjuries(week,season)
    df=df.merge(injuries,on=['week','season','gsis_id'],how='left')
    df=df[~df.injury_designation.isin(['Out','Doubtful'])]
    # Export Roster to weekly roster database
    try:
        df.to_csv(
            f"{datadir}/rosterData/weekly_rosters/{season}/week{week}/Week{week}RosterData.csv",
            index=False,
        )
        db=pd.read_csv(f"{datadir}/rosterData/{season}_SeasonGameRosters.csv")
        db=db[db.week!=week]
        db=pd.concat([db,df])
        db.to_csv(f"{datadir}/rosterData/{season}_SeasonGameRosters.csv",index=False)
    except OSError:
        print("Roster directory does not exist. Creating directory...")
        os.mkdir(f"{datadir}/rosterData/weekly_rosters/{season}/Week{week}/")
        df.to_csv(
            f"{datadir}/rosterData/weekly_rosters/{season}/week{week}/Week{week}RosterData.csv",
            index=False,
        )
        db=pd.read_csv(f"{datadir}/rosterData/{season}_SeasonGameRosters.csv")
        db=db[db.week!=week]
        db=pd.concat([db,df])
        db.to_csv(f"{datadir}/rosterData/{season}_SeasonGameRosters.csv")
    return df

def getWeeklyInjuries(week,season,season_type='REG'):
    URL=f'https://www.nfl.info/nfldataexchange/dataexchange.asmx/getInjuryDataJSON?lseason={season}&lweek={week}&lseasontype={season_type}'
    response = requests.get(URL, auth=HTTPBasicAuth("media", "media")).json()
    df = pd.DataFrame(response)
    if len(df)==0:
        df=pd.DataFrame(columns=['season',
                                 'week',
                                 'injury_designation',
                                 'gsis_id',
                                 'ModifiedDt'])
        return df
    df.rename(
        {
            "Season": "season",
            "Week": "week",
            "GsisID": "gsis_id",
            "Position": "position",
            "ClubCode": "team",
            "InjuryStatus": "injury_designation",
        },
        axis=1,
        inplace=True,
    )
    df["full_name"] = df.FootballName + " " + df.LastName
    df.full_name.replace({"Gabe Davis": "Gabriel Davis"}, inplace=True)
    Questionable=["Nico Collins",
                  'Ryan Tannehill']
    if week==8:
        for player in Questionable:
            df.loc[df.full_name==player,'injury_designation']='Out'
    # Rename some team abbreviations
    df.team.replace(
        {"OAK": "LV", "HST": "HOU", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE"},
        inplace=True,
    )
    df.position.replace("FB", "RB", inplace=True)
    df['ModifiedDt']=df.ModifiedDt.apply(lambda x: datetime.fromtimestamp(x))
    df=df[[
            'season',
            'week',
            'injury_designation',
            'gsis_id',
            'ModifiedDt',
            ]]
    return df

def getGameDate(week, season):
    schedule = getSchedule(week, season)
    schedule.game_day = pd.to_datetime(schedule.GameDate)
    game_date = (
        schedule[schedule.week == week].game_day.min().strftime("%Y-%m-%d")
    )
    return game_date


def getGameDates(week, season):
    schedule = getSchedule(week, season)
    schedule.game_day = pd.to_datetime(schedule.GameDate)
    game_dates = schedule.game_day.astype(str).unique()
    return game_dates
