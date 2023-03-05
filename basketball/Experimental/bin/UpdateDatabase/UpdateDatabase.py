import os
from os.path import exists
from datetime import datetime
import pandas as pd
import requests
import numpy as np
import unidecode
import sys
from getDKSalaries import getDKSalaries
from getDKPts import getDKPts

basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f"{basedir}/../../data"
libdir = f"{basedir}/../../lib"
sys.path.insert(0, f"{libdir}")
from nba_api.stats.endpoints import (
    leaguegamelog,
    boxscoretraditionalv2,
    boxscoresummaryv2,
    boxscoreusagev2,
)

basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f"{basedir}/../../data"
today = datetime.now().strftime("%Y-%m-%d")

USAGE_COLS = [
    "PLAYER_ID",
    "USG_PCT",
    "PCT_FGM",
    "PCT_FGA",
    "PCT_FG3A",
    "PCT_FTM",
    "PCT_FTA",
    "PCT_OREB",
    "PCT_DREB",
    "PCT_BLKA",
    "PCT_PF",
    "PCT_PFD",
]

def getMinutes(df):
    df["mins"] = df["min"].apply(lambda x: str(x).split(":")[0]).astype(float)
    df["seconds"] = (
        df["min"]
        .apply(lambda x: str(x).split(":")[1] if len(str(x).split(":")) > 1 else 0)
        .astype(float)
    )
    df["mins"] = df.mins + (df.seconds / 60)
    # df.drop(['seconds','min'],axis=1,inplace=True)
    return df

def reformatName(df):

    ## REFORMAT PLAYER NAMES BY REMOVING NON-ALPHA-NUMERICS
    df["first_name"] = df.PLAYER_NAME.apply(lambda x: x.split(" ")[0])
    df["last_name"] = df.PLAYER_NAME.apply(
        lambda x: " ".join(x.split(" ")[1::])
    )

    # Remove non-alpha numeric characters from first/last names.
    df["first_name"] = df.first_name.apply(
        lambda x: "".join(c for c in x if c.isalnum())
    )
    df["last_name"] = df.last_name.apply(
        lambda x: "".join(c for c in x if c.isalnum())
    )

    # Recreate PLAYER_NAME to fit format "Firstname Lastname" with no accents
    df["PLAYER_NAME"] = df.apply(
        lambda x: x.first_name + " " + x.last_name, axis=1
    )
    df["PLAYER_NAME"] = df.PLAYER_NAME.apply(lambda x: x.lower())
    df.drop(["first_name", "last_name"], axis=1, inplace=True)
    df["PLAYER_NAME"] = df.PLAYER_NAME.apply(
        lambda x: x.split(" ")[0][0].upper()
        + x.split(" ")[0][1::]
        + " "
        + x.split(" ")[-1][0].upper()
        + x.split(" ")[-1][1::]
    )
    df["PLAYER_NAME"] = df.PLAYER_NAME.apply(
        lambda x: unidecode.unidecode(x)
    )
    # Create Column to match with RotoGrinders
    df["RotoName"] = df.PLAYER_NAME.apply(
        lambda x: x.lower().split(" ")[0][0:4] + x.lower().split(" ")[1][0:5]
    )
    df.RotoName.replace(
        {
            "gabedavis": "gabrdavis",
            
        },
        inplace=True,
    )
    return df 
#%%
player_database = pd.read_csv(f"{datadir}/game_logs/PlayerStatsDatabase.csv").drop_duplicates()
team_database = pd.read_csv(f"{datadir}/game_logs/TeamStatsDatabase.csv").drop_duplicates()
odds_db=pd.read_csv(f'{datadir}/Database/OddsDatabase.csv')
odds_db.drop(['moneyline','opp'],axis=1,inplace=True)
player_frames = []
team_frames = []
player_frames.append(player_database)
team_frames.append(team_database)
season=2022
gamelog = leaguegamelog.LeagueGameLog(season=season).get_data_frames()[0]
contest_type='Classic'
for game_date in gamelog["GAME_DATE"].unique():
    print(game_date)
    if exists(f"{datadir}/game_logs/{season}/{game_date}/"):
        continue
    else:
        os.mkdir(f"{datadir}/game_logs/{season}/{game_date}/")
    if game_date >= today:
        sys.exit()
    date_player_frames = []
    date_team_frames = []
    gameday_frame = gamelog[gamelog.GAME_DATE == game_date]
    for game_id in gameday_frame.GAME_ID.unique():
        # try:
            print(game_id)
            players = boxscoretraditionalv2.BoxScoreTraditionalV2(
                game_id
            ).get_data_frames()[0]
            player_usage = boxscoreusagev2.BoxScoreUsageV2(
                game_id
            ).get_data_frames()[0][USAGE_COLS]
            players = players.merge(player_usage, on=["PLAYER_ID"], how="left")
            players["GAME_DATE"] = game_date

            teams = boxscoretraditionalv2.BoxScoreTraditionalV2(
                game_id
            ).get_data_frames()[1]
            teams["GAME_DATE"] = game_date
            team_info = boxscoresummaryv2.BoxScoreSummaryV2(
                game_id
            ).get_data_frames()[7]
            teams = teams.merge(
                    team_info[["GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]],
                    on=["GAME_ID"],
                )
            players = players.merge(
                    team_info[["GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]],
                    on=["GAME_ID"],
                )
            players.loc[
                    players.TEAM_ID == players.HOME_TEAM_ID, "GAME_LOCATION"
                ] = "home"
            players.loc[
                    players.TEAM_ID != players.HOME_TEAM_ID, "GAME_LOCATION"
                ] = "away"
            players.loc[
                    players.TEAM_ID != players.HOME_TEAM_ID, "OPP"
                ] = players.loc[players.TEAM_ID != players.HOME_TEAM_ID, "HOME_TEAM_ID"]
            players.loc[
                    players.TEAM_ID == players.HOME_TEAM_ID, "OPP"
                ] = players.loc[players.TEAM_ID == players.HOME_TEAM_ID, "VISITOR_TEAM_ID"]
            team_ids = players.groupby("TEAM_ID", as_index=False).first()[
                    ["TEAM_ID", "TEAM_ABBREVIATION"]
                ]
            players["OPP"] = players.apply(
                    lambda x: team_ids[team_ids.TEAM_ID == x.OPP].TEAM_ABBREVIATION.values[0], axis=1
                )
            teams.loc[teams.TEAM_ID == teams.HOME_TEAM_ID, "GAME_LOCATION"] = "home"
            teams.loc[teams.TEAM_ID != teams.HOME_TEAM_ID, "GAME_LOCATION"] = "away"
            teams.loc[teams.TEAM_ID != teams.HOME_TEAM_ID, "OPP"] = teams.loc[
                    teams.TEAM_ID != teams.HOME_TEAM_ID, "HOME_TEAM_ID"
                ]
            teams.loc[teams.TEAM_ID != teams.VISITOR_TEAM_ID, "OPP"] = teams.loc[
                    teams.TEAM_ID != teams.VISITOR_TEAM_ID, "VISITOR_TEAM_ID"
                ]
            teams["OPP"] = teams.apply(
                    lambda x: team_ids[team_ids.TEAM_ID == x.OPP].TEAM_ABBREVIATION.values[0], axis=1
                )
            players.fillna(0, inplace=True)
            players=reformatName(players)
            if len(gameday_frame.GAME_ID.unique())==1:
                salaries = getDKSalaries(game_date,contest='Showdown')
                salaries=salaries[salaries['Roster Position']=='FLEX']
                salaries['Roster Position']=salaries.position
            else:
                salaries = getDKSalaries(game_date)
            if salaries is not None:
                players = players.merge(
                    salaries[
                            [
                                "RotoName",
                                "position",
                                "Roster Position",
                                "Salary",
                                "TEAM_ABBREVIATION",
                            ]
                        ],
                        on=["RotoName", "TEAM_ABBREVIATION"],
                        how="left",
                    )
            players.loc[players.START_POSITION != "", "STARTED"] = True
            players.STARTED.fillna(False, inplace=True)
            players = getDKPts(players)
            team_stats = players.groupby(
                    ["TEAM_ABBREVIATION", "GAME_DATE"]
                ).sum()[[                           'PTS', 
                                                    'FG3M', 
                                                    'AST',
                                                    'STL',
                                                    'BLK',
                                                    'REB',
                                                    'TO',
                                                    "DKPts"]]
            team_stats=team_stats.add_prefix('team_')
            team_stats.reset_index(inplace=True)
            players = players.merge(team_stats,on=['TEAM_ABBREVIATION','GAME_DATE'],how='left')
            for stat in [                       'PTS', 
                                                'FG3M', 
                                                'AST',
                                                'STL',
                                                'BLK',
                                                'REB',
                                                'TO',
                                                "DKPts"]:
                if stat=='TO':
                    players[f'pct_TOV']=players[stat]/players[f'team_{stat}']
                else:
                    players[f'pct_{stat}']=players[stat]/players[f'team_{stat}']
            for c in players.columns.to_list():
                players.rename({c: c.lower()}, axis=1, inplace=True)
            players["season"] = season
            players= getMinutes(players)
            date_player_frames.append(players)
            
            pd.concat(date_player_frames).to_csv(
                    f"{datadir}/game_logs/{season}/{game_date}/{game_date}_PlayerStats.csv",
                    index=False,
                )
            for c in teams.columns.to_list():
                teams.rename({c: c.lower()}, axis=1, inplace=True)
            teams["season"] = season
            teams=teams.merge(odds_db[['team_abbreviation','game_date','proj_team_score','total_line']],on=['team_abbreviation','game_date'],how='left')
            stats=['pts','fg3m','reb','ast','stl','blk','to','dkpts']
            for stat in stats:
                teams.rename({stat:f'team_{stat}'},axis=1,inplace=True)
            teams.to_csv(
                    f"{datadir}/game_logs/{season}/{game_date}/{game_date}_TeamStats.csv",
                    index=False,
                )
            player_frames.append(players)
            team_frames.append(teams)
    pd.concat(player_frames).drop_duplicates().to_csv(
                    f"{datadir}/game_logs/PlayerStatsDatabase.csv", index=False,
                )
    team_db=pd.concat(team_frames)
    team_db.drop_duplicates().to_csv(
                    f"{datadir}/game_logs/TeamStatsDatabase.csv", index=False,
                )
        # except Exception:
        #     continue

## Merge Results with Projections Files
os.chdir("..")
#%%
from Optimizer.ProcessRankings import processRankings
player_db=pd.concat(player_frames)
for contest_type in ["Classic", "Showdown"]:
    dirs = os.listdir(f"../data/Projections/RealTime/{season}/{contest_type}")
    for game_date in dirs:
        files = os.listdir(
            f"../data/Projections/RealTime/{season}/{contest_type}/{game_date}"
        )
        for f in files:
            df = pd.read_csv(
                f"../data/Projections/RealTime/{season}/{contest_type}/{game_date}/{f}"
            )
            if "dkpts" not in df.columns.to_list():
                if game_date in player_db.game_date.unique():
                    players=player_db[player_db.game_date==game_date]
                    df = df.merge(
                        players.rename({'mins':
                                        'actual_mins'},axis=1)[
                                            ["player_id", "dkpts","actual_mins"]
                                            ], on="player_id", how="left"
                    )
                    df["Rank"] = df.dkpts.rank(ascending=False, method="min")
                    df = processRankings(df,Scaled=False)
                    df.loc[df.RG_projection==0,'Projection']=0
                    df.drop_duplicates(inplace=True)
                    df.to_csv(
                        f"../data/Projections/RealTime/{season}/{contest_type}/{game_date}/{f}",
                        index=False,
                    )