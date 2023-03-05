import os
from os.path import exists
from datetime import datetime
import pandas as pd
import requests
import numpy as np
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
    "PCT_FG3M",
    "PCT_FG3A",
    "PCT_FTM",
    "PCT_FTA",
    "PCT_OREB",
    "PCT_DREB",
    "PCT_REB",
    "PCT_AST",
    "PCT_TOV",
    "PCT_STL",
    "PCT_BLK",
    "PCT_BLKA",
    "PCT_PF",
    "PCT_PFD",
    "PCT_PTS",
]
def getMinutes(df):
    df["mins"] = df["min"].apply(lambda x: str(x).split(":")[0]).astype(float)
    df["seconds"] = (
        df["min"]
        .apply(lambda x: str(x).split(":")[1] if len(str(x).split(":")) > 1 else 0)
        .astype(float)
    )
    df["mins"] = df.mins + (df.seconds / 60)
    return df["mins"]
# Read in OddsDatabase mater file
odds_db=pd.read_csv(f'{datadir}/Database/OddsDatabase.csv')
odds_db.drop(['opp'],axis=1,inplace=True)
odds_db.rename({'team':'team_abbreviation'},axis=1,inplace=True)
# Loop through last 5 season
for season in range(2022, 2023):
    print(season)
    # Pull season game log from NBA API
    gamelog = leaguegamelog.LeagueGameLog(season=season).get_data_frames()[0]
    # Loop through each day in game log
    for game_date in gamelog["GAME_DATE"].unique():
        print(game_date)
        # If file exists for this date, continue to next day
        if exists(f"{datadir}/game_logs/{season}/{game_date}/"):
            continue
        else:
            os.mkdir(f"{datadir}/game_logs/{season}/{game_date}/")
        if game_date >= today:
            sys.exit()
        date_player_frames = []
        date_team_frames = []
        gameday_frame = gamelog[gamelog.GAME_DATE == game_date]
        try:
            for game_id in gameday_frame.GAME_ID.unique():
                    print(game_id)
                    game_frames=boxscoretraditionalv2.BoxScoreTraditionalV2(
                        game_id
                    ).get_data_frames()
                    players = game_frames[0]
                    player_usage = boxscoreusagev2.BoxScoreUsageV2(
                        game_id
                    ).get_data_frames()[0][USAGE_COLS]
                    players = players.merge(player_usage, on=["PLAYER_ID"], how="left")
                    players["GAME_DATE"] = game_date
    
                    teams = game_frames[1]
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
                    salaries = getDKSalaries(game_date)
                    if salaries is None:
                        salaries=pd.DataFrame({},columns=['Name','TeamAbbrev','Position','Salary','Roster Position'])
                    # salaries.rename({"Name":"PLAYER_NAME",
                    #                  "TeamAbbrev":"TEAM_ABBREVIATION",
                    #                  "Position":"position"},axis=1,inplace=True)
                    # salaries=salaries.groupby('PLAYER_NAME',as_index=False).first()
                    players = players.merge(
                        salaries[
                            [
                                "PLAYER_NAME",
                                "position",
                                "Roster Position",
                                "Salary",
                                "TEAM_ABBREVIATION",
                            ]
                        ],
                        on=["PLAYER_NAME", "TEAM_ABBREVIATION"],
                        how="left",
                    )
                    players.loc[players.START_POSITION != "", "STARTED"] = True
                    players.STARTED.fillna(False, inplace=True)
                    players = getDKPts(players)
                    teamDKPts = players.groupby(
                        ["TEAM_ABBREVIATION", "GAME_DATE"], as_index=False
                    ).sum()
                    for c in players.columns.to_list():
                        players.rename({c: c.lower()}, axis=1, inplace=True)
                    players["season"] = season
                    teams = teams.merge(
                        teamDKPts[["TEAM_ABBREVIATION", "GAME_DATE", "DKPts"]],
                        on=["TEAM_ABBREVIATION", "GAME_DATE"],
                        how="left",
                    )
                    for c in teams.columns.to_list():
                        teams.rename({c: c.lower()}, axis=1, inplace=True)
                    teams["season"] = season
                    teams=teams.merge(odds_db[['team_abbreviation',
                                               'game_date',
                                               'proj_team_score',
                                               'total_line',
                                               'spread_line',
                                               'moneyline']],on=['team_abbreviation','game_date'],how='left')
                    date_team_frames.append(teams)
    
                    stats=['pts','fg3m','reb','ast','stl','blk','to','dkpts']
                    for stat in stats:
                        teams.rename({stat:f'team_{stat}'},axis=1,inplace=True)
                    players=players.merge(teams[['game_id','team_abbreviation']+[f'team_{stat}' for stat in stats]],on=['game_id','team_abbreviation'],how='left')
                    players['mins']=getMinutes(players)
                    players['pct_dkpts']=players.dkpts/players.team_dkpts
                    date_player_frames.append(players)
            date_player_frames=pd.concat(date_player_frames)
            date_player_frames.to_csv(
                    f"{datadir}/game_logs/{season}/{game_date}/{game_date}_PlayerStats.csv",
                    index=False,
                )
            date_team_frames=pd.concat(date_team_frames)
            date_team_frames.to_csv(
                f"{datadir}/game_logs/{season}/{game_date}/{game_date}_TeamStats.csv",
                    index=False,)
        except Exception as e:
            os.system(f'rm -r {datadir}/game_logs/{season}/{game_date}/')
            print(e)
            continue
            
            # except Exception as e:
            #     print(e)
            #     continue

