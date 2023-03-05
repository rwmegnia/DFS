import os
from os.path import exists
from datetime import datetime
import pandas as pd
import requests
import numpy as np
from MLB_API_TOOLS import *
from DatabaseTools import *
from advancedHittingMetrics import getAdvancedHittingData
import sys

basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f"{basedir}/../../data"
API_BASEURL = "https://statsapi.mlb.com"
URL = "https://statsapi.mlb.com/api/v1/teams?sportId=1"
response = requests.get(URL).json()
team_ids = pd.DataFrame.from_dict(response["teams"])["id"]
today = datetime.now().strftime("%Y-%m-%d")


TEAM_BATTING_SHARE_STATS = [
    "hits",
    "singles",
    "rbi",
    "runs",
    "doubles",
    "triples",
    "homeRuns",
    "strikeOuts",
    "baseOnBalls",
    "hitByPitch",
    "stolenBases",
]

TEAM_PITCHING_SHARE_STATS = [
    "strikeOuts",
    "hits",
    "hitByPitch",
    "inningsPitched",
    "earnedRuns",
]
hitter_database = pd.read_csv(f"{datadir}/game_logs/batterstatsDatabase.csv")
pitcher_database = pd.read_csv(f"{datadir}/game_logs/pitcherStatsDatabase.csv")
team_batting_database = pd.read_csv(
    f"{datadir}/game_logs/TeamBattingStatsDatabase.csv"
)
team_pitching_database = pd.read_csv(
    f"{datadir}/game_logs/TeamPitchingStatsDatabase.csv"
)
pitch_hand_db = pd.read_csv(f"{datadir}/game_logs/PitcherThrowingHands.csv")
bat_side_db = pd.read_csv(f"{datadir}/game_logs/BatterBatSide.csv")
hit_metrics_db = pd.read_csv(f"{datadir}/game_logs/HittingStatsDatabase.csv")
player_frames = []
pitcher_frames = []
team_batting_frames = []
team_pitching_frames = []
at_bat_metrics_frames = []
player_frames.append(hitter_database)
pitcher_frames.append(pitcher_database)
team_batting_frames.append(team_batting_database)
team_pitching_frames.append(team_pitching_database)
at_bat_metrics_frames.append(hit_metrics_db)
#%%
for season in range(2016, 2022):
    print(season)
    schedule = getSeasonSchedule(season)
    for date in schedule["dates"]:
        game_date = date["date"]
        print(game_date)
        if exists(f"{datadir}/game_logs/{season}/{game_date}/"):
            continue
        if game_date >= today:
            sys.exit()
        date_player_frames = []
        date_pitcher_frames = []
        date_team_batting_frames = []
        date_team_pitching_frames = []
        date_at_bat_metrics_frames = []
        for game in date["games"]:
            game_link = game["link"]
            game_data = requests.get(API_BASEURL + game_link).json()["liveData"]
            # Get game ID
            game_id = game["gamePk"]
            # Retrieve BoxScore
            boxscore = game_data["boxscore"]["teams"]

            # Process Away Team Batting Data
            away_team = boxscore["away"]
            away_team_name = teamAbbrevsDict[away_team["team"]["id"]]
            away_team_batting_stats = pd.DataFrame.from_dict(
                [away_team["teamStats"]["batting"]]
            ).reset_index(drop=True)
            away_team_batting_stats["team"] = away_team_name
            away_team_batting_stats.set_index("team", inplace=True)
            away_team_batting_stats[
                "singles"
            ] = away_team_batting_stats.hits - away_team_batting_stats[
                ["doubles", "triples", "homeRuns"]
            ].sum(
                axis=1
            )
            # Process Away Team Pitching Data
            away_team = boxscore["away"]
            away_team_name = teamAbbrevsDict[away_team["team"]["id"]]
            away_team_pitching_stats = pd.DataFrame.from_dict(
                [away_team["teamStats"]["pitching"]]
            ).reset_index(drop=True)
            away_team_pitching_stats["team"] = away_team_name
            away_team_pitching_stats.set_index("team", inplace=True)

            # Process Away Team Batting Data
            home_team = boxscore["home"]
            home_team_name = teamAbbrevsDict[home_team["team"]["id"]]
            home_team_batting_stats = pd.DataFrame.from_dict(
                [home_team["teamStats"]["batting"]]
            ).reset_index(drop=True)
            home_team_batting_stats["team"] = home_team_name
            home_team_batting_stats.set_index("team", inplace=True)
            home_team_batting_stats[
                "singles"
            ] = home_team_batting_stats.hits - home_team_batting_stats[
                ["doubles", "triples", "homeRuns"]
            ].sum(
                axis=1
            )

            # Process Away Team Pitching Data
            home_team = boxscore["home"]
            home_team_name = teamAbbrevsDict[home_team["team"]["id"]]
            home_team_pitching_stats = pd.DataFrame.from_dict(
                [home_team["teamStats"]["pitching"]]
            ).reset_index(drop=True)
            home_team_pitching_stats["team"] = home_team_name
            home_team_pitching_stats.set_index("team", inplace=True)

            # Enter Opponents
            away_team_batting_stats["opp"] = home_team_name
            away_team_pitching_stats["opp"] = home_team_name
            home_team_batting_stats["opp"] = away_team_name
            home_team_pitching_stats["opp"] = away_team_name

            # Get batting Stats
            home_batter_stats = getPlayerStats("home", "batters", boxscore)
            if home_batter_stats is None:
                continue
            home_batter_stats.fillna(0, inplace=True)

            # Away batter Stats
            away_batter_stats = getPlayerStats("away", "batters", boxscore)
            away_batter_stats.fillna(0, inplace=True)

            # Merge a slice of team stats with player stats for share calculations
            away_batting_slice = away_team_batting_stats[
                TEAM_BATTING_SHARE_STATS
            ]
            away_batting_slice = away_batting_slice.add_prefix("team_")
            home_batting_slice = home_team_batting_stats[
                TEAM_BATTING_SHARE_STATS
            ]
            home_batting_slice = home_batting_slice.add_prefix("team_")

            # Merge a slice of team stats with player stats for share calculations
            away_pitching_slice = away_team_pitching_stats[
                TEAM_PITCHING_SHARE_STATS
            ]
            away_pitching_slice = away_pitching_slice.add_prefix("team_")
            home_pitching_slice = home_team_pitching_stats[
                TEAM_PITCHING_SHARE_STATS
            ]
            home_pitching_slice = home_pitching_slice.add_prefix("team_")
            # Merge batter Stats with Team Slice Stats
            home_batter_stats = home_batter_stats.merge(
                home_batting_slice, on="team", how="left"
            )
            away_batter_stats = away_batter_stats.merge(
                away_batting_slice, on="team", how="left"
            )
            # Get pitcher Stats
            home_pitcher_stats = getPlayerStats("home", "pitchers", boxscore)
            away_pitcher_stats = getPlayerStats("away", "pitchers", boxscore)
            home_pitcher_stats = home_pitcher_stats.merge(
                home_pitching_slice, on="team", how="left"
            )
            away_pitcher_stats = away_pitcher_stats.merge(
                away_pitching_slice, on="team", how="left"
            )

            # Combine home/away batters
            batters = pd.concat([home_batter_stats, away_batter_stats])
            batters["season"] = season
            batters["game_date"] = game_date
            batters["game_id"] = game_id
            team_DKPts = (
                batters.groupby("team")
                .DKPts.sum()
                .to_frame()
                .add_prefix("team_")
                .reset_index()
            )
            batters = batters.merge(team_DKPts, on="team", how="left")
            batting_metrics, at_bat_metrics = getAdvancedHittingData(
                game_data, batters, season
            )
            at_bat_metrics["game_date"] = game_date
            date_at_bat_metrics_frames.append(at_bat_metrics)
            batters = batters.merge(batting_metrics, on="player_id", how="left")
            batters = getWOBA(batters, season)
            batters = getAverages(batters)
            batters["SB_perAtBat"] = batters.stolenBases / batters.atBats
            # Compute share stats
            for share_stat in [
                "DKPts",
                "hits",
                "runs",
                "singles",
                "doubles",
                "triples",
                "homeRuns",
                "rbi",
                "stolenBases",
                "baseOnBalls",
            ]:
                try:
                    batters[f"{share_stat}_share"] = (
                        batters[share_stat] / batters[f"team_{share_stat}"]
                    )
                except ZeroDivisionError:
                    batters[f"{share_stat}_share"] = np.nan

            #
            pitchers = pd.concat([home_pitcher_stats, away_pitcher_stats])
            pitchers["season"] = season
            pitchers["game_date"] = game_date
            pitchers["game_id"] = game_id
            pitchers["K_prcnt"] = pitchers.strikeOuts / pitchers.battersFaced
            team_batting = pd.concat(
                [away_team_batting_stats, home_team_batting_stats]
            )
            team_batting["season"] = season
            team_batting["game_date"] = game_date
            team_pitching = pd.concat(
                [away_team_pitching_stats, home_team_pitching_stats]
            )
            team_pitching["season"] = season
            team_pitching["game_date"] = game_date
            date_player_frames.append(batters)
            date_pitcher_frames.append(pitchers)
            date_team_batting_frames.append(team_batting)
            date_team_pitching_frames.append(team_pitching)
        if len(date_player_frames) == 0:
            continue
        batters = pd.concat(date_player_frames)
        pitchers = pd.concat(date_pitcher_frames)
        # Rename Position Strings to align with draft kings format
        team_batting = pd.concat(date_team_batting_frames)
        team_pitching = pd.concat(date_team_pitching_frames)
        at_bats = pd.concat(date_at_bat_metrics_frames)
        try:
            batters = getPlayerSalaries(batters, game_date)
            pitchers = getPlayerSalaries(pitchers, game_date)
            if len(batters[batters.Salary.isna() == False]) == 0:
                continue
        except OSError:
            batters["Salary"] = np.nan
            pitchers["Salary"] = np.nan
        try:
            batters.to_csv(
                f"{datadir}/game_logs/{season}/{game_date}/{game_date}_batterstats.csv",
                index=False,
            )
            pitchers.to_csv(
                f"{datadir}/game_logs/{season}/{game_date}/{game_date}_pitcherStats.csv",
                index=False,
            )
            team_batting.to_csv(
                f"{datadir}/game_logs/{season}/{game_date}/{game_date}_TeamBattingStats.csv",
                index=False,
            )
            team_pitching.to_csv(
                f"{datadir}/game_logs/{season}/{game_date}/{game_date}_TeamPitchingStats.csv",
                index=False,
            )
            at_bats.to_csv(
                f"{datadir}/game_logs/{season}/{game_date}/{game_date}_HittingStats.csv",
                index=False,
            )
        except OSError:
            os.mkdir(f"{datadir}/game_logs/{season}/{game_date}")
            batters.to_csv(
                f"{datadir}/game_logs/{season}/{game_date}/{game_date}_batterstats.csv",
                index=False,
            )
            pitchers.to_csv(
                f"{datadir}/game_logs/{season}/{game_date}/{game_date}_pitcherStats.csv",
                index=False,
            )
            team_batting.to_csv(
                f"{datadir}/game_logs/{season}/{game_date}/{game_date}_TeamBattingStats.csv",
                index=False,
            )
            team_pitching.to_csv(
                f"{datadir}/game_logs/{season}/{game_date}/{game_date}_TeamPitchingStats.csv",
                index=False,
            )
            at_bats.to_csv(
                f"{datadir}/game_logs/{season}/{game_date}/{game_date}_HittingStats.csv",
                index=False,
            )
        player_frames.append(batters)
        pitcher_frames.append(pitchers)
        team_batting_frames.append(team_batting)
        team_pitching_frames.append(team_pitching)
        at_bat_metrics_frames.append(at_bats)
        batters = pd.concat(player_frames)
        pitchers = pd.concat(pitcher_frames)
        team_batting = pd.concat(team_batting_frames)
        team_pitching = pd.concat(team_pitching_frames)
        at_bats = pd.concat(at_bat_metrics_frames)
        batters.to_csv(
            f"{datadir}/game_logs/batterstatsDatabase.csv", index=False
        )
        pitchers.to_csv(
            f"{datadir}/game_logs/pitcherStatsDatabase.csv", index=False
        )
        team_batting.to_csv(
            f"{datadir}/game_logs/TeamBattingStatsDatabase.csv", index=False
        )
        team_pitching.to_csv(
            f"{datadir}/game_logs/TeamPitchingStatsDatabase.csv", index=False
        )
        at_bats.to_csv(
            f"{datadir}/game_logs/HittingStatsDatabase.csv", index=False
        )
