import os
from os.path import exists
from datetime import datetime
import pandas as pd
import requests
import numpy as np
from NHL_API_TOOLS import *
from advancedShootingMetrics import advancedShootingMetrics
from DatabaseTools import *
import sys

basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f"{basedir}/../../data"
API_BASEURL = "https://statsapi.web.nhl.com"
URL = "https://statsapi.web.nhl.com/api/v1/teams"
response = requests.get(URL).json()
team_ids = pd.DataFrame.from_dict(response["teams"])["id"]
today = datetime.now().strftime("%Y-%m-%d")
""" 
Stats to Scrape

['full_name',
 'id',
 'position',
 'team',
 'opp',
 'game_date',
 'game_location',
 'goals',
 'assists',
 'shots',
 'blocked_shots',
 'short_handed_goals',
 'short_handed_assists',
 'shootout_goals',
 'hat_trick'
 'five_shots',
 'three_blocks',
 'three_points',]
 
"""
player_database = pd.read_csv(f"{datadir}/game_logs/SkaterStatsDatabase.csv")
goalie_database = pd.read_csv(f"{datadir}/game_logs/GoalieStatsDatabase.csv")
team_database = pd.read_csv(f"{datadir}/game_logs/TeamStatsDatabase.csv")
player_frames = []
goalie_frames = []
team_frames = []
player_frames.append(player_database)
goalie_frames.append(goalie_database)
team_frames.append(team_database)
#%%
for season in range(2021, 2022):
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
        date_goalie_frames = []
        date_team_frames = []
        for game in date["games"]:
            game_link = game["link"]
            game_data = requests.get(API_BASEURL + game_link).json()["liveData"]
            shootingMetrics = advancedShootingMetrics(game_data, game_date)
            # Get game ID
            game_id = game["gamePk"]
            # Retrieve BoxScore
            boxscore = game_data["boxscore"]["teams"]

            # Get Final Score
            final_home_score = game["teams"]["home"]["score"]
            final_away_score = game["teams"]["away"]["score"]

            home_goals = boxscore["home"]["teamStats"]["teamSkaterStats"]["goals"]
            away_goals = boxscore["away"]["teamStats"]["teamSkaterStats"]["goals"]

            # Determine if there was overtime
            OT = getOT(home_goals, away_goals)

            # Process Away Team Data
            away_team = boxscore["away"]
            away_team_name = away_team["team"]["triCode"]
            away_team_stats = pd.DataFrame.from_dict(
                away_team["teamStats"]
            ).T.reset_index(drop=True)
            away_team_stats["team"] = away_team_name
            away_team_stats.set_index("team", inplace=True)

            # Process Home Team Data
            home_team = boxscore["home"]
            home_team_name = home_team["team"]["triCode"]
            home_team_stats = pd.DataFrame.from_dict(
                home_team["teamStats"]
            ).T.reset_index(drop=True)
            home_team_stats["team"] = home_team_name
            home_team_stats.set_index("team", inplace=True)

            # Enter Opponents
            away_team_stats["opp"] = home_team_name
            home_team_stats["opp"] = away_team_name
            # Add Penalty Kill Stats
            away_team_stats[
                "penaltyKillOpportunities"
            ] = home_team_stats.powerPlayOpportunities.astype(float).values[0]
            away_team_stats[
                "powerPlayGoalsAllowed"
            ] = home_team_stats.powerPlayGoals.astype(float).values[0]
            home_team_stats[
                "penaltyKillOpportunities"
            ] = away_team_stats.powerPlayOpportunities.astype(float).values[0]
            home_team_stats[
                "powerPlayGoalsAllowed"
            ] = away_team_stats.powerPlayGoals.astype(float).values[0]

            # Get Skater Stats
            home_skater_stats = getPlayerStats("home", "skaters", boxscore, OT)
            home_skater_stats = home_skater_stats.merge(
                shootingMetrics, on="player_id", how="left"
            )
            home_skater_stats.fillna(0, inplace=True)
            home_skater_stats["DKPts"] = home_skater_stats.DKPts + (
                home_skater_stats.shootout_goal * 1.5
            )

            # Away Skater Stats
            away_skater_stats = getPlayerStats("away", "skaters", boxscore, OT)
            away_skater_stats = away_skater_stats.merge(
                shootingMetrics, on="player_id", how="left"
            )
            away_skater_stats.fillna(0, inplace=True)
            away_skater_stats["DKPts"] = away_skater_stats.DKPts + (
                away_skater_stats.shootout_goal * 1.5
            )
            # Add Additional team stats from player stats
            away_team_stats = away_team_stats.join(
                away_skater_stats.groupby("team").agg(
                    {
                        "assists": np.sum,
                        "penaltyMinutes": np.sum,
                        "plusMinus": np.sum,
                        "shortHandedGoals": np.sum,
                        "team": "first",
                        "game_location": "first",
                        "DKPts": np.sum,
                        "Corsi": np.sum,
                        "Fenwick": np.sum,
                        "scoringChance": np.sum,
                        "HDSC": np.sum,
                        "HD_goals": np.sum,
                        "MDSC": np.sum,
                        "MD_goals": np.sum,
                        "LDSC": np.sum,
                        "LD_goals": np.sum,
                    }
                )
            )
            home_team_stats = home_team_stats.join(
                home_skater_stats.groupby("team").agg(
                    {
                        "assists": np.sum,
                        "penaltyMinutes": np.sum,
                        "plusMinus": np.sum,
                        "shortHandedGoals": np.sum,
                        "team": "first",
                        "game_location": "first",
                        "DKPts": np.sum,
                        "Corsi": np.sum,
                        "Fenwick": np.sum,
                        "scoringChance": np.sum,
                        "HDSC": np.sum,
                        "HD_goals": np.sum,
                        "MDSC": np.sum,
                        "MD_goals": np.sum,
                        "LDSC": np.sum,
                        "LD_goals": np.sum,
                    }
                )
            )
            # Merge a slice of team stats with player stats for share calculations
            away_slice = away_team_stats[
                [
                    "DKPts",
                    "shots",
                    "Corsi",
                    "Fenwick",
                    "goals",
                    "assists",
                    "blocked",
                    "scoringChance",
                    "HDSC",
                    "MDSC",
                    "LDSC",
                ]
            ]
            away_slice = away_slice.add_prefix("team_")
            home_slice = home_team_stats[
                [
                    "DKPts",
                    "shots",
                    "Corsi",
                    "Fenwick",
                    "goals",
                    "assists",
                    "blocked",
                    "scoringChance",
                    "HDSC",
                    "MDSC",
                    "LDSC",
                ]
            ]
            home_slice = home_slice.add_prefix("team_")
            # Merge Skater Stats with Team Slice Stats
            home_skater_stats = home_skater_stats.merge(
                home_slice, on="team", how="left"
            )
            away_skater_stats = away_skater_stats.merge(
                away_slice, on="team", how="left"
            )
            # Get Goalie Stats
            home_goalie_stats = getPlayerStats("home", "goalies", boxscore, OT)
            away_goalie_stats = getPlayerStats("away", "goalies", boxscore, OT)
            # Combine home/away skaters
            skaters = pd.concat([home_skater_stats, away_skater_stats])
            skaters["season"] = season
            skaters["game_date"] = game_date
            # Compute share stats
            for share_stat in [
                "DKPts",
                "shots",
                "goals",
                "assists",
                "blocked",
                "Corsi",
                "Fenwick",
                "scoringChance",
                "HDSC",
                "MDSC",
                "LDSC",
            ]:
                try:
                    skaters[f"{share_stat}_share"] = (
                        skaters[share_stat] / skaters[f"team_{share_stat}"]
                    )
                except ZeroDivisionError:
                    skaters[f"{share_stat}_share"] = np.nan

            # Compute per minute stats
            skaters["SPM"] = skaters.shots / skaters.timeOnIce
            skaters["BPM"] = skaters.blocked / skaters.timeOnIce
            skaters["PPM"] = skaters.points / skaters.timeOnIce
            skaters["GPM"] = skaters.goals / skaters.timeOnIce
            skaters["APM"] = skaters.assists / skaters.timeOnIce
            skaters["CPM"] = skaters.Corsi / skaters.timeOnIce
            skaters["FPM"] = skaters.Fenwick / skaters.timeOnIce
            skaters["SCPM"] = skaters.scoringChance / skaters.timeOnIce
            skaters["HDSCPM"] = skaters.HDSC / skaters.timeOnIce
            skaters["MDSCPM"] = skaters.MDSC / skaters.timeOnIce
            skaters["LDSCPM"] = skaters.LDSC / skaters.timeOnIce
            #
            goalies = pd.concat([home_goalie_stats, away_goalie_stats])
            goalies["season"] = season
            goalies["game_date"] = game_date
            teams = pd.concat([away_team_stats, home_team_stats])
            teams["season"] = season
            teams["game_date"] = game_date
            date_player_frames.append(skaters)
            date_goalie_frames.append(goalies)
            date_team_frames.append(teams)
        skaters = pd.concat(date_player_frames)
        goalies = pd.concat(date_goalie_frames)
        # Rename Position Strings to align with draft kings format
        skaters.position.replace({"L": "LW", "R": "RW"}, inplace=True)
        skaters["Roster Position"] = skaters.position.apply(lambda x: x[-1] + "/UTIL")
        goalies["Roster Position"] = "G"
        teams = pd.concat(date_team_frames)
        try:
            skaters = getPlayerSalaries(skaters, game_date)
            goalies = getPlayerSalaries(goalies, game_date)
            if len(skaters[skaters.Salary.isna() == False]) == 0:
                continue
            skaters["line"] = skaters.groupby(["position", "team"]).timeOnIce.apply(
                lambda x: x.rank(ascending=False, method="first")
            )
            skaters["powerPlayTimeOnIceRank"] = skaters.groupby(
                "team"
            ).powerPlayTimeOnIce.rank(ascending=False, method="first")
            skaters.loc[skaters.powerPlayTimeOnIceRank <= 10, "PowerPlayLine"] = True
            skaters.PowerPlayLine.fillna(False, inplace=True)
            skaters.loc[(skaters.position == "D") & (skaters.line <= 2), "line"] = 1
            skaters.loc[(skaters.position == "D") & (skaters.line == 3), "line"] = 2
            skaters.loc[(skaters.position == "D") & (skaters.line == 4), "line"] = 2
            skaters.loc[(skaters.position == "D") & (skaters.line >= 5), "line"] = 3
        except FileNotFoundError:
            skaters["Salary"] = np.nan
            skaters["line"] = np.nan
            skaters["PowerPlayLine"] = np.nan
            goalies["Salary"] = np.nan
        try:
            skaters.to_csv(
                f"{datadir}/game_logs/{season}/{game_date}/{game_date}_SkaterStats.csv",
                index=False,
            )
            goalies.to_csv(
                f"{datadir}/game_logs/{season}/{game_date}/{game_date}_GoalieStats.csv",
                index=False,
            )
            teams.to_csv(
                f"{datadir}/game_logs/{season}/{game_date}/{game_date}_TeamStats.csv",
                index=False,
            )
        except FileNotFoundError:
            os.mkdir(f"{datadir}/game_logs/{season}/{game_date}")
            skaters.to_csv(
                f"{datadir}/game_logs/{season}/{game_date}/{game_date}_SkaterStats.csv",
                index=False,
            )
            goalies.to_csv(
                f"{datadir}/game_logs/{season}/{game_date}/{game_date}_GoalieStats.csv",
                index=False,
            )
            teams.to_csv(
                f"{datadir}/game_logs/{season}/{game_date}/{game_date}_TeamStats.csv",
                index=False,
            )
        player_frames.append(skaters)
        goalie_frames.append(goalies)
        team_frames.append(teams)
        skaters = pd.concat(player_frames)
        goalies = pd.concat(goalie_frames)
        teams = pd.concat(team_frames)
        skaters.to_csv(f"{datadir}/game_logs/SkaterStatsDatabase.csv", index=False)
        goalies.to_csv(f"{datadir}/game_logs/GoalieStatsDatabase.csv", index=False)
        teams.to_csv(f"{datadir}/game_logs/TeamStatsDatabase.csv", index=False)
