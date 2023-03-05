#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 22:10:43 2022

@author: robertmegnia
"""
import pandas as pd
import scipy
import numpy as np
import os
basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f"{basedir}/../../../data"
etcdir = f"{basedir}/../../../etc/"
OffenseNonFeatures = [
    "full_name",
    "gsis_id",
    "opp",
    "injury_status",
    "rookie_year",
    "height",
    "weight",
    "draft_number",
    "draft_round",
    "college",
    "college_conference",
    "player_id",
    "week",
    "position",
    "season",
    "game_date",
    "start_time",
    "team",
    "game_location",
    "game_id",
    "game_day",
    "Slate",
    "poe",
    "exDKPts",
    "RosterPercent"
]
TeamShareFeatures = [
    "DKPts_share_pos",
    "DKPts_share_skill_pos",
    "DKPts_share",
    "rushing_DKPts_share",
    "receiving_DKPts_share",
    "pass_yards_share",
    "pass_td_share",
    "rush_yards_share",
    "rush_td_share",
    "rec_yards_share",
    "rec_td_share",
    "rec_share",
    "fumbles_lost_share",
    "int_share",
]
KnownFeatures = [
    "total_line",
    "proj_team_score",
    "spread_line",
    "opp_Rank",
    "Adj_opp_Rank",
    "salary",
    "depth_team"
]
def predictFeaturesStochastically(players,stats_df,STATS,week,season):
    N_games=8
    # Filter Players to Offense
    offense = players[players.position != "DST"]
    offense.loc[offense.position=='FB','position']='RB'
    stats_df.loc[stats_df.position=='FB','position']='RB'
    offense['depth_team']=offense.groupby(['position','team','season','week']).salary.apply(lambda x: x.rank(ascending=False,method='min'))
    offense.set_index(["gsis_id",'position'], inplace=True)
    stats_df.set_index(["gsis_id",'position'], inplace=True)
    offense = offense.join(stats_df[STATS], lsuffix="_a")

    # Get each offense player last 16 games
    offense = offense.groupby(["gsis_id","position"]).tail(16)

    # Scale offense stats frame to z scores to get opponent influence
    # Get offense stats form last 14 games
    scaled_offense = (
        stats_df
        .groupby(["gsis_id","position"])
        .tail(N_games*2)
    )
    scaled_offense.sort_index(inplace=True)
    scaled_offense.reset_index(inplace=True)
    
    # Get playeraverage/standard dev stats over last 20 games
    scaled_averages = (
        scaled_offense.groupby(["gsis_id",'position'])
        .rolling(window=N_games, min_periods=4)
        .mean()[STATS].reset_index()
        .groupby(["gsis_id",'position'])
        .tail(N_games)
    )
    scaled_averages.drop('level_2',axis=1,inplace=True)
    scaled_stds = (
        scaled_offense.groupby(["gsis_id",'position'])
        .rolling(window=N_games, min_periods=4)
        .std()[STATS].reset_index()
        .groupby(["gsis_id",'position'])
        .tail(N_games)
    )
    scaled_stds.drop('level_2',axis=1,inplace=True)

    scaled_offense = scaled_offense.groupby(["gsis_id","position"]).tail(N_games)
    # Get offense Z scores over last 7 games
    scaled_offense[STATS] = (
        (scaled_offense[STATS] - scaled_averages[STATS])
        / scaled_stds[STATS]
    ).values
    scaled_offense.drop(scaled_offense[(scaled_offense.position=='QB')&(scaled_offense.depth_team!=1)].index,inplace=True)
    opp_stats = (
        scaled_offense.groupby(["opp", "game_date","position"])
        .mean()
        .groupby(["opp","position"])
        .tail(N_games)[STATS]
        .reset_index()
    )
    opp_stats = opp_stats[opp_stats.opp.isin(players.opp)].groupby(["opp","position"]).mean()
    opp_stats.fillna(0,inplace=True)
    quantiles = scipy.stats.norm.cdf(opp_stats)
    quantiles = pd.DataFrame(quantiles, columns=STATS).set_index(
        opp_stats.index
    )
    averages = offense.groupby(["gsis_id", "opp","position"]).mean()
    averages = averages.reset_index().set_index(["opp","position"])
    stds = offense.groupby(["gsis_id","opp","position"]).std()
    stds = stds.reset_index().set_index(["opp","position"])
    quantiles = averages.join(quantiles[STATS], lsuffix="_quant")[
        STATS
    ]
    quantiles.fillna(0.5, inplace=True)
    averages.sort_index(inplace=True)
    stds.sort_index(inplace=True)
    sims = np.random.normal(
        averages[STATS],
        stds[STATS],
        size=(10000, len(averages), len(STATS)),
    )
    sims[sims < 0] = 0
    avg=pd.DataFrame(sims.mean(axis=0),columns=STATS).set_index(averages.gsis_id)
    #
    offense = players[players.position != "DST"]
    offense['depth_team']=offense.groupby(['position','team','season','week']).salary.apply(lambda x: x.rank(ascending=False,method='min'))
    offense.loc[offense.position=='FB','position']='RB'

    median = pd.concat(
        [
            pd.DataFrame(
                np.diag(
                    pd.DataFrame(
                        sims[:, i, :], columns=STATS
                    ).quantile(quantiles.values[i, :])
                ).reshape(1, -1),
                columns=STATS,
            )
            for i in range(0, len(offense))
        ]
    ).set_index(averages.gsis_id)
    median=pd.concat([median,avg])
    median=median.groupby(median.index).mean()
    median.reset_index(inplace=True)
    median['week']=week
    median['season']=season
    median.to_csv(f'{datadir}/FeaturePredictions/{season}/Week{week}_FeaturePredictions.csv',index=False)
    return median
#%%
db = pd.read_csv(
    f"{datadir}/game_logs/Full/Offense_Database.csv")
db["depth_team"] = db.groupby(["position", "team", "season", "week"]).salary.apply(lambda x: x.rank(ascending=False, method="min"))
db.game_date = pd.to_datetime(db.game_date)
#features_db=[]
for season in range(2019, 2022):
    print(season)
    for week in range(1, 19):
        print(week)
        if (season != 2021) & (
            week == 18
        ):  # 18 regular season weeks starting 2021 season
            break
        game_date = db[
            (db.season == season) & (db.week == week)
        ].game_date.min()
            # Filter out feature data prior to the week we're predicting and sort chronologically
        stats_df = db[(db.game_date < game_date)&(db.game_date>f'09-01-{season-3}')]
        stats_df.sort_values(by="game_date", inplace=True)
        games_played = (
            stats_df.groupby("gsis_id", as_index=False)
            .size()
            .rename({"size": "games_played"}, axis=1)
            )
        stats_df = stats_df.merge(
            games_played, on="gsis_id", how="left"
            )
        stats_db = db.merge(games_played, on="gsis_id", how="left")
        stats_db.games_played.fillna(0, inplace=True)
        # Only make ML Prediction for players who have played at least 3 games
        # Use Rookie ML model sfor players who have played less than 3 games
        players = stats_db[
            (stats_db.season == season)
            & (stats_db.week == week)
            & (stats_db.games_played >= 3)
            ]
        players = players[
            (players.position.isin(["QB", "RB", "WR", "TE"]))
            & (players.injury_status == "Active")
            & (players.offensive_snapcounts > 0)
            ]
        stats_df.drop("games_played", axis=1, inplace=True)
        STATS = stats_df.drop(KnownFeatures+OffenseNonFeatures+TeamShareFeatures,axis=1).columns
        features=predictFeaturesStochastically(players, stats_df, STATS, week, season)
        #features_db.append(features)
