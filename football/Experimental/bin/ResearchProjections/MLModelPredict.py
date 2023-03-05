#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 08:59:54 2021

@author: robertmegnia
"""
import numpy as np
import pandas as pd
# import nflfastpy as nfl
from scipy.stats import norm
import requests
import os
from sklearn.cluster import KMeans

basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
projdir = f"{datadir}/Projections"
#%%
# from config.MLModel_config import *
from config.ColumnMappings import *
from PMMPlayer import getPMM
from getDKPts import getDKPts
#%%
# from RosterUtils import *
os.chdir(f"{basedir}/../")
from Models.ModelDict import *


def rolling_average(df, window):
    return df.rolling(min_periods=window, window=window).mean()


def rolling_quantile(df, window=8):
    return df.rolling(min_periods=1, window=window).quantile(0.65)

def standardizeStats(df,DK_stats):
        averages=df.rolling(window=8,min_periods=4).mean()[DK_stats]
        stds=df.rolling(window=8,min_periods=4).mean()[DK_stats]
        df[DK_stats]=(df[DK_stats]-averages[DK_stats])/stds[DK_stats]
        df.fillna(0,inplace=True)
        return df

def MLPrediction(stats_df,weekly_proj_df,stat_type):
    # Columns that we know the values of going into a week
    if stat_type=='DST':
        NonFeatures=ML_DefenseNonFeatures
        DK_stats=DK_DST_Stoch_stats
        ml_feature_cols=ML_DEFENSE_FEATURE_COLS
    else:
        NonFeatures=ML_OffenseNonFeatures
        DK_stats=DK_Stoch_stats
        ml_feature_cols=ML_OFFENSE_FEATURE_COLS

    matchup_model=KMeans(n_clusters=4).fit(stats_df[['proj_team_score','total_line','spread_line']])     
    stats_df['matchup']=matchup_model.predict(stats_df[['proj_team_score','total_line','spread_line']])
    weekly_proj_df['matchup']=matchup_model.predict(weekly_proj_df[['proj_team_score','total_line','spread_line']])
    KnownFeatures=MLKnownFeatures
    TeamShareFeatures=MLTeamShareFeatures
    # Get Opp Stats First
    opp_features = stats_df[stats_df.season>=2020].groupby("gsis_id").apply(
        lambda x: standardizeStats(x, DK_stats)
    )
    opp_features = (
        opp_features.groupby(
            ["opp", "season", "week", "depth_team", "position","matchup"]
        )
        .mean()
        .drop(
            NonFeatures + KnownFeatures + TeamShareFeatures,
            axis=1,
            errors="ignore",
        )
    )
    opp_features = opp_features.groupby(
        ["opp", "depth_team", "position","matchup"]
    ).apply(lambda x: rolling_average(x,window=5))
    opp_features = opp_features.groupby(
        ["opp", "position", "depth_team","matchup"]
    ).last()
    # Add _allowed suffix for defense featurse
    opp_features = opp_features.add_suffix("_allowed")
    opp_features.reset_index(inplace=True)
    # Sort stats database chronologically and filter down to active players for the week
    stats_df.sort_values(by='game_date',inplace=True)
    stats_df=stats_df[stats_df.gsis_id.isin(weekly_proj_df.gsis_id)]
    stats_df.set_index(['gsis_id','depth_team','position','matchup'],inplace=True)

    features = (
        stats_df.drop(KnownFeatures, axis=1, errors="ignore")
        .groupby(["gsis_id","depth_team",'position','matchup'])
        .apply(lambda x: rolling_average(x,5))
    )

    # Reinsert gsis_id, get last of rolling averages, insert known features
    # features[["gsis_id","depth_team", "position"]] = stats_df[["gsis_id","depth_team", "position"]]
    features.reset_index(inplace=True)
    features = features.groupby(["gsis_id","depth_team","position",'matchup'], as_index=False).last()


    # Merge Offense and Defense features
    features.set_index(["gsis_id","depth_team","position","matchup"], inplace=True)
    features = (
        features.drop(KnownFeatures, axis=1, errors="ignore")
        .join(
            weekly_proj_df[KnownFeatures + ["gsis_id", "position","opp"]].set_index(
                ["gsis_id","depth_team","position","matchup"]
            )
        )
        .reset_index()
    )
    features = features.merge(
        opp_features, on=["opp", "position", "depth_team","matchup"], how="left"
    )
    features.rename({"DKPts": "avg_DKPts"}, axis=1, inplace=True)
    ProjFrame = weekly_proj_df[
        [c for c in NonFeatures if c in weekly_proj_df.columns.to_list()]
    ]
    ProjFrame.set_index("gsis_id", inplace=True)
    ProjFrame = ProjFrame.join(
        features.set_index("gsis_id").drop(NonFeatures, axis=1, errors="ignore")
    )
    ML = ProjFrame.dropna()
    ML_nan = ProjFrame[ProjFrame.isna().any(axis=1)]
    ML_nan=ML_nan[~ML_nan.index.isin(ML.index)]
    ML_nan=ML_nan.drop(KnownFeatures,axis=1,errors='ignore').join(
        weekly_proj_df.set_index('gsis_id')[KnownFeatures])
    ML_nan=ML_nan.groupby(ML_nan.index).first()
    if stat_type!='DST':
        for position in ML.position.unique():
            for stat in ['pass_yards','pass_td','rush_yards','rush_td','rec','rec_yards','rec_td']:
                models=PlayerStatsModelDict[position][stat]
                for method,model in models.items():
                    if (position=='QB')&(stat in ['rec','rec_yards','rec_td']):
                        ML.loc[ML.position==position,f'{method}_{stat}']=0
                        continue
                    if (position in ['RB','WR','TE'])&(stat in ['pass_yards','pass_td']):
                        ML.loc[ML.position==position,f'{method}_{stat}']=0
                        continue
                    ML.loc[ML.position==position,f'{method}_{stat}']=model.predict(
                        ML.loc[
                            ML.position==position,
                            ml_feature_cols])  
                    ML.loc[(ML.position==position)&(ML[f'{method}_{stat}']<0),f'{method}_{stat}']=0
                ML.loc[ML.position==position,stat]=ML.loc[ML.position==position][[f'BR_{stat}',f'EN_{stat}',f'NN_{stat}',f'RF_{stat}',f'GB_{stat}',f'Tweedie_{stat}']].mean(axis=1)
    for position in ML.position.unique():
        models = ModelDict[position]
        for method, model in models.items():
            if position=='DST':
                if 'matchup' in ml_feature_cols:
                    ml_feature_cols.remove('matchup')
            ML.loc[ML.position == position, method] = model.predict(
                ML.loc[
                    ML.position == position,
                    ml_feature_cols,
                ]
            )
            ML.loc[(ML.position == position) & (ML[method] < 0), method] = 0
    ML["ML"] = ML[["BR", "EN", "NN", "RF", "GB", "Tweedie"]].mean(axis=1)
    ProjFrame = pd.concat([ML, ML_nan])
    ProjFrame["DKPts"] = weekly_proj_df.set_index("gsis_id").DKPts
    if stat_type!='DST':
        for stat in ['pass_yards','pass_td','rush_yards','rush_td','rec','rec_yards','rec_td']:
            ProjFrame[f'{stat}_actual']=weekly_proj_df.set_index('gsis_id')[stat]
    ProjFrame.reset_index(inplace=True)
    return ProjFrame


#%%
# Build Machine Learning Projections Database
datadir = f"{basedir}/../../data"
# Loop through each season and week
for season in range(2022,2023):
    print(season)
    for week in range(1,10):
        if (season != 2021) & (
            week == 18
        ):  # 18 regular season weeks starting 2021 season
            break
        print(week)
        proj_frames = []  # List to place offense/dst projections frames
        for stat_type in ["Offense","DST"]:
            if stat_type == "Offense":
                db = pd.read_csv(
                    f"{datadir}/game_logs/Full/{stat_type}_Database.csv"
               )
                # Assign depth charts by salary rank
                db["depth_team"] = db.groupby(
                    ["position", "team", "season", "week"]
                ).salary.apply(lambda x: x.rank(ascending=False, method="min"))
            else:
                db = pd.read_csv(
                    f"{datadir}/game_logs/Full/{stat_type}_Database.csv"
                )
                # Assign depth team to 1
                db["depth_team"] = 1
            # convert game_date column to datetime object and retrieve game_date of the first game for week in iteration
            db.game_date = pd.to_datetime(db.game_date)
            game_date = db[
                (db.season == season) & (db.week == week)
            ].game_date.min()
            if pd.isnull(game_date):
                game_date = getGameDate(week, season)
            # Filter out feature data prior to the week we're predicting and sort chronologically
            stats_df = db[db.game_date < game_date]
            stats_df.sort_values(by="game_date", inplace=True)
            if stat_type != "DST":
                games_played = (
                    stats_df.groupby("gsis_id", as_index=False)
                    .size()
                    .rename({"size": "games_played"}, axis=1)
                )
                stats_df = stats_df.merge(
                    games_played, on="gsis_id", how="left"
                )
                db = db.merge(games_played, on="gsis_id", how="left")
                db.games_played.fillna(0, inplace=True)
                # Only make ML Prediction for players who have played at least 3 games
                # Use Rookie ML model sfor players who have played less than 3 games
                weekly_proj_df = db[
                    (db.season == season)
                    & (db.week == week)
                    & (db.games_played >= 3)
                ]
                weekly_proj_df = weekly_proj_df[
                    (weekly_proj_df.position.isin(["QB", "RB", "WR", "TE"]))
                    & (weekly_proj_df.injury_status == "Active")
                    & (weekly_proj_df.offensive_snapcounts > 0)
                ]
                stats_df.drop("games_played", axis=1, inplace=True)
            else:
                db["depth_chart_position"] = "DST"
                weekly_proj_df = db[(db.season == season) & (db.week == week)]
            weekly_proj_df.drop(weekly_proj_df[(weekly_proj_df.position=='QB')&(weekly_proj_df.depth_team>1)].index,inplace=True)
            weekly_proj_df.drop(weekly_proj_df[(weekly_proj_df.position=='RB')&(weekly_proj_df.depth_team>2)].index,inplace=True)
            weekly_proj_df.drop(weekly_proj_df[(weekly_proj_df.position=='WR')&(weekly_proj_df.depth_team>4)].index,inplace=True)
            weekly_proj_df.drop(weekly_proj_df[(weekly_proj_df.position=='TE')&(weekly_proj_df.depth_team>2)].index,inplace=True)
            stats_df.drop(stats_df[(stats_df.position=='QB')&(stats_df.depth_team>1)].index,inplace=True)
            stats_df.drop(stats_df[(stats_df.position=='RB')&(stats_df.depth_team>2)].index,inplace=True)
            stats_df.drop(stats_df[(stats_df.position=='WR')&(stats_df.depth_team>4)].index,inplace=True)
            stats_df.drop(stats_df[(stats_df.position=='Te')&(stats_df.depth_team>2)].index,inplace=True)

            ml_proj_df = MLPrediction(stats_df, weekly_proj_df, stat_type)
            print(ml_proj_df[ml_proj_df.position == "RB"].BR.max())
            proj_frames.append(ml_proj_df)
        if len(proj_frames) == 0:
            continue
        proj_df = pd.concat(proj_frames)
        proj_df = getPMM(proj_df, season, week)
        proj_df.to_csv(
            f"{projdir}/{season}/ML/{season}_Week{week}_MLProjections2.csv",
            index=False,
        )
