#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 08:59:54 2021

@author: robertmegnia
"""
import pandas as pd
import os
from sqlalchemy import create_engine
import pymysql
import mysql.connector
import nfl_data_py as nfl
basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
projdir = f"{basedir}/../../LiveProjections"
from ScrapeBettingOdds import ScrapeBettingOdds
from ScrapeStartingLineups import scrapeStartingLineups
from ScrapeFantasyPros import scrapeFantasyPros
from PMMPlayer import getPMM
from getDKSalaries import getDKSalaries
from Utils import *
from getDKPts import getDKPts
from RosterUtils import *
from ModelFunctions import *
from stochasticFeaturePrediction import predictFeaturesStochastically
import warnings
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
warnings.simplefilter("ignore")
#%%
# Make Live Projection
Slate = "Full"
week = 1
season = 2022

odds=ScrapeBettingOdds()
# odds = pd.read_csv(f"{datadir}/BettingOdds/Week1_BettingOdds.csv")
schedule = getSchedule(week, season)
game_date = getGameDate(week, season)
game_dates = getGameDates(week, season)
salaries = getDKSalaries(game_date, game_dates, contest="Classic")
salaries.loc[salaries.position == "DST", "full_name"] = salaries.loc[
    salaries.position == "DST", "team"
]
salaries.loc[salaries.position == "DST", "RotoName"] = salaries.loc[
    salaries.position == "DST", "team"
]

weekly_proj_df = getWeeklyRosters(season, week, schedule, odds, game_date)
# weekly_proj_df.loc[weekly_proj_df.position=='DST','RotoName']=weekly_proj_df.loc[weekly_proj_df.position=='DST','RotoName'].str.lower()
weekly_proj_df = weekly_proj_df.merge(
    salaries[["RotoName", "team", "position", "salary","ID",'Roster Position','dk_player_id']],
    on=["RotoName", "team", "position"],
    how="left",
)
weekly_proj_df = weekly_proj_df[weekly_proj_df.salary.isna() == False]
#%%
# Get Stochastic/ML Projections
def_df = pd.read_csv(f"{datadir}/game_logs/Full/DST_Database.csv")
def_df.rename({"DepthChart": "depth_team"}, axis=1, inplace=True)
def_df.game_date = pd.to_datetime(def_df.game_date)
stats_df = def_df[def_df.game_date < game_date]
defense = DefenseStochastic(weekly_proj_df, stats_df)
# dstTempSal=stats_df.groupby('gsis_id').tail(5)[['salary','gsis_id']].groupby('gsis_id',as_index=False).mean()
#
off_df = pd.read_csv(f"{datadir}/game_logs/Full/Offense_Database.csv")
off_df.game_date = pd.to_datetime(off_df.game_date)
stats_df = off_df[off_df.game_date < game_date]
STATS = stats_df.drop(KnownFeatures+OffenseNonFeatures+TeamShareFeatures,axis=1).columns
# predicted_features = predictFeaturesStochastically(weekly_proj_df, stats_df, STATS)
# new_cols={}
# for column in STATS:
#     new_cols[column]=f'proj_{column}'
# predicted_features.rename(new_cols,axis=1,inplace=True)
#%%
offense = OffenseStochastic(weekly_proj_df, stats_df)
# offTempSal=stats_df.groupby('gsis_id').tail(5)[['salary','gsis_id']].groupby('gsis_id',as_index=False).mean()
stochastic_proj_df = pd.concat([defense, offense])
stochastic_proj_df.UpsideProb.fillna(0, inplace=True)
stochastic_proj_df["DepthChart"] = (
    stochastic_proj_df.reset_index(drop=True)
    .groupby(["team", "position"])
    .salary.apply(lambda x: x.rank(ascending=False, method="min"))
)
stochastic_proj_df.to_csv(
    f"{datadir}/Projections/{season}/Stochastic/{season}_Week{week}_StochasticProjections.csv",
    index=False,
)
#%%
ml_proj_frames = []
for stat_type in ["DST", "Offense"]:
    if stat_type == "DST":
        ml_proj_df = stochastic_proj_df[stochastic_proj_df.position == "DST"]
        stats_df = pd.read_csv(
            f"{datadir}/game_logs/Full/{stat_type}_Database.csv"
        )
        stats_df["depth_team"] = 1
        ml_proj_df["team"] = ml_proj_df.gsis_id
        ml_proj_df["DepthTeam"] = 1
    else:
        ml_proj_df = stochastic_proj_df[stochastic_proj_df.position != "DST"]
        stats_df = pd.read_csv(
            f"{datadir}/game_logs/Full/{stat_type}_Database.csv"
        )
        ml_proj_df["DepthChart"] = ml_proj_df.depth_team
        ml_proj_df["player_id"] = ml_proj_df.gsis_id
    stats_df.game_date = pd.to_datetime(stats_df.game_date)
    game_date = getGameDate(week, season)
    stats_df = stats_df[stats_df.game_date < game_date]
    ml_proj_df = MLPrediction(stats_df, ml_proj_df, stat_type)
    ml_proj_frames.append(ml_proj_df)
ml_proj_df = pd.concat(ml_proj_frames)
ml_proj_df = getPMM(ml_proj_df, season, week)
#%%
# Get Rookie Predictions
games_played = (
    stats_df.groupby("gsis_id", as_index=False)
    .size()
    .rename({"size": "games_played"}, axis=1)
)
stats_df = stats_df.merge(games_played, on="gsis_id", how="left")
weekly_proj_df = weekly_proj_df.merge(games_played, on="gsis_id", how="left")
# tempSals=pd.concat([dstTempSal,offTempSal])
# weekly_proj_df=weekly_proj_df.merge(tempSals[['gsis_id','salary']],on='gsis_id',how='left')
## Fill salary NaNs
# weekly_proj_df.loc[
#     (weekly_proj_df.salary.isna() == True) & (weekly_proj_df.position == "QB"),
#     "salary",
# ] = 5000
# weekly_proj_df.loc[
#     (weekly_proj_df.salary.isna() == True) & (weekly_proj_df.position == "RB"),
#     "salary",
# ] = 4000
# weekly_proj_df.loc[
#     (weekly_proj_df.salary.isna() == True) & (weekly_proj_df.position == "WR"),
#     "salary",
# ] = 3000
# weekly_proj_df.loc[
#     (weekly_proj_df.sagglary.isna() == True) & (weekly_proj_df.position == "TE"),
#     "salary",
# ] = 2800
weekly_proj_df["depth_team"] = weekly_proj_df.groupby(
    ["team", "position"]
).salary.rank(ascending=False, method="max")
rookie_df = rookiePrediction(
    weekly_proj_df.drop("games_played", axis=1), stats_df
)
rookie_df = rookie_df[rookie_df.position != "DST"]
proj_df = ml_proj_df[~ml_proj_df.gsis_id.isin(rookie_df.gsis_id)]
proj_df = pd.concat([proj_df, rookie_df])
#%%
# Get Player Share Predictions
share_proj_df = PlayerSharesPrediction(stats_df, weekly_proj_df)
# Get Depth Chart Predictions
dc_proj = TeamDepthChartModelPredict(week, season, schedule, odds)
proj_df = pd.concat(
    [
        proj_df.groupby("gsis_id").apply(
            lambda x: merge_dc_projections(x, dc_proj)
        )
    ]
)
proj_df.reset_index(drop=True, inplace=True)
# Get TeamStats Predictions
team_stats_proj = TeamStatsModelPredict(week, season, schedule, odds)
top_down = share_proj_df.merge(
    team_stats_proj[
        [
            "team_fpts",
            "pass_attempt",
            "rush_attempt",
            "passing_fpts",
            "rushing_fpts",
            "receiving_fpts",
            "week",
            "season",
            "team",
        ]
    ],
    on=["week", "season", "team"],
    how="left",
)
top_down["TopDown"] = (
    (top_down.passing_DKPts_share * top_down.passing_fpts)
    + (top_down.rushing_DKPts_share * top_down.rushing_fpts)
    + (top_down.receiving_DKPts_share * top_down.receiving_fpts)
)
top_down.rename({'team_fpts':'proj_team_fpts',
                'passing_fpts':'proj_team_passing_fpts',
                'rushing_fpts':'proj_team_rushing_fpts',
                'receiving_fpts':'proj_team_receiving_fpts',
                'passing_DKPts_share':'proj_passing_DKPts_share',
                'rushing_DKPts_share':'proj_rushing_DKPts_share',
                'receiving_DKPts_share':'proj_receiving_DKPts_share'},axis=1,inplace=True)
proj_df.drop(['rushing_DKPts_share','receiving_DKPts_share'],axis=1,inplace=True)
proj_df = proj_df.merge(
    top_down[["gsis_id", 
              "TopDown",
              'proj_team_fpts',
              'proj_team_passing_fpts',
              'proj_team_rushing_fpts',
              'proj_team_receiving_fpts',
              'proj_passing_DKPts_share',
              'proj_rushing_DKPts_share',
              'proj_receiving_DKPts_share']], on="gsis_id", how="left"
)
#%%
proj_df["Projection"] = proj_df[
    ["Stochastic", "ML", "DC_proj", "TopDown", "PMM"]
].mean(axis=1)
proj_df.loc[
    (proj_df.position == "QB") & (proj_df.depth_team != 1), "Projection"
] = 0
dst = proj_df[proj_df.position == "DST"]
proj_df = pd.concat([proj_df, dst])
proj_df = reformatName(proj_df)
proj_df.drop_duplicates(inplace=True)
# Load Fantasy Pros
fp = pd.read_csv(
    f"{datadir}/FantasyPros/{season}/Week{week}/FP_Proj_Week{week}.csv"
)
fp = reformatFantasyPros(fp)
fp = fp[fp.FP_Proj > 0]
proj_df = proj_df.merge(
    fp[["position", "salary", "team", "opp", "RotoName", "FP_Proj"]],
    on=["salary", "team", "opp", "RotoName", "position"],
    how="left",
)
proj_df.rename({"FP_Proj": "FP_Proj2"}, axis=1, inplace=True)
fp = scrapeFantasyPros(week)
proj_df = proj_df.merge(
    fp, on=["team", "RotoName", "position", "week"], how="left"
)
# Load Rotogrinders
RG = scrapeStartingLineups()
proj_df = proj_df.merge(
    RG[["RotoName", "ownership_proj", "RG_projection", "team", "position"]],
    on=["RotoName", "team", "position"],
    how="left",
)
proj_df["Projection"] = proj_df[
    ["Projection", "FP_Proj", "FP_Proj2", "RG_projection"]
].mean(axis=1)
proj_df = getOwnership(proj_df)
# proj_df.Ownership.fillna(0.01,inplace=True)
proj_df = getImpliedOwnership(proj_df)
proj_df = getConsensusRanking(proj_df)
proj_df = getScaledProjection(proj_df)
#%%
# Weighted Projection
proj_df.loc[proj_df.position != "DST", "wProjection"] = (
    (proj_df.loc[proj_df.position != "DST", "TopDown"] * 0.25)
    + (proj_df.loc[proj_df.position != "DST", "ML"] * 0.25)
    + (proj_df.loc[proj_df.position != "DST", "PMM"] * 0.25)
    + (proj_df.loc[proj_df.position != "DST", "Stochastic"] * 0.15)
    + (proj_df.loc[proj_df.position != "DST", "DC_proj"] * 0.05)
    + (proj_df.loc[proj_df.position != "DST", "ScaledProj"] * 0.05)
)
proj_df.loc[proj_df.position == "DST", "wProjection"] = proj_df.loc[
    proj_df.position == "DST", "Projection"
]
proj_df["wProjection"] = proj_df[
    ["wProjection", "FP_Proj", "FP_Proj2", "RG_projection"]
].mean(axis=1)
proj_df.loc[proj_df.wProjection.isna() == True, "wProjection"] = proj_df.loc[
    proj_df.wProjection.isna() == True, "Projection"
]
# Merge draftkings ID back in
proj_df=proj_df.merge(weekly_proj_df[['gsis_id','ID','Roster Position','game_time']],on='gsis_id',how='left')

proj_df = proj_df[
    [
        "full_name",
        "opp",
        "position",
        "salary",
        "depth_team",
        "team",
        "game_location",
        "game_day",
        "Slate",
        "Floor",
        "Ceiling",
        "Projection",
        "TopDown",
        "DC_proj",
        "Stochastic",
        "ML",
        "wProjection",
        "FP_Proj",
        "FP_Proj2",
        "RG_projection",
        "DC_proj",
        "UpsideProb",
        "UpsideScore",
        "Leverage",
        "AvgOwnership",
        "ownership_proj",
        "Ownership",
        "ownership_proj_from_rank",
        "total_line",
        "proj_team_score",
        "spread_line",
        "opp_Rank",
        "Adj_opp_Rank",
        "rush_redzone_looks",
        "rush_value",
        "rec_air_yards",
        "yac",
        "yac_allowed",
        "targets",
        "target_share",
        "air_yards_share",
        "rec_redzone_looks",
        "wopr",
        "target_value",
        "Usage",
        "HVU",
        "HV_PPO_allowed",
        "adot",
        "DKPts_share",
        "offensive_snapcount_percentage",
        "pass_yards_allowed",
        "pass_td_allowed",
        "sacks_allowed",
        "rush_yards_allowed",
        "rush_td_allowed",
        "rush_redzone_looks_allowed",
        "rush_value_allowed",
        "pass_yds_per_att_allowed",
        "rush_yds_per_att_allowed",
        "passer_rating_allowed",
        "adot_allowed",
        "rec_air_yards_allowed",
        "RotoName",
        "ImpliedOwnership",
        "gsis_id",
        "ID",
        'Roster Position',
        'game_time',
        "season",
        "game_date",
        "week",
        'proj_team_fpts',
        'proj_team_rushing_fpts',
        'proj_team_receiving_fpts',
        'proj_team_passing_fpts',
        'proj_passing_DKPts_share',
        'proj_rushing_DKPts_share',
        'proj_receiving_DKPts_share',
    ]
]
proj_df = proj_df.round(2)
proj_df.AvgOwnership.fillna(0, inplace=True)
proj_df["week"] = week
proj_df["season"] = season
proj_df.to_csv(
    f"{datadir}/Projections/{season}/WeeklyProjections/{season}_Week{week}_Projections.csv",
    index=False,
)
#%%
try:
    gi_df = pd.read_csv(
        f"{datadir}/Projections/{season}/megnia_projections.csv"
    )
    past_gi_df = gi_df[gi_df.week != week]
    new_gi_df = proj_df[["full_name",
                        "week",
                        "season",
                        "gsis_id",
                        "position",
                        "team",
                        "wProjection",
                        "Floor",
                        'Ceiling',
                        'UpsideScore',
                        'UpsideProb',
                        "AvgOwnership",
                        "Leverage",
                        "HVU",
                        "target_value",
                        "rush_value",]]
    # new_gi_df=new_gi_df.merge(predicted_features[['gsis_id',
    #                                       'proj_target_value',
    #                                       'proj_rush_value',
    #                                       'proj_HVU',
    #                                       'week',
    #                                       'season']],on=['gsis_id','week','season'],how='left')
    ids=nfl.import_ids()
    new_gi_df=new_gi_df.merge(ids[['gsis_id','sleeper_id']],on='gsis_id',how='left')
    headshots=nfl.import_rosters([2022],columns=['sleeper_id','headshot_url','week','season'])
    headshots.sleeper_id=headshots.sleeper_id.astype(float)
    headshots = headshots[headshots.sleeper_id.isna()==False]
    new_gi_df=new_gi_df.merge(headshots,on=['sleeper_id','week','season'],how='left')
    new_gi_df.rename(
        {
            "wProjection": "Projection",
            "AvgOwnership": "Ownership",
            "HVU": "8gm_average_HVU",
            "target_value": "8gm_average_value_adjusted_targets",
            "rush_value": "8gm_average_value_adjusted_rush_attempts",
            "proj_target_value":"proj_value_adjusted_targets",
            "proj_rush_value":"proj_value_adjusted_rush_attempts",
        },
        axis=1,
        inplace=True,
        errors="ignore",
    )
    gi_df=pd.concat([past_gi_df,new_gi_df])
    gi_df.to_csv(
        f"{datadir}/Projections/{season}/megnia_projections.csv", index=False
    )
except FileNotFoundError:
    gi_df = proj_df[
        [
            "full_name",
            "week",
            "season",
            "gsis_id",
            "position",
            "team",
            "wProjection",
            "AvgOwnership",
            "Leverage",
            "target_value",
            "rush_value",
            "HVU"
        ]
    ]
    # gi_df=gi_df.merge(predicted_features[['gsis_id',
    #                                       'proj_target_value',
    #                                       'proj_rush_value',
    #                                       'proj_HVU']],on='gsis_id',how='left')
    headshots=nfl.import_rosters([2022],columns=['sleeper_id','headshot_url','week','season'])
    ids=nfl.import_ids()
    ids=ids[ids.position.isin(['QB','RB','WR','TE','FB'])]
    gi_df=gi_df.merge(ids[['sleeper_id','gsis_id']],on='gsis_id',how='left')
    headshots=nfl.import_rosters([2022],columns=['sleeper_id','headshot_url','week','season'])
    headshots.sleeper_id=headshots.sleeper_id.astype(float)
    headshots = headshots[headshots.sleeper_id.isna()==False]
    gi_df=gi_df.merge(headshots[['sleeper_id','week','season','headshot_url']],on=['sleeper_id','week','season'],how='left')
    gi_df.rename(
        {
            "wProjection": "Projection",
            "AvgOwnership": "Ownership",
            "HVU": "8gm_average_HVU",
            "target_value": "8gm_average_value_adjusted_targets",
            "rush_value": "8gm_average_value_adjusted_rush_attempts",
            "proj_target_value":"proj_value_adjusted_targets",
            "proj_rush_value":"proj_value_adjusted_rush_attempts",
        },
        axis=1,
        inplace=True,
        errors="ignore",
    )
    gi_df.to_csv(
        f"{datadir}/Projections/{season}/megnia_projections.csv", index=False
    )
#%%
# Send Projections to GI SQL Database
mydb = mysql.connector.connect(
    host="footballai-db-prod.cxgq1kandeps.us-east-2.rds.amazonaws.com",
    user="gridironai",
    password="thenameofthewind",
    database="gridironai",
)
sqlEngine = create_engine(
    "mysql+pymysql://gridironai:thenameofthewind@footballai-db-prod.cxgq1kandeps.us-east-2.rds.amazonaws.com/gridironai",
    pool_recycle=3600,
)
gi_df.to_sql(con=sqlEngine, name="megnia_projections", if_exists="replace")
