#%% Import External Modules
import pandas as pd
import numpy as np
import warnings
import nfl_data_py as nfl
from bs4 import BeautifulSoup as BS
from sklearn.cluster import KMeans
import requests
from datetime import datetime
import os
basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f"{basedir}/../../../data"
etcdir = f"{basedir}/../../../etc/"
import sys
import pickle
sys.path.append('../')
sys.path.append('../TeamDatabase')
from TeamDatabase import UpdateTeamDatabase
from DepthChartDatabase.UpdateDepthChartDatabase import updateDepthChartsDB
os.chdir(f'{basedir}')
import sys
import pickle
sys.path.append('../')
from TeamDatabase import UpdateTeamDatabase
from DepthChartDatabase.UpdateDepthChartDatabase import updateDepthChartsDB
# Import Internal Modules
from config import *
from DatabaseUtils.DraftKingsPoints import *
from DatabaseUtils.getEPA import getEPAPerPlay
from DatabaseUtils.AdvancedStats import *
from DatabaseUtils.SharedStats import getSharedStats
from DatabaseUtils.OpponentRanks import *
from DatabaseUtils.LoadData import *
from DatabaseUtils.getMillionairePlayerPool import getMilliPlayers


WR_exDKPts_model = pickle.load(
    open(f"{etcdir}/model_pickles/WR_exDKPts_model.pkl", "rb")
)
TE_exDKPts_model = pickle.load(
    open(f"{etcdir}/model_pickles/TE_exDKPts_model.pkl", "rb")
)


def rolling_average(df, window=4):
    return df.rolling(min_periods=4, window=window).mean().shift(1)


warnings.simplefilter(action="ignore", category=Warning)
# pd.set_option('display.max_columns', 5)
# pd.set_option('display.max_rows',60)
def __init__(season, update=False):
    start_time = datetime.utcnow()
    timer_start = datetime.utcnow()
    print("Started script at {} ".format(start_time))
    DatabasePath = os.getcwd()
    df = load_pbp_data(season, update=True)
    load_player_data(df, season, update)
    load_DST_data(df, season)
    print(f"{season} Season Database Update Complete!")
    timer_end = datetime.utcnow()
    script_time = (timer_end - timer_start).seconds
    minutes = script_time // 60
    seconds = script_time - minutes * 60
    print(f"Database update took {minutes} minutes {seconds} seconds")


def load_player_data(df, season, update):
    roster_data = load_roster_data(season, update)
    roster_data.fillna(-1, inplace=True)
    # Construct Passing Dataframe
    df["gsis_id"] = df["passer_player_id"]
    Pass_df = df.groupby(list(PBP_STATIC_COLUMNS.keys()), as_index=False)[
        list(PBP_PASS_STATS_COLUMNS.keys())
    ].sum()
    Pass_df = getEPAPerPlay(Pass_df, df)
    Pass_df.rename(PBP_PASS_STATS_COLUMNS, axis=1, inplace=True)
    Pass_df.rename(PBP_STATIC_COLUMNS, axis=1, inplace=True)
    Pass_df.set_index(list(PBP_STATIC_COLUMNS.values()), inplace=True)

    # Construct Rushing Dataframe
    df["gsis_id"] = df["rusher_player_id"]
    Rush_df = df.groupby(list(PBP_STATIC_COLUMNS.keys()), as_index=False)[
        list(PBP_RUSH_STATS_COLUMNS.keys())
    ].sum()
    Rush_df = getRushShare(Rush_df, df)
    Rush_df = getRushRedZoneLooks(Rush_df, df)
    Rush_df = getValuedRushes(Rush_df, df)
    Rush_df.rename(PBP_RUSH_STATS_COLUMNS, axis=1, inplace=True)
    Rush_df.rename(PBP_STATIC_COLUMNS, axis=1, inplace=True)
    Rush_df.set_index(list(PBP_STATIC_COLUMNS.values()), inplace=True)

    # Construct Receiving Dataframe
    df["gsis_id"] = df["receiver_player_id"]
    Rec_df = df.groupby(list(PBP_STATIC_COLUMNS.keys()), as_index=False)[
        list(PBP_REC_STATS_COLUMNS.keys())
    ].sum()
    Rec_df = getRecAdvancedStats(Rec_df, df)
    Rec_df = getValuedTargets(Rec_df, df)
    Rec_df["exDKPts"] = WR_exDKPts_model.predict(
        Rec_df[["air_yards", "targets"]]
    )
    # Rec_df["poe"] = Rec_df.DKPts - rec_df.exDKPts
    Rec_df.rename(PBP_REC_STATS_COLUMNS, axis=1, inplace=True)
    Rec_df.rename(PBP_STATIC_COLUMNS, axis=1, inplace=True)
    Rec_df.set_index(list(PBP_STATIC_COLUMNS.values()), inplace=True)

    # Merge Dataframes
    Pass_df_final = Pass_df.join(Rush_df).join(Rec_df)
    Rush_df_final = Rush_df.join(Pass_df).join(Rec_df)
    Rec_df_final = Rec_df.join(Pass_df).join(Rush_df)
    Stats_df = pd.concat(
        [Pass_df_final, Rush_df_final, Rec_df_final]
    ).drop_duplicates()
    Stats_df.reset_index(inplace=True)
    roster_data = roster_data.merge(
        Stats_df, on=["week", "team", "gsis_id"], how="left"
    )
    epas = Stats_df.groupby(["team", "week"], as_index=False)[
        [c for c in Stats_df.columns if "epa" in c]
    ].mean()
    other_known_features = Stats_df.groupby(["team", "week"])[
        [
            "game_date",
            "game_id",
            "game_location",
            "proj_team_score",
            "total_line",
            "opp",
            "start_time",
        ]
    ].first()
    roster_data = roster_data.drop(
        [c for c in Stats_df.columns if "epa" in c], axis=1
    ).merge(epas, on=["team", "week"], how="left")
    roster_data = roster_data.drop(
        other_known_features.columns, axis=1, errors="ignore"
    ).merge(other_known_features, on=["team", "week"], how="left")
    Stats_df = roster_data
    Stats_df = Stats_df[
        (Stats_df.offense_epa.isna() == False)
        & (Stats_df.proj_team_score.isna() == False)
    ]
    # Additional Statistics
    Stats_df["fumbles_lost"] = Stats_df[
        ["pass_fumble_lost", "rush_fumble_lost", "rec_fumble_lost"]
    ].sum(axis=1)
    Stats_df["Usage"] = Stats_df[["pass_att", "rush_att", "targets"]].sum(
        axis=1
    )
    Stats_df["HVU"] = Stats_df[["pass_att", "rush_value", "target_value"]].sum(
        axis=1
    )
    Stats_df["pass_yds_per_att"] = Stats_df.pass_yards / Stats_df.pass_att
    Stats_df["rush_yds_per_att"] = Stats_df.rush_yards / Stats_df.rush_att
    Stats_df["passer_rating"] = Stats_df.apply(
        lambda x: getPasserRating(
            x.pass_cmp, x.pass_td, x.int, x.pass_yards, x.pass_att
        ),
        axis=1,
    )
    Stats_df["adot"] = Stats_df.rec_air_yards / Stats_df.targets
    # Stats_df.loc[Stats_df.depth_chart_position.isna()==True,'depth_chart_position']=Stats_df.loc[Stats_df.depth_chart_position.isna()==True,'position']
    Stats_df.fillna(0, inplace=True)
    Stats_df["passing_DKPts"] = Stats_df.apply(
        lambda x: getDKPts_passing(
            x.pass_yards, x.pass_td, x.pass_fumble_lost, x.int,x.pass_two_point_conv
        ),
        axis=1,
    )
    Stats_df["rushing_DKPts"] = Stats_df.apply(
        lambda x: getDKPts_rushing(x.rush_yards, x.rush_td, x.rush_fumble_lost,x.rush_two_point_conv),
        axis=1,
    )
    Stats_df["receiving_DKPts"] = Stats_df.apply(
        lambda x: getDKPts_receiving(
            x.rec_yards, x.rec, x.rec_td, x.rec_fumble_lost,x.rec_two_point_conv
        ),
        axis=1,
    )
    Stats_df["DKPts"] = Stats_df[
        ["passing_DKPts", "rushing_DKPts", "receiving_DKPts"]
    ].sum(axis=1)
    Stats_df["poe"] = Stats_df.DKPts - Stats_df.exDKPts
    Stats_df["PPO"] = Stats_df.DKPts / Stats_df.Usage
    Stats_df["HV_PPO"] = Stats_df.DKPts / Stats_df.HVU

    Stats_df["game_date"] = pd.to_datetime(
        Stats_df.game_date + " " + Stats_df.start_time
    ).apply(lambda x: x.strftime("%a %m-%d-%Y %I:%M %p"))
    Stats_df = getSharedStats(Stats_df)
    Stats_df.fillna(0, inplace=True)
    Stats_df["Rank"] = Stats_df.groupby(["position", "week"]).DKPts.apply(
        lambda x: x.rank(ascending=False, method="min")
    )
    # Load Draftkings salary data (goes back to 2014)
    # Load Draftkings snapcount data (goes back to 2012)
    if season >= 2012:
        Stats_df = load_snapcount_data(Stats_df, season)
    else:
        Stats_df["offensive_snapcounts"] = np.nan
        Stats_df["offensive_snapcount_percentage"] = np.nan

    if season >= 2014:
        Stats_df = load_salary_data(Stats_df, season, "Offense")
    else:
        Stats_df["salary"] = np.nan

    if season >= 2020:
        Stats_df = load_ownership_data(Stats_df, season, last_week)
        Stats_df.sort_values(by="RosterPercent", ascending=False, inplace=True)
        Stats_df = Stats_df.groupby(
            ["gsis_id", "week", "season"], as_index=False
        ).first()
        Stats_df.sort_values(by="game_date", inplace=True)
    Stats_df.drop_duplicates(inplace=True)
    # Determine depth team by salary
    Stats_df.sort_values(
        by=["salary", "offensive_snapcounts", "DKPts"],
        ascending=False,
        inplace=True,
    )
    Stats_df["depth_team"] = Stats_df.groupby(
        ["team", "position", "week", "season"]
    ).DKPts.rank(ascending=False, method="first")

    Stats_df.to_csv(
        f"{datadir}/game_logs/{season}/{season}_Offense_GameLogs.csv",
        index=False,
    )
    return Stats_df


def load_DST_data(df, season):
    print(f"Processing DST Data {season}...")
    df["gsis_id"] = df.defteam
    DST_df = df.groupby(list(PBP_DST_STATIC_COLUMNS.keys()), as_index=False)[
        list(PBP_DST_STATS_COLUMNS.keys())
    ].sum()
    DST_df["posteam_int_return_td"] = DST_df[["posteam", "game_id"]].apply(
        lambda x: DST_df[
            (DST_df.defteam == x.posteam) & (DST_df.game_id == x.game_id)
        ].int_return_td.values[-1],
        axis=1,
    )
    DST_df["posteam_fumble_return_td"] = DST_df[["posteam", "game_id"]].apply(
        lambda x: DST_df[
            (DST_df.defteam == x.posteam) & (DST_df.game_id == x.game_id)
        ].fumble_return_td.values[-1],
        axis=1,
    )
    DST_df["defteam_fumble_return_td"] = DST_df[["defteam", "game_id"]].apply(
        lambda x: DST_df[
            (DST_df.defteam == x.defteam) & (DST_df.game_id == x.game_id)
        ].fumble_return_td.values[-1],
        axis=1,
    )
    DST_df["defteam_int_return_td"] = DST_df[["defteam", "game_id"]].apply(
        lambda x: DST_df[
            (DST_df.defteam == x.defteam) & (DST_df.game_id == x.game_id)
        ].int_return_td.values[-1],
        axis=1,
    )
    DST_df["defteam_fg_return_td"] = DST_df[["defteam", "game_id"]].apply(
        lambda x: DST_df[
            (DST_df.defteam == x.defteam) & (DST_df.game_id == x.game_id)
        ].fg_return_td.values[-1],
        axis=1,
    )
    DST_df["defteam_fg_block_return_td"] = DST_df[["defteam", "game_id"]].apply(
        lambda x: DST_df[
            (DST_df.defteam == x.defteam) & (DST_df.game_id == x.game_id)
        ].fg_block_return_td.values[-1],
        axis=1,
    )
    DST_df["defteam_punt_return_td"] = DST_df[["defteam", "game_id"]].apply(
        lambda x: DST_df[
            (DST_df.defteam == x.defteam) & (DST_df.game_id == x.game_id)
        ].punt_return_td.values[-1],
        axis=1,
    )
    DST_df["defteam_kick_return_td"] = DST_df[["posteam", "game_id"]].apply(
        lambda x: DST_df[
            (DST_df.defteam == x.posteam) & (DST_df.game_id == x.game_id)
        ].kick_return_td.values[-1],
        axis=1,
    )
    DST_df["defteam_punt_block_return_td"] = DST_df[
        ["posteam", "game_id"]
    ].apply(
        lambda x: DST_df[
            (DST_df.defteam == x.posteam) & (DST_df.game_id == x.game_id)
        ].punt_block_return_td.values[-1],
        axis=1,
    )
    DST_df["return_touchdown"] = (
        DST_df.defteam_int_return_td
        + DST_df.defteam_fumble_return_td
        + DST_df.defteam_fg_return_td
        + DST_df.defteam_fg_block_return_td
        + DST_df.defteam_punt_return_td
        + DST_df.defteam_kick_return_td
        + DST_df.defteam_punt_block_return_td
    )
    DST_df["blocks"] = DST_df.punt_blocked + DST_df.fg_blocked
    DST_df["defteam_type"] = DST_df.posteam_type
    DST_df.defteam_type[DST_df.posteam_type == "away"] = "home"
    DST_df.defteam_type[DST_df.posteam_type == "home"] = "away"
    DST_df["posteam_score"] = DST_df[DST_df.posteam_type == "home"].home_score
    DST_df["posteam_score"][DST_df.posteam_type == "away"] = DST_df[
        DST_df.posteam_type == "away"
    ].away_score
    DST_df["defteam_score"] = DST_df[DST_df.defteam_type == "home"].home_score
    DST_df["defteam_score"][DST_df.defteam_type == "away"] = DST_df[
        DST_df.defteam_type == "away"
    ].away_score
    DST_df["points_allowed"] = (
        DST_df["posteam_score"]
        - (DST_df.posteam_int_return_td * 6)
        - (DST_df.posteam_fumble_return_td * 6)
    )
    DST_df = getEPAPerPlay(DST_df, df)
    DST_df.rename(PBP_DST_STATIC_COLUMNS, axis=1, inplace=True)
    DST_df.rename(PBP_DST_STATS_COLUMNS, axis=1, inplace=True)

    # Format Dataframe and export to csv
    DST_df["position"] = "DST"
    DST_df["game_date"] = pd.to_datetime(
        DST_df.game_date + " " + DST_df.start_time
    ).apply(lambda x: x.strftime("%a %m-%d-%Y %I:%M %p"))
    DST_df = DST_df[
        [
            "gsis_id",
            "full_name",
            "position",
            "game_date",
            "game_id",
            "start_time",
            "week",
            "game_location",
            "opp",
            "fumble_recoveries",
            "sack",
            "qb_hit",
            "tfl",
            "interception",
            "blocks",
            "safety",
            "return_touchdown",
            "points_allowed",
            "offense_epa",
            "offense_pass_epa",
            "offense_rush_epa",
            "defense_epa",
            "defense_pass_epa",
            "defense_rush_epa",
            "proj_team_score",
            "total_line",
        ]
    ]
    DST_df.replace("home", "VS", inplace=True)
    DST_df.replace("away", "@", inplace=True)
    DST_df.fillna(0, inplace=True)
    DST_df["DKPts"] = DST_df.apply(
        lambda x: getDKPtsDST(
            x.fumble_recoveries,
            x.interception,
            x.sack,
            x.blocks,
            x.safety,
            x.return_touchdown,
            x.points_allowed,
        ),
        axis=1,
    )
    DST_df["Rank"] = DST_df.groupby("week").DKPts.apply(
        lambda x: x.rank(ascending=False, method="min")
    )
    DST_df["season"] = season
    DST_df["team"] = DST_df.gsis_id
    DST_df["DepthChart"] = 1
    DST_df["depth_chart_position"] = "DST"
    if season >= 2014:
        DST_df = load_salary_data(DST_df, season, "DST")
    else:
        DST_df["salary"] = np.nan

    if season >= 2020:
        DST_df = load_ownership_data(DST_df, season, last_week)
    DST_df.to_csv(
        f"{datadir}/game_logs/{season}/{season}_DST_GameLogs.csv", index=False
    )


def concatSeasons():
    for stat_type in ["Offense", "DST"]:
        print(f"Downloading Opponent Ranks and Adjusted Ranks for {stat_type}")
        frames = []
        for season in range(2001, 2023):
            print(season)
            df = pd.read_csv(
                f"{datadir}/game_logs/{season}/{season}_{stat_type}_GameLogs.csv"
            )
            frames.append(df)
        df = pd.concat(frames)
        df.reset_index(drop=True, inplace=True)
        df["spread_line"] = df["total_line"] - (2 * df.proj_team_score)
        df = getOppRanks(df, stat_type)
        df = getAdjOppRanks(df, stat_type)
        df.game_date = pd.to_datetime(df.game_date.values).strftime(
            "%A %m-%d-%Y %I:%M %p"
        )
        df = df[df.game_date.isna() == False]
        df["game_day"] = df.game_date.apply(lambda x: x.split(" ")[0])
        df.loc[
            (df.game_day == "Sunday")
            & (df.start_time >= "13:00:00")
            & (df.start_time < "17:00:00"),
            "Slate",
        ] = "Main"
        df.loc[df.Slate != "Main", "Slate"] = "Full"
        df.to_csv(
            f"{datadir}/game_logs/Full/{stat_type}_Database.csv", index=False
        )
        oppRanks = (
            df.groupby(["opp", "position"])
            .last()[["opp_Rank", "Adj_opp_Rank"]]
            .reset_index()
        )
        oppRanks.to_csv(
            f"/{datadir}/game_logs/Full/{stat_type}_Latest_OppRanks.csv",
            index=False,
        )


#%% Execution Line
current_season = 2022
last_week = 22
season_type = 'POST' # 'REG' or 'PRE'
updateDepthChartsDB(current_season, last_week,season_type)

for season in range(2022, 2023):
    print(season)
    if season != current_season:
        __init__(season=season, update=False)
    else:
        __init__(season=season, update=True)
        concatSeasons()
        proj = pd.read_csv(
            f"{datadir}/Projections/2022/WeeklyProjections/{season}_Week{last_week}_Projections.csv"
        )
        proj.drop("DKPts", axis=1, inplace=True, errors="ignore")
        off_db = pd.read_csv(
            f"{datadir}/game_logs/{season}/{season}_Offense_GameLogs.csv",
            usecols=["week", "season", "gsis_id", "DKPts", "RosterPercent"],
        )
        def_db = pd.read_csv(
            f"{datadir}/game_logs/{season}/{season}_DST_GameLogs.csv",
            usecols=["week", "season", "gsis_id", "DKPts", "RosterPercent"],
        )
        db = pd.concat([off_db, def_db])
        db = db[db.week == last_week]
        proj = proj.merge(db, on=["gsis_id", "week", "season"], how="left")
        proj.to_csv(
            f"{datadir}/Projections/2022/WeeklyProjections/{season}_Week{last_week}_Projections_verified.csv",
            index=False,
        )
        proj = pd.read_csv(
            f"{datadir}/Projections/2022/megnia_projections.csv"
        )
        proj.drop("DKPts", axis=1, inplace=True, errors="ignore")
        db=pd.concat([off_db,def_db])
        db = db[db.week <= last_week]
        proj = proj.merge(db, on=["gsis_id", "week", "season"], how="left")
        proj.to_csv(
            f"{datadir}/Projections/2022/megnia_projections_verified.csv",
            index=False,
        )
getMilliPlayers(last_week, season)
#%%
# os.chdir("../")
# from OwnershipDatabase.UpdateOwnershipDatabase import UpdateOwnershipDatabase
# from SnapcountDatabase.UpdateSnapCountDatabase import UpdateSnapCountDatabase

# UpdateOwnershipDatabase(current_season, last_week)
# UpdateSnapCountDatabase(current_season, last_week)
# stochastic_db = pd.read_csv(
#     f"{datadir}/Projections/StochasticProjectionDatabase.csv"
# )
# snapcount_db = pd.read_csv(
#     f"{datadir}/SnapCountData/2012_present_offensive_snapcounts.csv"
# )
# stochastic_db = stochastic_db.merge(
#     snapcount_db,
#     on=["full_name", "week", "season", "position", "team"],
#     how="left",
# )
# for pos in ["DST", "Offense"]:
#     if pos == "DST":
#         pos_frame = stochastic_db[stochastic_db.position == "DST"]
#         db = pd.read_csv(f"{datadir}/game_logs/Full/{stat_type}_Database.csv")
#         db = db[(db.season != current_season) & (db.week != last_week)]
#         db = db.merge(
#             pos_frame[
#                 [
#                     "gsis_id",
#                     "week",
#                     "season",
#                     "Stochastic",
#                     "Floor",
#                     "Ceiling",
#                     "UpsideProb",
#                     "UpsideScore",
#                     "RosterPercent",
#                 ]
#             ],
#             on=["gsis_id", "week", "season"],
#             how="left",
#         )
#         db.to_csv(f"{datadir}/game_logs/Full/{stat_type}_Database.csv")
#     else:
#         pos_frame = stochastic_db[stochastic_db.position != "DST"]
#         db = pd.read_csv(f"{datadir}/game_logs/Full/{stat_type}_Database.csv")
#         db = db[(db.season != current_season) & (db.week != last_week)]
#         db = db.merge(
#             pos_frame[
#                 [
#                     "gsis_id",
#                     "week",
#                     "season",
#                     "Stochastic",
#                     "Floor",
#                     "Ceiling",
#                     "UpsideProb",
#                     "UpsideScore",
#                     "RosterPercent",
#                 ]
#             ],
#             on=["gsis_id", "week", "season"],
#             how="left",
#         )
#         # db["avg_poe"] = db.groupby("gsis_id").apply(
#         #     lambda x: rolling_average(x, window=4)
#         # )
#         db.to_csv(f"{datadir}/game_logs/Full/{stat_type}_Database.csv")
