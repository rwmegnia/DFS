#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 08:06:23 2022

@author: robertmegnia
"""
import pandas as pd
import numpy as np
import os
import warnings
import unidecode
import nfl_data_py as nfl
import requests
from requests.auth import HTTPBasicAuth
from sqlalchemy import create_engine
import pymysql
import mysql.connector

warnings.simplefilter("ignore")
basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
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
def exportNicksAggs2DB(N_lineups,season,week,proj):
    proj=proj.groupby('gsis_id',as_index=False).first()
    proj=proj[[  'gsis_id',
                 'full_name',
                 'team',
                 'opp',
                 'position',
                 'total_line',
                 'proj_team_score',
                 'spread_line',
                 'proj_team_fpts',
                 'proj_team_rushing_fpts',
                 'proj_team_receiving_fpts',
                 'proj_team_passing_fpts',
                 'depth_team',
                 'injury_designation',
                 'game_location',
                 'game_day',
                 'UpsideScore',
                 'RotoName',
                 'game_time',
                 'season',
                 'game_date',
                 'week',
                 'ID',]]
    df=pd.read_csv('Lineups.csv')
    qb=df['QB'].to_frame()
    qb=(qb.groupby('QB').size()/N_lineups).sort_values()
    qb.name='optimal_exposure'
    qb=qb.to_frame().reset_index()
    qb.rename({"QB":"player"},axis=1,inplace=True)
    qb['position']='QB'
    #
    flex=df[['FLEX','FLEX Position']]
    flex.rename({'FLEX Position':'position',
                 'FLEX':'player'},axis=1,inplace=True)
    rb_flex=flex[flex.position=='RB']['player']
    wr_flex=flex[flex.position=='WR']['player']
    te_flex=flex[flex.position=='TE']['player']

    #
    rb=pd.concat([df['RB'],df['RB.1'],rb_flex]).to_frame()
    rb.rename({0:'player'},axis=1,inplace=True)
    rb=(rb.groupby('player').size()/N_lineups).sort_values()
    rb.name='optimal_exposure'
    rb=rb.to_frame().reset_index()
    rb['position']='RB'
    wr=pd.concat([df['WR'],df['WR.1'],df['WR.2'],wr_flex]).to_frame()
    wr.rename({0:'player'},axis=1,inplace=True)
    wr=(wr.groupby('player').size()/N_lineups).sort_values()
    wr.name='optimal_exposure'
    wr=wr.to_frame().reset_index()
    wr['position']='WR'
    te=pd.concat([df['TE'],te_flex]).to_frame()
    te.rename({0:'player'},axis=1,inplace=True)
    te=(te.groupby('player').size()/N_lineups).sort_values()
    te.name='optimal_exposure'
    te=te.to_frame().reset_index()
    te['position']='TE'
    dst=df['DST'].to_frame()
    dst=(dst.groupby('DST').size()/N_lineups).sort_values()
    dst.name='optimal_exposure'
    dst=dst.to_frame().reset_index()
    dst.rename({"DST":"player"},axis=1,inplace=True)
    dst['position']='DST'
    players=pd.concat([qb,rb,wr,te,dst])
    players['ID']=players.player.apply(lambda x: x.split("(")[1]).apply(lambda x: float(x.split(')')[0]))
    players.sort_values(by=['position','optimal_exposure'],ascending=False,inplace=True)
    players['full_name']=players.player.apply(lambda x: x.split('(')[0])
    players=reformatName(players)
    proj=proj.merge(players[['optimal_exposure','ID']],on='ID',how='left')
    off_db=pd.read_csv(f'{datadir}/game_logs/{season}/{season}_Offense_GameLogs.csv')
    off_db=reformatName(off_db)
    stats=off_db.groupby(['gsis_id']).agg({
        'pass_yds_per_att':np.mean,
        'passer_rating':np.mean,
        'pass_air_yards':np.mean,
        'DKPts':np.mean,
        'rush_yds_per_att':np.mean,
        'rush_value':np.mean,
        'Usage':np.mean,
        'rush_redzone_looks':np.mean,
        'targets':np.mean,
        'target_share':np.mean,
        'air_yards_share':np.mean,
        'target_value':np.mean,
        'wopr':np.mean,
        'exDKPts':np.mean,
        'poe':np.mean,
        'adot':np.mean,
        'yac':np.mean,
        'rec_redzone_looks':np.mean,
        'offensive_snapcounts':np.mean,
        'HVU':np.mean,
        'offensive_snapcount_percentage':np.mean,
        })
    stats=stats.reset_index().set_index(['gsis_id'])
    proj.set_index('gsis_id',inplace=True)
    proj=proj.join(stats)
    proj.loc[proj.position=='DST','team']=proj.loc[proj.position=='DST'].index
    proj['team']=proj['team'].apply(lambda x: x.upper())
    proj.reset_index(inplace=True)
    def_qb_stats=off_db[(off_db.position=='QB')&(off_db.depth_team==1)].groupby(['opp']).agg({
        'pass_yds_per_att':np.mean,
        'passer_rating':np.mean,
        'pass_air_yards':np.mean,
        'pass_td':np.sum,
        })
    def_qb_stats=def_qb_stats.add_prefix('opp_')
    def_qb_stats=def_qb_stats.add_suffix('_allowed')
    def_qb_stats_rank = def_qb_stats.rank().add_suffix('_rank')
    def_rb_stats=off_db[(off_db.position=='RB')&(off_db.depth_team<3)].groupby(['opp']).agg({
        'rush_yds_per_att':np.mean,
        'rush_td':np.sum,
        })
    def_rb_stats=def_rb_stats.add_prefix('opp_')    
    def_rb_stats=def_rb_stats.add_suffix('_allowed')
    def_rb_stats_rank=def_rb_stats.rank().add_suffix('_rank')
    def_wr_stats=off_db[(off_db.position=='WR')&(off_db.depth_team<4)].groupby(['opp']).agg({
        'adot':np.mean,
        'yac':np.mean,
        'rec_air_yards':np.mean,
        'rec_td':np.sum
        })
    def_wr_stats=def_wr_stats.add_prefix('opp_')
    def_wr_stats=def_wr_stats.add_suffix('_allowed')
    def_wr_stats_rank=def_wr_stats.rank().add_suffix('_rank')
    def_te_stats=off_db[(off_db.position=='TE')&(off_db.depth_team<=2)].groupby(['opp']).agg({
        'adot':np.mean,
        'yac':np.mean,
        'rec_air_yards':np.mean,
        'rec_td':np.sum
        })
    def_te_stats=def_te_stats.add_prefix('opp_')
    def_te_stats=def_te_stats.add_suffix('_allowed_te')
    def_te_stats_rank=def_te_stats.rank().add_suffix('_rank')
    proj=proj.merge(def_qb_stats,on='opp',how='left')
    proj=proj.merge(def_rb_stats,on='opp',how='left')
    proj=proj.merge(def_wr_stats,on='opp',how='left')
    proj=proj.merge(def_te_stats,on='opp',how='left')
    proj=proj.merge(def_qb_stats_rank,on='opp',how='left')
    proj=proj.merge(def_rb_stats_rank,on='opp',how='left')
    proj=proj.merge(def_wr_stats_rank,on='opp',how='left')
    proj=proj.merge(def_te_stats_rank,on='opp',how='left')
    # Points Allowed
    def_avg=off_db.groupby(['opp','position'],as_index=False).mean()
    def_avg['OppRank']=def_avg.groupby('position').DKPts.rank()   
    proj=proj.merge(def_avg[['opp','position','OppRank']],on=['opp','position'],how='left')
    #
    headshots=pd.read_csv(f'{datadir}/Projections/{season}/megnia_projections.csv')
    headshots=headshots.groupby(['gsis_id','week'],as_index=False).first()
    headshots=reformatName(headshots)
    headshots.rename({'GI':'GridIron',
                      'Projection':'RobsProjection',
                      'proj_team_score':'ImpliedTeamTotal',
                      'total_line':'Over/Under',
                      'spread_line':'spread'},axis=1,inplace=True)
    proj.loc[proj.position=='DST','RotoName']=proj.loc[proj.position=='DST','RotoName'].apply(lambda x: x.upper())
    proj=proj.merge(headshots[headshots.week==week][['RotoName',
                                                           'NicksAgg',
                                                           'GridIron',
                                                           'own',
                                                           'salary',]],on='RotoName',how='left')
    headshots=headshots.groupby('RotoName').first()['headshot_url'].to_frame()
    proj=proj.merge(headshots,on='RotoName',how='left')
    proj['week']=week
    proj['season']=season
    teams=nfl.import_team_desc()
    teams.replace({'LAR':'LA'},inplace=True)
    teams.rename({'team_abbr':'team'},axis=1,inplace=True)
    dst=proj[proj.position=='DST']
    dst.drop('headshot_url',axis=1,inplace=True)
    dst['team']=dst['RotoName']
    dst=dst.merge(teams[['team','team_logo_espn']],on='team',how='left')
    league_logo=teams.team_league_logo.unique()[0]
    dst.rename({'team_logo_espn':'headshot_url'},axis=1,inplace=True)
    dst=dst[proj.columns]
    proj=pd.concat([proj[proj.position!='DST'],dst])
    proj.loc[proj.headshot_url.isnull(),'headshot_url']=league_logo
    proj=proj.merge(teams[['team','team_logo_espn','team_wordmark']],on='team',how='left')
    proj.rename({'team_logo_espn':'team_logo',
                    'team_wordmark':'team_logo_wordmark'},axis=1,inplace=True)
    teams.rename({'team':'opp'},axis=1,inplace=True)
    proj=proj.merge(teams[['opp','team_logo_espn','team_wordmark']],on='opp',how='left')
    proj.rename({'team_logo_espn':'opp_logo',
                    'team_wordmark':'opp_wordmark'},axis=1,inplace=True)

    proj.drop_duplicates(inplace=True)
    # Export to Database
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
    last_export=pd.read_sql('Nicks_player_pools',con=sqlEngine)
    last_export=last_export[last_export.week!=week]
    players=pd.concat([players,last_export])
    players.to_sql(
        con=sqlEngine, name=f"Nicks_player_pools", if_exists="replace",index=False
    )