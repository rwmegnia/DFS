#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 08:02:03 2022

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
from ExportColumnMappings import *
from PMMPlayer import getPMM
from getDKSalaries import getDKSalaries
import Utils
import RosterUtils
from ModelFunctions import *
import warnings
import time
warnings.simplefilter("ignore")

#%%
class WeeklyProjections:
    def __init__(self, week, season):
        start=time.time()
        self.week = int(week)
        self.season = season
        self.RosterUtils = RosterUtils
        self.Utils = Utils
        self.game_date = self.RosterUtils.getGameDate(week, season)
        self.schedule = self.RosterUtils.getSchedule(week, season)
        self.game_date = self.RosterUtils.getGameDate(week, season)
        self.game_dates = self.RosterUtils.getGameDates(week, season)
        self.odds = ScrapeBettingOdds(week,season,self.game_date)
        print('Downloading Schedule...')
        print('Downloading DraftKings Salaries...')
        self.salaries = getDKSalaries(self.game_date, self.game_dates, self.week, self.season)
        self.salaries.loc[self.salaries.full_name=='Tj Hockenson','team']='MIN'
        print('Downloading Latest Rosters...')
        self.rosters = self.import_rosters()
        print('Importing Databases...')
        self.offense_db, self.defense_db = self.import_databases()
        print('Computing Stochastic Projections...')
        self.StochasticProjections = self.import_stochastic_projections()
        print('Computing Machine Learning Projections...')
        self.MLProjections = self.import_ml_projections()
        print('Computing Rookie Projections if Applicable...')
        self.RookieProjections = self.import_rookie_projections()
        print('Computing Player Share Projections...')
        self.PlayerShareProjections = PlayerSharesPrediction(
            self.offense_db.reset_index(), self.rosters
        )
        print('Computing Depth Chart Based Projections...')
        self.add_dc_projections()
        print('Computing Team Stats Projections...')
        self.TeamProjections = TeamStatsModelPredict(
            week, season, self.schedule, self.odds
        )
        print('Computing Top Down Projections...')
        self.TopDownProjections = self.import_topdown_projections()
        print('Scraping Rotogrinders and FantasyPros Projections...')
        self.FP_Proj2 = self.import_fantasy_pros2()
        self.FP_Proj1 = self.import_fantasy_pros1()
        self.RotoGrinders = self.import_rotogrinders()
        self.GridIronAI = self.import_gridiron()
        self.NicksAgg = self.import_nicks_aggregates()
        print('Merging All Projections...')
        self.AllProjections = self.merge_projections()
        print('Exporting Projections to Local Database...')
        self.export_projections()
        print('Exporting Projections to GridIron Database...')
        self.export_projections_gi()
        endtime=round(time.time()-start,2)
        print(f'Complete! Model took {endtime} seconds to run')
        return

    def import_rosters(self):
        rosters = self.RosterUtils.getWeeklyRosters(
            self.season, self.week, self.schedule, self.odds, self.game_date
        )
        rosters = rosters.merge(
            self.salaries[SALARY_COLUMNS],
            on=["RotoName", "team", "position"],
            how="left",
        )
        rosters = rosters[rosters.salary.isna() == False]
        rosters['depth_team']=rosters.groupby(['team','position']).salary.rank(ascending=False,method='first')
        if self.week==8:
            rosters.loc[rosters.full_name=='Sam Ehlinger','depth_team']=1
            rosters.loc[rosters.full_name=='Matt Ryan','depth_team']=2

        return rosters

    def import_databases(self):
        def_df = pd.read_csv(f"{datadir}/game_logs/Full/DST_Database.csv")
        def_df.rename({"DepthChart": "depth_team"}, axis=1, inplace=True)
        def_df.game_date = pd.to_datetime(def_df.game_date)
        def_df = def_df[def_df.game_date < self.game_date]
        def_df["depth_team"] = 1
        def_df["team"] = def_df.gsis_id
        off_df = pd.read_csv(f"{datadir}/game_logs/Full/Offense_Database.csv")
        off_df.game_date = pd.to_datetime(off_df.game_date)
        off_df = off_df[off_df.game_date < self.game_date]
        return off_df, def_df

    def import_stochastic_projections(self):
        self.defense_stochastic = DefenseStochastic(
            self.rosters,
            self.defense_db,
        )
        self.qb_stochastic = self.get_qb_stochastic(self.rosters,self.offense_db)
        self.rb_stochastic = self.get_rb_stochastic(self.rosters,self.offense_db)
        self.wr_stochastic = self.get_wr_stochastic(self.rosters,self.offense_db)
        self.te_stochastic = self.get_te_stochastic(self.rosters,self.offense_db)
        self.offense_stochastic = pd.concat([self.qb_stochastic,
                                             self.rb_stochastic,
                                             self.wr_stochastic,
                                             self.te_stochastic])
        return pd.concat([self.offense_stochastic, self.defense_stochastic])

    def get_qb_stochastic(self,qbs,db):
        qbs=qbs[(qbs.position=='QB')&(qbs.depth_team==1)]
        db=db[(db.position=='QB')&
              (db.depth_team==1)&
              (db.offensive_snapcount_percentage>0.85)]
        qbs=QBStochastic(qbs,db)
        return qbs

    def get_rb_stochastic(self,rbs,db):
        rbs=rbs[(rbs.position=='RB')&
                (rbs.depth_team<=2)]
        rb_stats=db[(db.position=='RB')&
                    (db.depth_team<=2)]
        rbs=RBStochastic(rbs, rb_stats)
        return rbs
    
    def get_wr_stochastic(self,wrs,db):
        wrs=wrs[(wrs.position=='WR')&(wrs.depth_team<=3)]
        db=db[(db.position=='WR')&
              (db.depth_team<=3)&
              (db.offensive_snapcount_percentage>0.25)]
        wrs=OffenseStochastic(wrs,db)
        return wrs
    
    def get_te_stochastic(self,tes,db):
        tes=tes[(tes.position=='TE')&(tes.depth_team<=2)]
        db=db[(db.position=='TE')&
              (db.depth_team<=2)&
              (db.offensive_snapcount_percentage>0.25)]
        tes=OffenseStochastic(tes,db)
        return tes   
    
    def import_ml_projections(self):
        ml_def = self.defense_stochastic
        def_stats = self.defense_db
        def_stats['DepthChart']=1
        ml_def['DepthChart'] = 1
        ml_def = MLPrediction(def_stats.reset_index(), ml_def.reset_index(), "DST")
        ml_off = self.offense_stochastic
        ml_off["DepthChart"] = ml_off.depth_team
        ml_off["player_id"] = ml_off.gsis_id
        ml_off = MLPrediction(self.offense_db.reset_index(), ml_off.reset_index(drop=True), "Offense")
        ml = pd.concat([ml_off, ml_def])
        self.offense_db.reset_index(inplace=True)
        # ml = getPMM(ml, self.season, self.week)
        return ml

    def import_rookie_projections(self):
        games_played = (
            self.offense_db.groupby("gsis_id", as_index=False)
            .size()
            .rename({"size": "games_played"}, axis=1)
        )
        rookie_stats = self.offense_db.merge(
            games_played, on="gsis_id", how="left"
        )
        rookie_rosters = self.rosters.merge(
            games_played, on="gsis_id", how="left"
        )
        rookie_rosters["depth_team"] = rookie_rosters.groupby(
            ["team", "position"]
        ).salary.rank(ascending=False, method="first")
        rookie_proj = rookiePrediction(
            rookie_rosters.drop("games_played", axis=1), rookie_stats
        )
        rookie_proj = rookie_proj[rookie_proj.position != "DST"]
        self.MLProjections = self.MLProjections[
            ~self.MLProjections.gsis_id.isin(rookie_proj.gsis_id)
        ]
        self.MLProjections = pd.concat([self.MLProjections, rookie_proj])
        self.MLProjections=getPMM(self.MLProjections,self.season,self.week)
        return rookie_proj

    def add_dc_projections(self):
        dc_proj = TeamDepthChartModelPredict(
            self.week, self.season, self.schedule, self.odds
        )
        self.MLProjections = pd.concat(
            [
                self.MLProjections.groupby("gsis_id").apply(
                    lambda x: self.Utils.merge_dc_projections(x, dc_proj)
                )
            ]
        )
        self.MLProjections.reset_index(drop=True,inplace=True)

    def import_topdown_projections(self):
        top_down = self.PlayerShareProjections.merge(
            self.TeamProjections[TEAM_STATS_COLUMNS],
            on=["week", "season", "team"],
            how="left",
        )
        top_down["TopDown"] = (
            (top_down.passing_DKPts_share * top_down.passing_fpts)
            + (top_down.rushing_DKPts_share * top_down.rushing_fpts)
            + (top_down.receiving_DKPts_share * top_down.receiving_fpts)
        )
        top_down.rename(
            RENAME_TOPDOWN,
            axis=1,
            inplace=True,
        )
        return top_down

    def import_fantasy_pros1(self):
        return scrapeFantasyPros(self.week)

    def import_fantasy_pros2(self):
        # Load Fantasy Pros Top Experts
        fp = pd.read_csv(
            f"{datadir}/FantasyPros/{self.season}/Week{self.week}/FP_Proj_Week{self.week}.csv"
        )
        fp = self.Utils.reformatFantasyPros(fp)
        fp = fp[fp.FP_Proj > 0]
        fp.rename({"FP_Proj": "FP_Proj2"}, axis=1, inplace=True)
        
        # Load Fantasy Pros Full Consensus
        fp2 = pd.read_csv(
            f"{datadir}/FantasyPros/{self.season}/Week{self.week}/FP_Proj_Full_Week{self.week}.csv"
        )
        fp2 = self.Utils.reformatFantasyPros(fp2)
        fp2 = fp2[fp2.FP_Proj > 0]
        fp2.rename({"FP_Proj": "FP_Proj_Full"}, axis=1, inplace=True)
        fp=fp.merge(fp2[['RotoName','FP_Proj_Full']],on='RotoName',how='left')
        return fp

    def import_rotogrinders(self):
        return scrapeStartingLineups(self.week,self.season)
    
    def import_gridiron(self):
        self.mydb = mysql.connector.connect(
            host="footballai-db-prod.cxgq1kandeps.us-east-2.rds.amazonaws.com",
            user="gridironai",
            password="thenameofthewind",
            database="gridironai",
        )
        self.sqlEngine = create_engine(
            "mysql+pymysql://gridironai:thenameofthewind@footballai-db-prod.cxgq1kandeps.us-east-2.rds.amazonaws.com/gridironai",
            pool_recycle=3600,
        )
        df=pd.read_sql('rankings_playergameprediction',con=self.sqlEngine)
        df = df[(df.season==self.season)&(df.week==self.week)]
        if len(df)==0:
            self.GI_available=False
        elif df.passing_yards.max()==0:
            self.GI_available=False
            return
        else:
            ids=pd.read_sql('dim_name_team_and_position',con=self.sqlEngine)
            df=self.Utils.reformatGridIron(df,ids)
            self.GI_available=True
            return df
        
    def import_nicks_aggregates(self):
        self.mydb = mysql.connector.connect(
            host="footballai-db-prod.cxgq1kandeps.us-east-2.rds.amazonaws.com",
            user="gridironai",
            password="thenameofthewind",
            database="gridironai",
        )
        self.sqlEngine = create_engine(
            "mysql+pymysql://gridironai:thenameofthewind@footballai-db-prod.cxgq1kandeps.us-east-2.rds.amazonaws.com/gridironai",
            pool_recycle=3600,
        )
        df=pd.read_sql('raw_gurol_draftkings_agg_projection',con=self.sqlEngine)
        df = df[(df.week==self.week)&(df.Player.isnull()==False)]
        if len(df)==0:
            self.Nick_available=False
        else:
            df=self.Utils.reformatNicksAgg(df)
            self.Nick_available=True
            return df
        
    def merge_projections(self):
        # Merge with top down projections
        projections = self.MLProjections.merge(
            self.TopDownProjections[TOPDOWN_MERGE_COLUMNS],
            on="gsis_id",
            how="left",
        )

        # Create Projections Column
        projections["Projection"] = projections[
            ["Stochastic","Median", "DC_proj", "TopDown","ML","PMM"]
        ].mean(axis=1)
        projections.loc[
            (projections.position == "QB") & (projections.depth_team != 1),
            "Projection",
        ] = 0

        projections = self.Utils.reformatName(projections)

        # Merge with Fantasy Pros and RotoGrinders
        projections = projections.merge(
            self.FP_Proj1,
            on=["team", "RotoName", "position", "week"],
            how="left",
        )

        projections = projections.merge(
            self.FP_Proj2[
                ["position",
                 "salary",
                 "team",
                 "opp",
                 "RotoName",
                 "FP_Proj2",
                 "FP_Proj_Full"]
            ],
            on=["salary", "team", "opp", "RotoName", "position"],
            how="left",
        )

        projections = projections.merge(
            self.RotoGrinders[
                [
                    "RotoName",
                    "ownership_proj",
                    "RG_projection",
                    "team",
                    "position",
                ]
            ],
            on=["RotoName", "team", "position"],
            how="left",
        )
        
        # Merge in GridIronAI and Nick Gurol Aggregates if Available
        if (self.GI_available==False):
            if ('GI' in PROJECTION_EXPORT_COLUMNS):
                PROJECTION_EXPORT_COLUMNS.remove('GI')
            if ('GI' in GI_EXPORT_COLUMNS):
                GI_EXPORT_COLUMNS.remove('GI')
        else:
            projections=projections.merge(self.GridIronAI[
                                        ['GI','RotoName','position']
                                        ],
                                          on=['RotoName','position'],
                                          how='left')
        if (self.Nick_available==False):
            if ('NicksAgg' in PROJECTION_EXPORT_COLUMNS):
                PROJECTION_EXPORT_COLUMNS.remove('NicksAgg')
                PROJECTION_EXPORT_COLUMNS.remove('own')

            if ('NicksAgg' in GI_EXPORT_COLUMNS):
                GI_EXPORT_COLUMNS.remove('NicksAgg')
                GI_EXPORT_COLUMNS.remove('own')

        else:
            projections=projections.merge(self.NicksAgg,
                                          on=['RotoName',
                                              'position',
                                              'team',
                                              'week'],
                                          how='left')
        if (self.GI_available==True)&(self.Nick_available==True):
            projections["Projection"] = projections[
                    ["Projection", 
                     "FP_Proj", 
                     "FP_Proj2",
                     "FP_Proj_Full", 
                     "RG_projection",
                     'GI',
                     'NicksAgg']
                ].mean(axis=1)
        elif (self.GI_available==False)&(self.Nick_available==True):
            projections["Projection"] = projections[
                    ["Projection", 
                     "FP_Proj", 
                     "FP_Proj2",
                     "FP_Proj_Full", 
                     "RG_projection",
                     'NicksAgg']
                ].mean(axis=1)
        elif (self.GI_available==True)&(self.Nick_available==False):
            projections["Projection"] = projections[
                    ["Projection", 
                     "FP_Proj", 
                     "FP_Proj2",
                     "FP_Proj_Full", 
                     "RG_projection",
                     'GI',]
                ].mean(axis=1)
        else:
            projections["Projection"] = projections[
                    ["Projection", 
                     "FP_Proj", 
                     "FP_Proj2",
                     "FP_Proj_Full", 
                     "RG_projection",]
                ].mean(axis=1)
        projections = self.Utils.getOwnership(projections)
        projections = self.Utils.getImpliedOwnership(projections)
        projections = self.Utils.getConsensusRanking(projections)
        # projections = self.Utils.getScaledProjection(projections)
        projections = self.Utils.getWeightedProjections(projections,
                                                        self.GI_available,
                                                        self.Nick_available)
        # Merge draftkings ID back in
        projections = projections.merge(
            self.rosters[["gsis_id", "ID","injury_designation","ModifiedDt", "Roster Position", "game_time"]],
            on="gsis_id",
            how="left",
        )
        return projections
    

    def export_projections(self):
        projections = self.AllProjections[PROJECTION_EXPORT_COLUMNS]
        projections = projections.round(2)
        projections["week"] = self.week
        projections["season"] = self.season
        projections = self.Utils.getPlayerVariance(projections,self.week)
        projections.to_csv(
            f"{datadir}/Projections/{self.season}/WeeklyProjections/{self.season}_Week{self.week}_Projections.csv",
            index=False,
        )
    
    def export_projections_gi(self):
        try:
            gi_df = pd.read_csv(
                f"{datadir}/Projections/{self.season}/megnia_projections.csv"
            )
            past_gi_df = gi_df[gi_df.week != self.week]
            new_gi_df = self.AllProjections[GI_EXPORT_COLUMNS]
            ids = nfl.import_ids()
            new_gi_df = new_gi_df.merge(
                ids[["gsis_id", "sleeper_id"]], on="gsis_id", how="left"
            )
            headshots = nfl.import_rosters(
                [2022], columns=["sleeper_id", "headshot_url", "week", "season"]
            )
            headshots.sleeper_id = headshots.sleeper_id.astype(float)
            headshots = headshots[headshots.sleeper_id.isna() == False]
            new_gi_df = new_gi_df.merge(
                headshots, on=["sleeper_id", "week", "season"], how="left"
            )
            new_gi_df.rename(
                {
                    "wProjection": "Projection",
                    "AvgOwnership": "Ownership",
                    "HVU": "8gm_average_HVU",
                    "target_value": "8gm_average_value_adjusted_targets",
                    "rush_value": "8gm_average_value_adjusted_rush_attempts",
                    "proj_target_value": "proj_value_adjusted_targets",
                    "proj_rush_value": "proj_value_adjusted_rush_attempts",
                },
                axis=1,
                inplace=True,
                errors="ignore",
            )
            gi_df = pd.concat([past_gi_df, new_gi_df])
            gi_df.to_csv(
                f"{datadir}/Projections/{self.season}/megnia_projections.csv",
                index=False,
            )
        except FileNotFoundError:
            gi_df = self.AllProjections[[EXPORT_GI_COLUMNS]]
            headshots = nfl.import_rosters(
                [2022], columns=["sleeper_id", "headshot_url", "week", "season"]
            )
            ids = nfl.import_ids()
            ids = ids[ids.position.isin(["QB", "RB", "WR", "TE", "FB"])]
            gi_df = gi_df.merge(
                ids[["sleeper_id", "gsis_id"]], on="gsis_id", how="left"
            )
            headshots = nfl.import_rosters(
                [2022], columns=["sleeper_id", "headshot_url", "week", "season"]
            )
            headshots.sleeper_id = headshots.sleeper_id.astype(float)
            headshots = headshots[headshots.sleeper_id.isna() == False]
            gi_df = gi_df.merge(
                headshots[["sleeper_id", "week", "season", "headshot_url"]],
                on=["sleeper_id", "week", "season"],
                how="left",
            )
            gi_df.rename(
                {
                    "wProjection": "Projection",
                    "AvgOwnership": "Ownership",
                    "HVU": "8gm_average_HVU",
                    "target_value": "8gm_average_value_adjusted_targets",
                    "rush_value": "8gm_average_value_adjusted_rush_attempts",
                    "proj_target_value": "proj_value_adjusted_targets",
                    "proj_rush_value": "proj_value_adjusted_rush_attempts",
                },
                axis=1,
                inplace=True,
                errors="ignore",
            )
            gi_df.to_csv(
                f"{datadir}/Projections/{self.season}/megnia_projections.csv",
                index=False,
            )

        gi_df.Leverage.replace(np.inf,np.nan,inplace=True)
        gi_df.to_sql(
            con=self.sqlEngine, name="megnia_projections", if_exists="replace"
        )
week=10
season=2022
if __name__=='__main__':
    week=input('Enter Week: ')
    proj = WeeklyProjections(int(week),season)
