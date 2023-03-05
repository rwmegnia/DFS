#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 15:10:27 2022

@author: robertmegnia
"""
import pandas as pd
import os
import Utils
import time
from getDKSalaries import getDKSalaries
from reformatNames import reformatNames
from datetime import datetime
from PMM import getPMM
basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
projdir = f"{datadir}/Projections"
from config.MLModel_config import *
from config.StochasticModel_config import *
from ScrapeStartingLineups import scrapeStartingLineups
from ScrapeBettingOdds import ScrapeBettingOdds
from ModelFunctions import *
#%%
class DailyProjections:
    def __init__(self,game_date,season,contestType):
        start=time.time()
        self.game_date=game_date
        self.season=season
        self.contestType=contestType
        self.Utils=Utils
        self.odds = ScrapeBettingOdds()
        self.salaries= getDKSalaries(game_date, contestType)
        self.import_databases()
        self.daily_grind = self.import_daily_grind()
        # self.fantasy_pros = self.import_fantasy_pros()

        self.startingLineups = self.getStartingLineups() 
        self.stochastic_projections = self.import_stochastic_projections()
        self.ml_projections = self.import_ml_projections()
        self.top_down_projections = self.import_top_down_projections()
        self.AllProjections = self.mergeProjections()
        self.exportProjections()
        endtime=round(time.time()-start,2)
        print(f'Complete! Model took {endtime} seconds to run')
        
    def import_databases(self):
        self.player_db = pd.read_csv(f"{datadir}/game_logs/PlayerStatsDatabase.csv")
        self.player_db.player_name.replace("Nene", "Nene Nene", inplace=True)
        self.player_db["game_date_string"] = self.player_db.game_date
        self.player_db.game_date = pd.to_datetime(self.player_db.game_date)
        self.player_db.sort_values(by="game_date", inplace=True)
        config=self.player_db[self.player_db.started==True].groupby(['game_id','team_abbreviation'],as_index=False).player_id.sum().reset_index(drop=True)
        config.rename({'player_id':'config'},axis=1,inplace=True)
        self.player_db=self.player_db.merge(config,on=['game_id','team_abbreviation'],how='left')
        self.player_db = self.player_db[self.player_db.game_date < self.game_date]
        self.player_db.sort_values(by='game_date',inplace=True)
        self.player_db['rotoname']=self.player_db.player_name.apply(lambda x: x.split(' ')[0][0:4].lower()+x.split(' ')[1][0:5].lower())
        # Import Team Database
        team_db = pd.read_csv(f"{datadir}/game_logs/TeamStatsDatabase.csv")
        team_db["game_date_string"] = team_db.game_date
        team_db.game_date = pd.to_datetime(team_db.game_date)
        team_db.sort_values(by="game_date", inplace=True)
        self.team_db = team_db[team_db.game_date < game_date]
        
    def getStartingLineups(self):
        lineups = scrapeStartingLineups(self.game_date)
        lineups=lineups.merge(self.odds,on=['team_abbreviation','game_date'])
        lineups.drop(['spread_line','moneyline'],axis=1,inplace=True)
        # Get Draftkings Salaries for the day
        if self.contestType=='Showdown':
            self.salaries=self.salaries[self.salaries['Roster Position']=='UTIL']
        
        lineups = lineups[lineups.team_abbreviation.isin(self.salaries.team)]

        player_IDs = self.player_db.groupby("player_id", as_index=False).last()[
            ["team_abbreviation", "player_name", "nickname", "player_id"]
        ]
        player_IDs = reformatNames(player_IDs)

        self.salaries = self.salaries.merge(
            player_IDs[["player_id", "RotoName"]],on="RotoName", how="left"
        )

        if self.contestType=='Showdown':
            lineups = lineups.drop('Salary',axis=1).merge(
                self.salaries[["position", "player_id", "Salary","Game Info","ID", "RotoName", "player_name"]],
                on=["RotoName"],
                how="left",
            )
        else:
            lineups = lineups.drop('Salary',axis=1).merge(
                self.salaries[["position", "player_id", "Salary","Game Info","ID", "RotoName", "player_name",'Roster Position']],
                on=["RotoName"],
                how="left",
            )


        config=lineups[lineups.starter==True].groupby('team_abbreviation').player_id.sum().reset_index()
        config.rename({'player_id':'config'},axis=1,inplace=True)
        lineups=lineups.merge(config,on='team_abbreviation',how='left')
        lineups=MLMinutesPrediction(self.player_db, lineups)
        return lineups
    
    def import_stochastic_projections(self):
        start_df = self.startingLineups.reset_index()
        stats_df=self.player_db
        stochastic_proj_df=StochasticPrediction(stats_df,start_df)
        stochastic_proj_df.reset_index(inplace=True)
        stochastic_proj_df=stochastic_proj_df.groupby(['player_id',
                                                       'position',
                                                       'RotoName', 
                                                       'RotoPosition',
                                                       'team_abbreviation',
                                                       'opp',
                                                       'player_name',
                                                       'Game Info',
                                                       'full_name',
                                                       'game_date',
                                                       ],as_index=False).mean()
        #QC
        stochastic_proj_df.loc[(stochastic_proj_df.RG_projection==0)&(stochastic_proj_df.Stochastic!=0),'Stochastic']=0
        stochastic_proj_df.loc[(stochastic_proj_df.RG_projection==0)&(stochastic_proj_df.Stochastic!=0),'Ceiling']=0
        stochastic_proj_df.loc[(stochastic_proj_df.RG_projection==0)&(stochastic_proj_df.Stochastic!=0),'Floor']=0
        return stochastic_proj_df
    
    def import_ml_projections(self):
        lineups = self.startingLineups.reset_index().merge(self.daily_grind[['RotoName',
                                        'DG_proj_mins']],on='RotoName',how='left')
        lineups['avg_proj_mins']=lineups[['DG_proj_mins']].mean(axis=1)
        ml_proj_df = MLPrediction(self.player_db, lineups)
        return ml_proj_df
    
    def import_top_down_projections(self):

        start_df= self.startingLineups.rename({'team':'team_abbreviation'},axis=1)
        team_games_df = (
            start_df.groupby("team_abbreviation")
            .first()[
                ["opp", "game_date", "proj_team_score",'total_line']
            ]
            .reset_index()
        )
        team_proj_df = TeamStatsPredictions(team_games_df, self.team_db)
        #
        ml_proj_df = self.ml_projections.merge(
            team_proj_df.reset_index()[
                [
                    "team_abbreviation",
                    "proj_pts",
                    "proj_fg3m",
                    "proj_ast",
                    "proj_reb",
                    "proj_blk",
                    "proj_stl",
                    "proj_to",
                    "proj_dkpts",
                ]
            ],
            on="team_abbreviation",
            how="left",
        )
        ml_proj_df = TopDownPrediction(ml_proj_df)
        ml_proj_df=ml_proj_df.groupby('player_id',as_index=False).mean()
        return ml_proj_df
    
    def import_fantasy_pros(self):
        game_date_file_string = '_'.join(self.game_date.split('-'))
        fp_proj=pd.read_csv(f'{datadir}/FantasyPros/{season}/{game_date}/FP_Proj_{game_date_file_string}.csv')
        fp_proj=self.Utils.reformatFantasyPros(fp_proj)
        return fp_proj
    
    def import_daily_grind(self):
        game_date_file_string = '_'.join(self.game_date.split('-'))
        dg_proj=pd.read_csv(f'{datadir}/DailyGrind/{season}/{game_date}/DG_Proj_{game_date_file_string}.csv')
        dg_proj = self.Utils.reformatDailyGrind(dg_proj)
        return dg_proj
    
    def mergeProjections(self):
        df = self.stochastic_projections.merge(self.ml_projections[['player_id',
                                                                    'ML',
                                                                    'DG_proj_mins',
                                                                    'EN',
                                                                    'RF',
                                                                    'GB']],on='player_id',how='left')
        df = df.merge(self.top_down_projections[['player_id','TD_Proj','TD_Proj2']],on='player_id',how='left')

        # df = df.merge(self.fantasy_pros[['RotoName','FP_Proj']],on=['RotoName'],how='left')
        df = df.merge(self.daily_grind[['RotoName',
                                        'DG_Proj',
                                        'DG_ownership_proj',
                                        'proj_usg']],on='RotoName',how='left')
        df['avg_proj_mins']=df[['proj_mins','DG_proj_mins']].mean(axis=1)
        df['avg_ownership']=df[['ownership_proj','DG_ownership_proj']].mean(axis=1)
        df=getPMM(df,self.game_date)
        df["Projection"] = df[
            ["Stochastic", "ML", "RG_projection", "PMM", "TD_Proj2","Median",'DG_Proj']
        ].mean(axis=1)
        df.loc[df.RG_projection==0,'Projection']=0
        df["game_date"] = game_date
        df = df[df.RG_projection.isna() == False]
        df.drop_duplicates(inplace=True)
        df = df.merge(self.odds[['team_abbreviation','spread_line','moneyline']],on=['team_abbreviation'],how='left')
        df=df.groupby('player_id',as_index=False).first()
        return df
    
    def exportProjections(self):
        projdir = f"{datadir}/Projections/RealTime/{self.season}/{self.contestType}"
        try:
            self.AllProjections.to_csv(
                    f"{projdir}/{game_date}/{game_date}_Projections.csv", index=False,
                )
        except OSError:
            os.mkdir(f"{projdir}/{game_date}")
            self.AllProjections.to_csv(
                f"{projdir}/{game_date}/{game_date}_Projections.csv", index=False,
            )
            
season = 2022
game_date = datetime.now().strftime("%Y-%m-%d")
contestType=None
while contestType not in ['Classic', 'Showdown']:
    contestType = input('Enter contest type: ')
    if contestType not in ['Classic','Showdown']:
        print('Invalid Selection, enter Classic or Showdown')
if __name__=='__main__':
    proj = DailyProjections(game_date,season,contestType)