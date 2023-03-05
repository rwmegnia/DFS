","  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:44:49 2021

@author: robertmegnia

Need to beat baseline
                All, Reduced
QB - RMSE =    65.5, 69.22
RB - RMSE =    50.33, 51.3
WR - RMSE =    53.21, 55.44
TE - RMSE =    27.57, 27.54
DST - RMSE =   29.84


"""
#%% External Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression,SelectFdr,SelectFpr
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import BayesianRidge, ElasticNet, TweedieRegressor,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
import warnings
import pickle
import os
import seaborn as sns

basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../../etc"
datadir = f"{basedir}/../../../data"
warnings.simplefilter("ignore")
from config import *

#
#%%


def mse(a, b):
    x = round(np.mean((a - b) ** 2), 2)
    return x


def mae(a, b):
    rms = round(np.mean(np.abs(a - b)), 2)
    return rms


def rolling_average(df, window):
    return df.rolling(min_periods=window, window=window).mean().shift(1)


class Model:
    def __init__(self, starter,position,target="dkpts", player_id=None, window=10):
        self.target = target
        self.window = window
        self.starter= starter
        self.player_id = player_id
        self.position = position
        self.Database = pd.read_csv(f"{datadir}/game_logs/PlayerStatsDatabase.csv")
        config=self.Database[self.Database.started==True].groupby(['team_abbreviation','game_id'],as_index=False).player_id.sum().reset_index()
        config.rename({'player_id':'config'},axis=1,inplace=True)
        self.Database=self.Database.merge(config[['team_abbreviation','config','game_id']],on=['team_abbreviation','game_id'],how='left')
        if self.starter==True:
            self.Database=self.Database[(self.Database.mins>self.Database.mins.quantile(0.25))]
        if position!=False:
            self.Database=self.Database[self.Database.position.str.contains(self.position)==True]
            self.position = position
        else:
            self.position=''
        self.Database=self.Database[self.Database[self.target].isna()==False]
        self.Database = self.Database[self.Database.started==starter]
        self.Database = self.Database[self.Database.mins > 0]
        self.Database['game_date_string']=self.Database.game_date
        self.Database.game_date = pd.to_datetime(self.Database.game_date)
        self.Database.sort_values(by="game_date", inplace=True)
        self.NonFeatures = NonFeatures
        self.Regressors = self.RegressorsDict()
        self.trainTestSplit()

    def generateFeatures(self, database):
        offenseFeatures = self.offenseFeatures(database)
        defenseFeatures = self.defenseFeatures(database)
        Features = self.mergeOffenseDefense(offenseFeatures, defenseFeatures)
        return Features

    def offenseFeatures(self, database):
        if self.player_id is not None:
            database = database[database.player_id == self.player_id]
        offenseFeatures = (
            database.groupby(["player_id",'config'])
            .apply(lambda x: rolling_average(x, self.window))
            .drop(self.NonFeatures + KnownFeatures, axis=1, errors="ignore")
        )
        offenseFeatures[KnownFeatures] = database[KnownFeatures]
        offenseFeatures[self.NonFeatures] = database[self.NonFeatures].values
        offenseFeatures.rename(
            {self.target: f"avg_{self.target}"}, axis=1, inplace=True
        )
        offenseFeatures[self.target] = database[self.target]
        return offenseFeatures

    def defenseFeatures(self, database):
        database = database.groupby("player_id").apply(
            lambda x: self.standardizeStats(x, DK_stats)
        )
        defenseFeatures = (
            database.groupby(["opp", "game_date"])
            .agg(DefenseFeatures)
        )
        defenseFeatures = defenseFeatures.groupby(["opp"]).apply(
            lambda x: rolling_average(x,self.window)
        )
        defenseFeatures = defenseFeatures.add_suffix("_allowed").reset_index()
        return defenseFeatures

    def mergeOffenseDefense(self, offense, defense):
        features = offense.merge(defense, on=["opp", "game_date"], how="left")
        # features.dropna(inplace=True)
        feature_list = features.drop(
            self.NonFeatures + [self.target], axis=1
        ).columns.to_list()
        features=features[features[feature_list].isna().any(axis=1)==False]
        return features, feature_list

    def RegressorsDict(self):
        return {
            "BR": {
                "regressor": BayesianRidge(
                    compute_score=False, fit_intercept=False, normalize=False
                ),
            },
            "EN": {"regressor": ElasticNet(),},
            "NN": {
                "regressor": MLPRegressor(
                    hidden_layer_sizes=10,
                    activation="identity",
                    solver="lbfgs",
                    alpha=1,
                    learning_rate="constant",
                    learning_rate_init=0.003,
                    power_t=0.1,
                    max_iter=1000,
                    random_state=2,
                )
            },
            "RF": {
                "regressor": RandomForestRegressor(
                    n_estimators=1000,
                    max_features="sqrt",
                    max_depth=7,
                    min_samples_split=5,
                    min_samples_leaf=20,
                    bootstrap=True,
                )},
            "GB": {
                "regressor": GradientBoostingRegressor(
                    n_estimators=200,
                    max_features="sqrt",
                    subsample=1,
                    max_depth=1,
                    min_samples_split=10,
                    min_samples_leaf=10,
                )
            },
            "Tweedie": {"regressor": TweedieRegressor(alpha=0)},
        }

    def trainTestSplit(self):
        self.Features, self.FeaturesList = self.generateFeatures(self.Database)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.Features.loc[self.Features.season < 2022, self.FeaturesList],
            self.Features.loc[self.Features.season < 2022, self.target],
            test_size=0.25,
            train_size=0.75,
            shuffle=False,
        )

    def fitModel(self, method, X_train=None, y_train=None, pipe=False, k=None):
        regressor = self.Regressors[method]["regressor"]
        self.method = method
        if X_train is None:
            X_train = self.X_train
        else:
            X_train = self.X_train[X_train]
        if y_train is None:
            y_train = self.y_train
        X_train.dropna(inplace=True)
        y_train = y_train[y_train.index.isin(X_train.index)]
        if pipe == True:
            pipe = make_pipeline(
                SelectFpr(f_regression),regressor
            )
            # pipe=make_pipeline(PolynomialFeatures(2),regressor)

            regressor = pipe
        if len(X_train.shape) == 1:
            regressor.fit(X_train.values.reshape(-1, 1), y_train)
        else:
            regressor.fit(X_train, y_train)
        self.Model = regressor

    def Predict(self, X_test=None, y_test=None):
        if X_test is None:
            X_test = self.X_test
        else:
            X_test = self.X_test[X_test]
        if y_test is None:
            y_test = self.y_test
        X_test.dropna(inplace=True)
        if len(X_test.shape) == 1:
            self.pred = self.Model.predict(X_test.values.reshape(-1, 1))
        else:
            self.pred = self.Model.predict(X_test)
        self.y_test = y_test[y_test.index.isin(X_test.index)]
        self.Proj_df = self.getProjDataFrame()
        return self.pred[0]
    
    def standardizeStats(self, df, DK_stats):
        averages = df.rolling(window=8, min_periods=4).mean()[DK_stats]
        stds = df.rolling(window=8, min_periods=4).mean()[DK_stats]
        df[DK_stats] = (df[DK_stats] - averages[DK_stats]) / stds[DK_stats]
        df.fillna(0, inplace=True)
        return df

    def getProjDataFrame(self):
        df = self.Database[self.Database.season < 2021]
        df.reset_index(drop=True, inplace=True)
        df = df[df.index.isin(self.y_test.index)]
        Proj_df = pd.DataFrame({"MyProj": self.pred, "Actual": self.y_test})
        if self.target!='upside':
            Proj_df["MyProj_mse"] = Proj_df.apply(lambda x: mse(x.Actual, x.MyProj), axis=1)
            Proj_df["MyProj_mae"] = Proj_df.apply(lambda x: mae(x.Actual, x.MyProj), axis=1)
        else:
            Proj_df['ProbUpside']=self.Model.predict_proba(self.X_test).T[1]
        df = df.join(Proj_df)
        df["MyRank"] = df.groupby(["game_date"]).MyProj.rank(
            ascending=False, method="min"
        )
        return df

    def scatterPlot(self):
        fig = plt.figure()
        y_pred = self.pred
        y_test = self.y_test
        m = self.method
        ms = mse(y_pred, y_test)
        ma = mae(y_pred, y_test)
        r2 = r2_score(y_test, y_pred)
        r2 = np.round(r2, 2)
        plt.scatter(y_pred, y_test, color="gray")
        plt.title(f"{m} R2={r2} MSE={ms} MAE={ma}")
        plt.xlabel("projected")
        plt.ylabel("actual")
        plt.xlim(0, np.nanmax(y_test))
        plt.ylim(0, np.nanmax(y_test))
        sns.regplot(y_pred, y_test, scatter=True, order=1)
        plt.show()

    def dump(self):
        if self.starter==True:
            start='starter'
        else:
            start='bench'
        pickle.dump(
            self.Model,
            open(
                f"{etcdir}/model_pickles/{self.method}_{self.position}_{self.target}_{start}_model.pkl", "wb",
            ),
        )
        
stats=['ast',
       'blk',
       'stl',
       'reb',
       'pts',
       'to',
       'fg3m',
       'dkpts',
       "pct_pts",
       "pct_fg3m",
       "pct_ast",
       "pct_reb",
       "pct_stl",
       "pct_blk",
       "pct_tov",
       "pct_dkpts"]

# for position in ['PG','SF','PF']:
#     print(position)
#     for starter in [True,False]:
#         print(starter)
#         for stat in stats:
#             if (position=='PG')&(starter==True)&(stat in ['ast','blk','stl','reb',
#                                                            'pts','to','fg3m','dkpts',
#                                                            'pct_pts']):
#                 continue
#             print(stat)
#             model=Model(starter=starter,position=position,target=stat,window=5)
#             for method in model.RegressorsDict():
#                 print(method)
#                 # model.trainTestSplit()
#                 model.fitModel(method)
#                 model.Predict()
#                 model.scatterPlot()
#                 model.dump()
