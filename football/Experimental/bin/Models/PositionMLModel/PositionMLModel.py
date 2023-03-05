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
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    SelectFdr,
    SelectFpr,
    mutual_info_regression,
)
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import BayesianRidge, ElasticNet, TweedieRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import warnings
import pickle
import os
import seaborn as sns

basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../../etc"
datadir = f"{basedir}/../../../data"
warnings.simplefilter("ignore")
os.chdir(f"{basedir}/../..")
from Models.PositionMLModel.config import *

#%%

def mse(a, b):
    x = round(np.mean((a - b) ** 2), 2)
    return x


def mae(a, b):
    rms = round(np.mean(np.abs(a - b)), 2)
    return rms


def rolling_average(df,window):
    return df.rolling(min_periods=window, window=window).mean().shift(1)


def rolling_quantile(df, window):
    q = np.random.choice(np.arange(0.40, 0.62, 0.02))
    return df.rolling(min_periods=int(window/2), window=window).quantile(q).shift(1)

# Window=5
class Model:
    def __init__(self, stat_type, pos, target="DKPts", window=8):
        self.target = target
        self.window = window
        self.Database = pd.read_csv(
            f"{datadir}/game_logs/Full/{stat_type}_Database.csv"
        )
        self.Database = self.Database[(self.Database.position == pos)]   
        self.Database['matchup']=KMeans(n_clusters=4).fit_predict(self.Database[['proj_team_score','total_line','spread_line']])
        self.Database.game_date = pd.to_datetime(self.Database.game_date)
        self.Database.sort_values(by=["game_date", "DKPts"], inplace=True)
        if stat_type == "Offense":
            self.Database = self.Database[
                self.Database.offensive_snapcounts > 0
            ]
            self.Database["depth_team"] = self.Database.groupby(
                ["team", "position", "week", "season"]
            ).salary.rank(ascending=False, method="first")
            self.NonFeatures = OffenseNonFeatures
            if pos=='QB':
                self.Database=self.Database[(self.Database.depth_team==1)&
                                            (self.Database.offensive_snapcount_percentage>=0.85)]
            elif pos=='RB':
                self.Database=self.Database[(self.Database.depth_team<=2)]
            elif pos=='WR':
                self.Database=self.Database[(self.Database.depth_team<=4)&
                                            (self.Database.offensive_snapcount_percentage>=0.25)]
        else:
            self.Database["depth_team"] = 1
            self.NonFeatures = DSTNonFeatures
            TeamShareFeatures = []
        self.Database.salary.fillna(-1, inplace=True)
        self.pos = pos
        self.Regressors = self.RegressorsDict()
        self.trainTestSplit()

    def generateFeatures(self, database):
        offenseFeatures = self.offenseFeatures(database)
        defenseFeatures = self.defenseFeatures(database)
        Features = self.mergeOffenseDefense(offenseFeatures, defenseFeatures)
        return Features


    def offenseFeatures(self, database):
        offenseFeatures = (
            database.groupby(["gsis_id","depth_team","matchup"])
            .apply(lambda x: rolling_average(x, self.window))
            .drop(self.NonFeatures + KnownFeatures, axis=1, errors="ignore")
        )
        offenseFeatures[KnownFeatures] = database[KnownFeatures]
        NonFeatures = [c for c in self.NonFeatures if c in database.columns]
        offenseFeatures[NonFeatures] = database[NonFeatures]
        offenseFeatures.rename(
            {self.target: f"avg_{self.target}"}, axis=1, inplace=True
        )
        offenseFeatures[self.target] = database[self.target]
        return offenseFeatures


    def defenseFeatures(self, database):
        if self.pos != "DST":
            DK_stats = DK_Stoch_stats
        else:
            DK_stats = DK_DST_Stoch_stats
        database = database.groupby("gsis_id").apply(
            lambda x: self.standardizeStats(x, DK_stats)
        )
        defenseFeatures = (
            database.groupby(["opp", "season", "week", "depth_team","matchup"])
            .mean()
            .drop(
                self.NonFeatures + KnownFeatures + TeamShareFeatures,
                axis=1,
                errors="ignore",
            )
        )
        defenseFeatures = defenseFeatures.groupby(["opp", "depth_team","matchup"]).apply(
            lambda x: rolling_average(x,window=5)
        )
        defenseFeatures = defenseFeatures.add_suffix("_allowed").reset_index()
        return defenseFeatures

    def mergeOffenseDefense(self, offense, defense):
        features = offense.merge(
            defense, on=["opp", "season", "week", "depth_team","matchup"], how="left"
        )
        # features.dropna(inplace=True)
        feature_list = features.drop(
            self.NonFeatures + [self.target],
            axis=1,
            errors="ignore",
        ).columns.to_list()
        return features, feature_list

    def RegressorsDict(self):
        return {
            "BR": {
                "regressor": BayesianRidge(
                    compute_score=False, fit_intercept=False, normalize=False
                ),
            },
            "EN": {
                "regressor": ElasticNet(),
            },
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
                ),
            },
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

    def trainTestSplit(self, season=False):
        self.season = season
        self.Features, self.FeaturesList = self.generateFeatures(self.Database)
        if season == False:
            (
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test,
            ) = train_test_split(
                self.Features.loc[
                    self.Features.season < 2023, self.FeaturesList
                ],
                self.Features.loc[self.Features.season < 2023, self.target],
                test_size=0.25,
                train_size=0.75,
                shuffle=False,
            )
        else:
            self.X_train = self.Features[self.Features.season < season][
                self.FeaturesList
            ]
            self.X_test = self.Features[self.Features.season == season][
                self.FeaturesList
            ]
            self.y_train = self.Features[self.Features.season < season][
                self.target
            ]
            self.y_test = self.Features[self.Features.season == season][
                self.target
            ]

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
                SelectFdr(f_regression),
                regressor,
            )
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
        averages = df.rolling(window=self.window*2, min_periods=self.window).mean()[DK_stats]
        stds = df.rolling(window=self.window*2, min_periods=self.window).mean()[DK_stats]
        df[DK_stats] = (df[DK_stats] - averages[DK_stats]) / stds[DK_stats]
        df.fillna(0, inplace=True)
        return df

    def getProjDataFrame(self):
        if self.season == False:
            df = self.Database[self.Database.season < 2022]
            df.reset_index(drop=True, inplace=True)
            df = df[df.index.isin(self.y_test.index)]
        else:
            df = self.Database[self.Database.season <= self.season]
            df.reset_index(drop=True, inplace=True)
            df = df[df.index.isin(self.y_test.index)]
        Proj_df = pd.DataFrame({"MyProj": self.pred, "Actual": self.y_test})
        # Proj_df["MyProj_mse"] = Proj_df.apply(
        #     lambda x: mse(x.Actual, x.MyProj), axis=1
        # )
        # Proj_df["MyProj_mae"] = Proj_df.apply(
        #     lambda x: mae(x.Actual, x.MyProj), axis=1
        # )
        df = df.join(Proj_df)
        # df["MyRank"] = df.groupby(["season", "week"]).MyProj.rank(
        #     ascending=False, method="min"
        # )
        return df

    def scatterPlot(self):
        fig = plt.figure()
        y_pred = self.pred
        y_test = self.y_test
        m = self.method
        pos = self.pos
        ms = mse(y_pred, y_test)
        ma = mae(y_pred, y_test)
        r2 = r2_score(y_test, y_pred)
        r2 = np.round(r2, 2)
        plt.scatter(y_pred, y_test, color="gray")
        plt.title(f"{m} {pos} R2={r2} MSE={ms} MAE={ma}")
        plt.xlabel("projected")
        plt.ylabel("actual")
        plt.xlim(0, np.nanmax(y_test))
        plt.ylim(0, np.nanmax(y_test))
        sns.regplot(y_pred, y_test, scatter=True, order=1)
        plt.show()

    def dump(self):
        pickle.dump(
            self.Model,
            open(
                f"{etcdir}/model_pickles/{self.pos}_{self.method}_{self.target}_model.pkl",
                "wb",
            ),
        )
