","  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:44:49 2021

@author: robertmegnia


"""
#%% External Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import BayesianRidge, ElasticNet, TweedieRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
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


def rolling_average(df, window=5):
    return df.rolling(min_periods=1, window=window).mean().shift(1)


class Model:
    def __init__(
        self, pitcher_hand, target="DKPts", window=5
    ):
        self.target = target
        self.pitcher_hand = pitcher_hand
        self.window = window
        self.Database = pd.read_csv(f"{datadir}/game_logs/pitcherStatsDatabase.csv")
        self.Database = self.Database[(self.Database.gamesStarted==1)&(self.Database.throws==pitcher_hand)]
        self.OppDatabase = pd.read_csv(
            f"{datadir}/game_logs/batterstatsDatabase.csv"
        )
        self.OppDatabase=self.OppDatabase[self.OppDatabase.opp_pitcher_hand==pitcher_hand]
        self.OppDatabase.game_date = pd.to_datetime(self.OppDatabase.game_date)
        self.OppDatabase.sort_values(by="game_date", inplace=True)
        self.NonFeatures = PitcherNonFeatures

        self.Database.game_date = pd.to_datetime(self.Database.game_date)
        self.Database.sort_values(by="game_date", inplace=True)
        self.Regressors = self.RegressorsDict()
        self.trainTestSplit()

    def generateFeatures(self, database, opp_database, pitcher_database=None):
        offenseFeatures = self.offenseFeatures(database)
        defenseFeatures = self.defenseFeatures(opp_database)
        Features = self.mergeOffenseDefense(offenseFeatures, defenseFeatures)
        return Features

    def offenseFeatures(self, database):
        offenseFeatures = (
            database.groupby(["player_id"])
            .apply(lambda x: rolling_average(x, self.window))
            .drop(self.NonFeatures, axis=1, errors="ignore")
        )
        offenseFeatures[self.NonFeatures] = database[self.NonFeatures]
        offenseFeatures["Salary"] = database["Salary"]
        offenseFeatures.rename(
            {self.target: f"avg_{self.target}"}, axis=1, inplace=True
        )
        offenseFeatures[self.target] = database[self.target]
        return offenseFeatures

    def defenseFeatures(self, opp_database, pitcher_database=None):
        defenseFeatures = opp_database.groupby(["opp", "game_date"]).sum()[
                OpposingTeamColumns
            ]
        defenseFeatures = (
                defenseFeatures.groupby("opp")
                .apply(lambda x: rolling_average(x,window=10))
                .add_prefix("opp_")
            )
        return defenseFeatures

    def mergeOffenseDefense(self, offense, defense, pitchers=None):
        features = offense.merge(
                defense.reset_index(), on=["opp", "game_date"], how="left"
            )
        feature_list = features.drop(
            self.NonFeatures + [self.target], axis=1
        ).columns.to_list()
        return features[feature_list+['season','DKPts']], feature_list

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
                    hidden_layer_sizes=5,
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

    def trainTestSplit(self):
        self.Features, self.FeaturesList = self.generateFeatures(
                self.Database, self.OppDatabase
            )
        self.Features.dropna(inplace=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.Features.loc[self.Features.season < 2022, self.FeaturesList],
            self.Features.loc[self.Features.season < 2022, self.target],
            test_size=0.25,
            train_size=0.75,
            shuffle=True,
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
            pipe = make_pipeline(SelectKBest(mutual_info_regression, k=k), regressor)
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

    def getProjDataFrame(self):
        df = self.Database[self.Database.season < 2021]
        df.reset_index(drop=True, inplace=True)
        df = df[df.index.isin(self.y_test.index)]
        Proj_df = pd.DataFrame({"MyProj": self.pred, "Actual": self.y_test})
        Proj_df["MyProj_mse"] = Proj_df.apply(lambda x: mse(x.Actual, x.MyProj), axis=1)
        Proj_df["MyProj_mae"] = Proj_df.apply(lambda x: mae(x.Actual, x.MyProj), axis=1)
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
        pos = 'SP'
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
                f"{etcdir}/model_pickles/Pitcher_{self.pitcher_hand}_{self.method}_{self.target}_model.pkl",
                "wb",
            ),
        )
