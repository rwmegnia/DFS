#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 19:24:46 2022

@author: robertmegnia
"""

#%% External Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    TweedieRegressor,
    LinearRegression,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
import warnings
import pickle
import os
import seaborn as sns

basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../../etc"
datadir = f"{basedir}/../../../data"
warnings.simplefilter("ignore")
os.chdir(f"{basedir}/../..")
from Models.OwnershipModel.config import *


def rolling_average(df, window=8):
    return df.rolling(min_periods=1, window=window).mean().shift(1)


def mse(a, b):
    x = round(np.mean((a - b) ** 2), 2)
    return x


def mae(a, b):
    rms = round(np.mean(np.abs(a - b)), 2)
    return rms


# Load in player database and filter down to 2020 since this is the earliest
# season we have ownership data for. Filter to Main slate players only.


class Model:
    def __init__(self, pos):
        self.target = "RosterPercent"
        self.Database = pd.read_csv(
                f"{datadir}/Ownership/OwnershipModel_TrainingData.csv"
            )
        self.Database = self.Database[self.Database.season >= 2020]
        self.Database["OwnershipRank"] = self.Database.groupby(
            ["position", "week", "season"]
        ).RosterPercent.rank(ascending=False,method='min')
        self.Database.sort_values(by="game_date", inplace=True)
        self.Database = self.Database[(self.Database.position == pos)]
        self.Database["Value"] = (
            self.Database.Projection / self.Database.salary
        ) * 1000
        self.pos = pos
        self.Regressors = self.RegressorsDict()
        self.trainTestSplit()

    def RegressorsDict(self):
        return {
            "BR": {
                "regressor": BayesianRidge(
                    compute_score=False, fit_intercept=False, normalize=False
                ),
            },
            "EN": {
                "regressor": ElasticNet(
                    alpha=0,
                    l1_ratio=0,
                    selection="random",
                    precompute=False,
                    max_iter=1000,
                    normalize=False,
                ),
            },
            "NN": {
                "regressor": MLPRegressor(
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
            "LR": {"regressor": LinearRegression()},
        }

    def trainTestSplit(self, season=False):
        self.season = season
        self.Features = self.Database[KnownFeatures + NonFeatures + ["RosterPercent"]]
        # self.Features = self.Database[['season','OwnershipRank','RosterPercent']]
        self.FeaturesList = KnownFeatures
        # self.FeaturesList=['OwnershipRank']
        if season == False:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.Features.loc[self.Features.season < 2022, self.FeaturesList],
                self.Features.loc[self.Features.season < 2022, self.target],
                test_size=0.5,
                shuffle=False,
            )
        else:
            self.X_train = self.Features[self.Features.season < season][
                self.FeaturesList
            ]
            self.X_test = self.Features[self.Features.season == season][
                self.FeaturesList
            ]
            self.y_train = self.Features[self.Features.season < season][self.target]
            self.y_test = self.Features[self.Features.season == season][self.target]

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
            pipe = make_pipeline(PolynomialFeatures(2), regressor)
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
        if self.season == False:
            df = self.Database[self.Database.season < 2021]
            df = df[df.index.isin(self.y_test.index)]
        else:
            df = self.Database[self.Database.season <= self.season]
            df.reset_index(drop=True, inplace=True)
            df = df[df.index.isin(self.y_test.index)]
        Proj_df = pd.DataFrame({"MyProj": self.pred, "Actual": self.y_test})
        Proj_df["MyProj_mse"] = Proj_df.apply(lambda x: mse(x.Actual, x.MyProj), axis=1)
        Proj_df["MyProj_mae"] = Proj_df.apply(lambda x: mae(x.Actual, x.MyProj), axis=1)
        df = df.join(Proj_df)
        df["MyRank"] = df.groupby(["season", "week"]).MyProj.rank(
            ascending=False, method="min"
        )
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
        sns.regplot(y_pred, y_test, scatter=True, order=2)
        plt.show()

    def dump(self):
        pickle.dump(
            self.Model,
            open(
                f"{etcdir}/model_pickles/{self.pos}_{self.method}_{self.target}_model.pkl",
                "wb",
            ),
        )
