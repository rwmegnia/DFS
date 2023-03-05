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
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression,SelectFdr
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

class MinutesModel:
    def __init__(self,starter):
        self.starter=starter
        self.Database = pd.read_csv(f"{datadir}/game_logs/PlayerStatsDatabase.csv")
        config=self.Database[self.Database.started==True].groupby(['team_abbreviation','game_id'],as_index=False).player_id.sum().reset_index()
        config.rename({'player_id':'config'},axis=1,inplace=True)
        self.Database=self.Database.merge(config[['team_abbreviation','config','game_id']],on=['team_abbreviation','game_id'],how='left')
        self.Database['game_date_string']=self.Database.game_date
        self.Database.game_date = pd.to_datetime(self.Database.game_date)
        self.Database=self.Database[self.Database.season>=2021]
        self.Database.sort_values(by="game_date", inplace=True)
        self.Database=self.Database[(self.Database.started==starter)]
        if starter==True:
            self.Database=self.Database[self.Database.mins>0]
            self.Database=self.Database[(self.Database.mins>self.Database.mins.quantile(0.01))]
        self.NonFeatures = MinutesNonFeatures
        self.KnownFeatures = MinutesKnownFeatures
        self.Regressors = self.RegressorsDict()
        self.trainTestSplit()

    def generateFeatures(self, database):
        frames=[]
        for window in [3,5]:
            print(window)
            Features = (
                database.groupby(["player_id","config"])
                .apply(lambda x: rolling_average(x, window))
                .drop(self.NonFeatures + self.KnownFeatures, axis=1, errors="ignore")
            )
            Features = Features.add_suffix(f'_{window}gm_mean')
            frames.append(Features)
        # for window in [3,5]:
        #     print(window)
        #     Features = (
        #         database.groupby(["player_id","config"])
        #         .apply(lambda x: rolling_std(x, window))
        #         .drop(self.NonFeatures + KnownFeatures, axis=1, errors="ignore")
        #     )
        #     Features = Features.add_suffix(f'_{window}gm_std')
        #     frames.append(Features)
        # for window in [3,5]:
        #     print(window)
        #     Features = (
        #         database.groupby(["player_id","config"])
        #         .apply(lambda x: rolling_median(x, window))
        #         .drop(self.NonFeatures + KnownFeatures, axis=1, errors="ignore")
        #     )
        #     Features = Features.add_suffix(f'_{window}gm_median')
        #     frames.append(Features)
        # for window in [3,5]:
        #     print(window)
        #     Features = (
        #         database.groupby(["player_id","config"])
        #         .apply(lambda x: rolling_min(x, window))
        #         .drop(self.NonFeatures + KnownFeatures, axis=1, errors="ignore")
        #     )
        #     Features = Features.add_suffix(f'_{window}gm_min')
        #     frames.append(Features)
        # for window in [3,5]:
        #     print(window)
        #     Features = (
        #         database.groupby(["player_id","config"])
        #         .apply(lambda x: rolling_max(x, window))
        #         .drop(self.NonFeatures + KnownFeatures, axis=1, errors="ignore")
        #     )
        #     Features = Features.add_suffix(f'_{window}gm_max')
        #     frames.append(Features)
        Features=pd.concat(frames,axis=1)
        Features['mins'] = database['mins']
        Features[self.KnownFeatures] = database[self.KnownFeatures]
        Features[self.NonFeatures] = database[self.NonFeatures].values
        return Features,Features.drop(self.NonFeatures+['mins'], axis=1).columns.to_list()


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
            self.Features.loc[self.Features.season < 2023, self.FeaturesList],
            self.Features.loc[self.Features.season < 2023, 'mins'],
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
        # X_train.fillna(0,inplace=True)
        X_train.dropna(inplace=True)
        y_train = y_train[y_train.index.isin(X_train.index)]
        if pipe == True:
            pipe = make_pipeline(
                SelectKBest(f_regression,k=k),regressor
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
        # X_test.fillna(0,inplace=True)
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
        pickle.dump(
            self.Model,
            open(
                f"{etcdir}/model_pickles/{self.method}_{self.starter}_minutes_model.pkl", "wb",
            ),
        )
for starter in [True,False]:
    model=MinutesModel(starter)
    for method in model.RegressorsDict():
        model.fitModel(method,pipe=True,k=20)
        model.Predict()
        model.scatterPlot()
        model.dump()