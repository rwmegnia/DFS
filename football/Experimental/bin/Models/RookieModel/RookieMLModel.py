#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:44:49 2021

@author: robertmegnia



"""
#%% External Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import BayesianRidge,ElasticNet,TweedieRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import warnings
import pickle
import os
import seaborn as sns
basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f'{basedir}/../../../etc'
datadir= f'{basedir}/../../../data'
warnings.simplefilter('ignore')
os.chdir('../')
from RookieModel.config import *


def mse(a,b):
    x=round(np.mean((a-b)**2),2)
    return x

def mae(a,b):
     rms=round(np.mean(np.abs(a-b)),2)
     return rms
 
def rolling_average(df,window=8):
    return df.rolling(min_periods=1, window=window).mean().shift(1)

class Model:
    def __init__(self,pos,target='DKPts',window=8):
        self.target=target
        self.window=window
        self.Database=pd.read_csv(f'{datadir}/game_logs/Full/Offense_Database.csv')
        self.Database['depth_team']=self.Database.groupby(['team','position','week','season']).salary.rank(ascending=False,method='first')
        self.Database.game_date=pd.to_datetime(self.Database.game_date)
        self.Database.sort_values(by='game_date',inplace=True)
        self.Database.salary.fillna(-1,inplace=True)
        self.Database=self.Database[self.Database.offensive_snapcounts>0]
        self.Database=self.Database[(self.Database.position==pos)&(self.Database.season==self.Database.rookie_year)]
        self.Database=self.Database.groupby('gsis_id').head(6)
        self.pos=pos
        self.Regressors=self.RegressorsDict()
        self.trainTestSplit()
        
    def generateFeatures(self,database):
        offenseFeatures=self.offenseFeatures(database)
        defenseFeatures=self.defenseFeatures(database)
        Features=self.mergeOffenseDefense(offenseFeatures,defenseFeatures)
        return Features
    
    def offenseFeatures(self,database):
        offenseFeatures=database.groupby(['gsis_id']).apply(lambda x: rolling_average(x,self.window)).drop(NonFeatures+KnownFeatures,axis=1,errors='ignore')    
        offenseFeatures[KnownFeatures]=database[KnownFeatures]
        offenseFeatures[NonFeatures]=database[NonFeatures]
        offenseFeatures.rename({self.target:f'avg_{self.target}'},axis=1,inplace=True)
        offenseFeatures[self.target]=database[self.target]
        return offenseFeatures
    
    
    def defenseFeatures(self,database):
        defenseFeatures=database.groupby(['opp','season','week']).sum().drop(NonFeatures+KnownFeatures+TeamShareFeatures,axis=1,errors='ignore')    
        defenseFeatures=defenseFeatures.groupby('opp').apply(lambda x: rolling_average(x))
        defenseFeatures=defenseFeatures.add_suffix('_allowed').reset_index()
        return defenseFeatures
    

    
    def mergeOffenseDefense(self,offense,defense):
        features=offense.merge(defense,on=['opp','season','week'],how='left')
        #features.dropna(inplace=True)
        feature_list=features.drop(NonFeatures+[self.target],axis=1).columns.to_list()
        return features,feature_list

    def RegressorsDict(self):
            return {                
                    'BR':{'regressor':BayesianRidge(compute_score=False,fit_intercept=False,normalize=False),},
                    'EN':{'regressor':ElasticNet(alpha=0,l1_ratio=0,selection='random',precompute=False,max_iter=1000,normalize=False),},
                    'NN':{'regressor':MLPRegressor(hidden_layer_sizes=10,activation='identity',solver='lbfgs',alpha=1,learning_rate='constant',learning_rate_init=.003,power_t=.1,max_iter=1000,random_state=2)},
                    'RF':{'regressor':RandomForestRegressor(n_estimators=1000,max_features='sqrt',max_depth=7,min_samples_split=5,min_samples_leaf=20,bootstrap=True),},
                    'GB':{'regressor':GradientBoostingRegressor(n_estimators=200,max_features='sqrt',subsample=1,max_depth=1,min_samples_split=10,min_samples_leaf=10)},
                    'Tweedie':{'regressor':TweedieRegressor(alpha=0)},
                    }

    def trainTestSplit(self,season=False):
        self.season=season
        self.Features,self.FeaturesList=self.generateFeatures(self.Database)
        if season==False:  
            self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.Features.loc[self.Features.season<2020,self.FeaturesList],self.Features.loc[self.Features.season<2020,self.target],test_size=0.25,shuffle=True)
        else:
            self.X_train=self.Features[self.Features.season<season][self.FeaturesList]
            self.X_test=self.Features[self.Features.season==season][self.FeaturesList]
            self.y_train=self.Features[self.Features.season<season][self.target]
            self.y_test=self.Features[self.Features.season==season][self.target]
        
    def fitModel(self,method,X_train=None,y_train=None,pipe=False,k=None):
        regressor=self.Regressors[method]['regressor']
        self.method=method
        if X_train is None:
            X_train=self.X_train
        else:
            X_train=self.X_train[X_train]
        if y_train is None:
            y_train=self.y_train
        X_train.dropna(inplace=True)
        y_train=y_train[y_train.index.isin(X_train.index)]
        if pipe==True:
            pipe=make_pipeline(SelectKBest(f_regression,k=k),regressor)
            #pipe=make_pipeline(SelectKBest(f_regression,k=k),PolynomialFeatures(2),regressor)

            regressor=pipe
        if len(X_train.shape)==1:
            regressor.fit(X_train.values.reshape(-1,1),y_train)
        else:
            regressor.fit(X_train,y_train)
        self.Model=regressor
    
    def Predict(self,X_test=None,y_test=None):
        if X_test is None:
            X_test=self.X_test
        else:
            X_test=self.X_test[X_test]
        if y_test is None:
            y_test=self.y_test
        X_test.dropna(inplace=True)
        if len(X_test.shape)==1:
            self.pred=self.Model.predict(X_test.values.reshape(-1,1))
        else:
            self.pred=self.Model.predict(X_test)
        self.y_test=y_test[y_test.index.isin(X_test.index)]
        self.Proj_df=self.getProjDataFrame()
        return self.pred[0]
    
    def getProjDataFrame(self):
        if self.season==False:
            df=self.Database[self.Database.season<2020]
            df.reset_index(drop=True,inplace=True)
            df=df[df.index.isin(self.y_test.index)]
        else:
            df=self.Database[self.Database.season<=self.season]
            df.reset_index(drop=True,inplace=True)
            df=df[df.index.isin(self.y_test.index)]      
        Proj_df=pd.DataFrame({'MyProj':self.pred,'Actual':self.y_test})
        Proj_df['MyProj_mse']=Proj_df.apply(lambda x: mse(x.Actual,x.MyProj),axis=1)
        Proj_df['MyProj_mae']=Proj_df.apply(lambda x: mae(x.Actual,x.MyProj),axis=1)
        df=df.join(Proj_df)
        df['MyRank']=df.groupby(['season','week']).MyProj.rank(ascending=False,method='min')
        return df
        
    def scatterPlot(self):
        fig=plt.figure()
        y_pred=self.pred
        y_test=self.y_test
        m=self.method
        pos=self.pos
        ms=mse(y_pred,y_test)
        ma=mae(y_pred,y_test)
        r2=r2_score(y_test,y_pred)
        r2=np.round(r2,2)
        plt.scatter(y_pred,y_test ,color='gray')
        plt.title(f'{m} {pos} R2={r2} MSE={ms} MAE={ma}')
        plt.xlabel('projected')
        plt.ylabel('actual')
        plt.xlim(0,np.nanmax(y_test))
        plt.ylim(0,np.nanmax(y_test))
        #m, b = np.polyfit(y_pred, y_test, 2)
        sns.regplot(y_pred,y_test,scatter=True,order=1)
        plt.show()
    
    def dump(self):
        pickle.dump(self.Model,open(f'{etcdir}/model_pickles/{self.pos}_{self.method}_Rookie_model.pkl','wb'))
    
    