# -*- coding: utf-8 -*-

from sklearn.linear_model import ElasticNet as EN
import os
import pandas as pd
import pickle
import numpy as np
basedir = os.path.dirname(os.path.abspath(__file__))
datadir= f'{basedir}/../../../data'
etcdir= f'{basedir}/../../../etc'

db=pd.read_csv(f'{datadir}/TopLineupPlayers/FirstPlaceLineups.csv')
db.position.replace('FB','RB',inplace=True)
db=db[db.salary.isna()==False]
for position in db.position.unique():
    pos_df=db[db.position==position]
    #pos_df=pos_df[pos_df.DKPts>=pos_df.DKPts.mean()]
    X_train=db[(db.position==position)].salary[:,np.newaxis]
    y_train=db[db.position==position].DKPts
    model=EN().fit(X_train,y_train)
    pickle.dump(model,open(f'{etcdir}/model_pickles/{position}_upside_score_model.pkl','wb'))