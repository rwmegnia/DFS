#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:10:02 2022

@author: robertmegnia
"""

import os
import pandas as pd
basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f'{basedir}/../../data'
projdir=f'{datadir}/Projections'
start_week=1
end_week=18
start_season=2014
end_season=2021
proj_frames=[]
for season in range(start_season,end_season+1):
    if season<2021:
        end_week=17
    else:
        end_week=18
    for week in range(start_week,end_week+1):
        st=pd.read_csv(f'{projdir}/{season}/Stochastic/{season}_Week{week}_StochasticProjections.csv')
        proj_frames.append(st)