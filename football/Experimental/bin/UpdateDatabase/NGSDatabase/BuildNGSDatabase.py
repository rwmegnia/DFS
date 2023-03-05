#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 06:38:21 2022

@author: robertmegnia
"""
import os
basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f'{basedir}/../../../data'
import nfl_data_py as nfl 
import pandas as pd
NGS_PASS_COLS = ['player_gsis_id',
                 'week',
                 'season',
                 'avg_time_to_throw',
                 'avg_completed_air_yards',
                 'avg_intended_air_yards',
                 'avg_air_yards_differential',
                 'aggressiveness',
                 'avg_air_yards_to_sticks',
                 'completion_percentage',
                 'expected_completion_percentage',
                 'completion_percentage_above_expectation',
                 'avg_air_distance']
NGS_RUSH_COLS = ['player_gsis_id',
                 'week',
                 'season',
                 'efficiency',
                 'percent_attempts_gte_eight_defenders',
                 'expected_rush_yards',
                 'rush_yards_over_expected',
                 'rush_yards_over_expected_per_att',
                 'rush_pct_over_expected']

NGS_RECEIVING_COLS = ['player_gsis_id',
                     'week',
                     'season',
                     'avg_cushion',
                     'avg_separation',
                     'avg_intended_air_yards',
                     'percent_share_of_intended_air_yards',
                     'avg_expected_yac',
                     'avg_yac_above_expectation']

def getNGSData(season):
    passing = nfl.import_ngs_data('passing',[season])[NGS_PASS_COLS]
    rushing = nfl.import_ngs_data('rushing',[season])[NGS_RUSH_COLS]
    receiving = nfl.import_ngs_data('receiving',[season])[NGS_RECEIVING_COLS]
    passing.rename({'player_gsis_id':'gsis_id',
                    'avg_intended_air_yards':'avg_intended_pass_air_yards'},axis=1,inplace=True)
    rushing.rename({'player_gsis_id':'gsis_id'},axis=1,inplace=True)
    receiving.rename({'player_gsis_id':'gsis_id',
                      'avg_intended_air_yards':'avg_intended_rec_air_yards'},axis=1,inplace=True)
    ngs=pd.concat([passing,rushing,receiving])
    return ngs
#%%
frames=[]
for season in range(2016,2022):
    print(season)
    ngs = getNGSData(season)
    ngs.to_csv(f'{datadir}/NextGenStats/{season}/NextGenStats_{season}.csv',index=False)
    frames.append(ngs)
master=pd.concat(frames)
master.to_csv(f'{datadir}/NextGenStats/NextGenStats_2016_present.csv',index=False)
    