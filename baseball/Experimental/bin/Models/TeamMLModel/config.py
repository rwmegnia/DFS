#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 06:49:19 2022

@author: robertmegnia
"""


#DK_
# Columns that can't/shouldn't be used in ML models
NonFeatures=['team',
             'game_location',
             'opp',
             'season',
             'game_date']



# Feature Lists for Top Down Model Building


team_fpts_features=[
    'avg_DKPts',
    'DKPts_allowed',
    'scoringChance',
    'scoringChance_allowed',
    'giveaways',
    'takeaways',
    'plusMinus']

# 
team_shot_features=[
                            'DKPts',
                            'avg_shots',
                            'Corsi',
                            'giveaways',
                            'takeaways',
                            'Fenwick',
                            'Corsi_allowed',
                            'Fenwick_allowed',
                            'shots_allowed']


TeamStatsModel_features_dict={
    'DKPts':team_fpts_features,
    'shots':team_shot_features,}








