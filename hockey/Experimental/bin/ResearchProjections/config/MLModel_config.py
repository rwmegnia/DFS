#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 05:43:13 2022

@author: robertmegnia

Configuration file for Position Machine Learning Model
"""



DK_Stats={'Skater':['assists',
                 'blocked',
                 'goals',
                 'powerPlayAssists',
                 'powerPlayGoals',
                 'powerPlayTimeOnIce',
                 'shortHandedAssists',
                 'shortHandedGoals',
                 'shortHandedTimeOnIce',
                 'shots',
                 'timeOnIce',
                 'full_name',
                 'player_id',
                 'position',
                 'team',
                 'game_location',
                 'opp',
                 'game_date',
                 'points',
                 'sh_points',
                 'five_shots',
                 'three_blocks',
                 'three_points',
                 'DKPts'],

          'Goalie':[ 'assists',
                     'decision',
                     'evenSaves',
                     'evenShotsAgainst',
                     'evenStrengthSavePercentage',
                     'goals',
                     'powerPlaySavePercentage',
                     'powerPlaySaves',
                     'powerPlayShotsAgainst',
                     'savePercentage',
                     'saves',
                     'shortHandedSaves',
                     'shortHandedShotsAgainst',
                     'shots',
                     'timeOnIce',
                     'even_goals_allowed',
                     'powerPlay_goals_allowed',
                     'shortHanded_goals_allowed',
                     'goals_allowed',
                     'points',
                     'shutout',
                     'saves_35',
                     'DKPts',
                     'shortHandedSavePercentage']}



# Columns that can't/shouldn't be used in ML models
NonFeatures=['full_name',
             'player_id',
             'position',
             'position_type',
             'team',
             'game_location',
             'opp',
             'game_date',]

TeamNonFeatures=[
             'team',
             'season',
             'game_location',
             'opp',
             'game_date',]

OpposingGoalieColumns=[  'savePercentage',
                         'shortHandedSaves',
                         'shortHandedShotsAgainst',
                         'shots',
                         'shortHanded_goals_allowed',
                         'goals_allowed',
                         'DKPts',
                         'shortHandedSavePercentage']

OpposingTeamColumns=[    'shots',
                         'goals',
                         'DKPts',
                         'giveaways',
                         'takeaways',
                         'plusMinus',
                         'HDSC',
                         'Corsi']









