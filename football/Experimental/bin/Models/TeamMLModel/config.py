#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 06:49:19 2022

@author: robertmegnia
"""

# DK Stats Columns
DK_stats=[
    'pass_yards',
    'pass_td',
    'rush_yards',
    'rush_td',
    'rec',
    'rec_yards',
    'rec_td',
    'fumbles_lost',
    'int',
    'pass_yards_allowed',
    'pass_td_allowed',
    'rush_yards_allowed',
    'rush_td_allowed',
    'rec_allowed',
    'rec_yards_allowed',
    'rec_td_allowed',
    'fumbles_lost_allowed',
    'int_allowed',
    'pass_yards_share',
    'pass_td_share',
    'rush_yards_share',
    'rush_td_share',
    'rec_share',
    'rec_yards_share',
    'rec_td_share',
    'fumbles_lost_share',
    'int_share']

#DK_
# Columns that can't/shouldn't be used in ML models
NonFeatures=['team',
             'week',
             'season',
             'game_date',
             'game_id',
             'Location',
             'opp',
             'QB1_full_name',
             'RB1_full_name',
             'RB2_full_name',
             'RB3_full_name',
             'WR1_full_name',
             'WR2_full_name',
             'WR3_full_name',
             'WR4_full_name',
             'WR5_full_name',
             'TE1_full_name',
             'TE2_full_name',
             'TE3_full_name',]

# Columns that we know the values of going into a week
# Columns that we know the values of going into a week
KnownFeatures=['total_line',
               'ImpliedPoints',
               'spread_line',
               'QB1_salary',
               'RB1_salary',
               'RB2_salary',
               'RB3_salary',
               'WR1_salary',
               'WR2_salary',
               'WR3_salary',
               'WR4_salary',
               'WR5_salary',
               'TE1_salary',
               'TE2_salary',
               'TE3_salary',
               'matchup']

# Feature Lists for Top Down Model Building

offense_snaps_features=[
    'ImpliedPoints',
    'total_line',
    'spread_line',
    'avg_offense_snaps',
    'offense_snaps_allowed']

# 
team_pass_attempt_features=[
                            'ImpliedPoints',
                            'total_line',
                            'spread_line',
                            'offense_snaps',
                            'offense_snaps_allowed',
                            'avg_pass_attempt',
                            'pass_attempt_allowed',
                            'rush_attempt',
                            'rush_attempt_allowed']

# 
team_rush_attempt_features=[
                            'ImpliedPoints',
                            'total_line',
                            'spread_line',
                            'team_fpts',
                            'offense_snaps',
                            'offense_snaps_allowed',
                            'pass_attempt',
                            'pass_attempt_allowed',
                            'avg_rush_attempt',
                            'rush_attempt_allowed',
                            'epa']

team_fpts_features=[
    'ImpliedPoints',
    'total_line',
    'spread_line',
    'avg_team_fpts',
    'team_fpts_allowed',
    'offense_snaps',
    'offense_snaps_allowed',
    'pass_attempt',
    'pass_attempt_allowed',
    'rush_attempt',
    'rush_attempt_allowed']

# Features to predict passing fantasy points
team_passing_DKPts_features=[
    'ImpliedPoints',
    'total_line',
    'spread_line',
    'team_fpts',
    'offense_snaps',
    'pass_attempt',
    'rush_attempt',
    'avg_passing_fpts',
    'passing_fpts_allowed',]

# Features to predict rushing fantasy points
team_rushing_DKPts_features=[
    'ImpliedPoints',
    'total_line',
    'spread_line',
    'team_fpts',
    'offense_snaps',
    'pass_attempt',
    'rush_attempt',
    'passing_fpts',
    'passing_fpts_allowed',
    'avg_rushing_fpts',
    'rushing_fpts_allowed',]

# Features to predict rushing fantasy points
team_receiving_DKPts_features=[
    'ImpliedPoints',
    'total_line',
    'spread_line',
    'team_fpts',
    'offense_snaps',
    'pass_attempt',
    'rush_attempt',
    'passing_fpts',
    'passing_fpts_allowed',
    'rushing_fpts',
    'rushing_fpts_allowed',
    'avg_receiving_fpts',
    'receiving_fpts_allowed']

TeamStatsModel_features_dict={
    'offense_snaps':offense_snaps_features,
    'pass_attempt':team_pass_attempt_features,
    'rush_attempt':team_rush_attempt_features,
    'team_fpts':team_fpts_features,
    'passing_fpts':team_passing_DKPts_features,
    'rushing_fpts':team_rushing_DKPts_features,
    'receiving_fpts':team_receiving_DKPts_features}








