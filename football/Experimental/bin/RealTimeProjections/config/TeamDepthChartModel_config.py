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
TDM_NonFeatures=['team',
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
TDM_KnownFeatures=['total_line',
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
               'TE3_salary']

# Feature Lists for Top Down Model Building


team_fpts_features=[
    'ImpliedPoints',
    'total_line',
    'spread_line',
    'avg_team_fpts',
    'team_fpts_allowed']

# 
team_pass_attempt_features=[
                            'ImpliedPoints',
                            'total_line',
                            'spread_line',
                            'team_fpts',
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
                            'pass_attempt',
                            'pass_attempt_allowed',
                            'avg_rush_attempt',
                            'rush_attempt_allowed',
                            'epa']

# Features to predict passing fantasy points
team_passing_DKPts_features=[
    'ImpliedPoints',
    'total_line',
    'spread_line',
    'team_fpts',
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
    'pass_attempt',
    'rush_attempt',
    'passing_fpts',
    'passing_fpts_allowed',
    'rushing_fpts',
    'rushing_fpts_allowed',
    'avg_receiving_fpts',
    'receiving_fpts_allowed']

TeamStatsModel_features_dict={
    'team_fpts':team_fpts_features,
    'pass_attempt':team_pass_attempt_features,
    'rush_attempt':team_rush_attempt_features,
    'passing_fpts':team_passing_DKPts_features,
    'rushing_fpts':team_rushing_DKPts_features,
    'receiving_fpts':team_receiving_DKPts_features}

feature_cols=['air_epa',
 'air_yards',
 'comp_air_epa',
 'comp_yac_epa',
 'complete_pass',
 'epa',
 'fumble',
 'fumble_lost',
 'interception',
 'pass_attempt',
 'passing_yards',
 'pass_touchdown',
 'qb_epa',
 'qb_hit',
 'sack',
 'yards_after_catch',
 'big_pass_play',
 'rush_attempt',
 'rush_touchdown',
 'rushing_yards',
 'tackled_for_loss',
 'big_run',
 'early_down_pass_rate',
 'early_down_success_rate',
 '3rdDown_success_rate',
 '4thDown_attempt_rate',
 '4thDown_success_rate',
 'EarlyGame_4thDown_attempt_rate',
 'EarlyGame_4thDown_success_rate',
 'Points',
 'TotalPoints',
 'OppPoints',
 'ADOT',
 'passer_rating',
 'ypa',
 'QB1_pass_cmp',
 'QB1_pass_att',
 'QB1_pass_yards',
 'QB1_pass_yds_per_att',
 'QB1_pass_air_yards',
 'QB1_pass_td',
 'QB1_int',
 'QB1_DKPts',
 'QB1_DKPts_share_pos',
 'QB1_passing_DKPts_share',
 'QB1_rushing_DKPts_share',
 'QB1_receiving_DKPts_share',
 'QB1_passer_rating',
 'RB1_offensive_snapcounts',
 'RB1_rush_att',
 'RB1_rush_share',
 'RB1_rush_yards',
 'RB1_rush_td',
 'RB1_targets',
 'RB1_rec',
 'RB1_rec_yards',
 'RB1_rec_td',
 'RB1_rush_redzone_looks',
 'RB1_PPO',
 'RB1_DKPts',
 'RB1_DKPts_share_pos',
 'RB1_passing_DKPts_share',
 'RB1_rushing_DKPts_share',
 'RB1_receiving_DKPts_share',
 'RB2_offensive_snapcounts',
 'RB2_rush_att',
 'RB2_rush_share',
 'RB2_rush_yards',
 'RB2_rush_td',
 'RB2_targets',
 'RB2_rec',
 'RB2_rec_yards',
 'RB2_rec_td',
 'RB2_rush_redzone_looks',
 'RB2_PPO',
 'RB2_DKPts',
 'RB2_DKPts_share_pos',
 'RB2_passing_DKPts_share',
 'RB2_rushing_DKPts_share',
 'RB2_receiving_DKPts_share',
 'RB3_offensive_snapcounts',
 'RB3_rush_att',
 'RB3_rush_share',
 'RB3_rush_yards',
 'RB3_rush_td',
 'RB3_targets',
 'RB3_rec',
 'RB3_rec_yards',
 'RB3_rec_td',
 'RB3_rush_redzone_looks',
 'RB3_PPO',
 'RB3_DKPts',
 'RB3_DKPts_share_pos',
 'RB3_passing_DKPts_share',
 'RB3_rushing_DKPts_share',
 'RB3_receiving_DKPts_share',
 'WR1_offensive_snapcounts',
 'WR1_targets',
 'WR1_rec',
 'WR1_rec_yards',
 'WR1_yac',
 'WR1_rec_air_yards',
 'WR1_wopr',
 'WR1_rec_td',
 'WR1_rec_redzone_looks',
 'WR1_PPO',
 'WR1_DKPts',
 'WR1_DKPts_share_pos',
 'WR1_passing_DKPts_share',
 'WR1_rushing_DKPts_share',
 'WR1_receiving_DKPts_share',
 'WR2_offensive_snapcounts',
 'WR2_targets',
 'WR2_rec',
 'WR2_rec_yards',
 'WR2_yac',
 'WR2_rec_air_yards',
 'WR2_wopr',
 'WR2_rec_td',
 'WR2_rec_redzone_looks',
 'WR2_PPO',
 'WR2_DKPts',
 'WR2_DKPts_share_pos',
 'WR2_passing_DKPts_share',
 'WR2_rushing_DKPts_share',
 'WR2_receiving_DKPts_share',
 'WR3_offensive_snapcounts',
 'WR3_targets',
 'WR3_rec',
 'WR3_rec_yards',
 'WR3_yac',
 'WR3_rec_air_yards',
 'WR3_wopr',
 'WR3_rec_td',
 'WR3_rec_redzone_looks',
 'WR3_PPO',
 'WR3_DKPts',
 'WR3_DKPts_share_pos',
 'WR3_passing_DKPts_share',
 'WR3_rushing_DKPts_share',
 'WR3_receiving_DKPts_share',
 'WR4_offensive_snapcounts',
 'WR4_targets',
 'WR4_rec',
 'WR4_rec_yards',
 'WR4_yac',
 'WR4_rec_air_yards',
 'WR4_wopr',
 'WR4_rec_td',
 'WR4_rec_redzone_looks',
 'WR4_PPO',
 'WR4_DKPts',
 'WR4_DKPts_share_pos',
 'WR4_passing_DKPts_share',
 'WR4_rushing_DKPts_share',
 'WR4_receiving_DKPts_share',
 'WR5_offensive_snapcounts',
 'WR5_targets',
 'WR5_rec',
 'WR5_rec_yards',
 'WR5_yac',
 'WR5_rec_air_yards',
 'WR5_wopr',
 'WR5_rec_td',
 'WR5_rec_redzone_looks',
 'WR5_PPO',
 'WR5_DKPts',
 'WR5_DKPts_share_pos',
 'WR5_passing_DKPts_share',
 'WR5_rushing_DKPts_share',
 'WR5_receiving_DKPts_share',
 'TE1_offensive_snapcounts',
 'TE1_targets',
 'TE1_rec',
 'TE1_rec_yards',
 'TE1_yac',
 'TE1_rec_air_yards',
 'TE1_wopr',
 'TE1_rec_td',
 'TE1_rec_redzone_looks',
 'TE1_PPO',
 'TE1_DKPts',
 'TE1_DKPts_share_pos',
 'TE1_passing_DKPts_share',
 'TE1_rushing_DKPts_share',
 'TE1_receiving_DKPts_share',
 'TE2_offensive_snapcounts',
 'TE2_targets',
 'TE2_rec',
 'TE2_rec_yards',
 'TE2_yac',
 'TE2_rec_air_yards',
 'TE2_wopr',
 'TE2_rec_td',
 'TE2_rec_redzone_looks',
 'TE2_PPO',
 'TE2_DKPts',
 'TE2_DKPts_share_pos',
 'TE2_passing_DKPts_share',
 'TE2_rushing_DKPts_share',
 'TE2_receiving_DKPts_share',
 'TE3_offensive_snapcounts',
 'TE3_targets',
 'TE3_rec',
 'TE3_rec_yards',
 'TE3_yac',
 'TE3_rec_air_yards',
 'TE3_wopr',
 'TE3_rec_td',
 'TE3_rec_redzone_looks',
 'TE3_PPO',
 'TE3_DKPts',
 'TE3_DKPts_share_pos',
 'TE3_passing_DKPts_share',
 'TE3_rushing_DKPts_share',
 'TE3_receiving_DKPts_share',
 'passing_fpts',
 'rushing_fpts',
 'receiving_fpts',
 'team_fpts',
 'total_line',
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
 'air_epa_allowed',
 'air_yards_allowed',
 'comp_air_epa_allowed',
 'comp_yac_epa_allowed',
 'complete_pass_allowed',
 'epa_allowed',
 'fumble_allowed',
 'fumble_lost_allowed',
 'interception_allowed',
 'pass_attempt_allowed',
 'passing_yards_allowed',
 'pass_touchdown_allowed',
 'qb_epa_allowed',
 'qb_hit_allowed',
 'sack_allowed',
 'yards_after_catch_allowed',
 'big_pass_play_allowed',
 'rush_attempt_allowed',
 'rush_touchdown_allowed',
 'rushing_yards_allowed',
 'tackled_for_loss_allowed',
 'big_run_allowed',
 'early_down_pass_rate_allowed',
 'early_down_success_rate_allowed',
 '3rdDown_success_rate_allowed',
 '4thDown_attempt_rate_allowed',
 '4thDown_success_rate_allowed',
 'EarlyGame_4thDown_attempt_rate_allowed',
 'EarlyGame_4thDown_success_rate_allowed',
 'Points_allowed',
 'TotalPoints_allowed',
 'OppPoints_allowed',
 'ADOT_allowed',
 'passer_rating_allowed',
 'ypa_allowed',
 'QB1_pass_cmp_allowed',
 'QB1_pass_att_allowed',
 'QB1_pass_yards_allowed',
 'QB1_pass_yds_per_att_allowed',
 'QB1_pass_air_yards_allowed',
 'QB1_pass_td_allowed',
 'QB1_int_allowed',
 'QB1_DKPts_allowed',
 'QB1_DKPts_share_pos_allowed',
 'QB1_passing_DKPts_share_allowed',
 'QB1_rushing_DKPts_share_allowed',
 'QB1_receiving_DKPts_share_allowed',
 'QB1_passer_rating_allowed',
 'RB1_offensive_snapcounts_allowed',
 'RB1_rush_att_allowed',
 'RB1_rush_share_allowed',
 'RB1_rush_yards_allowed',
 'RB1_rush_td_allowed',
 'RB1_targets_allowed',
 'RB1_rec_allowed',
 'RB1_rec_yards_allowed',
 'RB1_rec_td_allowed',
 'RB1_rush_redzone_looks_allowed',
 'RB1_PPO_allowed',
 'RB1_DKPts_allowed',
 'RB1_DKPts_share_pos_allowed',
 'RB1_passing_DKPts_share_allowed',
 'RB1_rushing_DKPts_share_allowed',
 'RB1_receiving_DKPts_share_allowed',
 'RB2_offensive_snapcounts_allowed',
 'RB2_rush_att_allowed',
 'RB2_rush_share_allowed',
 'RB2_rush_yards_allowed',
 'RB2_rush_td_allowed',
 'RB2_targets_allowed',
 'RB2_rec_allowed',
 'RB2_rec_yards_allowed',
 'RB2_rec_td_allowed',
 'RB2_rush_redzone_looks_allowed',
 'RB2_PPO_allowed',
 'RB2_DKPts_allowed',
 'RB2_DKPts_share_pos_allowed',
 'RB2_passing_DKPts_share_allowed',
 'RB2_rushing_DKPts_share_allowed',
 'RB2_receiving_DKPts_share_allowed',
 'RB3_offensive_snapcounts_allowed',
 'RB3_rush_att_allowed',
 'RB3_rush_share_allowed',
 'RB3_rush_yards_allowed',
 'RB3_rush_td_allowed',
 'RB3_targets_allowed',
 'RB3_rec_allowed',
 'RB3_rec_yards_allowed',
 'RB3_rec_td_allowed',
 'RB3_rush_redzone_looks_allowed',
 'RB3_PPO_allowed',
 'RB3_DKPts_allowed',
 'RB3_DKPts_share_pos_allowed',
 'RB3_passing_DKPts_share_allowed',
 'RB3_rushing_DKPts_share_allowed',
 'RB3_receiving_DKPts_share_allowed',
 'WR1_offensive_snapcounts_allowed',
 'WR1_targets_allowed',
 'WR1_rec_allowed',
 'WR1_rec_yards_allowed',
 'WR1_yac_allowed',
 'WR1_rec_air_yards_allowed',
 'WR1_wopr_allowed',
 'WR1_rec_td_allowed',
 'WR1_rec_redzone_looks_allowed',
 'WR1_PPO_allowed',
 'WR1_DKPts_allowed',
 'WR1_DKPts_share_pos_allowed',
 'WR1_passing_DKPts_share_allowed',
 'WR1_rushing_DKPts_share_allowed',
 'WR1_receiving_DKPts_share_allowed',
 'WR2_offensive_snapcounts_allowed',
 'WR2_targets_allowed',
 'WR2_rec_allowed',
 'WR2_rec_yards_allowed',
 'WR2_yac_allowed',
 'WR2_rec_air_yards_allowed',
 'WR2_wopr_allowed',
 'WR2_rec_td_allowed',
 'WR2_rec_redzone_looks_allowed',
 'WR2_PPO_allowed',
 'WR2_DKPts_allowed',
 'WR2_DKPts_share_pos_allowed',
 'WR2_passing_DKPts_share_allowed',
 'WR2_rushing_DKPts_share_allowed',
 'WR2_receiving_DKPts_share_allowed',
 'WR3_offensive_snapcounts_allowed',
 'WR3_targets_allowed',
 'WR3_rec_allowed',
 'WR3_rec_yards_allowed',
 'WR3_yac_allowed',
 'WR3_rec_air_yards_allowed',
 'WR3_wopr_allowed',
 'WR3_rec_td_allowed',
 'WR3_rec_redzone_looks_allowed',
 'WR3_PPO_allowed',
 'WR3_DKPts_allowed',
 'WR3_DKPts_share_pos_allowed',
 'WR3_passing_DKPts_share_allowed',
 'WR3_rushing_DKPts_share_allowed',
 'WR3_receiving_DKPts_share_allowed',
 'WR4_offensive_snapcounts_allowed',
 'WR4_targets_allowed',
 'WR4_rec_allowed',
 'WR4_rec_yards_allowed',
 'WR4_yac_allowed',
 'WR4_rec_air_yards_allowed',
 'WR4_wopr_allowed',
 'WR4_rec_td_allowed',
 'WR4_rec_redzone_looks_allowed',
 'WR4_PPO_allowed',
 'WR4_DKPts_allowed',
 'WR4_DKPts_share_pos_allowed',
 'WR4_passing_DKPts_share_allowed',
 'WR4_rushing_DKPts_share_allowed',
 'WR4_receiving_DKPts_share_allowed',
 'WR5_offensive_snapcounts_allowed',
 'WR5_targets_allowed',
 'WR5_rec_allowed',
 'WR5_rec_yards_allowed',
 'WR5_yac_allowed',
 'WR5_rec_air_yards_allowed',
 'WR5_wopr_allowed',
 'WR5_rec_td_allowed',
 'WR5_rec_redzone_looks_allowed',
 'WR5_PPO_allowed',
 'WR5_DKPts_allowed',
 'WR5_DKPts_share_pos_allowed',
 'WR5_passing_DKPts_share_allowed',
 'WR5_rushing_DKPts_share_allowed',
 'WR5_receiving_DKPts_share_allowed',
 'TE1_offensive_snapcounts_allowed',
 'TE1_targets_allowed',
 'TE1_rec_allowed',
 'TE1_rec_yards_allowed',
 'TE1_yac_allowed',
 'TE1_rec_air_yards_allowed',
 'TE1_wopr_allowed',
 'TE1_rec_td_allowed',
 'TE1_rec_redzone_looks_allowed',
 'TE1_PPO_allowed',
 'TE1_DKPts_allowed',
 'TE1_DKPts_share_pos_allowed',
 'TE1_passing_DKPts_share_allowed',
 'TE1_rushing_DKPts_share_allowed',
 'TE1_receiving_DKPts_share_allowed',
 'TE2_offensive_snapcounts_allowed',
 'TE2_targets_allowed',
 'TE2_rec_allowed',
 'TE2_rec_yards_allowed',
 'TE2_yac_allowed',
 'TE2_rec_air_yards_allowed',
 'TE2_wopr_allowed',
 'TE2_rec_td_allowed',
 'TE2_rec_redzone_looks_allowed',
 'TE2_PPO_allowed',
 'TE2_DKPts_allowed',
 'TE2_DKPts_share_pos_allowed',
 'TE2_passing_DKPts_share_allowed',
 'TE2_rushing_DKPts_share_allowed',
 'TE2_receiving_DKPts_share_allowed',
 'TE3_offensive_snapcounts_allowed',
 'TE3_targets_allowed',
 'TE3_rec_allowed',
 'TE3_rec_yards_allowed',
 'TE3_yac_allowed',
 'TE3_rec_air_yards_allowed',
 'TE3_wopr_allowed',
 'TE3_rec_td_allowed',
 'TE3_rec_redzone_looks_allowed',
 'TE3_PPO_allowed',
 'TE3_DKPts_allowed',
 'TE3_DKPts_share_pos_allowed',
 'TE3_passing_DKPts_share_allowed',
 'TE3_rushing_DKPts_share_allowed',
 'TE3_receiving_DKPts_share_allowed',
 'passing_fpts_allowed',
 'rushing_fpts_allowed',
 'receiving_fpts_allowed',
 'team_fpts_allowed']








