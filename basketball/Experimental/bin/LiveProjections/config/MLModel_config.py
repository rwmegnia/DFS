#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 05:43:13 2022

@author: robertmegnia

Configuration file for Position Machine Learning Model
"""


# Columns that can't/shouldn't be used in ML models
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 05:43:13 2022

@author: robertmegnia

Configuration file for Position Machine Learning Model
"""
import numpy as np

# Columns that can't/shouldn't be used in ML models
NonFeatures = [
    "game_id",
    "team_id",
    "team_abbreviation",
    "team_city",
    "player_id",
    "player_name",
    "nickname",
    "start_position",
    "comment",
    "game_date",
    "home_team_id",
    "visitor_team_id",
    "game_location",
    "opp",
    "position",
    "roster position",
    "started",
    "doublepoints",
    "doubleassists",
    "doublesteals",
    "doublerebounds",
    "doubleblocks",
    "doublestats",
    "doubledouble",
    "tripledouble",
    "season",
    "game_date_string",
    "rotoname",
    "comment",
    "config"
]

MinutesNonFeatures = [
    "game_id",
    "team_id",
    "team_abbreviation",
    "team_city",
    "player_id",
    "player_name",
    "nickname",
    "start_position",
    "comment",
    "game_date",
    "home_team_id",
    "visitor_team_id",
    "game_location",
    "opp",
    "position",
    "roster position",
    "started",
    "doublepoints",
    "doubleassists",
    "doublesteals",
    "doublerebounds",
    "doubleblocks",
    "doublestats",
    "doubledouble",
    "tripledouble",
    "season",
    "game_date_string",
    'rotoname',
    "comment",
    "min",
    "config"
]

DefenseFeatures = {'fgm':np.sum,
                   'fga':np.sum,
                   'fg_pct':np.mean,
                   'fg3m':np.sum,
                   'fg3a':np.sum,
                   'fg3_pct':np.mean,
                   'reb':np.sum,
                   'ast':np.sum,
                   'stl':np.sum,
                   'blk':np.sum,
                   'to':np.sum,
                   'pts':np.sum,
                   'plus_minus':np.sum,
                   'dkpts':np.sum,}

KnownFeatures = ["salary",
                 "mins"]

MinutesKnownFeatures = ["salary"]

TeamNonFeatures = [
    "team_abbreviation",
    "opp",
    "game_date",
    "game_id",
    "team_id",
    "home_team_id",
    "visitor_team_id",
    "season"
]

TeamKnownFeatures = ['proj_team_score',
                     'total_line']
feats=['fgm',
 'fga',
 'fg_pct',
 'fg3m',
 'fg3a',
 'fg3_pct',
 'ftm',
 'fta',
 'ft_pct',
 'oreb',
 'dreb',
 'reb',
 'ast',
 'stl',
 'blk',
 'to',
 'pf',
 'pts',
 'plus_minus',
 'usg_pct',
 'pct_fgm',
 'pct_fga',
 'pct_fg3m',
 'pct_fg3a',
 'pct_ftm',
 'pct_fta',
 'pct_oreb',
 'pct_dreb',
 'pct_reb',
 'pct_ast',
 'pct_tov',
 'pct_stl',
 'pct_blk',
 'pct_blka',
 'pct_pf',
 'pct_pfd',
 'pct_pts',
 'dkpts',
 'team_pts',
 'team_fg3m',
 'team_reb',
 'team_ast',
 'team_stl',
 'team_blk',
 'team_to',
 'team_dkpts',
 'seconds',
 'pct_dkpts',
 'salary',
 'mins',
 'fgm_allowed',
 'fga_allowed',
 'fg_pct_allowed',
 'fg3m_allowed',
 'fg3a_allowed',
 'fg3_pct_allowed',
 'reb_allowed',
 'ast_allowed',
 'stl_allowed',
 'blk_allowed',
 'to_allowed',
 'pts_allowed',
 'plus_minus_allowed',
 'dkpts_allowed']


minutes_feats=[
    ]