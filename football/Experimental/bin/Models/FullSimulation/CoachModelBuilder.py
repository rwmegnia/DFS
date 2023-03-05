#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 19:44:59 2022

@author: robertmegnia
"""

import pandas as pd
import nfl_data_py as nfl
from sklearn.neural_network import MLPClassifier as NN
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
import pickle
import os
import numpy as np
from datetime import datetime
import warnings

warnings.simplefilter("ignore")
basedir = os.path.dirname(os.path.abspath(__file__))
model_dir = f"{basedir}/ml_models"
#%%
# Builds Necessary Models for game simulation
pbp = nfl.import_pbp_data(range(2016, 2023))
try:
    pbp["time"] = pbp.time.apply(
        lambda x: datetime.strptime(x, "%M:%S") if x is not np.nan else x
    )
    pbp["yardline"] = pbp.yrdln.apply(
        lambda x: int(x.split(" ")[-1]) if x is not np.nan else x
    )
except:
    pbp["time"] = pbp.time.apply(
        lambda x: datetime.strptime(x, "%M:%S") if x is not None else x
    )
    pbp["yardline"] = pbp.yrdln.apply(
        lambda x: int(x.split(" ")[-1]) if x is not None else x
    )
pbp["end_time"] = pbp.time.shift(-1)
pbp["elapsed_time"] = (pbp.time - pbp.end_time).apply(
    lambda x: x.total_seconds()
)

pbp["territory"] = pbp.yrdln.apply(
    lambda x: x.split(" ")[0] if x not in [None,np.nan, "50"] else x
)
pbp.loc[pbp.territory == pbp.defteam, "yardline"] = (
    100 - pbp.loc[pbp.territory == pbp.defteam, "yardline"]
)
pbp["score_differential"] = pbp.posteam_score - pbp.defteam_score
#


# Remove Funky Personnel
pbp = pbp[
    (pbp.offense_personnel.str.contains("DL") == False)
    & (pbp.offense_personnel.str.contains("LB") == False)
    & (pbp.offense_personnel.str.contains("2 QB") == False)
    & (pbp.offense_personnel.str.contains("3 QB") == False)
    & (pbp.offense_personnel.str.contains("DB") == False)
    & (pbp.offense_personnel.str.contains("LS") == False)
    & (pbp.offense_personnel.str.contains("K") == False)
    & (pbp.offense_personnel.str.contains("P") == False)
]

away_coaches = pbp.away_coach.unique()
home_coaches = pbp.home_coach.unique()
coaches = np.concatenate((away_coaches, home_coaches))
coaches = np.unique(coaches)

#%%
for coach_name in coaches:
    coach_string=''.join(coach_name.split(' '))
    coach_home = pbp[pbp.home_coach == coach_name]
    coach_home = coach_home[coach_home.posteam == coach_home.home_team]
    coach_away = pbp[pbp.away_coach == coach_name]
    coach_away = coach_away[coach_away.posteam == coach_away.away_team]
    coach = pd.concat([coach_home, coach_away])
    print(coach_name, len(coach))
    #4th Quarter Run/Pass Model
    temp = coach[coach.play_type.isin(["pass", "run","qb_kneel","qb_spike"])&(coach.qtr==4)]
    temp = temp[temp.play_type_nfl != "PAT2"]
    temp = temp[temp.down < 4]
    if len(temp) != 0:
        model = RF().fit(
            temp[
                [
                    "qtr",
                    "down",
                    "ydstogo",
                    "yardline",
                    "half_seconds_remaining",
                    "game_seconds_remaining",
                    "score_differential",
                ]
            ],
            temp.play_type,
        )
        coach_name = coach_name.replace(" ", "")
        pickle.dump(
            model,
            open(
                f"{model_dir}/CoachingModels/{coach_string}_4thQtrRunPass_model.pkl",
                "wb",
            ),
        )   
    #Run/Pass Model for 3rd down or less
    temp = coach[coach.play_type.isin(["pass", "run"])]
    temp = temp[temp.play_type_nfl != "PAT2"]
    temp = temp[temp.down < 4]
    if len(temp) != 0:
        model = LR().fit(
            temp[
                [
                    "qtr",
                    "down",
                    "ydstogo",
                    "yardline",
                    "half_seconds_remaining",
                    "game_seconds_remaining",
                    "score_differential",
                ]
            ],
            temp.play_type,
        )
        coach_name = coach_name.replace(" ", "")
        pickle.dump(
            model,
            open(
                f"{model_dir}/CoachingModels/{coach_string}_RunPass_model.pkl",
                "wb",
            ),
        )

    # Run Play Formation Model
    temp = coach[coach.play_type == "run"]
    temp = temp[temp.play_type_nfl != "PAT2"]
    temp = temp[temp.down < 4]
    temp = temp[temp.offense_formation.isna() == False]
    if len(temp) != 0:
        model = NN().fit(
            temp[
                [
                    "down",
                    "ydstogo",
                    "half_seconds_remaining",
                    "score_differential",
                ]
            ],
            temp.offense_formation,
        )
        coach_name = coach_name.replace(" ", "")
        pickle.dump(
            model,
            open(
                f"{model_dir}/CoachingModels/{coach_string}_RunFormation_model.pkl",
                "wb",
            ),
        )

    # Run Play Personnel/Formation Model
    # np.where(np.abs(random-probs)==np.abs(random-probs).min())[1][0]
    for formation in coach[
        (coach.play_type == "run") & (coach.offense_formation.isnull() == False)
    ].offense_formation.unique():
        temp = coach[coach.play_type == "run"]
        temp = temp[temp.play_type_nfl != "PAT2"]
        temp = temp[temp.down < 4]
        temp = temp[temp.offense_personnel.isna() == False]
        temp = temp[temp.offense_formation == formation]
        if len(temp) != 0:
            model = NN().fit(temp[["down", "ydstogo"]], temp.offense_personnel)
            pickle.dump(
                model,
                open(
                    f"{model_dir}/CoachingModels/{coach_string}_Run{formation}_personnel_model.pkl",
                    "wb",
                ),
            )

    # Pass Play Formation Model
    temp = coach[coach.play_type == "pass"]
    temp = temp[temp.play_type_nfl != "PAT2"]
    temp = temp[temp.down < 4]
    temp = temp[temp.offense_formation.isna() == False]
    if len(temp) != 0:
        model = NN().fit(
            temp[
                [
                    "down",
                    "ydstogo",
                    "half_seconds_remaining",
                    "score_differential",
                ]
            ],
            temp.offense_formation,
        )
        coach_name = coach_name.replace(" ", "")
        pickle.dump(
            model,
            open(
                f"{model_dir}/CoachingModels/{coach_string}_PassFormation_model.pkl",
                "wb",
            ),
        )

    # Pass Play Personnel/Formation Model
    # np.where(np.abs(random-probs)==np.abs(random-probs).min())[1][0]
    for formation in coach[
        (coach.play_type == "pass")
        & (coach.offense_formation.isnull() == False)
    ].offense_formation.unique():
        temp = coach[coach.play_type == "pass"]
        temp = temp[temp.play_type_nfl != "PAT2"]
        temp = temp[temp.down < 4]
        temp = temp[temp.offense_personnel.isna() == False]
        temp = temp[temp.offense_formation == formation]
        if len(temp) != 0:
            model = NN().fit(temp[["down", "ydstogo"]], temp.offense_personnel)
            pickle.dump(
                model,
                open(
                    f"{model_dir}/CoachingModels/{coach_string}_Pass{formation}_personnel_model.pkl",
                    "wb",
                ),
            )

    # 4th down play model
    temp = coach[coach.down == 4]
    temp = temp[temp.play_type.isin(["pass", "run"])]
    if len(temp) != 0:
        model = NN().fit(
            temp[
                [
                    "ydstogo",
                    "yardline",
                    "half_seconds_remaining",
                    "game_seconds_remaining",
                    "score_differential",
                ]
            ],
            temp.play_type,
        )
        pickle.dump(
            model,
            open(
                f"{model_dir}/CoachingModels/{coach_string}_4thDownPlayType_model.pkl",
                "wb",
            ),
        )

    # 4th down Run Play formation model
    temp = coach[coach.down == 4]
    temp = temp[temp.play_type == "run"]
    temp = temp[temp.offense_formation.isna() == False]
    if len(temp) != 0:
        model = NN().fit(
            temp[
                [
                    "ydstogo",
                    "yardline",
                    "half_seconds_remaining",
                    "game_seconds_remaining",
                    "score_differential",
                ]
            ],
            temp.offense_formation,
        )
        pickle.dump(
            model,
            open(
                f"{model_dir}/CoachingModels/{coach_string}_4thDownRunFormation_model.pkl",
                "wb",
            ),
        )

    for formation in coach[
        (coach.play_type == "run")
        & (coach.offense_formation.isnull() == False)
        & (coach.down == 4)
    ].offense_formation.unique():
        temp = coach[coach.play_type == "run"]
        temp = temp[temp.down == 4]
        temp = temp[temp.play_type_nfl != "PAT2"]
        temp = temp[temp.offense_personnel.isna() == False]
        temp = temp[temp.offense_formation == formation]
        if len(temp) != 0:
            model = NN().fit(temp[["down", "ydstogo"]], temp.offense_personnel)
            pickle.dump(
                model,
                open(
                    f"{model_dir}/CoachingModels/{coach_string}_4thDownRunFormation_personnel_model.pkl",
                    "wb",
                ),
            )

    # 4th down Pass Play formation model
    temp = coach[coach.down == 4]
    temp = temp[temp.play_type == "pass"]
    temp = temp[temp.offense_formation.isna() == False]
    if len(temp) != 0:
        model = NN().fit(
            temp[
                [
                    "ydstogo",
                    "yardline",
                    "half_seconds_remaining",
                    "game_seconds_remaining",
                    "score_differential",
                ]
            ],
            temp.offense_formation,
        )
        pickle.dump(
            model,
            open(
                f"{model_dir}/CoachingModels/{coach_string}_4thDownPassFormation_model.pkl",
                "wb",
            ),
        )

    for formation in coach[
        (coach.play_type == "pass")
        & (coach.offense_formation.isnull() == False)
        & (coach.down == 4)
    ].offense_formation.unique():
        temp = coach[coach.play_type == "pass"]
        temp = temp[temp.down == 4]
        temp = temp[temp.play_type_nfl != "PAT2"]
        temp = temp[temp.offense_personnel.isna() == False]
        temp = temp[temp.offense_formation == formation]
        if len(temp) != 0:
            model = NN().fit(temp[["down", "ydstogo"]], temp.offense_personnel)
            pickle.dump(
                model,
                open(
                    f"{model_dir}/CoachingModels/{coach_string}_4thDownPassFormation_personnel_model.pkl",
                    "wb",
                ),
            )
#%% Defensive Personnel, Pass Rushers, Defenders in Box
pbp = nfl.import_pbp_data(range(2016, 2023))
pbp["time"] = pbp.time.apply(
    lambda x: datetime.strptime(x, "%M:%S") if x is not None else x
)
pbp["end_time"] = pbp.time.shift(-1)
pbp["elapsed_time"] = (pbp.time - pbp.end_time).apply(
    lambda x: x.total_seconds()
)
pbp["yardline"] = pbp.yrdln.apply(
    lambda x: int(x.split(" ")[-1]) if x is not None else x
)
pbp["territory"] = pbp.yrdln.apply(
    lambda x: x.split(" ")[0] if x not in [None, "50"] else x
)
pbp.loc[pbp.territory == pbp.defteam, "yardline"] = (
    100 - pbp.loc[pbp.territory == pbp.defteam, "yardline"]
)
pbp["score_differential"] = pbp.posteam_score - pbp.defteam_score
pbp = pbp[
    (pbp.defense_personnel.str.contains("WR") == False)
    & (pbp.defense_personnel.str.contains("RB") == False)
    & (pbp.defense_personnel.str.contains("TE") == False)
    & (pbp.defense_personnel.str.contains("QB") == False)
    & (pbp.defense_personnel.str.contains("OL") == False)
    & (pbp.defense_personnel != "")
]
away_coaches = pbp.away_coach.unique()
home_coaches = pbp.home_coach.unique()
coaches = np.concatenate((away_coaches, home_coaches))
coaches = np.unique(coaches)
for coach_name in coaches:
    coach_string=''.join(coach_name.split(' '))
    coach_home = pbp[pbp.home_coach == coach_name]
    coach_home = coach_home[coach_home.defteam == coach_home.home_team]
    coach_away = pbp[pbp.away_coach == coach_name]
    coach_away = coach_away[coach_away.defteam == coach_away.away_team]
    coach = pd.concat([coach_home, coach_away])
    print(coach_name, len(coach))
    # Personnel Model
    temp = coach[coach.play_type.isin(["run", "pass"])]
    temp = temp[temp.play_type_nfl != "PAT2"]
    temp = temp[temp.defense_personnel.isna() == False]
    if len(temp) != 0:
        model = NN().fit(
            temp[
                [
                    "yardline",
                    "down",
                    "ydstogo",
                    "half_seconds_remaining",
                    "score_differential",
                ]
            ],
            temp.defense_personnel,
        )
        coach_name = coach_name.replace(" ", "")
        pickle.dump(
            model,
            open(
                f"{model_dir}/CoachingModels/{coach_string}_defense_personnel_model.pkl",
                "wb",
            ),
        )
    temp = coach[coach.play_type.isin(["run", "pass"])]
    temp = temp[temp.play_type_nfl != "PAT2"]
    temp = temp[temp.defenders_in_box.isna() == False]
    if len(temp) != 0:
        model = NN().fit(
            temp[
                [
                    "yardline",
                    "down",
                    "ydstogo",
                    "half_seconds_remaining",
                    "score_differential",
                ]
            ],
            temp.defenders_in_box,
        )
        coach_name = coach_name.replace(" ", "")
        pickle.dump(
            model,
            open(
                f"{model_dir}/CoachingModels/{coach_string}_defenders_in_box_model.pkl",
                "wb",
            ),
        )
    temp = coach[coach.play_type.isin(["run", "pass"])]
    temp = temp[temp.play_type_nfl != "PAT2"]
    temp = temp[temp.number_of_pass_rushers.isna() == False]
    if len(temp) != 0:
        model = NN().fit(
            temp[
                [
                    "yardline",
                    "down",
                    "ydstogo",
                    "half_seconds_remaining",
                    "score_differential",
                ]
            ],
            temp.number_of_pass_rushers,
        )
        coach_name = coach_name.replace(" ", "")
        pickle.dump(
            model,
            open(
                f"{model_dir}/CoachingModels/{coach_string}_n_pass_rushers_model.pkl",
                "wb",
            ),
        )

    # Redzone Personnel Model
    temp = coach[coach.play_type.isin(["run", "pass"])]
    temp = temp[temp.play_type_nfl != "PAT2"]
    temp = temp[temp.defense_personnel.isna() == False]
    temp = temp[temp.yardline <= 20]
    if len(temp) != 0:
        model = NN().fit(
            temp[
                [
                    "yardline",
                    "down",
                    "ydstogo",
                    "half_seconds_remaining",
                    "score_differential",
                ]
            ],
            temp.defense_personnel,
        )
        coach_name = coach_name.replace(" ", "")
        pickle.dump(
            model,
            open(
                f"{model_dir}/CoachingModels/{coach_string}_redzone_defense_personnel_model.pkl",
                "wb",
            ),
        )
    temp = coach[coach.play_type.isin(["run", "pass"])]
    temp = temp[temp.play_type_nfl != "PAT2"]
    temp = temp[temp.defenders_in_box.isna() == False]
    temp = temp[temp.yardline <= 20]

    if len(temp) != 0:
        model = NN().fit(
            temp[
                [
                    "yardline",
                    "down",
                    "ydstogo",
                    "half_seconds_remaining",
                    "score_differential",
                ]
            ],
            temp.defenders_in_box,
        )
        coach_name = coach_name.replace(" ", "")
        pickle.dump(
            model,
            open(
                f"{model_dir}/CoachingModels/{coach_string}_redzone_defenders_in_box_model.pkl",
                "wb",
            ),
        )
    temp = coach[coach.play_type.isin(["run", "pass"])]
    temp = temp[temp.play_type_nfl != "PAT2"]
    temp = temp[temp.number_of_pass_rushers.isna() == False]
    temp = temp[temp.yardline <= 20]

    if len(temp) != 0:
        model = NN().fit(
            temp[
                [
                    "yardline",
                    "down",
                    "ydstogo",
                    "half_seconds_remaining",
                    "score_differential",
                ]
            ],
            temp.number_of_pass_rushers,
        )
        coach_name = coach_name.replace(" ", "")
        pickle.dump(
            model,
            open(
                f"{model_dir}/CoachingModels/{coach_string}_redzone_n_pass_rushers_model.pkl",
                "wb",
            ),
        )
#%% Punt or Field Goal Model
pbp = nfl.import_pbp_data(range(2016, 2023))
pbp["time"] = pbp.time.apply(
    lambda x: datetime.strptime(x, "%M:%S") if x is not None else x
)
pbp["end_time"] = pbp.time.shift(-1)
pbp["elapsed_time"] = (pbp.time - pbp.end_time).apply(
    lambda x: x.total_seconds()
)
pbp["yardline"] = pbp.yrdln.apply(
    lambda x: int(x.split(" ")[-1]) if x is not None else x
)
pbp["territory"] = pbp.yrdln.apply(
    lambda x: x.split(" ")[0] if x not in [None, "50"] else x
)
pbp.loc[pbp.territory == pbp.defteam, "yardline"] = (
    100 - pbp.loc[pbp.territory == pbp.defteam, "yardline"]
)
pbp["score_differential"] = pbp.posteam_score - pbp.defteam_score
#%%
for coach_name in coaches:
    coach_string=''.join(coach_name.split(' '))
    coach_home = pbp[pbp.home_coach == coach_name]
    coach_home = coach_home[coach_home.defteam == coach_home.home_team]
    coach_away = pbp[pbp.away_coach == coach_name]
    coach_away = coach_away[coach_away.defteam == coach_away.away_team]
    coach = pd.concat([coach_home, coach_away])
    print(coach_name, len(coach))
    # temp = coach[coach.play_type.isin(['punt','field_goal'])][
    #    ['yardline',
    #          'down',
    #          'ydstogo',
    #          'half_seconds_remaining',
    #          'score_differential',
    #          'play_type']
    #    ].dropna()
    # if len(temp)!=0:
    #     model = RF().fit(
    #         temp[['yardline',
    #               'down',
    #               'ydstogo',
    #               'half_seconds_remaining',
    #               'score_differential']],
    #         temp.play_type)
    #     pickle.dump(
    #                     model,
    #                     open(
    #                         f"{model_dir}/CoachingModels/{coach_string}_PuntFieldGoal_model.pkl",
    #                         "wb",
    #                     ),
    #                 )
    #4th Quarter 4th down decision model

    temp = coach[coach.down == 4]
    temp = temp[temp.play_type.isin(["punt", "field_goal", "pass", "run"])&(temp.qtr==4)]
    temp.loc[temp.play_type.isin(["punt", "field_goal"]), "go_for_it"] = 0
    temp.loc[~temp.play_type.isin(["punt", "field_goal"]), "go_for_it"] = 1
    if len(temp) > 0:
        try:
            model = LR().fit(
                temp[
                    [
                        "ydstogo",
                        "yardline",
                        "half_seconds_remaining",
                        "game_seconds_remaining",
                        "score_differential",
                    ]
                ],
                temp.go_for_it,
            )
            pickle.dump(
                    model,
                    open(
                        f"{model_dir}/CoachingModels/{coach_string}_4thQtr4thDown_decision_model.pkl",
                        "wb",
                    ),
                )
        except Exception as error:
            print(error)
            pass
    #4th down decision model

    # temp = coach[coach.down == 4]
    # temp = temp[temp.play_type.isin(["punt", "field_goal", "pass", "run"])]
    # temp.loc[temp.play_type.isin(["punt", "field_goal"]), "go_for_it"] = 0
    # temp.loc[~temp.play_type.isin(["punt", "field_goal"]), "go_for_it"] = 1
    # if len(temp) != 0:
    #     if len(temp.go_for_it.unique())==1:
    #         print('Ay fuck head')
    #     model = LR().fit(
    #         temp[
    #             [
    #                 "ydstogo",
    #                 "yardline",
    #                 "half_seconds_remaining",
    #                 "game_seconds_remaining",
    #                 "score_differential",
    #             ]
    #         ],
    #         temp.go_for_it,
    #     )
    #     pickle.dump(
    #         model,
    #         open(
    #             f"{model_dir}/CoachingModels/{coach_string}_4thDown_decision_model.pkl",
    #             "wb",
    #         ),
    #     )
