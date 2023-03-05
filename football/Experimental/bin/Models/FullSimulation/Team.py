#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 19:18:55 2023

@author: robertmegnia
"""

import pickle
import os
basedir = os.path.dirname(os.path.abspath(__file__))
model_dir = f"{basedir}/ml_models"
import pandas as pd
import re
class Player(pd.DataFrame):
    def __init__(self, player):
        super().__init__(player)
        self["depth_team"] = player.depth_team.values[0]
        self["first_name"] = player.FirstName.values[0]
        self["last_name"] = player.LastName.values[0]
        self["football_name"] = player.FootballName.values[0]
        self["gsis_id"] = player.gsis_id.values[0]
        self["jersey_number"] = player.JerseyNumber.values[0]
        self["position"] = player.position.values[0]
        self["full_name"] = player.full_name.values[0]
        self.name = self.full_name.values[0]
        self.pos = self.position.values[0]

    def __repr__(self):
        return f"{self.name} - {self.pos}"
    
class Team(object):
    def __init__(self, team, week, season, coach):
        self.coach = coach
        self.coach_string = "".join(self.coach.split(" "))
        self.team_abbrev = team
        self.import_coach_models()
        self.depth_chart = self.import_roster(week, season)
        self.kicker_id = self.depth_chart['K'][0].gsis_id.values[0]
        self.fg_model=pickle.load(
            open(
                f"{model_dir}/FieldGoalModels/{self.kicker_id}_FieldGoal_model.pkl",
                "rb",
                ))
    def __repr__(self):
        return self.team_abbrev

    def import_coach_models(self):
        print(self.coach_string)
        self.punt_fg_model = pickle.load(
            open(
                f"{model_dir}/CoachingModels/{self.coach_string}_PuntFieldGoal_model.pkl",
                "rb",
                ))
        self.defense_personnel_model = pickle.load(
            open(
                f"{model_dir}/CoachingModels/{self.coach_string}_defense_personnel_model.pkl",
                "rb",
            )
        )
        self.defenders_in_box_model = pickle.load(
            open(
                f"{model_dir}/CoachingModels/{self.coach_string}_defenders_in_box_model.pkl",
                "rb",
            )
        )
        self.n_pass_rushers_model = pickle.load(
            open(
                f"{model_dir}/CoachingModels/{self.coach_string}_n_pass_rushers_model.pkl",
                "rb",
            )
        )

        self.RunPass_model = pickle.load(
            open(
                f"{model_dir}/CoachingModels/{self.coach_string}_RunPass_model.pkl",
                "rb",
            )
        )
        
        self.RunPass4thQtr_model = pickle.load(
            open(
                f"{model_dir}/CoachingModels/{self.coach_string}_4thQtrRunPass_model.pkl",
                "rb",
            )
        )
        
        self.FourthDownDecisionModel = pickle.load(
            open(
                f"{model_dir}/CoachingModels/{self.coach_string}_4thDown_decision_model.pkl",
                "rb",
            )
        )
        self.FourthDown4thQtrDecisionModel = pickle.load(
            open(
                f"{model_dir}/CoachingModels/{self.coach_string}_4thQtr4thDown_decision_model.pkl",
                "rb",
            )
        )
        self.FourthDownRunPass_model = pickle.load(
            open(
                f"{model_dir}/CoachingModels/{self.coach_string}_4thDownPlayType_model.pkl",
                "rb",
            )
        )

        self.RunFormation_model = pickle.load(
            open(
                f"{model_dir}/CoachingModels/{self.coach_string}_RunFormation_model.pkl",
                "rb",
            )
        )
        self.FourthDownRunFormation_model = pickle.load(
            open(
                f"{model_dir}/CoachingModels/{self.coach_string}_4thDownRunFormation_model.pkl",
                "rb",
            )
        )
        self.FourthDownRunFormation_personnel_model = pickle.load(
            open(
                f"{model_dir}/CoachingModels/{self.coach_string}_4thDownRunFormation_personnel_model.pkl",
                "rb",
            )
        )       
        
        self.PassFormation_model = pickle.load(
            open(
                f"{model_dir}/CoachingModels/{self.coach_string}_PassFormation_model.pkl",
                "rb",
            )
        )
        self.FourthDownPassFormation_model = pickle.load(
            open(
                f"{model_dir}/CoachingModels/{self.coach_string}_4thDownPassFormation_model.pkl",
                "rb",
            )
        )
        self.FourthDownPassFormation_personnel_model = pickle.load(
            open(
                f"{model_dir}/CoachingModels/{self.coach_string}_4thDownPassFormation_personnel_model.pkl",
                "rb",
            )
        )
        self.Personnel_models = {"Run": {}, "Pass": {}}
        for model in os.listdir(f"{model_dir}/CoachingModels"):
            if (f"{self.coach_string}_Run" in model) & ("_personnel" in model):
                formation = re.findall(
                    f"{self.coach_string}_Run([A-Z].+)_personnel", model
                )[0]
                self.Personnel_models["Run"][formation] = pickle.load(
                    open(
                        f"{model_dir}/CoachingModels/{self.coach_string}_Run{formation}_personnel_model.pkl",
                        "rb",
                    )
                )
            elif (f"{self.coach_string}_Pass" in model) & (
                "_personnel" in model
            ):
                formation = re.findall(
                    f"{self.coach_string}_Pass([A-Z].+)_personnel", model
                )[0]
                self.Personnel_models["Pass"][formation] = pickle.load(
                    open(
                        f"{model_dir}/CoachingModels/{self.coach_string}_Pass{formation}_personnel_model.pkl",
                        "rb",
                    )
                )
        return

    def import_roster(self, week, season):
        roster = pd.read_csv("./depth_charts/RosterModelData.csv")
        roster["depth_team"] = roster.depth_team.astype(float)
        roster = roster[
            (roster.season == season)
            & (roster.week == week)
            & (roster.team == self.team_abbrev)
            & (roster.injury_status=='Active')
        ]
        depth_chart = {}
        depth_chart["K"] = []
        depth_chart["OL"] = []
        depth_chart["DL"] = []
        depth_chart["LB"] = []
        depth_chart["DB"] = []
        depth_chart["WR"] = []
        depth_chart["TE"] = []
        depth_chart["RB"] = []
        depth_chart["QB"] = []
        for player in roster.iterrows():
            position = player[1]["position"]
            if position not in depth_chart.keys():
                continue
            gsis_id = player[1]["gsis_id"]
            player = roster[roster.gsis_id == gsis_id]
            depth_chart[position].append(Player(player))
        return depth_chart