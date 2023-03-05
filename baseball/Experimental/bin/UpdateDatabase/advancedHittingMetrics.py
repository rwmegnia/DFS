#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 20:57:05 2022

@author: robertmegnia
"""
import pandas as pd
import numpy as np
import pickle
import os
from MLB_API_TOOLS import getDKPts
from DatabaseTools import getExWOBA

basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"


def getAdvancedHittingData(game_data, df, season):
    player_ids = []
    bat_hands = []
    splits = []
    launchSpeed = []
    launchAngle = []
    distance = []
    results = []
    rbis = []
    plate_appearances = []
    plays = game_data["plays"]["allPlays"]
    for play in plays:
        if 'event' not in play['result'].keys():
            continue
        result = play["result"]["event"]
        # print(result)
        if result == "Strikeout Double Play":
            result = "Strikeout"
        if (
            ("Caught Stealing" in result)
            | ("Pickoff" in result)
            | ("Stolen Base" in result)
            | (
                result
                in [
                    "Catcher Interference",
                    "Runner Out",
                    "Wild Pitch",
                    "Other Advance",
                    "Passed Ball",
                    "Runner Double Play",
                    "Balk",
                    "Pitching Substitution",
                    "Field Out",
                    "Ejection",
                    "Game Advisory",

                ]
            )
        ):
            continue
        player_id = play["matchup"]["batter"]["id"]
        bat_hand = play["matchup"]["batSide"]["code"]
        split = play["matchup"]["splits"]["batter"]
        playEvents = play["playEvents"]
        rbi = play["result"]["rbi"]
        player_ids.append(player_id)
        bat_hands.append(bat_hand)
        splits.append(split)
        results.append(result)
        rbis.append(rbi)
        plate_appearances.append(1)
        if result in [
            "Strikeout",
            "Walk",
            "Intent Walk",
            "Hit By Pitch",            
        ]:
            launchSpeed.append(np.nan)
            launchAngle.append(np.nan)
            distance.append(np.nan)
            continue
        for playEvent in playEvents:
            if "hitData" in playEvent.keys():
                if (
                    ("launchSpeed" in playEvent["hitData"].keys())
                    & ("totalDistance" in playEvent["hitData"].keys())
                    & ("launchAngle" in playEvent["hitData"].keys())
                ):
                    launchSpeed.append(playEvent["hitData"]["launchSpeed"])
                    launchAngle.append(playEvent["hitData"]["launchAngle"])
                    distance.append(playEvent["hitData"]["totalDistance"])
                else:
                    launchSpeed.append(np.nan)
                    launchAngle.append(np.nan)
                    distance.append(np.nan)
    metrics = pd.DataFrame(
        {
            "player_id": player_ids,
            "bats": bat_hands,
            "splits": splits,
            "launchSpeed": launchSpeed,
            "launchAngle": launchAngle,
            "distance": distance,
            "bat_result": results,
            "rbi": rbis,
            "plateAppearances": plate_appearances,
        }
    )
    metrics.loc[metrics.bat_result == "Single", "singles"] = 1
    metrics.singles.fillna(0, inplace=True)
    metrics.loc[metrics.bat_result == "Double", "doubles"] = 1
    metrics.doubles.fillna(0, inplace=True)

    metrics.loc[metrics.bat_result == "Triple", "triples"] = 1
    metrics.triples.fillna(0, inplace=True)

    metrics.loc[metrics.bat_result == "Home Run", "homeRuns"] = 1
    metrics.homeRuns.fillna(0, inplace=True)

    metrics.loc[metrics.bat_result == "Walk", "baseOnBalls"] = 1
    metrics.baseOnBalls.fillna(0, inplace=True)

    metrics.loc[metrics.bat_result == "Hit By Pitch", "hitByPitch"] = 1
    metrics.hitByPitch.fillna(0, inplace=True)

    metrics.loc[metrics.bat_result == "Strikeout", "strikeOuts"] = 1
    metrics.strikeOuts.fillna(0, inplace=True)

    metrics.loc[metrics.bat_result == "Intent Walk", "intentionalWalks"] = 1
    metrics.intentionalWalks.fillna(0, inplace=True)

    metrics.loc[metrics.bat_result == "Flyout", "flyOuts"] = 1
    metrics.flyOuts.fillna(0, inplace=True)

    metrics.loc[metrics.bat_result == "Groundout", "groundOuts"] = 1
    metrics.groundOuts.fillna(0, inplace=True)

    metrics.loc[metrics.bat_result == "Sac Fly", "sacFlies"] = 1
    metrics.sacFlies.fillna(0, inplace=True)

    metrics.loc[metrics.bat_result == "Sac Bunt", "sacBunts"] = 1
    metrics.sacBunts.fillna(0, inplace=True)

    if len(metrics.dropna()) != 0:
        model = pickle.load(
            open(f"{etcdir}/model_pickles/exWOBA_model.pkl", "rb")
        )
        metrics.loc[
            metrics.distance.isna() == False,
            [
                "probDouble",
                "probHomeRun",
                "probSingle",
                "probTriple",
                "probOut",
            ],
        ] = model.predict_proba(
            metrics.loc[
                metrics.distance.isna() == False,
                ["launchSpeed", "launchAngle", "distance"],
            ]
        )
    if "probDouble" in metrics.columns:
        total_metrics = metrics.groupby(
            ["player_id", "splits"], as_index=False
        ).agg(
            {
                "plateAppearances": np.sum,
                "singles": np.sum,
                "bats": "first",
                "doubles": np.sum,
                "triples": np.sum,
                "homeRuns": np.sum,
                "rbi": np.sum,
                "baseOnBalls": np.sum,
                "hitByPitch": np.sum,
                "strikeOuts": np.sum,
                "intentionalWalks": np.sum,
                "flyOuts": np.sum,
                "groundOuts": np.sum,
                "sacFlies": np.sum,
                "sacBunts": np.sum,
                "launchSpeed": np.mean,
                "launchAngle": np.mean,
                "distance": np.mean,
                "probSingle": np.sum,
                "probDouble": np.sum,
                "probTriple": np.sum,
                "probHomeRun": np.sum,
            }
        )
        total_metrics = getExWOBA(total_metrics, season)
        total_metrics.drop(
            [
                "probSingle",
                "probDouble",
                "probTriple",
                "probHomeRun",
                "plateAppearances",
            ],
            axis=1,
            inplace=True,
        )
    else:
        total_metrics = metrics.groupby(
            ["player_id", "splits"], as_index=False
        ).agg(
            {
                "singles": np.sum,
                "bats": "first",
                "doubles": np.sum,
                "triples": np.sum,
                "homeRuns": np.sum,
                "rbi": np.sum,
                "baseOnBalls": np.sum,
                "hitByPitch": np.sum,
                "strikeOuts": np.sum,
                "intentionalWalks": np.sum,
                "flyOuts": np.sum,
                "groundOuts": np.sum,
                "sacFlies": np.sum,
                "sacBunts": np.sum,
                "launchSpeed": np.mean,
                "launchAngle": np.mean,
                "distance": np.mean,
            }
        )
    total_metrics = total_metrics.merge(
        df[["player_id", "runs", "stolenBases"]], on="player_id", how="left"
    )
    DKPts_splits = getDKPts(total_metrics, "batters")
    DKPts_splits.name = "DKPts_splits"
    total_metrics["DKPts_splits"] = DKPts_splits.values
    total_metrics.drop(
        ["runs", "stolenBases", "DKPts"], axis=1, errors="ignore", inplace=True
    )
    return total_metrics, metrics
