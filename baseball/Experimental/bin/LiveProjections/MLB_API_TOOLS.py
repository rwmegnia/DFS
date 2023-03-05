# -*- coding: utf-8 -*-
import requests
import pandas as pd

teamAbbrevsDict = {
    133: "OAK",
    134: "PIT",
    135: "SD",
    136: "SEA",
    137: "SF",
    138: "STL",
    139: "TB",
    140: "TEX",
    141: "TOR",
    142: "MIN",
    143: "PHI",
    144: "ATL",
    145: "CWS",
    146: "MIA",
    147: "NYY",
    158: "MIL",
    108: "LAA",
    109: "ARI",
    110: "BAL",
    111: "BOS",
    112: "CHC",
    113: "CIN",
    114: "CLE",
    115: "COL",
    116: "DET",
    117: "HOU",
    118: "KC",
    119: "LAD",
    120: "WSH",
    121: "NYM",
}


def getSeasonSchedule(season):
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule?season={season}&gameType=R&sportId=1"
    )
    response = requests.get(url).json()
    return response


def getDKPts(df, player_type):
    if player_type == "batters":
        df["DKPts"] = (
            (df.singles * 3)
            + (df.doubles * 5)
            + (df.triples * 8)
            + (df.homeRuns * 10)
            + (df.rbi * 2)
            + (df.hitByPitch * 2)
            + (df.baseOnBalls * 2)
            + (df.runs * 2)
            + (df.stolenBases * 5)
        )
        return df.DKPts
    else:
        df.loc[
            (df.completeGames == 1) & (df.shutouts == 1), "complete_game_shutout"
        ] = 1
        df.complete_game_shutout.fillna(0, inplace=True)
        df["DKPts"] = (
            (df.wins * 4)
            + (df.inningsPitched * 2.25)
            + (df.strikeOuts * 2)
            + (df.earnedRuns * -2)
            + (df.hits * -0.6)
            + (df.hitByPitch * -0.6)
            + (df.baseOnBalls * -0.6)
            + (df.completeGames * 2.5)
            + (df.complete_game_shutout * 2.5)
        )
        return df.DKPts
