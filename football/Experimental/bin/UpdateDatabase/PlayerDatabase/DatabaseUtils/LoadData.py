#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 04:02:34 2022

@author: robertmegnia
"""
import pandas as pd
import nfl_data_py as nfl
import os
import unidecode
import sys
basedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{basedir}/../../OwnershipDatabase')
from UpdateOwnershipDatabase import UpdateOwnershipDatabase

datadir = f"{basedir}/../../../../data"
#%%
rename_players = {
    "Beanie Wells": "Chris Wells",
    "Ben Watson": "Benjamin Watson",
    "Tim Wright": "Timothy Wright",
    "Danny Vitale": "Dan Vitale",
    "Chris Herndon": "Christopher Herndon",
    "Jeff Wilson": "Jeffery Wilson",
    "Jonathan Baldwin": "Jon Baldwin",
    "Matt Mcgloin": "Matthew Mcgloin",
    "Matthew Slater": "Matt Slater",
    "Michael Higgins": "Mike Higgins",
    "Josh Cribbs": "Joshua Cribbs",
    "Walt Powell": "Walter Powell",
    "Pj Walker": "Phillip Walker",
    "Robert Kelley": "Rob Kelley",
    "Clyde Gates": "Edmond Gates",
    "Drew Davis": "Dj Davis",
    "Gerrell Robinson": "Gerell Robinson",
}


def reformatNames(df):
    df["first_name"] = df.full_name.apply(lambda x: x.split(" ")[0])
    df["last_name"] = df.full_name.apply(lambda x: " ".join(x.split(" ")[1::]))

    # Remove suffix from last name but keep prefix
    df["last_name"] = df.last_name.apply(
        lambda x: x if x in ["St. Brown", "Vander Laan"] else x.split(" ")[0]
    )

    # Remove non-alpha numeric characters from first names.
    df["first_name"] = df.first_name.apply(
        lambda x: "".join(c for c in x if c.isalnum())
    )
    df["last_name"] = df.last_name.apply(
        lambda x: "".join(c for c in x if c.isalnum())
    )
    # Recreate full_name
    df["full_name"] = df.apply(
        lambda x: x.first_name + " " + x.last_name, axis=1
    )
    df["full_name"] = df.full_name.apply(lambda x: x.lower())
    df.drop(["first_name", "last_name"], axis=1, inplace=True)
    df.full_name = df.full_name.apply(
        lambda x: x.split(" ")[0][0].upper()
        + x.split(" ")[0][1::]
        + " "
        + x.split(" ")[-1][0].upper()
        + x.split(" ")[-1][1::]
    )
    df.full_name.replace(rename_players, inplace=True)
    return df.full_name


def get_game_snapcounts(season):
    # Get snap counts
    df = nfl.import_snap_counts([season])
    df = df[df.position.isin(["QB", "RB", "FB", "WR", "TE"])]
    df.team.replace({"SD": "LAC", "STL": "LA", "OAK": "LV"}, inplace=True)

    # Rename columns
    df.rename(
        {
            "player": "full_name",
            "offense_snaps": "offensive_snapcounts",
            "offense_pct": "offensive_snapcount_percentage",
            "pfr_player_id": "pfr_id",
        },
        axis=1,
        inplace=True,
    )
    ids = nfl.import_ids()
    df = df.merge(ids[["pfr_id", "gsis_id"]], on="pfr_id", how="left")
    df["full_name"] = reformatNames(df)
    df = df[df.season == season]
    return df


def UpdateSnapCountDatabase(season):
    snapcounts = get_game_snapcounts(season)
    snapcounts['RotoName']=snapcounts.full_name.apply(lambda x: x.split(' ')[0].lower()[0:4]+x.split(' ')[1].lower()[0:5])

    snapcounts.to_csv(
        f"{datadir}/SnapCountData/{season}_offensive_snapcounts.csv",
        index=False,
    )

    # Update Master Database
    db = pd.read_csv(
        f"{datadir}/SnapCountData/2012_present_offensive_snapcounts.csv"
    )
    db = db[db.season != season]
    db = pd.concat([db, snapcounts])
    db.to_csv(
        f"{datadir}/SnapCountData/2012_present_offensive_snapcounts.csv",
        index=False,
    )


def reformatName(df):
    ## REFORMAT PLAYER NAMES BY REMOVING NON-ALPHA-NUMERICS
    df["first_name"] = df.full_name.apply(lambda x: x.split(" ")[0])
    df["last_name"] = df.full_name.apply(lambda x: " ".join(x.split(" ")[1::]))

    # Remove non-alpha numeric characters from first/last names.
    df["first_name"] = df.first_name.apply(
        lambda x: "".join(c for c in x if c.isalnum())
    )
    df["last_name"] = df.last_name.apply(
        lambda x: "".join(c for c in x if c.isalnum())
    )

    # Recreate full_name to fit format "Firstname Lastname" with no accents
    df["full_name"] = df.apply(
        lambda x: x.first_name + " " + x.last_name, axis=1
    )
    df["full_name"] = df.full_name.apply(lambda x: x.lower())
    df.drop(["first_name", "last_name"], axis=1, inplace=True)
    df.loc[df.position != "DST", "full_name"] = df.loc[
        df.position != "DST"
    ].full_name.apply(
        lambda x: x.split(" ")[0][0].upper()
        + x.split(" ")[0][1::]
        + " "
        + x.split(" ")[-1][0].upper()
        + x.split(" ")[-1][1::]
    )
    df["full_name"] = df.full_name.apply(lambda x: unidecode.unidecode(x))

    # Create Column to match with RotoGrinders
    df["RotoName"] = df.full_name.apply(
        lambda x: x.lower().split(" ")[0][0:4] + x.lower().split(" ")[1][0:5]
    )
    df.loc[df.position == "DST", "RotoName"] = df.loc[
        df.position == "DST", "team"
    ].str.upper()
    return df


def load_pbp_data(season, update=False):
    print(f"Downloading Play-By-Play Data {season}...")
    # if update==True:
    # if season==2022:
    roster_data = load_roster_data(season, update)

    replace_names = {
        "Br.Hall": "Breece Hall",
        "G.Wilson": "Garrett Wilson",
        "R.Doubs": "Romeo Doubs",
        "C.Watson": "Christian Watson",
        "D.Pierce": "Dameon Pierce",
        "J.Dotson": "Jahan Dotson",
        "S.Moore": "Skyy Moore",
        "D.London": "Drake London",
        "C.Olave": "Chris Olave",
        "G.Pickens": "George Pickens",
        "T.Hairston": "Troy Hairston",
        "A.Rogers": "Armani Rogers",
        "Z.Horvath": "Zander Horvath",
        "T.Burks": "Treylon Burks",
        "W.Robinson": "Wandale Robinson",
        "K.Philips": "Kyle Philips",
        "C.Okonkwo": "Chigoziem Okonkwo",
        "D.Houston": "Dennis Houston",
        "R.White": "Rachaad White",
        "J.Warren": "Jaylen Warren",
        "I.Pacheco": "Isiah Pacheco",
        "I.Likely": "Isaiah Likely",
        "A.Pierce": "Alec Pierce",
        "K.Pickett": "Kenny Pickett",
    }
    df = nfl.import_pbp_data([season])
    df.two_point_conv_result.replace('success',1,inplace=True)
    df.two_point_conv_result.replace('failure',0,inplace=True)

    # potential_missing_passer_ids = df[(df.pass_attempt==1)&(df.passer_player_id.isnull()==True)]['passer_player_name'].unique()
    # potential_missing_receiver_ids = df[(df.pass_attempt==1)&(df.receiver_player_id.isnull()==True)]['receiver_player_name'].unique()
    # potential_missing_rusher_ids = df[(df.rush_attempt==1)&(df.rusher_player_id.isnull()==True)]['rusher_player_name'].unique()

    # print('Potential Missing Player Ids....')
    # print(potential_missing_passer_ids)
    # print(potential_missing_receiver_ids)
    # print(potential_missing_rusher_ids)

    # missing_ids = (df[df.receiver_player_name.isin(replace_names.keys())|
    #                   df.rusher_player_name.isin(replace_names.keys())|
    #                   df.passer_player_name.isin(replace_names.keys())].drop_duplicates())
    # rec = missing_ids[missing_ids.pass_attempt==1]
    # rec['index']=rec.index

    # rec.drop('receiver_player_id',axis=1,inplace=True)
    # rec.replace(replace_names,inplace=True)
    # roster_data.rename({'gsis_id':'receiver_player_id',
    #                     'team':'posteam',
    #                     'full_name':'receiver_player_name'},axis=1,inplace=True)
    # rec=rec.merge(roster_data[['receiver_player_name','receiver_player_id','week','season','posteam']],
    #               on=['receiver_player_name','posteam','week','season'],how='left')
    # rec.set_index('index',inplace=True)
    # #
    # rush = missing_ids[missing_ids.rush_attempt==1]
    # rush['index']=rush.index
    # rush.drop('rusher_player_id',axis=1,inplace=True)
    # rush.replace(replace_names,inplace=True)
    # roster_data.rename({'receiver_player_id':'rusher_player_id',
    #                     'team':'posteam',
    #                     'receiver_player_name':'rusher_player_name'},axis=1,inplace=True)
    # rush=rush.merge(roster_data[['rusher_player_name','rusher_player_id','week','season','posteam']],
    #               on=['rusher_player_name','posteam','week','season'],how='left')
    # rush.set_index('index',inplace=True)
    # #
    # passer = missing_ids[missing_ids.pass_attempt==1]
    # passer['index']=passer.index
    # passer.drop('passer_player_id',axis=1,inplace=True)
    # passer.replace(replace_names,inplace=True)
    # roster_data.rename({'rusher_player_id':'passer_player_id',
    #                      'team':'posteam',
    #                      'rusher_player_name':'passer_player_name'},axis=1,inplace=True)
    # passer=passer.merge(roster_data[['passer_player_name','passer_player_id','week','season','posteam']],
    #                on=['passer_player_name','posteam','week','season'],how='left')
    # passer.set_index('index',inplace=True)
    # df.drop(missing_ids.index,inplace=True)
    # df=pd.concat([df,rec,rush,passer])

    # else:
    #     df=pd.read_csv(f'{datadir}/pbp_data/{season}_pbpData.csv')
    # # Don't include playoff games
    # if season<2021:
    #     df=df[df.week<18]
    # else:
    #     df=df[df.week<19]
    # Add int/fumble return td columns for more accurate DST points allowed computation
    df["int_return_td"] = 0
    df["fumble_return_td"] = 0
    df["fg_block_return_td"] = 0
    df["punt_block_return_td"] = 0
    df["fg_return_td"] = 0
    df["punt_return_td"] = 0
    df["kick_return_td"] = 0
    df["fg_blocked"] = 0
    df.loc[df.field_goal_result == "blocked", "fg_blocked"] = 1
    df.loc[
        (df.touchdown == 1) & (df.interception == 1), "int_return_td"
    ] = 1
    df.loc[
        (df.touchdown == 1) & (df.fumble_lost == 1), "fumble_return_td"
    ] = 1
    df.loc[
        (df.touchdown == 1) & (df.field_goal_result == "blocked"),
        "fg_block_return_td",
    ] = 1
    df.loc[
        (df.touchdown == 1) & (df.punt_blocked == 1),
        "punt_block_return_td",
    ] = 1
    df.loc[
        (df.touchdown == 1) & (df.field_goal_result == "missed"),
        "fg_return_td",
    ] = 1
    df.loc[
        (df.return_touchdown == 1) & (df.punt_attempt == 1), "punt_return_td"
    ] = 1
    df.loc[
        (df.return_touchdown == 1) & (df.kickoff_attempt == 1), "kick_return_td"
    ] = 1
    # Fill indoor/outdoor column as indoor ('dome') where NaN
    df.roof.fillna("dome", inplace=True)
    df["Total"] = df.home_score + df.away_score
    df["proj_away_score"] = ((df.total_line) / 2) - (df.spread_line / 2)
    df["proj_home_score"] = ((df.total_line) / 2) + (df.spread_line / 2)
    df["proj_posteam_score"] = None
    df["proj_posteam_score"][df.posteam_type == "away"] = df[
        df.posteam_type == "away"
    ]["proj_away_score"]
    df["proj_posteam_score"][df.posteam_type == "home"] = df[
        df.posteam_type == "home"
    ]["proj_home_score"]
    df["proj_defteam_score"] = None
    df["proj_defteam_score"][df.posteam_type == "away"] = df[
        df.posteam_type == "away"
    ]["proj_home_score"]
    df["proj_defteam_score"][df.posteam_type == "home"] = df[
        df.posteam_type == "home"
    ]["proj_away_score"]
    return df


def load_roster_data(season, update):

    # Reformat names
    # Create first/last name columns for roster_data
    if season < 2022:
        roster_data = pd.read_csv(
            f"{datadir}/rosterData/{season}_SeasonGameRosters.csv"
        )
        roster_data = roster_data[
            roster_data.position.isin(["QB", "RB", "FB", "WR", "TE"])
        ]
        roster_data = roster_data[roster_data.injury_status == "Active"]
        roster_data["first_name"] = roster_data.full_name.apply(
            lambda x: x.split(" ")[0]
        )
        roster_data["last_name"] = roster_data.full_name.apply(
            lambda x: " ".join(x.split(" ")[1::])
        )

        # Remove suffix from last name but keep prefix
        roster_data["last_name"] = roster_data.last_name.apply(
            lambda x: x
            if x in ["St. Brown", "Vander Laan"]
            else x.split(" ")[0]
        )

        # Remove non-alpha numeric characters from first names.
        roster_data["first_name"] = roster_data.first_name.apply(
            lambda x: "".join(c for c in x if c.isalnum())
        )
        roster_data["last_name"] = roster_data.last_name.apply(
            lambda x: "".join(c for c in x if c.isalnum())
        )
        # Recreate full_name
        roster_data["full_name"] = roster_data.apply(
            lambda x: x.first_name + " " + x.last_name, axis=1
        )
        roster_data["full_name"] = roster_data.full_name.apply(
            lambda x: x.lower()
        )
        roster_data.drop(["first_name", "last_name"], axis=1, inplace=True)
        roster_data.full_name = roster_data.full_name.apply(
            lambda x: x.split(" ")[0][0].upper()
            + x.split(" ")[0][1::]
            + " "
            + x.split(" ")[-1][0].upper()
            + x.split(" ")[-1][1::]
        )
    else:
        roster_data = pd.read_csv(
            f"{datadir}/rosterData/{season}_SeasonGameRosters.csv"
        )
        roster_data = roster_data[
            roster_data.position.isin(["QB", "RB", "FB", "WR", "TE"])
        ]
        roster_data.rename(
            {
                "Height": "height",
                "College": "college",
                "CollegeConference": "college_conference",
            },
            axis=1,
            inplace=True,
        )
        roster_data = roster_data[
            [
                "full_name",
                "RotoName",
                "position",
                "team",
                "season",
                "week",
                "gsis_id",
                "injury_status",
                "rookie_year",
                "height",
                "weight",
                "draft_number",
                "draft_round",
                "college",
                "college_conference",
                "player_id",
            ]
        ]
    return roster_data


def load_salary_data(df, season, pos):
    salaries = pd.read_csv(f"{datadir}/DKSalaries/{season}_salaries.csv")
    if season < 2022:
        if pos != "DST":
            players = salaries[salaries.position.isin(["QB", "RB", "WR", "TE"])]
            df = df.merge(
                players[["full_name", "team", "season", "week", "salary"]],
                on=["full_name", "team", "season", "week"],
                how="left",
            )
            return df
        else:
            dst = salaries[salaries.position == "DST"]
            df = df.merge(
                dst[["full_name", "team", "season", "week", "salary"]],
                on=["full_name", "team", "season", "week"],
                how="left",
            )
            return df
    else:
        if pos != "DST":
            players = salaries[salaries.position.isin(["QB", "RB", "WR", "TE"])]
            print("Mismatch RotoName columns")
            print(
                df[~df.RotoName.isin(players.RotoName)][
                    ["full_name", "RotoName", "team", "position"]
                ]
            )

            df = df.merge(
                players[["RotoName", "team", "season", "week", "salary"]],
                on=["RotoName", "team", "season", "week"],
                how="left",
            )
            return df
        else:
            dst = salaries[salaries.position == "DST"].dropna()
            df = df.merge(
                dst[["team", "season", "week", "salary"]],
                on=["team", "season", "week"],
                how="left",
            )
            return df


def load_snapcount_data(df, season):
    UpdateSnapCountDatabase(season)
    snapcounts = pd.read_csv(
        f"{datadir}/SnapCountData/{season}_offensive_snapcounts.csv"
    )
    missing=snapcounts[snapcounts.gsis_id.isna()==True]
    snapcounts=snapcounts[snapcounts.gsis_id.isna()==False]
    df1 = df[df.gsis_id.isin(snapcounts.gsis_id)]
    df1 = df1.merge(
            snapcounts[
                [
                    "offensive_snapcounts",
                    "offensive_snapcount_percentage",
                    "season",
                    "week",
                    "gsis_id",
                ]
            ],
            on=["season", "week", "gsis_id"],
            how="left",
        )
    df2 = df.merge(
            missing[
                [
                    "offensive_snapcounts",
                    "offensive_snapcount_percentage",
                    "season",
                    "week",
                    "RotoName",
                ]
            ],
            on=["season", "week", "RotoName"],
            how="left",
        )
    df2=df2[~df2.gsis_id.isin(df1.gsis_id)]
    df=pd.concat([df1,df2])
    return df


def load_ownership_data(df, week, season):
    try:
        UpdateOwnershipDatabase(week,season)
    except FileNotFoundError:
        print('Millionaire Contets Not Available Yet!')
        pass
    df = reformatName(df)
    ownership = pd.read_csv(
        f"{datadir}/Ownership/2020_present_MainSlate_Ownership.csv"
    )
    if "DST" in df.position.unique():
        ownership["RotoName"] = ownership.RotoName.str.upper()
    df = df.merge(ownership, on=["RotoName", "week", "season"], how="left")
    df.drop("RotoName", axis=1, inplace=True)
    return df
