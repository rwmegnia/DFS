# -*- coding: utf-8 -*-

import pandas as pd
import unidecode
import os
import sys
import requests
from sqlalchemy import create_engine
import pymysql
import mysql.connector
basedir = os.path.dirname(os.path.abspath(__file__))
datadir= f'{basedir}/../../../../data'
db=pd.read_csv(f'{datadir}/game_logs/Full/Offense_Database.csv')
dst_db=pd.read_csv(f'{datadir}/game_logs/Full/DST_Database.csv')
db=pd.concat([db,dst_db[['full_name','position','team','week','season','salary','DKPts']]])
frames=[]

def getDKPlayerData(contest_id):
    print(contest_id)
    # Pull contest info from draftkings api with contest_id
    contest_url = f"https://api.draftkings.com/contests/v1/contests/{contest_id}?format=json"
    contestInfo = requests.get(contest_url).json()

    # get draftGroupId from contestInfo json
    draftGroup = contestInfo["contestDetail"]["draftGroupId"]

    # Use draftGroup Number to pull seach players DK Player ID from DK API
    draftGroup_url = f"https://api.draftkings.com/draftgroups/v1/draftgroups/{draftGroup}/draftables"
    draftGroupInfo = pd.DataFrame(
        requests.get(draftGroup_url).json()["draftables"]
    )[["displayName", "playerId", "teamAbbreviation", "salary"]]

    # Rename some columns for merging purposes
    draftGroupInfo.rename(
        {
            "displayName": "draftkings_name",
            "playerId": "draftkings_player_id",
            "teamAbbreviation": "team",
            "salary": "Salary",
        },
        axis=1,
        inplace=True,
    )

    # Pull Salary csv data so we have each players salary for FLEX/CPT positions
    csv_url = f"https://www.draftkings.com/lineup/getavailableplayerscsv?draftGroupId={draftGroup}"
    salaries = pd.read_csv(csv_url)

    # rename columns for merging purposes
    salaries.rename(
        {
            "Roster Position": "roster_position",
            "Name": "draftkings_name",
            "TeamAbbrev": "team",
        },
        axis=1,
        inplace=True,
    )

    # Filter salaries to columns that we will use
    salaries = salaries[
        [
            "draftkings_name",
            "team",
            "Position",
            "roster_position",
            "Salary",
            "ID",
        ]
    ]

    # Need to add a space at the end of the team names. i.e. convert "Cowboys " to "Cowboys"
    # In order to merge all draftkings_player_ids correctly
    draftGroupInfo["draftkings_name"] = draftGroupInfo["draftkings_name"].apply(
        lambda x: x + " " if len(x.split(" ")) == 1 else x
    )

    # drop suffix from name columns i.e. 'Ronald Jones II' becomes 'Ronald Jones'
    # for merging purpsoses
    salaries["draftkings_name"] = salaries.draftkings_name.apply(
        lambda x: " ".join(x.split(" ")[0:2]) if len(x.split(" ")) > 1 else x
    )
    draftGroupInfo["draftkings_name"] = draftGroupInfo.draftkings_name.apply(
        lambda x: " ".join(x.split(" ")[0:2]) if len(x.split(" ")) > 1 else x
    )
    draftGroupInfo["draftkings_name"] = draftGroupInfo.draftkings_name.apply(
        lambda x: unidecode.unidecode(x)
    )
    # Have found that in some case the draftkings names in "draftGroupInfo" and salaries
    # don't matchup. Replace names as necessary such that they match the Player names in the
    # draftkings contest results file
    salaries.draftkings_name.replace(
        {
            "Dan Brown": "Daniel Brown",
            "Van Jefferson Jr.": "Van Jefferson",
        },
        inplace=True,
    )

    # merge salaries with draftGroupInfo
    salaries = salaries.merge(
        draftGroupInfo, on=["draftkings_name", "team", "Salary"], how="left"
    )
    salaries['contest_id']=contest_id
    salaries.set_index('draftkings_name',inplace=True)
    # converting player_id to int may raise exception if there was a merging
    # mismatch and one of the player_ids were filled as NaN
    try:
        salaries["draftkings_player_id"] = salaries.draftkings_player_id.astype(
            int
        )
    except:
        mismatch = salaries[
            salaries.draftkings_player_id.isna() == True
        ].index.unique()[0]
        print(f"Mismatch in playername {mismatch}")
        sys.exit()

    return salaries
def parseNFLLineup(df):
        def_dict={'WAS Football Team ':'WAS',
              'Saints ':'NO',
              'Chargers ':'LAC',
              'Ravens ':'BAL',
              'Jets ':'NYJ',
              'Patriots ':'NE',
              'Dolphins ':'MIA',
              'Colts ':'IND',
              'Lions ':'DET',
              'Cardinals ':'ARI',
              'Buccaneers ':'TB',
              'Giants ':'NYG',
              'Bears ':'CHI',
              'Chiefs ':'KC',
              'KC':'KC',
              'Packers ':'GB',
              'Titans ':'TEN',
              'Eagles ':'PHI',
              'Steelers ':'PIT',
              'Rams ':'LA',
              'Panthers ':'CAR',
              '49ers ':'SF',
              'Browns ':'CLE',
              'Falcons ':'ATL',
              'Seahawks ':'SEA',
              'Bengals ':'CIN',
              'Vikings ':'MIN',
              'Raiders ':'OAK',
              'Jaguars ':'JAX',
              'Texans ':'HOU',
              'Cowboys ':'DAL',
              'Bills ':'BUF',
              'Broncos ':'DEN'}
        qb = pd.DataFrame(
            df[df.Lineup.isna() == False]
            .Lineup.str.findall("QB ([A-Za-z\.'-]+ [A-Za-z\.'-]+)")
            .to_list(),
            columns=["Player"],
        )
        rbs = pd.DataFrame(
            df[df.Lineup.isna() == False]
            .Lineup.str.findall("RB ([A-Za-z\.'-]+ [A-Za-z\.'-]+)")
            .to_list(),
            columns=["Player", "Player"],
        )
        rb1 = rbs.iloc[:, 0].to_frame()
        rb2 = rbs.iloc[:, 1].to_frame()

        wrs = pd.DataFrame(
            df[df.Lineup.isna() == False]
            .Lineup.str.findall("WR ([A-Za-z\.'-]+ [A-Za-z\.'-]+)")
            .to_list(),
            columns=["Player", "Player", "Player"],
        )
        wr1 = wrs.iloc[:, 0].to_frame()
        wr2 = wrs.iloc[:, 1].to_frame()
        wr3 = wrs.iloc[:, 2].to_frame()
        te = pd.DataFrame(
            df[df.Lineup.isna() == False]
            .Lineup.str.findall("TE ([A-Za-z\.'-]+ [A-Za-z\.'-]+)")
            .to_list(),
            columns=["Player"],
        )
        flex = pd.DataFrame(
            df[df.Lineup.isna() == False]
            .Lineup.str.findall("FLEX ([A-Za-z\.'-]+ [A-Za-z\.'-]+)")
            .to_list(),
            columns=["Player"],
        )
        dst = pd.DataFrame(
            df[df.Lineup.isna() == False]
            .Lineup.str.findall("DST (\w+ )")
            .to_list(),
            columns=["Player"],
        )
        lineups = pd.concat(
            [
                qb,
                rb1,
                rb2,
                wr1,
                wr2,
                wr3,
                te,
                flex,
                dst,
            ]
        )
        lineups["lineup_id"] = lineups.index+1
        df = lineups.merge(
            df[
                [
                    "lineup_id",
                    "Points",
                    "Roster Position",
                    "%Drafted",
                    "FPTS",
                    "Prize",
                    "cashLine",
                    "contestName",
                    "contestKey",
                    "singleEntry",
                ]
            ],
            on="lineup_id",
        )
        df.set_index('Player',inplace=True)
        return df

def reformatNames(df):
    ## REFORMAT PLAYER NAMES BY REMOVING NON-ALPHA-NUMERICS
    df["first_name"] = df.full_name.apply(lambda x: x.split(" ")[0])
    df["last_name"] = df.full_name.apply(
        lambda x: " ".join(x.split(" ")[1::])
    )

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
    df.loc[df.position!='DST',"full_name"] = df.loc[df.position!='DST'].full_name.apply(
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
    df.loc[df.position=='DST','RotoName']=df.loc[df.position=='DST','team'].str.upper()
    df.loc[df.position=='DST','full_name']=df.loc[df.position=='DST','team'].str.lower()

    return df 

def getMilliPlayers(week,season):
    # Read in contest results
    df=pd.read_csv(f'{datadir}/MillionaireMakerContestResults/{season}/Week{week}_Millionaire_Results.csv')
    # Create a dataframe with player ownership
    players = df[df.Player.isna() == False]
    players.loc[players["%Drafted"].isna() == True, "%Drafted"] = "0%"
    players["ownership"] = players["%Drafted"].apply(
        lambda x: float(x.split("%")[0])
    )
    players = players[["Player", "contestKey", "ownership", "FPTS"]]
    players["Player"] = players.Player.apply(lambda x: " ".join(x.split(" ")[0:2]))
    players.set_index('Player',inplace=True)
    
    # Get data frames for Top100 Lineups and Lineups outside top 100
    not100_df=df[100::]
    df=df.head(100)
    df['lineup_id']=df.index+1
    not100_df['lineup_id']=not100_df.index+1
    # Parse players from the Top10 Lineups
    df=parseNFLLineup(df)
    not100_df=parseNFLLineup(not100_df)
    ##
    players_in_top_100=df.groupby(df.index).first()
    players_in_top_100['frequency_top_100']=df.groupby(df.index).size()
    players_in_top_100['top100']=True
    players_not_in_top_100=not100_df.groupby(not100_df.index).first()
    players_not_in_top_100['top100']=False
    players_not_in_top_100['frequency_top_100']=0
    players_not_in_top_100=players_not_in_top_100[~players_not_in_top_100.index.isin(players_in_top_100.index)]
    df=pd.concat([players_in_top_100,players_not_in_top_100])
    contestKey = df.contestKey.unique()[0]
    dk_data=getDKPlayerData(contestKey)
    dk_data.drop_duplicates(inplace=True)
    df=df.join(dk_data)    
    df = df.drop('FPTS',axis=1).join(players[['ownership','FPTS']])
    df.reset_index(inplace=True)
    df.rename({'index':'full_name',
               'Position':'position',
               'Player':'full_name'},axis=1,inplace=True)
    df['week']=week
    df['season']=season
    df.full_name.replace({    'Jeff Wilson':'Jeffery Wilson',
                              'Eli Mitchell':'Elijah Mitchell',
                              'Amonra St': 'Amonra Stbrown',
                              'Joshua Palmer':'Josh Palmer',
                              'Gabe Davis':'Gabriel Davis',},inplace=True)
    df.team.replace({'LAR':'LA'},inplace=True)
    
    # remove alphanumerics
    df=reformatNames(df)
    df.rename({'FPTS':'DKPts'},axis=1,inplace=True)
    df=df[
        ['full_name',
         'cashLine',
         'frequency_top_100',
         'top100',
         'team',
         'position',
         'Salary',
         'ownership',
         'DKPts',
         'week',
         'season',
         'RotoName']
        ]
    df.frequency_top_100/=100
    df.ownership/=100
    df.to_csv(f'{datadir}/TopLineupPlayers/{season}_Week{week}_MilliPlayers.csv',index=False)
    mydb = mysql.connector.connect(
        host="footballai-db-prod.cxgq1kandeps.us-east-2.rds.amazonaws.com",
        user="gridironai",
        password="thenameofthewind",
        database="gridironai",
    )
    sqlEngine = create_engine(
        "mysql+pymysql://gridironai:thenameofthewind@footballai-db-prod.cxgq1kandeps.us-east-2.rds.amazonaws.com/gridironai",
        pool_recycle=3600,
    )
    nick=pd.read_sql('Nicks_player_pools',con=sqlEngine)
    nick.drop(['DKPts','salary'],axis=1,inplace=True)
    nick.loc[nick.position=='DST','full_name']=nick.loc[nick.position=='DST','full_name'].apply(lambda x: x.split(' ')[0])
    nick=nick.merge(df,on=['RotoName','week','full_name','team','position','season'],how='left')
    nick.to_sql(con=sqlEngine, name="Nicks_player_pools_verification", if_exists="replace")

    df.to_sql(
        con=sqlEngine, name=f"{season}_Week{week}_Milli_Players", if_exists="replace"
    )