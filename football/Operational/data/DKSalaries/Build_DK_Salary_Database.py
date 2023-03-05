#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 09:13:41 2022

@author: robertmegnia
Download Archived Draftkings Salary Data
"""
from nfldfs import games
import pandas as pd
dst_names_dict={  'Arizona':'ARI',
                  'Atlanta':'ATL',
                  'Baltimore':'BAL',
                  'Buffalo':'BUF',
                  'Carolina':'CAR',
                  'Chicago':'CHI',
                  'Cincinnati':'CIN',
                  'Cleveland':'CLE',
                  'Dallas':'DAL',
                  'Denver':'DEN',
                  'Detroit':'DET',
                  'Green Bay':'GB',
                  'Houston':'HOU',
                  'Indianapolis':'IND',
                  'Jacksonville':'JAX',
                  'Kansas City':'KC',
                  'LA Chargers':'LAC',
                  'San Diego':'LAC',
                  'LA Rams':'LA',
                  'Los Angeles':'LA',
                  'St. Louis':'LA',
                  'Las Vegas':'LV',
                  'Oakland':'LV',
                  'Miami':'MIA',
                  'Minnesota':'MIN',
                  'New England':'NE',
                  'New Orleans':'NO',
                  'New York G':'NYG',
                  'New York J':'NYJ',
                  'Philadelphia':'PHI',
                  'Pittsburgh':'PIT',
                  'San Francisco':'SF',
                  'Seattle':'SEA',
                  'Tampa Bay':'TB',
                  'Tennessee':'TEN',
                  'Washington':'WAS'}
rename_teams={'GNB':'GB',
              'JAC':'JAX',
              'KAN':'KC',
              'LAR':'LA',
              'LVR':'LV',
              'NOR':'NO',
              'NWE':'NE',
              'SDG':'LAC',
              'STL':'LA',
              'OAK':'LV',
              'SFO':'SF',
              'TAM':'TB',}

rename_players={'Joshua Perkins':'Josh Perkins',
                    'Ben Watson':'Benjamin Watson',
                    'Tim Wright':'Timothy Wright',
                    'Damier Byrd':'Damiere Byrd',
                    'Dan Vitale':'Danny Vitale',
                    'Tj Graham':'Trevor Graham',
                    'Philly Brown':'Corey Brown',
                    'David Williams':'Dave Williams',
                    'Matthew Dayes':'Matt Dayes',
                    'Joshua Cribbs':'Josh Cribbs',
                    'Walter Powell':'Walt Powell',
                    'Pj Walker':'Phillip Walker',
                    'Kenny Gainwell':'Kenneth Gainwell',
                    'Nick Westbrook':'Nick Westbrookikhine'
                    }
# Get 2014-2020 first
g = games.find_games('dk', 2014, 1, 2020, 17)
stats = games.get_game_data(g)

# Get 2021 separately since weeks per season increases to 18
g = games.find_games('dk',2021,1,2021,18)
stats=pd.concat([stats,games.get_game_data(g)])

# rename columns
stats.rename({'year':'season','player_name':'full_name','team_name':'team'},axis=1,inplace=True)

#make team column  uppercase and rename teams
stats['team']=stats.team.apply(lambda x: x.upper())
stats.team.replace(rename_teams,inplace=True)

# Create Separate Frames for players and DST
players=stats[stats.position.isin(['QB','RB','WR','TE'])]
dst=stats[stats.position=='Def']

# Reformat dst names
dst['full_name']=dst.full_name.apply(lambda x: x.split(' Defense')[0])
dst['full_name']=dst.full_name.apply(lambda x: dst_names_dict[x])
dst.position.replace('Def','DST',inplace=True)
# Create first/last name columns for players
players['first_name']=players.full_name.apply(lambda x: x.split(', ')[1])
players['last_name']=players.full_name.apply(lambda x: x.split(', ')[0])

# Remove suffix from last name but keep prefix
players['last_name']=players.last_name.apply(lambda x: x if x in ['St. Brown','Vander Laan'] else x.split(' ')[0])

# Remove non-alpha numeric characters from first names.
players['first_name']=players.first_name.apply(lambda x: ''.join(c for c in x if c.isalnum()))
players['last_name']=players.last_name.apply(lambda x: ''.join(c for c in x if c.isalnum()))
# Recreate full_name
players['full_name']=players.apply(lambda x: x.first_name+' '+x.last_name,axis=1)
players['full_name']=players.full_name.apply(lambda x: x.lower())

players.drop(['first_name','last_name'],axis=1,inplace=True)
players['full_name']=players.full_name.apply(lambda x: x.split(' ')[0][0].upper()+x.split(' ')[0][1::]+' '+x.split(' ')[-1][0].upper()+x.split(' ')[-1][1::])
players.full_name.replace(rename_players,inplace=True)
#
stats=pd.concat([players,dst])
stats.to_csv('./DKSalary_Database.csv',index=False)