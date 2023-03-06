# NBA MODEL 3.0
import numpy as np
import pandas as pd

import re

import requests
from bs4 import BeautifulSoup

import seaborn as sns
import matplotlib.pyplot as plt


# =============================================== [--- Downloading data ---] ===============================================

def scrape_basketball_stats(url):
    # Send a request to the URL and retrieve the HTML content
    html_content = requests.get(url).content
    
    # Create a BeautifulSoup object from the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the advanced stats table and convert it to a pandas DataFrame
    adv_table = soup.find('table', {'id': 'advanced-team'})
    adv_df = pd.read_html(str(adv_table), header=None)[0]
    
    adv_df = adv_df.droplevel(0, axis=1)
    adv_df = adv_df.loc[:, ~adv_df.columns.str.contains('^Unnamed')] # drop all columns with 'Unnamed' in the name
    adv_df = adv_df.drop(columns='Arena')
    adv_df = adv_df.drop(columns='Attend.')
    adv_df = adv_df.drop(columns='Rk')
    
    # Find the shooting stats table and convert it to a pandas DataFrame
    shooting_table = soup.find('table', {'id': 'shooting-opponent'})
    shooting_df = pd.read_html(str(shooting_table))[0]
    
    shooting_df = shooting_df.droplevel(0, axis=1)
    shooting_df = shooting_df.loc[:, ~shooting_df.columns.str.contains('^Unnamed')]  # drop all columns with 'Unnamed' in the name
    shooting_df = shooting_df.drop(columns='Rk')
    
    # merge the two dataframes
    merged_df = pd.merge(adv_df, shooting_df, on='Team')
    
    # remove league average row
    merged_df = merged_df[merged_df.Team != 'League Average']
    
    return merged_df

def get_elo():
    url = 'https://projects.fivethirtyeight.com/2023-nba-predictions/'
    html = requests.get(url).content
    soup = BeautifulSoup(html, 'html.parser')
    
    # get the table (id = standings-table)
    table = soup.find('table', attrs={'id':'standings-table'})
    
    # go into the table body
    table_body = table.find('tbody')
    
    # every row in the table, get the team name and the spi rating
    rows = table_body.find_all('tr')
    elo = {}
    for row in rows:
        # remove the number at the end of the team name (there is no space between the name and the number)
        elo[re.split('(\d+)', row.find('td', attrs={'class':'team'}).text)[0]] = row.find('td', attrs={'class':'num elo carmelo-current'}).text
        
    # elo rating are currently between 1200 and 1700 change it to be between 0 and 1
    elo = {key: ((float(value) - 1200) / 500) * 100 for key, value in elo.items()}
    
    
    return elo

def downloadPastData(seasons):
    past_data = []
    for season in seasons:
        df = scrape_basketball_stats('https://www.basketball-reference.com/leagues/NBA_' + str(season) + '.html').drop(columns='Team')
        past_data.append(df)
    
    # save the data to a csv file
    pd.concat(past_data).to_csv('pastData/past_data.csv', index=False)
    
def downloadCurrentData():
    df = scrape_basketball_stats('https://www.basketball-reference.com/leagues/NBA_2023.html')
    
    # save the data to a csv file
    df.to_csv('currentData/current_data.csv', index=False)
    
def add_elo(df, elo):
    # convert elo values to floats
    elo = {key: float(value) for key, value in elo.items()}
    df['ELO'] = df.index.map(lambda x: elo[get_team_from_team_name(elo, x)])
    return df

def get_team_from_team_name(elo, team):
    for key in elo:
        if key in team and key != '':
            return key
        
    return "Bulls"

# downloadCurrentData()
# downloadPastData([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])

# =============================================== [--- Data Preprocessing ---] ===============================================
past_data = pd.read_csv('pastData/past_data.csv')
curr_data = pd.read_csv('currentData/current_data.csv')
teams = curr_data['Team']
actual_ORtg = curr_data['ORtg']
actual_DRtg = curr_data['DRtg']

# ----- Data Cleaning -----------------------------------------------------------------------------------------------
# remove the % sign from the columns
past_data = past_data.apply(lambda x: x.str.replace('%', '') if x.dtype == 'object' else x)
curr_data = curr_data.apply(lambda x: x.str.replace('%', '') if x.dtype == 'object' else x)

# convert nan values to 0
past_data = past_data.fillna(0)
curr_data = curr_data.fillna(0)

# ----- Data Normalization -----------------------------------------------------------------------------------------------
# normalize the data
past_data = (past_data - past_data.mean()) / past_data.std()
curr_data = (curr_data - curr_data.mean()) / curr_data.std()

# ----- Data Splitting -----------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

# split the data into training and testing sets
X_train_offense, X_test_offense, y_train_offense, y_test_offense = train_test_split(past_data.drop(columns='ORtg'), past_data['ORtg'], test_size=0.2, random_state=42)
X_train_defense, X_test_defense, y_train_defense, y_test_defense = train_test_split(past_data.drop(columns='DRtg'), past_data['DRtg'], test_size=0.2, random_state=42)

# =============================================== [--- Model Training ---] ===============================================
from sklearn.linear_model import Ridge

# build the model
model_off = Ridge(alpha=1.0)
model_def = Ridge(alpha=1.0)

# train the model
model_off.fit(X_train_offense, y_train_offense)
model_def.fit(X_train_defense, y_train_defense)

# =============================================== [--- Model Evaluation ---] ===============================================
# evaluate the model with continuous data
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# make predictions
y_pred_offense = model_off.predict(X_test_offense)
y_pred_defense = model_def.predict(X_test_defense)

# calculate the error
print('Mean Squared Error:', mean_squared_error(y_test_offense, y_pred_offense))
print('Mean Absolute Error:', mean_absolute_error(y_test_offense, y_pred_offense))
print('R2 Score:', r2_score(y_test_offense, y_pred_offense))
print(model_off.score(X_test_offense, y_test_offense))

print('Mean Squared Error:', mean_squared_error(y_test_defense, y_pred_defense))
print('Mean Absolute Error:', mean_absolute_error(y_test_defense, y_pred_defense))
print('R2 Score:', r2_score(y_test_defense, y_pred_defense))
print(model_def.score(X_test_defense, y_test_defense))

# =============================================== [--- Model Prediction ---] ===============================================

predictions_df = pd.DataFrame(columns=['Team', 'Predicted ORtg', 'Predicted DRtg'])

predictions_df['Team'] = teams
predictions_df.set_index('Team', inplace=True)

# make predictions
curr_data.drop(columns='Team', inplace=True)
predictions_df['Predicted ORtg'] = model_off.predict(curr_data.drop(columns='ORtg').reindex(X_train_offense.columns, axis=1))
predictions_df['Predicted DRtg'] = model_def.predict(curr_data.drop(columns='DRtg').reindex(X_train_defense.columns, axis=1))

# reformalize the data
# add absolute value of the minimum value to all values, then divide by the maximum value and multiply by 100
predictions_df['Predicted ORtg'] = (predictions_df['Predicted ORtg'] + abs(predictions_df['Predicted ORtg'].min()) * 2) / predictions_df['Predicted ORtg'].max() * 100
predictions_df['Predicted DRtg'] = (predictions_df['Predicted DRtg'] + abs(predictions_df['Predicted DRtg'].min()) * 2) / predictions_df['Predicted DRtg'].max() * 100

# make the spread between max and min less by applying log
predictions_df['Predicted ORtg'] = np.log(predictions_df['Predicted ORtg'])
predictions_df['Predicted DRtg'] = np.log(predictions_df['Predicted DRtg'])

# apply linear transformation to scale the data between 80 and 120
predictions_df['Predicted ORtg'] = (predictions_df['Predicted ORtg'] - predictions_df['Predicted ORtg'].min()) / (predictions_df['Predicted ORtg'].max() - predictions_df['Predicted ORtg'].min()) * 25 + 95
predictions_df['Predicted DRtg'] = (predictions_df['Predicted DRtg'] - predictions_df['Predicted DRtg'].min()) / (predictions_df['Predicted DRtg'].max() - predictions_df['Predicted DRtg'].min()) * 25 + 95

elo = get_elo()
print(elo)

# add elo to the strengths dataframe
predictions_df = add_elo(predictions_df, elo)

# show heatmap of the predictions
def illustrate():
    plt.figure(figsize=(10, 8))
    sns.heatmap(predictions_df.sort_values(by='ELO', ascending=False), annot=True, fmt='.1f', cmap='Blues')
    plt.show()

print(predictions_df)
illustrate()

# =============================================== [--- Predict Games  ---] ===============================================
from scipy.stats import poisson

# using the strengths of each team, predict the outcome of each game
def predict_game(home_team, away_team):
    # get the strengths of each team
    home_strength = predictions_df.loc[home_team]
    away_strength = predictions_df.loc[away_team]
    
    # calculate the expected goals for each team
    home_expected_gf = home_strength['Predicted ORtg'] * away_strength['Predicted DRtg'] / 100
    away_expected_gf = away_strength['Predicted ORtg'] * home_strength['Predicted DRtg'] / 100
    
    # apply elo to the expected goals for each team
    home_expected_gf = home_expected_gf * (1 + (home_strength['ELO'] - away_strength['ELO']) / 500)
    away_expected_gf = away_expected_gf * (1 + (away_strength['ELO'] - home_strength['ELO']) / 500)
    
    # calculate the probability of each outcome
    away_prob = 0
    home_prob = 0
    tie_prob = 0
    
    print(home_expected_gf, away_expected_gf)
    
    for i in range(0, 200):
        for j in range(0, 200):
            prob = poisson.pmf(i, home_expected_gf) * poisson.pmf(j, away_expected_gf)
            if i > j:
                home_prob += prob
            elif j > i:
                away_prob += prob
            else:
                tie_prob += prob
                
    away_prob = away_prob + (tie_prob / 2)
    home_prob = home_prob + (tie_prob / 2)
                
    return home_prob, away_prob

# predict the outcome of each game
print(predict_game('Denver Nuggets', 'Boston Celtics'))