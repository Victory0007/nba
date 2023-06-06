# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:19:40 2023

@author: DELL  LATITUDE E5480
"""

from PIL import Image
import streamlit as st
import numpy as np
import pandas as pd
import pickle

#Loading in the data

#Playoff
#model
with open("C:/Users\DELL  LATITUDE E5480\Documents\Job Interview/model_east_po.pkl", "rb") as f:
    model_east_po = pickle.load(f)
with open("C:/Users\DELL  LATITUDE E5480\Documents\Job Interview/model_west_po.pkl", "rb") as f:
    model_west_po = pickle.load(f)
#Test data
with open("C:/Users\DELL  LATITUDE E5480\Documents\Job Interview/western_test_data", "rb") as f:
    western_test_data = pickle.load(f)
with open("C:/Users\DELL  LATITUDE E5480\Documents\Job Interview/eastern_test_data", "rb") as f:
    eastern_test_data = pickle.load(f)

#finals
with open("C:/Users\DELL  LATITUDE E5480\Documents\Job Interview/model_east_finals.pkl", "rb") as f:
    model_east_finals = pickle.load(f)
with open("C:/Users\DELL  LATITUDE E5480\Documents\Job Interview/model_west_finals.pkl", "rb") as f:
    model_west_finals = pickle.load(f)

#General/winner
with open("C:/Users\DELL  LATITUDE E5480\Documents\Job Interview/final_model.pkl", "rb") as f:
    final_model = pickle.load(f)
    
#Encoder
with open("C:/Users\DELL  LATITUDE E5480\Documents\Job Interview/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
    
#Full list of nba teams
nba_teams = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "LA Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards"
}

    
#Play_off prediction function
def make_predictions(model, data, encoder_test):
    eastern_teams = []
    predictions = model.predict_proba(data)
    one_pred = predictions[:,1]
    team_indices = np.argsort(one_pred)[:8]
    team_index = data.iloc[team_indices]
    teams = encoder_test.classes_
    abbrev_team = list(teams[list(team_index["franch_id"])])
    for i in abbrev_team:
        eastern_teams.append(nba_teams[i])
    return eastern_teams, abbrev_team

#Data of playoff teams
def playoff_teams(model, data, encoder_test):
    #Teams data that qualified for the playoffs
    full_names, pred_po = make_predictions(model, data, encoder_test)
    po = list(encoder_test.transform(pred_po))
    return data[data["franch_id"].apply(lambda x : x in po)]

#Final two teams prediction
def finals_prediction(model, data, encoder_test):
    eastern_teams = []
    predictions = model.predict_proba(data)
    one_pred = predictions[:,1]
    team_indices = np.argsort(one_pred)[:1]
    team_index = data.iloc[team_indices]
    teams = encoder_test.classes_
    abbrev_team = list(teams[list(team_index["franch_id"])])
    for i in abbrev_team:
        eastern_teams.append(nba_teams[i])
    return eastern_teams, abbrev_team

def playoff_data(model, data, encoder_test):
    #Teams data that qualified for the playoffs
    full_names, pred_po = finals_prediction(model, data, encoder_test)
    po = list(encoder_test.transform(pred_po))
    return data[data["franch_id"].apply(lambda x : x in po)]

#Play off teams prediction
eastern_playoff_teams = make_predictions(model_east_po, eastern_test_data, encoder)[0]
western_playoff_teams = make_predictions(model_west_po, western_test_data, encoder)[0]
#Playoff teams data
eastern_pred_po = playoff_teams(model_east_po, eastern_test_data, encoder)
western_pred_po = playoff_teams(model_west_po, western_test_data, encoder)
#Playoff teams final
eastern_pred_final = finals_prediction(model_east_finals, eastern_pred_po, encoder)[0]
western_pred_final = finals_prediction(model_west_finals, western_pred_po, encoder)[0]
#playoff teams final data
eastern_pred_final_data = playoff_data(model_east_finals, eastern_pred_po, encoder)
western_pred_final_data = playoff_data(model_west_finals, western_pred_po, encoder)
#Winner of the nba
winner = finals_prediction(final_model, pd.concat((eastern_pred_final_data, western_pred_final_data)), encoder)[0]
#print(eastern_pred_po)

#img_contact_form = Image.open("C:/Users\DELL  LATITUDE E5480\Documents\Job Interview\model deployment\images/He's On Fire wallpaper by Z_Studios - Download on ZEDGE™ _ 4c5e.jpeg")
#print(winner)
#st.image(img_contact_form)

# Add custom CSS styles to set the background image
page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Your Streamlit app code goes here


image_column, text_column = st.columns((1,2))
with image_column:
    nba_logo = Image.open("C:/Users\DELL  LATITUDE E5480\Documents\Job Interview\model deployment\images/7-feet-tall Letters.jpeg")
    st.image(nba_logo)
with text_column:
    st.title("NBA PREDICTIONS")
st.write("---")
st.subheader("This web app predicts teams that will make it to the 2023 playoffs based on past performances in previous years and the eventual winners of the playoff campaign.")
st.write("##")

image_column_east, text_column_east = st.columns((10,40))
with image_column_east:
    eastern_logo = Image.open("C:/Users\DELL  LATITUDE E5480\Documents\Job Interview\model deployment\images/nba-eastern-conference-logo-0B7E499625-seeklogo.com.png")
    st.image(eastern_logo)
with text_column_east:
    if st.button(":orange[PREDICT PLAYOFF TEAMS FROM THE EASTERN CONFERENCE]"):
        st.success(eastern_playoff_teams)
    if st.button(":orange[PREDICT WHICH EASTERN TEAM ADVANCES TO THE FINALS]"):
        st.success(eastern_pred_final)
        
st.write("###")

image_column_west, text_column_west = st.columns((10,40))
with image_column_west:
    western_logo = Image.open("C:/Users\DELL  LATITUDE E5480\Documents\Job Interview\model deployment\images/NBA Western Conference Logo PNG Vector (AI) Free Download.png")    
    st.image(western_logo)
with text_column_west:
    if st.button(":orange[PREDICT PLAYOFF TEAMS FROM THE WESTERN CONFERENCE]"):
        st.success(western_playoff_teams)
    if st.button(":orange[PREDICT WHICH WESTERN TEAM ADVANCES TO THE FINALS]"):
        st.success(western_pred_final)
st.warning("Please make sure you have run the predictions button for the eastern and western conference before proceeding")   
if st.button(":orange[PREDICT THE WINNER OF THE NBA FINALS]"):
    st.success(winner)
    

    
