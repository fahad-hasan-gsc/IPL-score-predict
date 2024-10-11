import streamlit as st
import pickle
import pandas as pd
import numpy as np

lr=pickle.load(open('lr.pkl','rb'))
y_pred=pickle.load(open('y_pred.pkl','rb'))
teams=[
       'Royal Challengers Bangalore', 
       'Kings XI Punjab',
       'Kolkata Knight Riders', 
       'Rajasthan Royals',
       'Mumbai Indians',
       'Chennai Super Kings', 
       'Sunrisers Hyderabad',
       'Delhi Daredevils'
]
vanue=['M Chinnaswamy Stadium',
       'Punjab Cricket Association Stadium, Mohali', 'Feroz Shah Kotla',
       'Wankhede Stadium', 'Eden Gardens', 'Sawai Mansingh Stadium',
       'Rajiv Gandhi International Stadium, Uppal',
       'MA Chidambaram Stadium, Chepauk', 'Dr DY Patil Sports Academy',
       'Newlands', "St George's Park", 'Kingsmead', 'SuperSport Park',
       'Buffalo Park', 'New Wanderers Stadium', 'De Beers Diamond Oval',
       'OUTsurance Oval', 'Brabourne Stadium',
       'Sardar Patel Stadium, Motera', 'Barabati Stadium',
       'Vidarbha Cricket Association Stadium, Jamtha',
       'Himachal Pradesh Cricket Association Stadium', 'Nehru Stadium',
       'Holkar Cricket Stadium',
       'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
       'Subrata Roy Sahara Stadium',
       'Shaheed Veer Narayan Singh International Stadium',
       'JSCA International Stadium Complex', 'Sheikh Zayed Stadium',
       'Sharjah Cricket Stadium', 'Dubai International Cricket Stadium',
       'Maharashtra Cricket Association Stadium',
       'Punjab Cricket Association IS Bindra Stadium, Mohali',
       'Saurashtra Cricket Association Stadium', 'Green Park']
st.title('IPL Score Predictor')
col1,col2 =st.columns(2)
with col1:
    batting_team=st.selectbox('Batting Team',sorted(teams))
with col2:
    bowling_team=st.selectbox('Bowling Team',sorted(teams))
city=st.selectbox('Venue',sorted(vanue))
col3,col4,col5=st.columns(3)
with col3:
    runs=st.number_input('Runs')
with col4:
    Overs=st.number_input('Overs (overs>5)')
with col5:
    wickets=st.number_input('Wickets')
col6,col7=st.columns(2)
with col6:
    Runs_last_5=st.number_input('Runs Last 5 overs')
with col7:
    Wickets_last_5=st.number_input('Wickets Last 5 overs')
if st.button('Predict Score'):
    # Create a DataFrame from the inputs
    input_data = pd.DataFrame(
        [[batting_team, bowling_team, city, runs, Overs, wickets, Runs_last_5, Wickets_last_5]],
        columns=['batting_team', 'bowling_team', 'city', 'current_score', 'overs', 'wickets', 'runs_last_5', 'wickets_last_5']
    )
    
    # Perform one-hot encoding to match the model's expected input features
    input_data = pd.get_dummies(input_data, columns=['batting_team', 'bowling_team', 'city'])

    # Align the columns with the training set
    model_features = lr.feature_names_in_
    input_data = input_data.reindex(columns=model_features, fill_value=0)
    
    # Perform prediction
    predicted_score = lr.predict(input_data)
    st.write(f"The predicted final score is: {runs+predicted_score[0]:.0f} runs")