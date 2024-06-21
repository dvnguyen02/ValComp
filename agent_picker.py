import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
try:
    model = joblib.load('best_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'best_model.pkl' is in the same directory as this script.")
    st.stop()

# Define the list of agents and maps
agents = ['omen', 'skye', 'raze', 'viper', 'cypher', 'breach', 'brimstone', 'killjoy',
          'jett', 'astra', 'fade', 'yoru', 'kayo', 'sova', 'harbor', 'sage', 'neon',
          'phoenix', 'chamber', 'gekko', 'reyna', 'clove', 'deadlock']

maps = ['Fracture', 'Pearl', 'Haven', 'Lotus', 'Bind', 'Split', 'Ascent', 'Icebox',
        'Sunset', 'Breeze']

# Create a function to preprocess the user input
def preprocess_input(team_a_agents, team_b_agents, selected_map):
    input_data = pd.DataFrame(index=[0])
    
    # Set the selected map
    input_data['Map'] = selected_map
    
    # Set the selected agents to 1
    for agent in agents:
        input_data[f'Team A {agent}'] = 1 if agent in team_a_agents else 0
        input_data[f'Team B {agent}'] = 1 if agent in team_b_agents else 0
    
    return input_data

# Create the Streamlit app
def main():
    st.title('Match Predictor')
    
    # Create three columns
    col1, col2, col3 = st.columns(3)
    
    # Team A selection (left column)
    with col1:
        st.header('Team A')
        team_a_agents = st.multiselect('Select 5 agents for Team A', agents, key='team_a', max_selections=5)
        st.write(f"Selected agents: {', '.join(team_a_agents)}")
    
    # Map selection (middle column)
    with col2:
        st.header('Map')
        selected_map = st.selectbox('Select a map', maps)
        
        # Add some space
        st.write("")
        st.write("")
        
        # Predict button in the middle column
        predict_button = st.button('Predict Match Outcome')
    
    # Team B selection (right column)
    with col3:
        st.header('Team B')
        team_b_agents = st.multiselect('Select 5 agents for Team B', agents, key='team_b', max_selections=5)
        st.write(f"Selected agents: {', '.join(team_b_agents)}")
    
    # Prediction logic
    if predict_button:
        if len(team_a_agents) != 5 or len(team_b_agents) != 5:
            st.warning('Please select exactly 5 agents for each team.')
        else:
            # Preprocess the user input
            input_data = preprocess_input(team_a_agents, team_b_agents, selected_map)
            
            try:
                # Make predictions using the loaded model
                prediction = model.predict(input_data)
                probabilities = model.predict_proba(input_data)
                
                # Display the prediction
                st.header('Match Prediction')
                if prediction[0] == 1:
                    st.success(f'The predicted outcome is: Team A wins! (With the winning rate of : {probabilities[0][1]:.2%})')
                else:
                    st.success(f'The predicted outcome is: Team B wins! (With the winning rate of : {probabilities[0][0]:.2%})')
                
                # Display feature importances if available
                if hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'feature_importances_'):
                    st.subheader('Feature Importances')
                    feature_imp = pd.DataFrame({'feature': input_data.columns, 
                                                'importance': model.best_estimator_.feature_importances_})
                    feature_imp = feature_imp.sort_values('importance', ascending=False).head(10)
                    st.bar_chart(feature_imp.set_index('feature'))
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")

if __name__ == '__main__':
    main()