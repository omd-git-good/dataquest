import streamlit as st
import pandas as pd

# Load the trained model
model = joblib.load('accident_severity_model.joblib')

st.title('Accident Severity Prediction')

# Create input fields for each feature
speed = st.slider('Speed of the vehicle', 0, 200, 50)
age = st.number_input('Age', 16, 100, 30)
gender = st.selectbox('Gender', ['Male', 'Female'])
vehicle_type = st.selectbox('Vehicle type', ['T.W', 'Car', 'Heavy Vehicle'])
lanes = st.number_input('Number of lanes', 1, 6, 2)
lane_width = st.slider('Lane width', 2.0, 5.0, 3.5)
road_type = st.selectbox('Road type', ['Urban', 'Rural'])
alcohol = st.selectbox('Alcohol consumption', ['Yes', 'No'])
crash_type = st.selectbox('Type of crash', ['Head-on', 'Rear-end'])
seatbelt = st.selectbox('Seatbelt usage', ['Yes', 'No'])
speed_limit = st.slider('Speed Limit on the road', 20, 120, 60)
road_condition = st.selectbox('Road surface condition', ['Dry', 'Wet', 'Icy'])

# Create a dataframe with the input data
input_data = pd.DataFrame({
    'Speed of the vehicle': [speed],
    'Age': [age],
    'Gender': [gender],
    'Vehicle type': [vehicle_type],
    'Number of lanes': [lanes],
    'Lane width': [lane_width],
    'Road type': [road_type],
    'Alcohol consumption': [alcohol],
    'Type of crash': [crash_type],
    'Seatbelt usage': [seatbelt],
    'Speed Limit on the road': [speed_limit],
    'Road surface condition': [road_condition]
})

# Make prediction when the user clicks the button
if st.button('Predict Severity'):
    prediction = model.predict(input_data)
    st.write(f'Predicted Accident Severity: {prediction[0]}')

# Display safety recommendations
st.header('Safety Recommendations')
st.write("""
1. Always wear a seatbelt.
2. Never drink and drive.
3. Obey speed limits and adjust speed according to road conditions.
4. Maintain a safe following distance.
5. Be extra cautious in adverse weather conditions.
6. Avoid distractions while driving (e.g., using mobile phones).
7. Ensure regular vehicle maintenance.
8. Use appropriate child restraints.
9. Take breaks during long drives to avoid fatigue.
10. Be aware of blind spots, especially around heavy vehicles.
""")
