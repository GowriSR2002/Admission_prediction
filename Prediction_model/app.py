import streamlit as st
import pickle
import numpy as np

# Load the trained model
load_model = pickle.load(open('D:/SYSTEM DATA/Documents/GAYATHRI/Prediction_model/model.sav', 'rb'))

def chance_pred(input_data):
    input_data = np.asarray(input_data, dtype=float)  
    input_reshape = input_data.reshape(1, -1)
    prediction = load_model.predict(input_reshape)
    return prediction

def main():
    st.title("Chance of admission")
      
    GRE_Score = st.number_input("Enter GRE Score", min_value=0, max_value=500, format="%d")
    TOEFL_Score = st.number_input("Enter TOEFL Score", min_value=0, max_value=500, format="%d")
    University_Rating = st.number_input("Enter University Rating", min_value=0, max_value=10, format="%d")
    SOP = st.number_input("Enter SOP", min_value=0.0, max_value=10.0,format="%.1f")
    LOR = st.number_input("Enter LOR", min_value=0.0, max_value=10.0,  format="%.1f")
    CGPA = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0,  format="%.1f")
    Research = st.number_input("Research", min_value=0.0, max_value=100.0,  format="%.3f")
    
    diagnosis = ""
    if st.button("Check"):
        diagnosis = chance_pred([GRE_Score, TOEFL_Score, University_Rating, SOP, LOR, CGPA, Research])
        result = diagnosis[0] * 100  # Convert to percentage
        st.success(f"Congratulations! The predicted chance of admission is {result:.1f}%")


if __name__ == "__main__":
    main()
