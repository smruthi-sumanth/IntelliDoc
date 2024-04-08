import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


class ImageProcessor:
    def __init__(self, input_size=(224, 224), normalize=True):
        self.input_size = input_size
        self.normalize = normalize

    def process_image(self, uploaded_file):
        # Read the image file
        image = Image.open(uploaded_file)
        
        # Resize the image to the expected input size of your model
        image = image.resize(self.input_size)
        
        # Convert the image to a numpy array
        image = np.array(image)
        
        # Normalize the image if required by your model
        if self.normalize:
            image = image / 255.0
        
        # Expand dimensions to match the input shape of your model
        image = np.expand_dims(image, axis=0)
        
        return image

diabetes_model = pickle.load(open('/home/smruthi_rao/Desktop/IntelliDoc/Saved models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('IntelliDoc/Saved models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('IntelliDoc/Saved models/parkinsons_classifier.sav', 'rb'))
pneumonia_model = load_model('/home/smruthi_rao/Desktop/IntelliDoc/Saved models/pneumonia_cnn.h5')



with st.sidebar:
    selected = option_menu('IntelliDoc',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Pneumonia Prediction'], # Add this line
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person', 'lungs'], # Add this line
                           default_index=0)

# Pneumonia Prediction Page
if selected == 'Pneumonia Prediction':
    # Page title
    st.title('Pneumonia Prediction using CNN')

    # Image uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    # Button for prediction
    if st.button('Predict Pneumonia'):
        if uploaded_file is not None:
            loaded_processor = ImageProcessor()
            processed_image = loaded_processor.process_image(uploaded_file)
            prediction = pneumonia_model.predict(processed_image)
            # Assuming `prediction` is a 1D array where each element is the probability of the positive class

            if prediction[0][0] > 0.5: 
                st.success("The person has pneumonia.")
            else:
                st.success("The person does not have pneumonia.")

        else:
            st.warning("Please upload an image to predict pneumonia.")



# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP-Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP-Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP-Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP-Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP-Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP-RAP')

    with col2:
        PPQ = st.text_input('MDVP-PPQ')

    with col3:
        DDP = st.text_input('Jitter-DDP')

    with col4:
        Shimmer = st.text_input('MDVP-Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP-Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer-APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer-APQ5')

    with col3:
        APQ = st.text_input('MDVP-APQ')

    with col4:
        DDA = st.text_input('Shimmer-DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)   


