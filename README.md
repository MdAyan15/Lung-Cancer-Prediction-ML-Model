# Lung-Cancer-Prediction-ML-Model
A web-based application that predicts the likelihood of a patient having lung cancer based on 15 input parameters. The model utilizes a Logistic Regression classifier trained on a labeled dataset for prediction accuracy.


## Features
The website predicts lung cancer status using the following 15 parameters:

    AGE, GENDER, ANXIETY, COUGHING, SMOKING, WHEEZING, YELLOW_FINGERS, PEER_PRESSURE, CHRONIC DISEASE, FATIGUE, ALCOHOL CONSUMING, SHORTNESS OF BREATH, ALLERGY, SWALLOWING DIFFICULTY, CHEST PAIN.
_______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
_______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

## Requirements : 

* Python: Version 3.7 or above
* Flask: For deploying the web application
* Materialize CSS: For frontend styling
* Scikit-learn: For Logistic Regression and model training.

_______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
_______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

## SETUP INSTRUCTIONS : 
1.  Environment Setup
    1.  Clone this repository :
        ```bash
        git clone https://github.com/your-repo/Lung-Cancer-Project.git
        cd Lung-Cancer-Project
        ```
    2.  Create and activate a virtual environment:
        * On windows :
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
        
        * On macOS/Linux : 
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    3.  Install dependencies :
    ```bash
    pip install -r requirements.txt
    ```
2.  Train the model
    1.  Open the notebook Lung Cancer Prediction.ipynb and train the Logistic Regression model with the dataset provided (survey lung cancer.csv).
    2. Save the trained model as a pickle file (Lung_Cancer.pkl) in the root directory:
    ```python 
    import pickle

    filename = 'Lung_Cancer.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(trained_model, file)
    ```

3. Flask Application :
    1. Ensure the following directory structure : <br>
    Lung-Cancer-Project/ <br>
    ├── app.py                 
    ├── Lung_Cancer.pkl <br>
    ├── requirements.txt <br>
    ├── static/             
│   ├── css/ <br>
│   ├── js/ <br>
│   └── images/<br>
    ├── templates/<br>
│       ├── lung_cancer.html 

    2. Start the Flask server : 
    ```
    python app.py
    ```
    3. Visit http://127.0.0.1:5000/ in your browser to use application.
    
## FRONTEND
* Materialize CSS is used for styling the webpage. You can get the required files from Materialize.

## SCREENSHOTS :
Initial User Interface : <br>
![Screenshot (215)](https://github.com/user-attachments/assets/c79616e4-0e71-4f4e-b0c4-e57e7524849b)

Prediction Results : <br>
![Screenshot (216)](https://github.com/user-attachments/assets/3b96aa66-3b34-485a-9ad9-6425b1a6e850) 
<br>
![Screenshot (217)](https://github.com/user-attachments/assets/893337e5-e8fb-41e5-9f02-05b20fc50f31) <br>


## HOW IT WORKS
1. The user inputs 15 health-related parameters into a form.
2. The Flask app processes the inputs and passes them to the Logistic Regression model.
3. The trained model predicts whether the person has lung cancer or not.

## ACKNOWLEDGEMENTS
This project uses publicly available data for educational purposes. The UI design leverages Materialize CSS for a clean and responsive layout.

