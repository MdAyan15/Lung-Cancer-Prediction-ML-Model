from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd  # Add this line to import pandas
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
filename = 'Lung_Cancer.pkl'
with open(filename, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def hello_world():
    return render_template("lung_cancer.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Collecting input features from the form
        int_features = [int(x) for x in request.form.values()]
        final = np.reshape(int_features, (1, -1))
        
        # Convert to DataFrame with the same column names as used in training
        column_names = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 
                        'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 
                        'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 
                        'SWALLOWING DIFFICULTY', 'CHEST PAIN']
        
        final_df = pd.DataFrame(final, columns=column_names)  # Convert input to DataFrame

        # Make prediction
        prediction = model.predict(final_df)
        
        # Displaying the result
        output = prediction[0]  # We expect only one prediction
        if output == 0:
            return render_template('lung_cancer.html', pred='Person Has Lung Cancer')
        else:
            return render_template('lung_cancer.html', pred='Person Does Not Have Lung Cancer')

    else:
        return render_template('lung_cancer.html')

if __name__ == '__main__':
    app.run(debug=True)
