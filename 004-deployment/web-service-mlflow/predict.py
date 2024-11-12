

import pickle
import mlflow
from flask import Flask, request, jsonify

RUN_ID = '5e47b3ff09524d12a7e92f0284352a09'
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

# load the model 
model_path = "/home/akogo/Desktop/MLOps/004-deployment/mlruns/1/5e47b3ff09524d12a7e92f0284352a09/artifacts/model/model.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

def prepare_features(ride):
    features = {
        'PU_DO': f"{ride['PULocationID']}_{ride['DOLocationID']}",
        'trip_distance': ride['trip_distance']
    }
    return features

def predict(features):
    
    preds = model.predict([features]) 
    return float(preds[0])

# Flask app setup
app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)
    result = {'duration': pred,
              "model_version": RUN_ID
              
              }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
