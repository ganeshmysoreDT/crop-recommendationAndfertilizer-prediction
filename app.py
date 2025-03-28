from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os
from flask_cors import CORS
from azure.cosmos import CosmosClient
from azure.cosmos import PartitionKey
import openai


app = Flask(__name__)
CORS(app)

# Load Crop Prediction Model (Logistic Regression)
with open("backend/model_crop.pkl", "rb") as f:
    crop_model_data = pickle.load(f)

crop_model = crop_model_data["model"]
scaler = crop_model_data["scaler"]
columns = crop_model_data["columns"]

# Load Fertilizer Prediction Model (RandomForestClassifier)
with open("backend/model_fertilizer.pkl", "rb") as f:
    fert_model_data = pickle.load(f)

fert_model = fert_model_data["model"]
ordinal_encoder_crop = fert_model_data["ordinal_encoder_crop"]
ordinal_encoder_soil = fert_model_data["ordinal_encoder_soil"]
ordinal_encoder_fertilizer = fert_model_data["ordinal_encoder_fertilizer"]



# Azure Cosmos DB Configuration
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
DATABASE_NAME = "CropFertilizerDB"
CONTAINER_NAME = "Predictions"

cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
database = cosmos_client.create_database_if_not_exists(DATABASE_NAME)
container = database.create_container_if_not_exists(id=CONTAINER_NAME, partition_key=PartitionKey(path="/id") )

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict_crop", methods=["POST"])
def predict_crop():
    try:
        data = request.json
        features = [[
            data["N"], data["P"], data["K"], 
            data["temperature"], data["humidity"], 
            data["ph"], data["rainfall"]
        ]]

        # Scale input features before prediction
        features_scaled = scaler.transform(features)
        predicted_crop = crop_model.predict(features_scaled)[0]

        # Store in Cosmos DB
        prediction_data = {
            "id": str(os.urandom(16).hex()),
            "type": "crop_prediction",
            "input": data,
            "result": predicted_crop,
        }
        container.upsert_item(prediction_data)

        return jsonify({"recommended_crop": predicted_crop})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/predict_fertilizer", methods=["POST"])
def predict_fertilizer():
    try:
        data = request.json

        # Encode categorical inputs
        crop_encoded = ordinal_encoder_crop.transform([[data["CropType"]]])[0][0]
        soil_encoded = ordinal_encoder_soil.transform([[data["SoilType"]]])[0][0]

        # Prepare input for prediction
        input_data = pd.DataFrame([[
            crop_encoded, soil_encoded, data["Temperature"], 
            data["Humidity"], data["Moisture"], data["Nitrogen"], 
            data["Potassium"], data["Phosphorous"]
        ]], columns=["Crop Type", "Soil Type", "Temperature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"])

        # Predict fertilizer and decode result
        fertilizer_encoded = fert_model.predict(input_data)[0]
        predicted_fertilizer = ordinal_encoder_fertilizer.inverse_transform([[fertilizer_encoded]])[0][0]

        # Store in Cosmos DB
        prediction_data = {
            "id": str(os.urandom(16).hex()),
            "type": "fertilizer_prediction",
            "input": data,
            "result": predicted_fertilizer,
        }
        container.upsert_item(prediction_data)

        return jsonify({"recommended_fertilizer": predicted_fertilizer})
    except Exception as e:
        return jsonify({"error": str(e)})

 
@app.route("/get_predictions", methods=["GET"])
def get_predictions():
    try:
        query = "SELECT * FROM c"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))

        # Structure the response correctly
        predictions = []
        for item in items:
            if item["type"] == "crop_prediction":
                predictions.append({
                    "crop": item["result"],  
                    "fertilizer": None
                })
            elif item["type"] == "fertilizer_prediction":
                predictions.append({
                    "crop": None,
                    "fertilizer": item["result"]
                })

        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
