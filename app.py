from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model from the pickle file
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json()

    # Perform the prediction using the loaded model
    prediction = model.predict(data)

    # Create the response JSON
    response = {
        'prediction': prediction.tolist()  # Convert the prediction to a list if needed
    }

    # Return the response as JSON
    return jsonify(response)

if __name__ == '__main__':
    app.run()
