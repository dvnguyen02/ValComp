# Valorant Match Predictor

## Overview

The Valorant Match Predictor is a machine learning project that predicts the outcome of Valorant matches based on team compositions and map selection. It uses a neural network trained on historical match data to make predictions.

## Components

1. **Data Preprocessing**: Transforms raw match data into a format suitable for machine learning.
2. **Neural Network Model**: A PyTorch-based neural network for predicting match outcomes.
3. **Streamlit Web Application**: A user-friendly interface for making predictions.

## Requirements

- Python 3.7+
- PyTorch
- Streamlit
- pandas
- numpy
- scikit-learn
- joblib

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/valorant-match-predictor.git
   cd valorant-match-predictor
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Prepare your data:
   - Ensure you have a CSV file named `vct-data.csv` in the project root directory.
   - This file should contain historical Valorant match data.

## Usage

### Training the Model

1. Run the training script:
   ```
   python pytorch_nn_training.py
   ```
   This will train the neural network and save the model as `valorant_nn_model.pth` and the preprocessor as `valorant_preprocessor.pkl`.

### Running the Predictor App

1. Start the Streamlit app:
   ```
   streamlit run agent_picker.py
   ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Use the interface to select agents for Team A and Team B, and choose a map.

4. Click the "Predict Match Outcome" button to see the prediction.

## File Descriptions

- `pytorch_nn_training.py`: Script for training the neural network model.
- `agent_picker.py`: Streamlit application for the user interface.
- `valorant_nn_model.pth`: Trained PyTorch model (generated after training).
- `valorant_preprocessor.pkl`: Saved preprocessor for input data (generated after training).
- `vct-data.csv`: Historical match data for training (not included in the repository).

## Model Details

The predictor uses a neural network with the following architecture:
- Input layer: Size depends on the number of features after preprocessing
- Hidden layer 1: 64 neurons with ReLU activation
- Hidden layer 2: 32 neurons with ReLU activation
- Output layer: 1 neuron with Sigmoid activation

The model is trained to predict whether Team A will win (output > 0.5) or lose (output <= 0.5).

## Contributing

Contributions to improve the model or extend the functionality are welcome. Please feel free to submit pull requests or open issues for any bugs or feature requests.


