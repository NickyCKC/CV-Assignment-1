# Lunar Lander Behavioral Cloning: Classification and Prediction

---

## Project Overview
This project involves applying Machine Learning techniques to perform classification and prediction tasks in the `LunarLander-v3` environment. The objective is to learn a policy from recorded gameplay data—a process known as behavioral cloning—to build an automatic agent that can land the spacecraft safely

## Phases of Development

### Phase 1: Data Collection and State Representation 
* **Data Generation**: Instances were collected using both manual keyboard control and a rule-based heuristic agent. 
* **State Representation**: The model uses the 8 raw variables provided by the environment (position, velocity, angle, angular velocity, and leg contacts)
* **Feature Engineering**: Three additional features were engineered to capture complex dynamics
    * **Predicted_x**: $x = x\_position + x\_velocity \times 0.5$ (estimates future horizontal drift)
    * **Time_to_ground**: $y\_position / |y\_velocity|$ (estimates seconds until impact)
    * **Tilt_worsening**: $Sign(angle \times angular\_velocity)$ (indicates if rotation is self-correcting)
* **Storage**: Data is stored simultaneously in `.arff` (for Weka) and `.csv` (for Scikit-learn) formats via a modified `print_line_data()` function

### Phase 2: Classification with Weka 
* **Objective**: Predict the next action of the lander given the current state
* **Evaluation**: Tested J48, Random Forest, IBk ($k=1$), and Multilayer Perceptron using 10-fold cross-validation
* **Top Performer**: Random Forest achieved **99.90% accuracy** on agent-generated data
* **Finding**: Engineered features provided no statistically significant improvement for classification, so the raw feature set was selected for deployment

### Phase 3: Deployment with Scikit-learn [cite: 118]
* **Implementation**: A `DecisionTreeClassifier` was trained in Scikit-learn using the agent-based dataset
* **Model Integration**: The model was serialized using `joblib` and integrated into the Lunar Lander game loop to predict actions in real-time
* **Online Performance**: The ML agent achieved a **90% success rate** and an average reward of **180.85**, outperforming the rule-based agent (75% success, 166.75 reward)

### Phase 4: Regression (Reward Prediction)
* **Objective**: Predict the immediate reward of the next timestep given the current state and action
* **Algorithms**: Compared Linear Regression and Random Forest
* **Conclusion**: Random Forest proved superior with significantly lower Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) across all datasets

## Files in this Repository
* `Assignment1.pdf`: Detailed project guidelines and requirements
* `ML_Assignment_1.pdf`: Final report containing experimental results and analysis
* `LunarLander_Modified.py`: Environment code with the integrated ML agent and data collection functions
* `training_agent.csv/arff`: Datasets used for model training
* `model.pkl`: The final trained classification model

## Installation & Usage
To run the automated agent, ensure you have the required Python libraries installed:
```bash
pip install scikit-learn pandas joblib
