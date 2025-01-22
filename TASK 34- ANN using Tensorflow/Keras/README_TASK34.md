# TASK 34- ANN USING TENSORFLOW/KERAS

This task shows the application of TensorFlow and Keras on both regression and classification tasks using real-world datasets. It covers data preprocessing, model building, training, evaluation, and performance metrics.

## 1. Regression Task: University Admission Prediction

This task aims to predict the probability of a student’s admission based on various academic features such as GRE score, TOEFL score, CGPA, and university rating.

### Key Steps:
- **Data Loading & Preprocessing**: 
  - The dataset is loaded from `Admission_Predict_Ver1.1.csv`.
  - The `Serial No.` column is dropped as it's irrelevant.
  - Features (X) are scaled using `MinMaxScaler` to ensure all features are on the same scale, which improves model performance.
  - The data is split into training and testing sets (80/20 split).

- **Model Building**: 
  - A simple feedforward neural network (Sequential model) with 1 hidden layer consisting of 7 neurons (matching the number of input features).
  - The output layer uses a linear activation function suitable for regression.

- **Model Compilation**: 
  - The model is compiled using the `Adam` optimizer and `mean_squared_error` loss function.

- **Model Training**:
  - The model is trained for 100 epochs with a validation split of 20%.
  - During training, loss decreases from 0.3341 to 0.0046 on the training set, and validation loss reduces from 0.2674 to 0.0043.

- **Evaluation**:
  - **R² Score**: 0.7801 (78% variance in the target variable is explained by the model).
  - **Mean Squared Error (MSE)**: 0.004247.
  - These statistics indicate a good fit for the regression model, with low error and strong explanatory power.

### Model Results:
- **R² Score**: 78.01%
- **Mean Squared Error (MSE)**: 0.004247
- **Training Loss (final epoch)**: 0.0046
- **Validation Loss (final epoch)**: 0.0043


____________________________________________________
## 2. Classification Task: Customer Churn Prediction

This task predicts whether a customer will churn (leave the service) based on demographic and account information.

### Key Steps:
- **Data Loading & Preprocessing**: 
  - The dataset is loaded from `Churn_Modelling.csv` and unnecessary columns (`RowNumber`, `CustomerId`, and `Surname`) are dropped.
  - Categorical variables like `Geography` and `Gender` are encoded using one-hot encoding to create binary variables.
  - The dataset is split into training and testing sets (80/20 split), and features are standardized using `StandardScaler`.

- **Model Building**:
  - A simple feedforward neural network (Sequential model) is created with 1 hidden layer of 3 neurons and an output layer with a sigmoid activation function for binary classification.

- **Model Compilation**:
  - The model is compiled using the `Adam` optimizer and `binary_crossentropy` loss function, with accuracy as the evaluation metric.

- **Model Training**:
  - The model is trained for 100 epochs with a batch size of 50, achieving an accuracy of 79.75% on the test set.
  - Training accuracy increases from 63.31% to 80.56% over 100 epochs.
  - Validation accuracy stabilizes around 79.69% from epoch 5 onwards.

- **Evaluation**:
  - **Accuracy**: 79.75% on the test set.
  - **Loss**: 0.4848 (final loss).
  - The model demonstrates good performance in predicting customer churn.

### Model Results:
- **Test Accuracy**: 79.75%
- **Training Accuracy (final epoch)**: 80.56%
- **Validation Accuracy (final epoch)**: 79.69%
- **Final Loss**: 0.4848

### Visualizations:
- Training and validation loss curves show the model's decreasing error over time.
- Accuracy plots demonstrate the model's steady improvement in both training and validation sets.

## Conclusion:
- **Regression Task**: The model successfully predicts university admission probabilities, with an R² score of 78%, indicating a good fit. Further improvement can be made by tweaking the model architecture.
- **Classification Task**: The model achieved an accuracy of 79.75% on predicting customer churn, demonstrating solid performance. More advanced models or hyperparameter tuning could push the accuracy further.

