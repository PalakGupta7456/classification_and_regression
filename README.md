##  Problem statement for classification problem

"Trips and Travel.com" company wants to enable and establish a viable business model to expand the customer base. One of the ways to expand the customer base is to introduce a new offering of packages. Currently, there are 5 types of packages the company is offering * Basic, Standard, Deluxe, Super Deluxe, King. Looking at the data of the last year, we observed that 18% of the customers purchased the packages. However, the marketing cost was quite high because customers customers were contacted at random without looking at the available information. The company is now planning to launch a new product i.e, Wellness Tourism Package. Wellness Tourism is defined as Travel that allows the traveller to maintain, enhance and kick-start a healthy lifestyle, and support or increase one's sense of well-being. However, this time company wants to harness the available data of existing and potential customers to make the marketing expenditure more efficient. 

## Customer Purchase Prediction Using Classification Models

## Project Overview

This project aims to predict whether a customer will purchase a product (ProdTaken) based on various demographic and behavioral features. The dataset includes attributes like age, occupation, number of follow-ups, and pitch satisfaction scores. The target variable, ProdTaken, is a binary classification problem where:

1 indicates the customer purchased the product.

0 indicates the customer did not purchase the product.

Several classification models were implemented, tuned, and evaluated to determine the best-performing algorithm.

## Dataset

The dataset contains the following columns:

CustomerID->Unique ID for each customer

ProdTaken (Target)
1 = Purchased, 0 = Not Purchased

Age->Age of the customer

TypeofContact->Contact type (e.g., Online, Offline)

CityTier->Tier of the city the customer belongs to

DurationOfPitch->Duration of the sales pitch in minutes

Occupation->Customer’s occupation

Gender->Gender of the customer

NumberOfPersonVisiting->Number of persons visiting with the customer

NumberOfFollowups->Number of follow-ups made

ProductPitched->Product name pitched to the customer

PreferredPropertyStar->Preferred hotel star rating

MaritalStatus->Marital status of the customer

NumberOfTrips->Number of trips the customer has taken

Passport
1 = Has passport, 0 = No passport

PitchSatisfactionScore->Satisfaction score for the pitch

OwnCar

1 = Owns a car, 0 = Does not own a car

NumberOfChildrenVisiting->Number of children accompanying the customer

Designation->Customer’s job designation

MonthlyIncome->Monthly income of the customer

## Data Preprocessing & Cleaning

The dataset underwent extensive data preprocessing and cleaning:

Handling Missing Values: Missing values were identified and handled appropriately using imputation techniques.

Encoding Categorical Features:

One-hot encoding for nominal variables.

Label encoding for ordinal variables.

Feature Scaling:

Standardization (Z-score normalization) for numerical features.

Outlier Detection & Removal:

Outliers were detected using box plots and handled accordingly.

## Feature Engineering:

Created new derived features for better prediction.

## Model Implementation

Various classification models were implemented and evaluated:

Logistic Regression
Decision Tree Classifier
Random Forest Classifier
Gradient Boosting Classifier
AdaBoost Classifier
XGBoost Classifier

Each model was trained on the preprocessed dataset, and their performance was evaluated.

## Hyperparameter Tuning

Each model was hypertuned RandomizedSearchCV to optimize hyperparameters and improve performance.

## Model Evaluation Metrics

The models were evaluated using the following metrics:

Accuracy

Precision

Recall

F1 Score

ROC AUC Score

Final results were tabulated for comparison.

## Results & Best Model Selection

After evaluating all models, the best model was selected based on:

Highest ROC AUC Score

Best balance between Precision and Recall

Overall performance across metrics

The final test scores of the best-performing model that was Gradient BoostingClassifier were:

- Accuracy: 0.9611
- F1 score: 0.9597
- Precision: 0.9873
- Recall: 0.8115
- Roc Auc Score: 0.9045

Finally,
A ROC AUC curve was plotted to visualize the model’s ability to distinguish between the two classes.

## Conclusion

This project successfully built a classification model for predicting customer purchase behavior. The best-performing model can be used for targeted marketing and improving customer acquisition strategies.

## Future Enhancements

Implement deep learning models for improved prediction.
Use feature selection techniques to optimize feature importance.
Collect more data to improve generalization.







2) # Used Car Price Prediction

Project Overview

This project aims to predict the selling price of used cars based on various features such as brand, model, vehicle age, mileage, fuel type, and more. The dataset was scraped from CarDekho.com, and predictive modeling was performed using multiple machine learning algorithms to provide price suggestions for sellers based on market conditions.

Dataset

Source: CarDekho website (scraped data)

Size: 15,411 rows and 13 columns

Columns:

car_name - Name of the car

brand - Brand of the car

model - Model name

vehicle_age - Age of the vehicle (in years)

km_driven - Total distance driven (in kilometers)

seller_type - Type of seller (Individual/Dealer)

fuel_type - Fuel type (Petrol/Diesel/CNG/Electric)

transmission_type - Manual or Automatic transmission

mileage - Fuel efficiency (km/l)

engine - Engine displacement (cc)

max_power - Maximum power output (bhp)

seats - Number of seats

selling_price - Actual selling price (Target variable)

# Data Preprocessing

Data preprocessing steps included:
Handling missing values
Converting categorical variables into numerical form
Feature scaling where necessary
Splitting data into training and testing sets

# Model Training

The following regression models were trained and evaluated:

Random Forest Regressor
Linear Regression
AdaBoost Regressor
Gradient Boosting Regressor
XGBoost Regressor
K-Nearest Neighbors Regressor
Decision Tree Regressor

# Model Evaluation Metrics

The models were evaluated using:

R² Score (Coefficient of Determination)
Mean Absolute Error (MAE)
Mean Squared Error (MSE)

# Hyperparameter Tuning

RandomizedSearchCV was used to optimize hyperparameters.

The best performance was achieved using Random Forest Regressor, which provided the highest accuracy.

# Conclusion

The project successfully developed a predictive model that estimates used car prices with high accuracy. The Random Forest Regressor was found to be the best-performing model after hyperparameter tuning.


