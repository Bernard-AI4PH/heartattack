# Machine Learning with Tidymodels
This code script can be modified and used in research studies. I will appreciate it if you clone and send pull request on how best the model can be improved. 


# Project Description

This project applies machine learning techniques using the tidymodels framework in R to analyze heart attack data. It includes both logistic regression for classification and linear regression for prediction. The analysis involves data preprocessing, model training, hyperparameter tuning, and evaluation using best practices in machine learning.

## Dataset Used

The dataset used for this project is Heart Attack.csv, which contains patient health information related to heart conditions. Key variables include:

Age: Age of the patient.

Gender: Categorical variable representing male or female.

Pulse: Renamed from "impluse" in the original dataset.

Blood Pressure: Systolic (sys) and diastolic (dias) values.

Glucose Levels: Converted to mmol/L.

Troponin: A key biomarker for heart attacks.

Class: Binary outcome variable indicating the presence or absence of a heart attack.

## Files Attached

The project contains the following files:

1. Machine_Learning_Tidymodel_Logistic.Rmd

Purpose: Performs logistic regression to classify patients based on heart attack risk.

Key Steps:

Data preprocessing: Renaming columns, handling missing values, encoding categorical variables.

Exploratory Data Analysis (EDA): Checking distributions, visualizing class imbalance.

Model building: Creating a logistic regression model using tidymodels.

Cross-validation: Tuning the model for best performance.

Model evaluation: Checking accuracy, sensitivity, specificity, and ROC curve analysis.

Output: An HTML file [Machine_Learning_Tidymodel_Logistic.html](https://github.com/Bernard-AI4PH/heartattack/blob/main/Machine_Learning_Tidymodel_Logistic.html) for web-based viewing.

2. Machine_Learning_Tidymodel_Linear.Rmd

Purpose: Performs linear regression to predict troponin levels based on patient health metrics.

Key Steps:

Data preprocessing: Similar to logistic regression.

Exploratory Data Analysis (EDA): Correlation analysis and visualization.

Model building: Ridge regression using glmnet.

Hyperparameter tuning: Selecting the best penalty and mixture values.

Model evaluation: Assessing root mean squared error (RMSE).

Output: An HTML file [Machine_Learning_Tidymodel_Linear.html](https://github.com/Bernard-AI4PH/heartattack/blob/main/Machine_Learning_Tidymodel_Linear.html) for web-based viewing.

Viewing the Project

Both .Rmd files generate HTML reports, making it easy to view results interactively in a web browser. The HTML files allow for navigation using a floating table of contents for better readability.

## Repository Structure

/Project_Folder
│-- Heart Attack.csv  # Dataset used for analysis
│-- Machine_Learning_Tidymodel_Linear.Rmd  
│-- Machine_Learning_Tidymodel_Linear.html  
│-- Machine_Learning_Tidymodel_Logistic.Rmd  
│-- Machine_Learning_Tidymodel_Logistic.html 

## How to Run the Project

Clone the repository.

Open RStudio and install necessary packages (tidymodels, tidyverse, parsnip, glmnet, themis).

Load the .Rmd file and run it to generate the HTML report.

View the HTML file in a browser for an interactive report.


Thank you