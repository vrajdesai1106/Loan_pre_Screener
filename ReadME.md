# Customer Churn Prediction

## Project Overview
This project predicts the likelihood of a loan applicant getting approval based on their personal, financial and credit details. It helps banks and financial institutions quickly identify high-risk and low-risk applicants.

The project includes two main components:

1. Model Training (main.py) - Uses PyCaret for automated machine learning, selecting and finalizing the best-performing clissification model.

2. Web Application(app.py) - Built with Gradio to allow interactive loan pre-screening and predictions. 

## Project files
1. Dataset: Loan Dataset from Kaggle
    
    The dataset contains information about loan applicants including demographics, financial details and credit history. Key columns include:
    * Applicant Details : Applicant_ID, Gender, Age, Marital_Status
    * Employment & Income: Employment_Status, Occupation_Type, Residential_Status, City/Town, Annual_Income, Monthly_Expenses
    * Credit & Loans: Credit_Score, Existing_Loans, Total_Existing_Loan_Amount, Outstanding_Debt, Loan_History
    * Loan Request: Loan_Amount_Requested, Loan_Term, Loan_Purpose, Interest_Rate, Loan_Type, Co-Applicant
    * Banking History: Bank_Account_History, Transaction_Frequency
    * Target Column: Loan_Approval_Status (Approved / Rejected)
    * Risk Indicator: Default_Risk

    The dataset is structured as a CSV file and provides comprehensive details required for predicting loan approval status.
2. main.py:
    * Libraries Used : 
        * pandas : for data manipulation
        * pycaret.classification : for automated machine learning workflow.

    * Workflow:
        1. Load Dataset: Reads applicant data from loan_dataset.csv
        2. Preprocessing and feature selection: Handles missing values, categorical encoding, and excludes non-predictive columns like Applicant_ID.
        3. Initialize PyCaret: Sets up the classification experiment with normalization, target variable Loan_Approval_Status.
        4. Model Comparison: Uses PyCaret’s compare_models() to test multiple classification algorithms and selects the best performing one.
        5. Model Finalization: Finalizes the best model to prepare it for deployment.
        6. Model Evaluation: Provides an interactive interface to evaluate the finalized model's performance metrics.
        7. Model Saving: Saves the finalized model as churn_model for later use.

    * Output:

        * Trained and saved loan approval model ready for interactive use.

3. app.py:
    * Libraries Used
        * gradio: For building an interactive web interface
        * pandas: To organize user inputs into a DataFrame.
        * pycaret.classification: To load the saved model and perform prediction.

    * Workflow
        1. Load Model: Loads the trained loan_model.
        2. User Input: Accepts applicant details through Gradio input widgets (e.g. age, income, credit_score, existing loans, loan purpose). 
        3. Input Processing: Converts categorical inputs into numerical/encoded values as required by the model.
        4. Prediction: Generates loan approval prediction (Approved/ Rejected) and default risk if available.
        5. Output Display: Shows the prediction clearly to the user.

    * User Interaction
        * Simple and interactive interface for inputting applicant details.
        * Immediate loan approval status displayed with clear labeling.

## How To Run

1. Ensure Python environment has required packages : pandas, pycaret, gradio. (if not, install them using pip install <package_name>)
2. Place the dataset in "Dataset/loan_dataset.csv.
3. Run main.py to train and save the model(first time only).
4. Run the gradio app.
5. Use the web app interface to input applicant details to get loan approval predictions.


### NOTES
* The model training uses automated machine learning to select the best classification model for the churn dataset.
* The app expects inputs to be consistent with the training data features and formats. 
* Columns ignored during training are excluded in prediction inputs to avoid data leakage.
        