import pandas as pd
from pycaret.classification import *

data=pd.read_csv("Dataset\\Loan Dataset.csv")

features = ['Credit_Score','Annual_Income','Loan_Amount_Requested','Existing_Loans','Loan_Term','Outstanding_Debt','Monthly_Expenses']

data=data[features + ['Loan_Approval_Status']]

clf_setup= setup(data=data, target='Loan_Approval_Status',session_id=123, normalize=True,verbose=False)

best_model=compare_models()

final_model=finalize_model(best_model)

save_model(final_model, "Loan_model")

print("Model Training completed and saved as Loan_model.pkl")