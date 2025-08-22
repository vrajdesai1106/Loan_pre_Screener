import gradio as gr
import joblib
import pandas as pd
from pycaret.classification import load_model, predict_model

model=load_model("Loan_model")

def predict_loan(Credit_Score , Annual_Income, Loan_Amount_Requested, Existing_Loans, Loan_Term, Outstanding_Debt, Monthly_Expenses ):
    import numpy as np

    input_df=pd.DataFrame([[Credit_Score , Annual_Income, Loan_Amount_Requested, Existing_Loans, Loan_Term, Outstanding_Debt, Monthly_Expenses]], 
    columns=['Credit_Score','Annual_Income','Loan_Amount_Requested','Existing_Loans','Loan_Term','Outstanding_Debt','Monthly_Expenses'])

    prediction=predict_model(model,data=input_df)

    pred_label=prediction['prediction_label'][0]
    pred_prob=prediction['prediction_score'][0]

    status= "Approved" if pred_label==1 else "Rejected"

    return f"Loan Status : {status} \n Approval Probability : {pred_prob:.2f}"

inputs=[
    gr.Number(label= "Credit Score"),
    gr.Number(label= " Annual Income"),
    gr.Number(label= "Loan Amount Requested"),
    gr.Number(label= "Existing loans"),
    gr.Number(label= "Loan Term"),
    gr.Number(label= "Outstanding Debt"),
    gr.Number(label= "Monthly Expenses")
]

outputs=gr.Textbox(label="Prediction Result")

app=gr.Interface(fn=predict_loan, inputs=inputs, outputs=outputs, title='AI Loan Pre-Screener',description="Enter loan applicants details to predict loan approval status and probability")

app.launch(share=True)