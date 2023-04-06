"""
Python Script to be Invoked by UiPath in Order to Perform a Logistic Regression Classification Model on a UiPath Datatable

 """


def predict():
    import pickle
    import pandas as pd
    import os
    import sklearn

    #Load in Scale and Model
    relative_path = os.path.dirname(os.path.realpath(__file__))
    scale = pickle.load(open(relative_path+"\Input\cancer_scale.pickle",'rb'))
    logr = pickle.load(open(relative_path+"\Input\cancer_pred.pickle",'rb'))
    
    print(logr.classes_)
    
    #Load in General Population
    median= pd.read_csv(relative_path+"\Input\cancer_gen_pop.csv")
    median.columns = ["Variable", "Coefficient"]
    median_key = dict(zip(median.Variable, median.Coefficient))
    
    #general_pred = logr.predict_proba(scale.transform(median.reshape(1,-1)))
    #Load in New Patient
    new_patient = pd.read_csv(os.path.dirname(relative_path) + "\ML_RPA_UiPathCode_Process_Legacy\Input\Temp_Patient.csv" )

    for val in new_patient.loc[new_patient.Coefficient.isna(), 'Variable']:
        new_patient.loc[new_patient["Coefficient"].isna(), "Coefficient"] = median_key[val]
        
    
    new_patient['Coefficient'] = new_patient['Coefficient'].astype(float)
 
    
    #Reindex New Patient Data
    new_patient = new_patient.set_index('Variable')
    new_patient = new_patient.reindex(index = median['Variable'])
    new_patient = new_patient.reset_index()
    
    #Reformat Before Passing to Scalar
    new_patient = new_patient.T
    new_patient.columns = new_patient.iloc[0]
    new_patient = new_patient[1:]
    
    median = median.T
    median.columns = median.iloc[0]
    median = median[1:]
    
    print(new_patient)
    
    #Scale Data Points
    scale_median = scale.transform(median)
    scale_new_patient = scale.transform(new_patient)
    
    #Perform Logistic Regression
    median_pred = logr.predict_proba(scale_median)
    new_pred = logr.predict_proba(scale_new_patient)

    print(median_pred)
    print(new_pred)
    risk_new = new_pred[0][0]
    risk_med = median_pred[0][0]
    
    #Relative Risk https://www.ncbi.nlm.nih.gov/books/NBK63647/#:~:text=For%20example%2C%20when%20the%20RR,1%2C%20the%20risk%20is%20unchanged.
    relative_risk = risk_new/risk_med
    
    with open(relative_path+'\Output\patient_risk.csv', 'w') as output:
        output.write(str(relative_risk))

    
if __name__ == "__main__":
    predict()
    print("Prediction Script Invoked")

