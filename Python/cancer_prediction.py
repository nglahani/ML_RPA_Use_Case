"""
Python Script to be Invoked by UiPath in Order to Perform a Logistic Regression Classification Model on a UiPath Datatable

 """
 
import pickle

def predict():
    
    median= pickle.load(open("cancer_gen_pop.pickle", 'rb'))
    scale = pickle.load(open("cancer_scale.pickle", 'rb'))
    logr = pickle.load(open("cancer_pred.pickle", 'rb'))
    
    
    general_pred = logr.predict_proba(scale.transform(median.reshape(1,-1)))
    print(general_pred)
    
    #Input Data from New Patient
    
    return 0

if __name__ == '__main__':
    
    predict()
    print('hello')