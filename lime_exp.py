import pandas as pd 
import joblib
from sklearn.calibration import LabelEncoder
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

#mlp = joblib.load('mlp_model.plk')
#scaler  = joblib.load('scaler.plk')

data =  pd.read_csv('drug_consumption.data', sep=",")
data.columns = ["ID", "age", "gender", "education", "country", "ethnicity", "nscore", "escore", 
                "oscore", "ascore", "cscore", "impulsive", "ss", "alcohol", "amphet", "amyl", 
                "benzos", "caff", "cannabis", "choc", "coke", "crack", "ecstasy", "heroin", 
                "ketamine", "legalh", "lsd", "meth", "mushrooms", "nicotine", "semer", "vsa"]
#print(data.head())

data = data[data.semer == "CL0"]

# Extract features
features = data[["age", "gender", "education", "country", "ethnicity", "nscore", "escore", 
                "oscore", "ascore", "cscore", "impulsive", "ss"]].copy()
#print(features.head())

categorical = data.select_dtypes(include=['object']).columns.tolist()
for i in categorical:
    data[i] = LabelEncoder().fit_transform(data[i])

#standardize 
X = scaler.transform(features)

#LIME explainer 
explainer = LimeTabularExplainer(X, feature_names = features.columns, class_names=['Not used', 'Used'], discretize_continuous = True)
instance = X[0]
exp = explainer.explain_instance(instance, mlp.predict_proba, num_features=12)
#exp.show_in_notebook(show_all =False)

expl_detail = exp.as_list()
for feature, weight in expl_detail:
    print(f"{feature}: {weight}")

# Extract feature names and weights
features = [x[0] for x in expl_detail]
weights = [x[1] for x in expl_detail]

plt.figure(figsize=(10, 6))
plt.barh(features, weights)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()