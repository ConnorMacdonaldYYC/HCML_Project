import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

mlp = joblib.load('mlp_model.pkl')
scaler = joblib.load('scaler.pkl')

#if not isinstance(scaler, StandardScaler):
#    raise TypeError("StandardScaler problem")

data = pd.read_csv('drug_consumption.data', sep=",")
data.columns = ["ID", "age", "gender", "education", "country", "ethnicity", "nscore", "escore",
                "oscore", "ascore", "cscore", "impulsive", "ss", "alcohol", "amphet", "amyl",
                "benzos", "caff", "cannabis", "choc", "coke", "crack", "ecstasy", "heroin",
                "ketamine", "legalh", "lsd", "meth", "mushrooms", "nicotine", "semer", "vsa"]

data = data[data.semer == "CL0"]

features = data[["age", "gender", "education", "country", "ethnicity", "nscore", "escore",
                "oscore", "ascore", "cscore", "impulsive", "ss"]].copy()
categorical = ["gender", "education", "country", "ethnicity"]

label_encoder = LabelEncoder()
for column in categorical:
    data[column] = label_encoder.fit_transform(data[column])

#Standardize 
#X = scaler.transform(features)
X = features.values

#Lime Explainer
explainer = LimeTabularExplainer(X, feature_names=features.columns, class_names=['Not used', 'Used'], discretize_continuous=True)

#explaining an instance
def explain_instance(instance):
    exp = explainer.explain_instance(instance, mlp.predict_proba, num_features=12)
    return exp

#explaining multiple instances
def explain_multiple_instances(instances, num_samples=5):
    for i in range(min(num_samples, len(instances))):
        print(f"Explanation for instance {i}:")
        exp = explain_instance(instances[i])
        expl_detail = exp.as_list()
        for feature, weight in expl_detail:
            print(f"{feature}: {weight}")
        features = [x[0] for x in expl_detail]
        weights = [x[1] for x in expl_detail]
        plt.figure(figsize=(10, 6))
        plt.barh(features, weights)
        plt.xlabel('importance')
        plt.title(f'feature importance for instance {i}')
        plt.show()

#explaining the first instance
instance = X[0]
exp = explain_instance(instance)

#feature importance for the first instance
expl_detail = exp.as_list()
for feature, weight in expl_detail:
    print(f"{feature}: {weight}")
features = [x[0] for x in expl_detail]
weights = [x[1] for x in expl_detail]

plt.figure(figsize=(10, 6))
plt.barh(features, weights)
plt.xlabel('Importance')
plt.title('feature importance for first instance')
plt.show()

#explain_multiple_instances(X, num_samples=5)