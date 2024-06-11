import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
    print(data[i].value_counts())
# This shows that nicotine and cannabis have the most even spread of numbers over the classes, so these
# might be the most interesting to use for our classification

# Transform categorical columns to numerical
label_encoder = LabelEncoder()
for column in categorical:
    data[column] = label_encoder.fit_transform(data[column])

# Turn the 7 class classification into binary classification
def binary(row):
    if row['cannabis'] < 3:
        return 0
    return 1
data['cannabis_binary'] = data.apply(binary, axis=1)

# Extract labels
labels = data[["cannabis_binary"]].copy()

# Split in training and test data
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create model
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='logistic', max_iter=1000, random_state=42)

# Train model
mlp.fit(X_train, y_train.values.ravel())

# Predict output for the test data
y_pred = mlp.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
