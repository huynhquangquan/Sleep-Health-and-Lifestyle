import joblib
import pandas as pd

model = joblib.load("rf_model.pkl")
onehot_encoder = joblib.load("onehot_encoder.pkl")
label_encoder = joblib.load("label_encoder.pkl")
numeric_features = [
    "Age", "Physical Activity Level", "Sleep Duration",
    "Systolic", "Diastolic", "Heart Rate", "Daily Steps"
]
category_features = ['Gender', 'Occupation', 'BMI Category']

# Load test data
test = pd.read_csv("processed_test.csv")
# print(test.columns)

X_test = test.drop('Sleep Disorder encoded', axis=1)
y_test = test['Sleep Disorder encoded']

pred = model.predict(X_test)
predicted_label = label_encoder.inverse_transform(pred)
true_label = label_encoder.inverse_transform(y_test)

results_df = pd.DataFrame({
    'predicted': predicted_label,
    'actual': true_label
})

final_df = pd.concat([X_test, results_df], axis=1)

final_df.to_csv("results.csv", index=False)