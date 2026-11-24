import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


data = pd.read_csv("knn_k_customers.csv")

print("First five customers:")
print(data.head())

print("\nDataset shape:", data.shape)

print("\nMissing values:")
print(data.isna().sum())

print("\nDuplicate rows:", data.duplicated().sum())

print("\nPurchase counts:")
print(data["Purchased"].value_counts())


features = [
    "Age",
    "Monthly_Spending",
    "Website_Visits",
    "Time_on_Website",
    "Previous_Purchases"
]

target = "Purchased"

X = data[features]
y = data[target].map({"No": 0, "Yes": 1})

print("\nFeatures:", features)
print("Target:", target)
print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nClass distribution:")
print(pd.DataFrame({
    "Train": y_train.value_counts().sort_index(),
    "Test": y_test.value_counts().sort_index()
}))


k_values = [1, 3, 5, 7, 9, 11]
results = []

for k in k_values:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=k,
            metric="euclidean"
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions) * 100

    results.append({
        "K Value": k,
        "Test Accuracy (%)": round(accuracy, 2)
    })

accuracy_table = pd.DataFrame(results)

print("\nK value comparison:")
print(accuracy_table.to_string(index=False))

best_row = accuracy_table.loc[
    accuracy_table["Test Accuracy (%)"].idxmax()
]

best_k = int(best_row["K Value"])

print("\nBest K:", best_k)
print("Best test accuracy:", best_row["Test Accuracy (%)"], "%")


plt.figure(figsize=(8, 5))
plt.plot(
    accuracy_table["K Value"],
    accuracy_table["Test Accuracy (%)"],
    marker="o"
)

plt.title("K Value vs Test Accuracy")
plt.xlabel("K Value")
plt.ylabel("Test Accuracy (%)")
plt.xticks(k_values)
plt.grid(True)
plt.tight_layout()
plt.show()

final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(
        n_neighbors=best_k,
        metric="euclidean"
    ))
])

final_model.fit(X_train, y_train)

new_customers = pd.DataFrame({
    "Age": [22, 34, 45, 29, 58, 41, 25, 63, 37, 50],
    "Monthly_Spending": [1500, 4000, 8500, 2500, 9200, 6500, 1800, 7800, 5000, 9800],
    "Website_Visits": [2, 10, 18, 5, 19, 12, 3, 15, 8, 20],
    "Time_on_Website": [8, 25, 50, 12, 55, 35, 10, 45, 20, 60],
    "Previous_Purchases": [0, 3, 7, 1, 9, 4, 0, 6, 2, 10]
})

new_predictions = final_model.predict(new_customers)

new_customers["Predicted"] = new_predictions
new_customers["Predicted_Result"] = [
    "Yes" if prediction == 1 else "No"
    for prediction in new_predictions
]
print("\nPredictions for new customers:")
print(new_customers.to_string(index=False))