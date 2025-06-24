import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
filename = input("Enter the filename :")
fhand = open(filename)
df = pd.read_csv(fhand)
print("Data shape:", df.shape)
print("First few rows:")
print(df.head())
feature_names = df.columns[:-1].tolist()
target_name = df.columns[-1]
X = df[feature_names]
y = df[target_name]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")
def plot_feature(X, y, feature_name, model, feature_names):
    plt.figure(figsize=(10, 6), facecolor='white')
    feature_idx = feature_names.index(feature_name)
    X_feat = X[feature_name]
    plt.scatter(X_feat, y, color='blue', marker='o', s=100, label='Actual')
    X_line = np.linspace(X_feat.min(), X_feat.max(), 100)
    X_pred = pd.DataFrame({name: X_line if name == feature_name 
                          else X[name].mean() * np.ones_like(X_line) 
                          for name in feature_names})
    y_line = model.predict(X_pred)
    plt.plot(X_line, y_line, color='red', linewidth=2, label='Regression Line')
    plt.grid(True, linestyle='-', alpha=0.2)
    plt.title(f'Salary Prediction by {feature_name}', pad=15, fontsize=12)
    plt.xlabel(feature_name, fontsize=10)
    plt.ylabel('Salary (₹)', fontsize=10)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
for feature in feature_names:
    plot_feature(X, y, feature, model, feature_names)
print(f"\nMean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.4f}")
example_input = pd.DataFrame([[5, 16]], columns=feature_names)
predicted_salary = model.predict(example_input)[0]
print(f"\nPredicted salary for {feature_names[0]}=5 and {feature_names[1]}=16: ₹{predicted_salary:.2f}")