import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("study_hours.csv")
df.head()
X = df[['StudyHours']]
y = df['ExamScore']

plt.scatter(X, y, color="b", marker="*")
plt.xlabel("Study Hours")
plt.ylabel("Exam Scores")

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)

model = LinearRegression()

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

plt.title("Comparing Actual and Predicted Linear Regression Data")
plt.xlabel("Study Hours")
plt.ylabel("Exam Scores")
plt.scatter(X_test, y_test, color="b", label="Actual Data")
plt.plot(X_test, y_pred, color="r", label="Predicted Data")
plt.legend()

r2 = r2_score(y_pred, y_test)
mse = mean_squared_error(y_pred, y_test)
print(f"Mean Squared Error = {mse}\nr^2 = {r2}")
