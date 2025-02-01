import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("solar_efficiency_temp.csv")
df.head()

X = df[['temperature']]
y = df['efficiency']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

LinearRegression()


y_pred = model.predict(X_test)

plt.title("Comparing Test Data with Predicted Data")
plt.xlabel("Temperature")
plt.ylabel("Solar Panel Efficiency")
plt.scatter(X_test, y_test, color="r", label="Actual Data")
plt.plot(X_test, y_pred, color="b", label="Predicted Data")
plt.legend()



mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error = {mse}\nr^2 = {r2}")


import statsmodels.api as sm

X = sm.add_constant(df[['temperature']])  # Adding a constant term for the intercept
Y = df['efficiency']

model = sm.OLS(Y, X).fit()  # Fitting the Ordinary Least Squares (OLS) model


t_statistic = model.tvalues['temperature']
p_value_t = model.pvalues['temperature']

f_statistic = model.fvalue
p_value_f = model.f_pvalue

print(f"F-statistic = {f_statistic}\nt-statistic = {t_statistic}")

if p_value_t < 0.05:
    print("The regression coefficient for temperature is statistically significant.")
else:
    print("The regression coefficient for temperature is NOT statistically significant.")

if p_value_t < 0.05:
    print("The temperature significantly predicts the efficiency of solar panels.")
else:
    print("The temperature does NOT significantly predicts the efficiency of solar panels.")

