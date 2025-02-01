import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("sales.csv")
df.head()
df = pd.read_csv("sales.csv")  # Replace with correct file

# Define features (X) and target (y)
X = df[['AdvertisingExpenditure', 'Competition', 'StoreLocation']]
y = df['SalesRevenue']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Scatter plots for each predictor vs SalesRevenue
predictors = ["AdvertisingExpenditure", "Competition", "StoreLocation"]
for predictor in predictors:
    plt.figure(figsize=(8, 6))
    plt.title(f"{predictor} vs Sales Revenue")
    plt.xlabel(predictor)
    plt.ylabel("Sales Revenue")
    plt.scatter(X_test[predictor], y_test, color="b", label="Actual Data")
    plt.scatter(X_test[predictor], y_pred, color="r", label="Predicted Data")
    plt.legend()
    plt.show()

import statsmodels.api as sm

X_with_const = sm.add_constant(X)

model = sm.OLS(Y, X_with_const).fit()

#for f-test
for predictor in predictors:
    t_statistic = model.tvalues[predictor]
    p_value_t = model.pvalues[predictor]

    print(f"t-statistic for {predictor} = {t_statistic}")

    if p_value_t < 0.05:
        print(f"{predictor} is a statistically significant predictor of SalesRevenue.")
    else:
        print(f"{predictor} is NOT a statistically significant predictor of SalesRevenue.")


#for t-test
X_with_const = sm.add_constant(X[predictor])
model = sm.OLS(Y, X_with_const).fit()

f_statistic = model.fvalue
p_value_f = model.f_pvalue

print(f"F-statistic for {predictor} = {f_statistic}")

if p_value_f < 0.05:
    print(f"{predictor} is a statistically significant predictor of SalesRevenue.")
else:
    print(f"{predictor} is NOT a statistically significant predictor of SalesRevenue.")

