# Step 1: Import necessary libraries  
import pandas as pd  
import matplotlib.pyplot as plt  
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split  
from sklearn.neighbors import KNeighborsClassifier  

# Step 2: Load the Iris dataset  
iris = load_iris()  

# Step 3: Create a DataFrame from the Iris dataset  
df = pd.DataFrame(iris.data, columns=iris.feature_names)  

# Step 4: Add the target variable (species) to the DataFrame  
df['target'] = iris.target  

# Step 5: Explore the DataFrame (optional)  
print(df.head())  # Display the first few rows  
print(df.tail())  # Display the last few rows (useful for checking target values)  

# Step 6: Split the DataFrame into different species for visualization  
df0 = df[df.target == 0]  
df1 = df[df.target == 1]  
df2 = df[df.target == 2]  

# Step 7: Visualize data using scatterplots  
# Scatter plot for Sepal Length vs Sepal Width  
plt.figure(figsize=(12, 5))  

plt.subplot(1, 2, 1)  
plt.xlabel('Sepal Length (cm)')  
plt.ylabel('Sepal Width (cm)')  
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color="green", marker='+', label='Setosa')  
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color="blue", marker='.', label='Versicolor')  
plt.scatter(df2['sepal length (cm)'], df2['sepal width (cm)'], color="red", marker='o', label='Virginica')  
plt.title('Sepal Dimensions')  
plt.legend()  

# Scatter plot for Petal Length vs Petal Width  
plt.subplot(1, 2, 2)  
plt.xlabel('Petal Length (cm)')  
plt.ylabel('Petal Width (cm)')  
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color="green", marker='+', label='Setosa')  
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color="blue", marker='.', label='Versicolor')  
plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'], color="red", marker='o', label='Virginica')  
plt.title('Petal Dimensions')  
plt.legend()  

plt.tight_layout()  
plt.show()  

# Step 8: Prepare data for model training  
X = df.drop(['target'], axis='columns')  # Features  
y = df.target  # Target variable (species)  

# Step 9: Split the dataset into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)  

# Step 10: Create and train the k-NN classifier  
knn = KNeighborsClassifier(n_neighbors=10)  
knn.fit(X_train, y_train)  

# Step 11: Evaluate the model's performance  
accuracy = knn.score(X_test, y_test)  
print(f'Accuracy of the k-NN classifier: {accuracy * 100:.2f}%')
