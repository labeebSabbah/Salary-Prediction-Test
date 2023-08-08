import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Data Preprocessing
ds = pd.read_csv('./salary.csv')
print("##################################################")
print(ds)
print("##################################################")
ds.info()
print("##################################################")
print(ds.describe())
print("##################################################")

# Data Cleaning
print(ds.isnull().sum())
print("##################################################")
print("Number of duplicate rows: %d" % ds.duplicated().sum())
print("##################################################")
ds = ds.drop(columns="Unnamed: 0")
print(ds)
print("##################################################")

# Data Visualization
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# Box Plot - YearsExperience
axes[0].boxplot(ds['YearsExperience'])
axes[0].set_title('Box Plot YearsExperience')

# Box Plot - Salary
axes[1].boxplot(ds['Salary'])
axes[1].set_title('Box Plot - Salary')

plt.show()

# Salary Histogram
plt.hist(ds['Salary'], bins=10)
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.title('Histogram - Salary Distribution')
plt.show()

# YearsExperience Histogram
plt.hist(ds['YearsExperience'], bins=10)
plt.xlabel('YearsExperience')
plt.ylabel('Frequency')
plt.title('Histogram - YearsExperience Distribution')
plt.show()

# Splitting the dataset
X = ds[['YearsExperience']]
y = ds['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# The R2 score
print('R2 score: %.2f' % r2_score(y_test, y_pred))
