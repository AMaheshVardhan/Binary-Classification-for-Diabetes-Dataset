import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
sns.set_style('whitegrid')

import warnings
warnings.filterwarnings('ignore')

from google.colab import files
uploaded = files.upload()

diabetes_data = pd.read_csv('preprocessed_diabetes_data.csv')

# View top 10 rows of the Diabetes dataset
diabetes_data.head(10)

diabetes_data.shape
diabetes_data.describe().T

fig,axes = plt.subplots(nrows=1,ncols=2,figsize = (8,6))

plot00=sns.distplot(diabetes_data['Pregnancies'],ax=axes[0],color='b')
axes[0].set_title('Distribution of Pregnancy',fontdict={'fontsize':8})
axes[0].set_xlabel('No of Pregnancies')
axes[0].set_ylabel('Frequency')
plt.tight_layout()


plot01=sns.boxplot(data=diabetes_data['Pregnancies'], ax=axes[1],orient = 'v', color='r')
plt.tight_layout()

diabetes_data.corr()

plt.figure(figsize=(12,6))
sns.countplot(x='Outcome',data=diabetes_data, palette='bright')
plt.title("Output class distribution")

print(diabetes_data['Outcome'].value_counts())
sns.pairplot(diabetes_data, hue="Outcome")

plt.figure(figsize=(12,8))
sns.boxplot(x='Outcome', y='BMI',data=diabetes_data, hue='Outcome')

plt.figure(figsize=(12,8))
sns.boxplot(x='Outcome', y='DiabetesPedigreeFunction',data=diabetes_data, hue="Outcome")
plt.figure(figsize=(12,8))
sns.boxplot(x='Outcome', y='Pregnancies',data=diabetes_data, hue="Outcome")
normalBMIData = diabetes_data[(diabetes_data['BMI'] >= 18.5) & (diabetes_data['BMI'] <= 25)]
normalBMIData['Outcome'].value_counts()

notNormalBMIData = diabetes_data[(diabetes_data['BMI'] < 18.5) | (diabetes_data['BMI'] > 25)]
notNormalBMIData['Outcome'].value_counts()

plt.figure(figsize=(12,8))
sns.boxplot(x='Outcome', y='BMI',data=notNormalBMIData)
plt.figure(figsize=(12,8))
sns.boxplot(x='Outcome', y='Age',data=diabetes_data, hue="Outcome")
unchanged_data = diabetes_data.drop('Outcome',axis=1)

unchanged_data

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

def plot_KNN_error_rate(xdata,ydata):
  error_rate = []
  test_scores = []
  train_scores = []
  X_train, X_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.3, random_state=42) ## Write your code here (expected lines ~ 1)

  for i in range(1,40):
      ## [REQUIRED] Complete the code in the next three lines
      knn = KNeighborsClassifier(n_neighbors=i)  ## Write your code here. Initialize the KNN classifier with 'i' neighbours (expected lines ~ 1)
      ## Write your code here. Fit the KNN model on the training set (expected lines ~ 1)
      knn.fit(X_train, y_train)
      pred_i = knn.predict(X_test)## Write your code here. Make predictions on the test set using KNN (expected lines ~ 1)

      error_rate.append(np.mean(pred_i != y_test))
      train_scores.append(knn.score(X_train,y_train))
      test_scores.append(knn.score(X_test,y_test))

  plt.figure(figsize=(12,8))
  plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
          markerfacecolor='red', markersize=10)
  plt.title('Error Rate vs. K Value')
  plt.xlabel('K')
  plt.ylabel('Error Rate')
  print()
  ## score that comes from testing on the same datapoints that were used for training
  max_train_score = max(train_scores)
  train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
  print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))
  print()
  ## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely
  max_test_score = max(test_scores)
  test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
  print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))

  return test_scores

#@title Answer to Task-6. Refer to and run this only if you are unable to complete the task in the previous cell.
def plot_KNN_error_rate(xdata,ydata):
  error_rate = []
  test_scores = []
  train_scores = []

  X_train, X_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.3, random_state=101)

  for i in range(1,40):
      knn = KNeighborsClassifier(n_neighbors=i)
      knn.fit(X_train, y_train)
      pred_i = knn.predict(X_test)

      error_rate.append(np.mean(pred_i != y_test))
      train_scores.append(knn.score(X_train,y_train))
      test_scores.append(knn.score(X_test,y_test))

  plt.figure(figsize=(12,8))
  plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
          markerfacecolor='red', markersize=10)
  plt.title('Error Rate vs. K Value')
  plt.xlabel('K')
  plt.ylabel('Error Rate')
  print()
  ## score that comes from testing on the same datapoints that were used for training
  max_train_score = max(train_scores)
  train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
  print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))
  print()
  ## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely
  max_test_score = max(test_scores)
  test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
  print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))

  return test_scores

unchanged_test_scores = plot_KNN_error_rate(unchanged_data,diabetes_data['Outcome'])

"""## Standardize the Variables
Standardization (also called z-score normalization) is the process of putting different variables on the same scale. Standardization transforms your data such that the resulting distribution has a mean of 0 and a standard deviation of 1.

$$ Z = {X - \mu \over \sigma}$$

"""

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(diabetes_data.drop('Outcome',axis=1))

scaled_data = scaler.transform(diabetes_data.drop('Outcome',axis=1))

df_feat = pd.DataFrame(scaled_data,columns=diabetes_data.columns[:-1])
df_feat.head()

scaled_test_scores = plot_KNN_error_rate(scaled_data,diabetes_data['Outcome'])

"""## Comparing Accuracy before and after Standardization"""

plt.figure(figsize=(20,8))
plt.title('Accuracy vs. K Value')
sns.lineplot(unchanged_test_scores,marker='o',label='Unscaled data test score')
sns.lineplot(scaled_test_scores,marker='o',label='Scaled data test Score')

from sklearn.preprocessing import MinMaxScaler
minMaxScaler = MinMaxScaler()
minMaxScaler.fit(diabetes_data.drop('Outcome', axis =1))
minMaxedScaledData = minMaxScaler.transform(diabetes_data.drop('Outcome', axis=1))

df_feat = pd.DataFrame(minMaxedScaledData, columns=diabetes_data.columns[:-1])
df_feat.head()

new_test_scores = plot_KNN_error_rate(minMaxedScaledData, diabetes_data['Outcome'])

len(new_test_scores)

from sklearn.model_selection import KFold
from statistics import mean

def plot_KNN_error_rate_new(xdata,ydata):
  cv_test = []
  cv_train = []
  cv_error = []
  kf = KFold(n_splits=10)
  count = 0
  for i in range(1,40):
      ## [REQUIRED] Complete the code in the next three lines
      knn = KNeighborsClassifier(n_neighbors=i)
      test_scores = []
      train_scores = []
      error_rate = []
      for train_index, test_index in kf.split(xdata, ydata):
        x_train_fold, x_test_fold = xdata.iloc[train_index], xdata.iloc[test_index]
        y_train_fold, y_test_fold = ydata.iloc[train_index], ydata.iloc[test_index]
        knn.fit(x_train_fold, y_train_fold)
        pred_i = knn.predict(x_test_fold)## Write your code here. Make predictions on the test set using KNN (expected lines ~ 1)
        error_rate.append(np.mean(pred_i != y_test_fold))
        train_scores.append(knn.score(x_train_fold,y_train_fold))
        test_scores.append(knn.score(x_test_fold,y_test_fold))
      count += 1
      cv_test.append(mean(test_scores))
      cv_train.append(mean(train_scores))
      cv_error.append(mean(error_rate))
  plt.figure(figsize=(12,8))
  plt.plot(range(1,40),cv_error,color='blue', linestyle='dashed', marker='o',
          markerfacecolor='red', markersize=10)
  plt.title('Error Rate vs. K Value')
  plt.xlabel('K')
  plt.ylabel('Error Rate')
  print()
  ## score that comes from testing on the same datapoints that were used for training
  max_train_score = max(cv_train)
  train_scores_ind = [i for i, v in enumerate(cv_train) if v == max_train_score]
  print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))
  print()
  ## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely
  max_test_score = max(cv_test)
  test_scores_ind = [i for i, v in enumerate(cv_test) if v == max_test_score]
  print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))

  return cv_test

unchanged_test_scores = plot_KNN_error_rate_new(unchanged_data,diabetes_data['Outcome'])
