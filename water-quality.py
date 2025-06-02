# Importing necessary libraries
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm_notebook
import plotly.figure_factory as ff

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
%matplotlib inline

# Loading the dataset
data = pd.read_csv('/kaggle/input/water-potability/water_potability.csv')

# Basic information
print('There are {} data points and {} features in the data'.format(data.shape[0], data.shape[1]))

# First few rows
print(data.head())

# Data description
print(data.describe())

# Data info
print(data.info())

# Visualizing null values
sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Counting nulls
for i in data.columns:
    if data[i].isnull().sum() > 0:
        print("There are {} null values in {} column".format(data[i].isnull().sum(), i))

# ------------------------------
# Handling Null Values
# ------------------------------

# Function for random value imputation
def impute_nan(df, variable):
    df[variable + "_random"] = df[variable]
    random_sample = df[variable].dropna().sample(df[variable].isnull().sum(), random_state=0)
    random_sample.index = df[df[variable].isnull()].index
    df.loc[df[variable].isnull(), variable + '_random'] = random_sample

# Impute ph
data['ph_mean'] = data['ph'].fillna(data['ph'].mean())
impute_nan(data, "ph")

# Plot KDE for ph
fig = plt.figure()
ax = fig.add_subplot(111)
data['ph'].plot(kind='kde', ax=ax)
data.ph_mean.plot(kind='kde', ax=ax, color='red')
data.ph_random.plot(kind='kde', ax=ax, color='green')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
plt.title("KDE Plot of ph: Original vs Mean Imputed vs Random Imputed")
plt.show()

# Impute Sulfate
impute_nan(data, "Sulfate")
fig = plt.figure()
ax = fig.add_subplot(111)
data['Sulfate'].plot(kind='kde', ax=ax)
data.Sulfate_random.plot(kind='kde', ax=ax, color='green')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
plt.title("KDE Plot of Sulfate: Original vs Random Imputed")
plt.show()

# Impute Trihalomethanes
impute_nan(data, "Trihalomethanes")
fig = plt.figure()
ax = fig.add_subplot(111)
data['Trihalomethanes'].plot(kind='kde', ax=ax)
data.Trihalomethanes_random.plot(kind='kde', ax=ax, color='green')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')
plt.title("KDE Plot of Trihalomethanes: Original vs Random Imputed")
plt.show()

# Drop old columns and mean imputed ph
data = data.drop(['ph', 'Sulfate', 'Trihalomethanes', 'ph_mean'], axis=1)

# Check if any missing values remain
print(data.isnull().sum())

# ------------------------------
# Correlation Matrix
# ------------------------------
plt.figure(figsize=(20, 17))
matrix = np.triu(data.corr())
sns.heatmap(data.corr(), annot=True, linewidth=.8, mask=matrix, cmap="rocket", cbar=False)
plt.title("Correlation Heatmap")
plt.show()

# ------------------------------
# Pairplot
# ------------------------------
sns.pairplot(data, hue="Potability", palette="husl")
plt.suptitle("Pairplot of Features vs Potability", y=1.02)
plt.show()

# ------------------------------
# Distribution Plot of all features
# ------------------------------
def distributionPlot(data):
    fig = plt.figure(figsize=(20, 20))
    for i in tqdm_notebook(range(0, len(data.columns))):
        fig.add_subplot(np.ceil(len(data.columns)/3), 3, i+1)
        sns.histplot(data.iloc[:, i], kde=True)
        plt.title(data.columns[i])
    plt.tight_layout()
    plt.show()

distributionPlot(data)
