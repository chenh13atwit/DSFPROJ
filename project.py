#code pasted from colab 

from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Food_Delivery_Times.csv")
df.head()
df.columns

#1
plt.figure()
sns.scatterplot(
    data=df,
    x="Distance_km",
    y="Delivery_Time_min",
    hue="Traffic_Level"
)
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (min)")
plt.title("Distance vs Delivery Time by Traffic Level")
plt.show()

#2
plt.figure()

sns.kdeplot(
    data=df,
    x="Delivery_Time_min",
    hue="Traffic_Level",
    fill=True
)
plt.xlabel("Delivery Time (min)")
plt.ylabel("Density")
plt.title("Distribution of Delivery Time by Traffic Level")

plt.show()

#Regression
from sklearn.linear_model import LinearRegression

df["Traffic_Code"] = df["Traffic_Level"].astype("category").cat.codes
X = df[["Distance_km", "Traffic_Code"]]
y = df["Delivery_Time_min"]
model = LinearRegression()
model.fit(X, y)
print("Distance coefficient:", model.coef_[0])
print("Traffic coefficient:", model.coef_[1])
print("Intercept:", model.intercept_)
