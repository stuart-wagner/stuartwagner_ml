from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()

print(california.DESCR)

print(california.data.shape)
print(california.feature_names)

import pandas as pd

pd.set_option('display.precision', 4)
pd.set_option('display.max_columns', 8)
pd.set_option('display.width', None)

california_df = pd.DataFrame(california.data, columns=california.feature_names)

california_df['MedHouseVal'] = pd.Series(california.target)
print(california_df.head())

print(california_df.describe())

sample_df = california_df.sample(frac=0.1, random_state=17)

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(font_scale=2)
sns.set_style("whitegrid")

for feature in california.feature_names:
    plt.figure(figsize=(8, 4.5))
    sns.scatterplot(
        data=sample_df, 
        x=feature, 
        y='MedHouseVal', 
        hue='MedHouseVal',
        palette='cool',
        legend=False
    )
    plt.title(f'MedHouseVal vs {feature}')
    plt.xlabel(feature)
    plt.ylabel('MedHouseVal')
    plt.show()







from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(california.data, 
                                                    california.target, 
                                                    random_state=11)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train, y_train)

for i, name in enumerate(california.feature_names):
    print(f'{name}: {lr.coef_[i]}')

predicted = lr.predict(x_test)
print(predicted[:5])
expected = y_test
print(expected[:5])

df = pd.DataFrame()

df["Expected"] = pd.Series(expected)
df["Predicted"] = pd.Series(predicted)

import matplotlib.pyplot as plt2

figure = plt2.figure(figsize=(9, 9))

axes = sns.scatterplot(data=df, x="Expected", y="Predicted", 
                       hue="Predicted", legend=False)

start = min(expected.min(), predicted.min())
end = max(expected.max(), predicted.max())

axes.set_xlim(start, end)
axes.set_ylim(start, end)

line = plt2.plot([start, end], [start, end], 'k--')
plt2.show()








