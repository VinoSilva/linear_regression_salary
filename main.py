import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv('salaries.csv')

# row 1 column 2
# x = df.iloc[0,1]
# x = df.iloc[0:3,-2]
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

# print(x[0:5])
# print(y[0:5])
# print(df.head())

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

plt.scatter(x,y)
plt.show()

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

# print(y_pred)
error = y_pred - y_test;

plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred,color="yellow")

r2 = r2_score(y_test,y_pred)
print(r2)