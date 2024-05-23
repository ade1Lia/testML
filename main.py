import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#столбец в тестовых данных
test_data['SalePrice'] = 0

#обучающие + тестовые данные
all_data = pd.concat([train_data, test_data], axis=0)

all_data = pd.get_dummies(all_data)

#обучающие и тестовые
train_data = all_data.iloc[:train_data.shape[0], :]
test_data = all_data.iloc[train_data.shape[0]:, :]

X = train_data.drop(columns=['SalePrice'])
y = train_data['SalePrice']

#разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

model = LinearRegression()
model.fit(X_train_imputed, y_train)

#производительность модели на тестовых данных
y_pred = model.predict(X_test_imputed)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

test_data_imputed = imputer.transform(test_data.drop(columns=['SalePrice']))

#прогнозы
predictions = model.predict(test_data_imputed)

#сохранение
submission_df = pd.DataFrame({'Id': test_data.index + 1461, 'SalePrice': predictions})

submission_df.to_csv('submission.csv', index=False)