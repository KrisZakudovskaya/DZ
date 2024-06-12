#Описание колонок
# id
# diagnosis - Диагноз (M - злокачественный, B - доброкачественный)
# Характеристики клеток
# radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave points_mean, symmetry_mean, fractal_dimension_mean
# Стандартные ошибки различных характеристик клеток
# radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave points_se, symmetry_se, fractal_dimension_se
# Наибольшие значения различных характеристик клеток
# radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave points_worst, symmetry_worst, fractal_dimension_worst

# Импорт необходимых библиотек
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import IsolationForest

# Импорт данных
data = pd.read_csv('C:/Users/kris.zakudovskaya/PycharmProjects/pythonProject10/data.csv')
df = pd.read_csv('data.csv')

# Просмотр первых нескольких строк данных
print(df.head())

# Описательная статистика
print(df.describe())

# Гистограммы для числовых переменных
df.hist(figsize=(10,10))
plt.show()

# Распределение диагноза
df['diagnosis'].value_counts().plot(kind='bar')
plt.title('Распределение диагноза')
plt.xlabel('Диагноз')
plt.ylabel('Количество')
plt.show()

# Выводы
# Датасет содержит 569 наблюдений и 33 переменных. Подавляющее большинство диагнозов относится к доброкачественным
# Некоторые из числовых переменных имеют широкий диапазон значений (radius_mean). Есть значительная вариация характеристик клеток в датасете
# Столбец Unnamed: 32 полностью состоит из пропущенных значений, можно удалить

# Предварительная обработка данных

# Удаление столбца Unnamed: 32
data = data.drop(['Unnamed: 32'], axis=1)

# Отбор числовых столбцов
numeric_columns = data.select_dtypes(include=['number']).columns

# Заполнение пропущенных значений средним для числовых столбцов
imputer_numeric = SimpleImputer(strategy='mean')
data[numeric_columns] = imputer_numeric.fit_transform(data[numeric_columns])

# Замена категориальных признаков на числовые (M - 1, B - 0)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
print(data.head())

# Визуализация корреляционной матрицы
corr_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Распределение категориального признака
print(data['diagnosis'].value_counts())

# Разделение данных на признаки (X) и целевую переменную (y)
X = data.drop(['diagnosis'], axis=1)
y = data['diagnosis']

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
logreg = LogisticRegression()

X_train_imputed = imputer.fit_transform(X_train)
X_train_scaled = scaler.fit_transform(X_train_imputed)
logreg.fit(X_train_scaled, y_train)

# Предсказание на тестовом наборе и оценка качества
X_test_imputed = imputer.transform(X_test)
X_test_scaled = scaler.transform(X_test_imputed)
y_pred = logreg.predict(X_test_scaled)

# Оценка качества модели
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Вывод результатов
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC:", roc_auc)

# Классификатор градиентного бустинга
# Обучение модели
model_gb = GradientBoostingClassifier()
model_gb.fit(X_train, y_train)
# Прогнозирование на тестовом наборе
y_pred_gb = model_gb.predict(X_test)
# Оценка точности модели
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print("Accuracy of Gradient Boosting:", accuracy_gb)

# Классификатор K Neighbors
# Обучение модели
model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train)
# Прогнозирование на тестовом наборе
y_pred_knn = model_knn.predict(X_test)
# Оценка точности модели
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("Accuracy of K Nearest Neighbors:", accuracy_knn)

# Классификатор дерева решений
# Создание и обучение модели
decision_tree_clf = DecisionTreeClassifier()
decision_tree_clf.fit(X_train, y_train)
#Прогнозирование на тестовом наборе
y_pred = decision_tree_clf.predict(X_test)
# Оценка качества модели с помощью метрик
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of Decision Tree Classifier: {accuracy}')
print(classification_report(y_test, y_pred))

# Регрессия Лассо
# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(data.drop('diagnosis', axis=1), data['diagnosis'], test_size=0.2, random_state=42)
# Создание и обучение модели
model_lasso = Lasso(alpha=0.1)
model_lasso.fit(X_train, y_train)
# Прогнозирование на тестовом наборе
y_pred_lasso = model_lasso.predict(X_test)
# Оценка модели
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print("Mean Squared Error of Lasso Regression:", mse_lasso)

# Регрессор случайного леса
# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Создание и обучение модели
random_forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)  # n_estimators - количество деревьев
random_forest_reg.fit(X_train, y_train)
# Прогнозирование на тестовом наборе
y_pred = random_forest_reg.predict(X_test)
# Оценка качества модели с помощью среднеквадратичной ошибки
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error of Random Forest: {mse}')

# Поиск аномалий

# Обучение модели Isolation Forest
clf = IsolationForest(contamination=0.1, random_state=42)  # contamination - ожидаемая доля аномалий в данных
clf.fit(X)

# Предсказание аномалий
y_pred = clf.predict(X)

# Отображение результатов
plt.title("Isolation Forest для обнаружения аномалий")
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c='white', s=20, edgecolor='k')

# Отображение нормальных точек
normal_points = X[y_pred == 1]
plt.scatter(normal_points.iloc[:, 0], normal_points.iloc[:, 1], c='blue', s=20, edgecolor='k', label="Нормальные точки")

# Отображение аномалий
anomalies = X[y_pred == -1]
plt.scatter(anomalies.iloc[:, 0], anomalies.iloc[:, 1], c='red', s=20, edgecolor='k', label="Аномалии")

plt.legend()
plt.show()

# Визуализация ошибок прогнозирования
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Фактические значения', color='blue')
plt.plot(y_pred_gb, label='Прогнозные значения', color='red')
plt.title('Прогнозирование диагноза')
plt.xlabel('Наблюдения')
plt.ylabel('Диагноз')
plt.legend()
plt.show()

# Визуализация метрик качества обученной модели
models = ['Gradient Boosting', 'K Nearest Neighbors', 'Decision Tree', 'Lasso Regression', 'Random Forest']
accuracies = [accuracy_gb, accuracy_knn, accuracy, 1 - mse_lasso, 1 - mse]

plt.figure(figsize=(8, 6))
plt.bar(models, accuracies, color='blue')
plt.title('Accuracy of Different Models')
plt.ylabel('Accuracy')
plt.xlabel('Models')
plt.show()

# Визуализация важности признаков для модели случайного леса
plt.figure(figsize=(10, 6))
plt.barh(X.columns, random_forest_reg.feature_importances_, color='blue')
plt.title('Feature Importances for Random Forest')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Визуализация важности признаков для модели градиентного бустинга
plt.figure(figsize=(10, 6))
plt.barh(X.columns, model_gb.feature_importances_, color='blue')
plt.title('Feature Importances for Gradient Boosting')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Визуализация важности признаков для модели линейной регрессии Лассо
plt.figure(figsize=(10, 6))
plt.barh(X.columns, np.abs(model_lasso.coef_), color='blue')
plt.title('Feature Importances for Lasso Regression')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()