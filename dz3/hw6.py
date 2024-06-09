import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Подключение к SQLite базе данных (создание)
conn = sqlite3.connect('mydatabase.db')
cursor = conn.cursor()
# Создание таблицы sales
create_table_query = '''
CREATE TABLE IF NOT EXISTS sales (
 order_id INTEGER PRIMARY KEY,
    Date DATE,
    Product Category TEXT,
    Units Sold INTEGER,
    Unit Price FLOAT,
    Total Revenue FLOAT,
    Region TEXT
);
'''
cursor.execute(create_table_query)
conn.commit()

#Импорт данных
df = pd.read_csv('C:/Users/kris.zakudovskaya/PycharmProjects/pythonProjecthw6/Online Sales Data.csv')
print (df)

# Вставка данных в таблицу базы данных
df.to_sql('sales', conn, if_exists='replace', index=False)

#SQL-запросы для извлечения данных из таблицы базы данных sales

cursor.execute("SELECT * FROM sales WHERE Date > '2024-04-20';")
result = cursor.fetchall()
for row in result:
    print(row)

cursor.execute("SELECT * FROM sales WHERE Region = 'Europe';")
result = cursor.fetchall()
for row in result:
    print(row)

cursor.execute("SELECT SUM(\"Units Sold\") AS Total_Units_Sold FROM sales;")
result = cursor.fetchall()
for row in result:
    print(row)

cursor.execute("SELECT AVG(\"Unit Price\") AS Avg_Unit_Price FROM sales;")
result = cursor.fetchall()
for row in result:
    print(row)

cursor.execute("SELECT COUNT(*) AS Total_Orders FROM sales;")
result = cursor.fetchall()
for row in result:
    print(row)

#Визуализация данных

conn = sqlite3.connect('mydatabase.db')
query = "SELECT * FROM sales;"
df = pd.read_sql_query(query, conn)
plt.figure(figsize=(10, 6))
df.groupby('Region')['Units Sold'].sum().plot(kind='bar', color='skyblue')
plt.title('Total Units Sold by Region')
plt.xlabel('Region')
plt.ylabel('Total Units Sold')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()












