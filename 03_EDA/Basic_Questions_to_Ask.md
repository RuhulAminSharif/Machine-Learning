## Basic Questions to ask

### How big is the data?

```python
# To know the dimensions of the dataset
df.shape
# To know only the number of rows
df.shape[0]
# To know only the number of cols
df.shape[1]
```

### How does the data look like? 

```python
# To see the first 5rows of the dataset
df.head()
# To see the last 5rows of the dataset
df.tail()
# To see any random 5rows of the dataset
df.sample(5)
```

### What is the data types of the columns?
```python
# To see the column name, non-null count, and data type
df.info()
```

### Are there any missing values?
```python
# To count the number of missing values in each column
df.isnull().sum()
# To see the percentage of missing values in each column
(df.isnull().sum()/df.shape[0])*100
``` 

### How does the data look mathematically?
```python
df.describe()
``` 

### Are there duplicated values?
```python
# To see the number of duplicate rows
df.duplicated().sum()
``` 

### How is the correlation between cols?
```python
# To see the correlation of all column with a specific columns named 'colName'
df.corr()['colName']
``` 