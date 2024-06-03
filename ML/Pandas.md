
1. Loading the dataset to a pandas DataFrame
```python
dataset = pd.read_csv('/content/train.csv')
```
2. Shape of dataframe
```python
dataset.shape
```
3. View first rows
```python
dataset.head()
```
4. counting the number of missing values in the dataset
```python
dataset.isnull().sum()
```
5. fill null values
```python
dataset = dataset.fillna("[filler]")
```
6. check column types
```python
dataset.dtypes
```
7. unique values of each column
```python
dataset.unique()
```
8. info
```python
dataset.info()
```
9. describe (count, min, max, mean)
```python
dataset.describe()
```

[kaggle cheatsheet](https://www.kaggle.com/code/grroverpr/pandas-cheatsheet)

![[Pandas_Cheat_Sheet.pdf]]

