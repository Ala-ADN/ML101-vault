#### Count NULL
```python
dataset.isnull().sum()
```

![[651px-Relationship_between_mean_and_median_under_different_skewness 1.png]]
### filling missing values with Mean value:
```python
#mean
dataset['NAME OF COLUMN'].fillna(dataset['NAME OF COLUMN'].mean(),inplace=True)
#mode
dataset['NAME OF COLUMN'].fillna(dataset['NAME OF COLUMN'].mode(),inplace=True)
```

### Missing Header
```python
data = pd.read_csv(path, names=columns)
```