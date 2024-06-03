#### Split testing and training data before standardizing
```python
ds_train, ds_test, target_train, target_test =train_test_split(ds, target, test_size=0.2, stratify = target, random_state=3)
#optional stratify param for even splitting of target data

scaler = StandardScaler()
#assign mean and std to the scaler
ds_train_standardized = scaler.fit_transform(ds_train)
ds_test_standardized = scaler.fit_transform(ds_test)
```
