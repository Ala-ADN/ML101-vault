turn text column to numbers
```python
from sklearn.preprocessing import LabelEncoder
data['labelled_column'].value_counts() #count labels 
encoded = label_encode.fit_transform(data.labelled_column)
cancer_data['unlabelled'] = encoded
```