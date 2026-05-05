import pandas as pd
from sklearn.metrics import classification_report

# Your evaluation code here
print(model.predictions.value_counts())
print(classification_report(y_true, y_pred))
