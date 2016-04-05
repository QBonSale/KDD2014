import pandas as pd
import numpy as np

def cutoff(preds, threshold):
    return (preds > threshold).astype(int)

print('saving prediction to file')
sample = pd.read_csv('gbc_predictions.csv')
sample = sample.sort_values('projectid')
sample.set_index('projectid', inplace=True)
preds = sample['is_exciting']
for threshold in np.arange(0.05, 0.1, 0.01):
    sample['is_exciting'] = cutoff(preds, 0.5)
    sample.to_csv(str(threshold)+'_gbc_predictions.csv', index=True)