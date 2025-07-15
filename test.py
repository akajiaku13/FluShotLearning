import pandas as pd
import numpy as np
from flushot.utils.main_utils.utils import load_object

test_df = pd.read_csv('FluShot_Data/test_processed.csv')
respondent_ids = test_df['respondent_id']
X_test = test_df.drop(columns=['respondent_id'])

best_model = load_object('final_model/model.pkl')

# Get probability predictions for each label
probas = best_model.predict_proba(X_test.values)
# Each element in probas is an array of shape (n_samples, 2)
# We want the probability for class 1 for each label
h1n1_proba = probas[0][:, 1]
seasonal_proba = probas[1][:, 1]

submission = pd.DataFrame({
    'respondent_id': respondent_ids,
    'h1n1_vaccine': h1n1_proba,
    'seasonal_vaccine': seasonal_proba
})

submission.to_csv('submission.csv', index=False)