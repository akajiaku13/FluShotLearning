import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")


train = pd.read_csv("FluShot_Data/training_set_features.csv")
train_label = pd.read_csv("FluShot_Data/training_set_labels.csv")
test = pd.read_csv("FluShot_Data/test_set_features.csv")


train_full = train.merge(train_label, on="respondent_id")

train_full['is_train'] = 1
test['is_train'] = 0

test['h1n1_vaccine'] = -1
test['seasonal_vaccine'] = -1

df = pd.concat([train_full, test], axis=0, ignore_index=True)

# Drop unnecessary columns
df.drop(columns=['respondent_id', 'employment_industry', 'employment_occupation'], inplace=True)

#region Handle missing values

print("\nMissing values in each column:")
print(df.isnull().sum())

# Fill binary columns with mode
binary_cols = [
    'behavioral_antiviral_meds', 'behavioral_avoidance', 'behavioral_face_mask',
    'behavioral_wash_hands', 'behavioral_large_gatherings', 'behavioral_outside_home',
    'behavioral_touch_face', 'doctor_recc_h1n1', 'doctor_recc_seasonal',
    'chronic_med_condition', 'child_under_6_months', 'health_worker', 'health_insurance'
]
for col in binary_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Fill h1n1 concern with mode
h1n1 = ['h1n1_concern', 'h1n1_knowledge']
for col in h1n1:
    df[col] = df[col].fillna(df[col].mode()[0])

# Fill ordinal opinion columns with median
opinion_cols = [col for col in df.columns if col.startswith('opinion_')]
for col in opinion_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with mode
cat_cols = ['education', 'marital_status', 'income_poverty', 'employment_status', 'rent_or_own']
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Fill household numeric columns with median
df['household_adults'] = df['household_adults'].fillna(df['household_adults'].median())
df['household_children'] = df['household_children'].fillna(df['household_children'].median())

print("\nMissing values in each column After cleaning:")
print(df.isnull().sum())

#endregion

#region Data types
print("\nData types of each column:")
print(df.dtypes)

categorical_cols = df.select_dtypes(include=['object']).columns
print("\nUnique values in categorical columns:")
for col in categorical_cols:
    print(f"{col}: {df[col].unique()}")

#endregion

#region Performing feature encoding

# Ordinal encoding
df['age_group'] = df['age_group'].map({'18 - 34 Years': 0, '35 - 44 Years': 1,
                                       '45 - 54 Years': 2, '55 - 64 Years': 3, '65+ Years': 4})
df['education'] = df['education'].map({'< 12 Years': 0, '12 Years': 1,
                                       'Some College': 2, 'College Graduate': 3})
df['income_poverty'] = df['income_poverty'].map({'Below Poverty': 0,
                                                 '<= $75,000, Above Poverty': 1,
                                                 '> $75,000': 2})

# One-hot encoding for nominal columns
nominal_cols = ['race', 'sex', 'marital_status', 'rent_or_own',
                'employment_status', 'hhs_geo_region', 'census_msa']

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

encoded_array = encoder.fit_transform(df[nominal_cols])
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(nominal_cols))
encoded_df.index = df.index
df = df.drop(columns=nominal_cols)
df = pd.concat([df, encoded_df], axis=1)

# Convert all float columns to int (after all preprocessing)
float_cols = df.select_dtypes(include=['float']).columns
df[float_cols] = df[float_cols].astype('int32')

print("\nData types of each column:")
print(df.dtypes)


#endregion

df.to_csv("FluShot_Data/flu_shot_data.csv", index=False)