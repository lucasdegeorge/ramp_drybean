import pandas as pd
import numpy as np

df = pd.read_csv('Dry_Beam_Dataset.csv')

msk = np.random.rand(len(df)) < 0.8
train_df = df[msk]
test_df = df[~msk]

train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

public_train_df = train_df.iloc[:len(train_df)//2]
public_test_df = train_df.iloc[len(train_df)//2:len(train_df)//2 + len(train_df)//4]

public_train_df.to_csv('data/public/train.csv', index=False)
public_test_df.to_csv('data/public/test.csv', index=False)