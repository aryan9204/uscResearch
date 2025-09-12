import pandas as pd

df = pd.read_csv("persona_sample.csv")

for index, row in df.iterrows():
    if row["Opinion"] == 1 or row["Opinion"] == 2:
        df.loc[index, "Willingness"] = 3
    elif row["Opinion"] == -1 or row["Opinion"] == -2:
        df.loc[index, "Willingness"] = 7

df.to_csv('persona_sample.csv', index=False)
