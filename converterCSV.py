import re
import pandas as pd
import numpy as np

df = pd.read_csv("mbti_1.csv")

df['posts'] = df['posts'].apply(lambda x: x.split("|||"))
df = df.explode('posts')
df = df.reset_index(drop=True).dropna()
df = df[['posts', 'type']]

def convert(string):
    conversion = '-"/.$*()@#%^&+=}\'{|:;?_<>]['
    newtext = ''
    outputLines = []
    temp = string
    temp = re.sub('http://\S+|https://\S+', '', temp)
    for c in conversion:
        temp = temp.replace(c, newtext)
    return temp

df['posts'] = df['posts'].apply(convert)

df['posts'].replace('', np.nan, inplace=True)
df['type'].replace('', np.nan, inplace=True)

df = df.dropna(axis=0, how='any')



df.to_csv("converted.csv", index=False)

print(df)