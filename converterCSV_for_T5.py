import re
import pandas as pd
import numpy as np

df = pd.read_csv("mbti_500.csv")


df = df.sample(frac = 1, random_state = 42)


def convert(string):
    conversion = '-"/.$*()@#%^&+=}\'{|:;?_<>]['
    newtext = ''
    outputLines = []
    temp = string
    temp = re.sub('http://\S+|https://\S+', '', temp)
    temp = re.sub(r'\d', '', temp)
    temp = ' '.join(temp.split())
    temp = temp.strip()
    for c in conversion:
        temp = temp.replace(c, newtext)
    if temp == ' ':
        temp = ''
    return temp

df['posts'] = df['posts'].apply(convert)
df['posts'] = "Context: " + df['posts'] + "predict the label: "

df['type'] = df['type'].str.lower()


df['posts'].replace('', np.nan, inplace=True)
df['posts'].replace(' ', np.nan, inplace=True)
df['posts'].replace('NA', np.nan, inplace=True)
df = df.reset_index(drop=True)


df = df.dropna(axis=0, how='any')
df['new_col'] = range(1, len(df) + 1)
df['pad'] = " "


df.to_csv("converted_new_for_T5.csv", index=False)



#%%
