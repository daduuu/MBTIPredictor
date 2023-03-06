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
    temp = re.sub(r'\d', '', temp)
    temp = ' '.join(temp.split())
    temp = temp.strip()
    for c in conversion:
        temp = temp.replace(c, newtext)
    if temp == ' ':
        temp = ''
    return temp

df['posts'] = df['posts'].apply(convert)



def toInt(string):
    if string == 'ISTJ':
        return 0
    if string == 'ISFJ':
        return 1
    if string == 'INFJ':
        return 2
    if string == 'INTJ':
        return 3
    if string == 'ISTP':
        return 4
    if string == 'ISFP':
        return 5
    if string == 'INFP':
        return 6
    if string == 'INTP':
        return 7
    if string == 'ESTP':
        return 8
    if string == 'ESFP':
        return 9
    if string == 'ENFP':
        return 10
    if string == 'ENTP':
        return 11
    if string == 'ESTJ':
        return 12
    if string == 'ESFJ':
        return 13
    if string == 'ENFJ':
        return 14
    if string == 'ENTJ':
        return 15
    else:
        return ''


df['type'] = df['type'].apply(toInt)


df['posts'].replace('', np.nan, inplace=True)
df['posts'].replace(' ', np.nan, inplace=True)
df['posts'].replace('NA', np.nan, inplace=True)
df = df.reset_index(drop=True)


df = df.dropna(axis=0, how='any')
df['new_col'] = range(1, len(df) + 1)
df['pad'] = " "


df.to_csv("converted.csv", index=False)



#%%
