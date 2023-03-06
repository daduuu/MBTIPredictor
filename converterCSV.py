import re

lines = []
with open('trains_data.csv',encoding="utf8") as input:
    lines = input.readlines()

conversion = '-"/.$*()@#%^&+=}{|:;\?_<>]['
newtext = ''
outputLines = []
for line in lines:
    line = str(line.encode('utf-8'))
    temp = line
    temp = re.sub('http://\S+|https://\S+', '', temp)
    for c in conversion:
        temp = temp.replace(c, newtext)
    outputLines.append(temp)

with open('converted.csv', 'w') as output:
    for line in outputLines:
        output.write(line + "\n")