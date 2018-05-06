import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

with open("lstm_result.txt") as fp:
    lines = fp.readlines()
    results = []
    for line in lines:
        token_dict = {}
        tokens = line.rstrip().split("-")
        for token in tokens:
            item = token.lstrip().split(":")
            token_dict[item[0]] = float(item[1])
        results.append(token_dict)
print results
y1 = []
y2 = []
for item in results:
    y1 += [item["loss"]]
    y2 += [item["acc"]]
x = [z + 1 for z in range(24)]
plt.plot(x, y1, 'r--', x, y2, 'g--')
red_patch = mpatches.Patch(color='red', label='Training Loss')
green_patch = mpatches.Patch(color='green', label='Training Accuracy')
plt.legend(handles=[red_patch, green_patch])
plt.xlabel('Epochs')
plt.show()
