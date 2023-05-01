import numpy as np
import scikitplot as skplt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

"""
In the prob % of cases, it will replace the value with a random one
"""
def randomize(list, label, prob=0.2):
    output = []
    for item in list:
        if np.random.random() > prob:
            output.append(item)
        else:
            output.append(np.random.choice(label))
    return output

labels = ['political', 'sport', 'tech', 'entertainment', 'business']

y = np.random.choice(labels, 1000)
p = randomize(y, labels)

acc = accuracy_score(y, p)

print(f"Accuracy: {acc}")
print(f"Misclassification: {1-acc}")

report = classification_report(y, p)
print(report)