import os
import glob
import matplotlib.pyplot as plt

classes = dict()
# for path in glob.glob("medical-reports" + "/**/*.txt", recursive=True):
#     filename = os.path.basename(path)
#     classname = filename[5:-4]
#     classes[classname] = classes.get(classname, 0) + 1

for path in glob.glob("medical-reports/train/" + "*.txt", recursive=False):
    filename = os.path.basename(path)
    classname = filename[5:-4]
    classes[classname] = classes.get(classname, 0) + 1

plt.bar(range(len(classes)), height=list(classes.values()), align='center')
plt.xticks(range(len(classes)), list(classes.keys()))

for index, value in enumerate(list(classes.values())):
    plt.text(index, value, str(value))

plt.show()