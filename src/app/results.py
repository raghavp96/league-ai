import csv
import os
dirname = os.path.dirname(os.path.abspath(__file__))

results_file = dirname + '/results.csv'

def writeDict(resultDict={}):
    rows = []
    rows.append(["Model Name", "Epoch", "Accuracy"])
    for modelName in resultDict:
        for result in resultDict[modelName]:
            rows.append([modelName, result["Epoch"], result["Accuracy"]])
    write(rows=rows)

def write(rows=[]):
    with open(results_file, 'w+') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(rows)
    writeFile.close()