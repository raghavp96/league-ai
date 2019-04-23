import csv
import os
dirname = os.path.dirname(os.path.abspath(__file__))

results_file = dirname + '/results.csv'

def writeDict(resultDict={}, epochs=[]):
    rows = []

    header = ["Model Name"]
    for epoch in epochs:
        header.append("After " + str(epoch) + " epochs")
    rows.append(header)

    for modelName in resultDict:
        row = [modelName]
        for result in resultDict[modelName]:
            row.append(result["Accuracy"])
        rows.append(row)

    write(rows=rows)

# https://www.programiz.com/python-programming/working-csv-files#existing-files
def write(rows=[]):
    with open(results_file, 'w+') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(rows)
    writeFile.close()
