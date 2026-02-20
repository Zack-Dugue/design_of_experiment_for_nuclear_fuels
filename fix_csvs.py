import csv
import os

def add_pattern_column(paths):
    for path in paths:
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            reader.__iter__()
            row_0 = reader.__next__()
            if "26" in row_0:
