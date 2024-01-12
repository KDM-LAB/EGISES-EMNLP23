import csv
import os.path
import traceback

import numpy as np


def divide_with_exception(x, y):
    try:
        return x / y
    except ZeroDivisionError as err:
        return 0
    except Exception as error:
        print(traceback.format_exc())
    return 0


def calculate_minmax_proportion(x, y, epsilon=0):
    try:
        return (min(x, y)+epsilon) / (max(x, y)+epsilon)
    except ZeroDivisionError as err:
        # print(f"x:{x}, y:{y}")
        # print(traceback.format_exc())
        return 0
    except Exception as err:
        print(f"x:{x}, y:{y}")
        print(traceback.format_exc())

        return 0


def write_scores_to_csv(rows, fields=None, filename="scores.csv"):
    # print(type(rows))
    if not rows:
        return
    if fields:
        try:
            assert rows and len(rows[0]) == len(fields)
        except AssertionError as err:
            print(traceback.format_exc())
            print(f"fields: {fields}")
            print(rows[0])

            return
    if os.path.exists(filename):
        # append to existing file
        with open(filename, 'a') as f:
            # using csv.writer method from CSV package
            if fields:
                write = csv.writer(f)
            write.writerows(rows)
    else:  # create new file
        with open(filename, 'w') as f:
            # using csv.writer method from CSV package
            if fields:
                write = csv.writer(f)
            write.writerow(fields)
            write.writerows(rows)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def custom_sigmoid(x, alpha=4.0, beta=1.0):
    return 1 / (1 + ((10 ** alpha) * np.exp(-(10 ** beta) * x)))
