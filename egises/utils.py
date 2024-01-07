import csv
import os.path
import traceback


def divide_with_exception(x, y):
    try:
        return x / y
    except ZeroDivisionError as err:
        return 0
    except Exception as error:
        print(traceback.format_exc())
    return 0


def calculate_proportion(x, y):
    try:
        return min(x, y) / max(x, y)
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
