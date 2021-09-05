import csv
import numpy

INPUT_FILE = "./data/samples_final.csv"
OUTPUT_FILE = "./data/proportions.csv"

with open(INPUT_FILE, "r") as csv_input, open(OUTPUT_FILE, "w", newline='') as csv_output:
    # configure reader
    csvReader = csv.reader(csv_input)

    # configure writer
    csvWriter = csv.writer(csv_output)
    header = ['13-4', '13-4', '9-31', '9-31', 'A/B', 'A/B', '58-52', '58-52', '67-63', '67-63', 'D/E', 'D/E']
    csvWriter.writerow(header)

    for i, row in enumerate(csvReader):
        if i == 0: # trash header
            continue

        # aggregate relevant points
        relevant_pts = [4, 13, 9, 31, 52, 58, 63, 67]
        x_y_pts = {}
        for p in relevant_pts:
            x = int(row[3+p*2])
            y = int(row[4+p*2])
            x_y_pts[p] = (x,y)

        # perform relevant calculations and add to OUTPUT_FILE
        relevant_calcs = {}
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" 
        letters_iter = 0
        for i, calc in enumerate(header):
            if i % 2 != 0:
                continue
            calc_x, calc_y = "ERROR", "ERROR"
            if '-' in calc: # subtract
                calc_items = calc.split("-")
                print("sub", x_y_pts[int(calc_items[0])], x_y_pts[int(calc_items[1])])
                calc_x, calc_y = numpy.subtract(x_y_pts[int(calc_items[0])], x_y_pts[int(calc_items[1])])
            elif '/' in calc: # divide
                calc_items = calc.split("/")
                print("divide ", calc_items)
                calc_x, calc_y = numpy.divide(relevant_calcs[calc_items[0]], relevant_calcs[calc_items[1]])
            relevant_calcs[letters[letters_iter]] = (calc_x, calc_y)
            letters_iter += 1
        
        print(relevant_calcs)
        write_row = []
        for i in relevant_calcs.values():
            print(i)
            for j in i:
                write_row.append(j)
        csvWriter.writerow(write_row)
    
