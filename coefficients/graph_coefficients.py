import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import csv
import math

# Description: graphically display the coefficients on a 68-pt labeled face

# definition: weigh the (x, y) coefficients accordingly
# param: coeffs: array of floats for coefficients from labeled regression model
# return: tktk: array of floats with weighted coefficients for corresponding x, y points
# TODO: this function may need to be adjusted depending on methodology
def weigh_coeffs(coeffs):
    weighted = []

    for i in range(0, len(coeffs), 2):
        coeff_x = float(coeffs[i])
        coeff_y = float(coeffs[i+1])

        avg = (abs(coeff_x) + abs(coeff_y)) / 2 * 50
        weighted.append(avg)

    return weighted

# button class & click event
class ModelButton(Button):
    lbl = ""
    coeffs = []

    # constructor
    # param: axes: positioning for button
    # param: label: button label
    # param: coeffs: array of weighted coefficient floats from labeled regression model
    def __init__(self, axes, label, coeffs):
        self.lbl = label
        self.coeffs = coeffs
        self.btn = Button(axes, label)
        self.btn.on_clicked(self.enlarge_pts)

    # definition: graphically weight points based on weights provided
    def enlarge_pts(self, event):
        ax.clear()
        ax.imshow(img)
        ax.title.set_text(self.lbl)
        print(self.lbl)
        ax.scatter(x=marker_x, y=marker_y, c="r", s=self.coeffs)

# load image, points
img = image.imread('face.png')
markers_file = "gui points.csv"

# points array
marker_x = []
marker_y = []

# append (x, y) to array
with open(markers_file, "r") as csvfile:
    csvReader = csv.reader(csvfile)
    points = [row for i, row in enumerate(csvReader) if i == 2][0] # relevant row
    for i in range(1, len(points), 2):
        marker_x.append(int(points[i]))
        marker_y.append(int(points[i+1]))    

# load coefficients
coefficients_file = "coefficients.csv"
models = {}

with open(coefficients_file, "r") as csvfile:
    csvReader = csv.reader(csvfile)
    
    for i, row in enumerate(csvReader):
        if i == 0: # throw header
            continue

        # adjust coefficients
        models[row[0]] = weigh_coeffs([row[val] for val in range(1, len(points))])

# display image, model points
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
ax.imshow(img)
ax.scatter(x=marker_x, y=marker_y, c="r", s=10)

# create and display buttons
btns = []
spacing = 0 # btwn buttons
for m in models.keys():
    btn = ModelButton(plt.axes([0.2+spacing, 0.05, 0.3, 0.05]), m, models[m])
    btns.append(btn)
    spacing += 0.31

plt.show()
