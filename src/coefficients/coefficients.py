from src.models.initialize_models import initialize_models
from src.read_write import write_csv_header
import csv
from matplotlib import image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

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

        # highlight largest and smallest points in blue
        max_coeff = self.coeffs.index(max(self.coeffs))
        min_coeff = self.coeffs.index(min(self.coeffs))
        ax.scatter(x=[marker_x[max_coeff], marker_x[min_coeff]], y=[marker_y[max_coeff], marker_y[min_coeff]], c="b", s=[max(self.coeffs), min(self.coeffs)])
        print("Largest: ", max(self.coeffs))
        print("Smallest: ", min(self.coeffs))

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

def generate_coefficients(data):
    OUTPUT = "data/coefficients/coefficients.csv"

    # fit models
    models = initialize_models()
    for name, m in models.items():
        m.fit(data.X_train, data.y_train)
    
    # coefficients dictionary
    coeffs = {}
    for name, m in models.items():
        try:
            coeffs[name] = m.coef_.tolist()[0] # NOTE: first two coeffs are for race, age
        except (AttributeError):
            continue

    print(coeffs.keys())

    # write coefficient arrays to csv
    write_csv_header(OUTPUT)

    with open(OUTPUT, 'w', newline='') as csvfile:
        csvWriter = csv.writer(csvfile)

        # store coefficients
        for name, coeff_arr in coeffs.items():
            row = [name]
            for c in coeff_arr:
                row.append(c)
            csvWriter.writerow(row)

# Description: graphically display the coefficients on a 68-pt labeled face
def graph_coefficients():
    # load image, points
    img = image.imread('data/coefficients/face.png')
    markers_file = "data/coefficients/gui points.csv"

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
    coefficients_file = "data/coefficients/coefficients.csv"
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