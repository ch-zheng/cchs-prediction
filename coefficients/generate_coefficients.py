from sklearn.model_selection import train_test_split
import csv

OUTPUT = "data/coefficients.csv"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# fit models
for name, m in models.items():
  m.fit(X_train, y_train)

# coefficients dictionary
coeffs = {}
for name, m in models.items():
  try:
    coeffs[name] = m.coef_.tolist()[0] # NOTE: first two coeffs are for race, age
  except (AttributeError):
    continue

print(coeffs.keys())

# write coefficient arrays to csv
with open(OUTPUT, 'w') as csvfile:
  csvWriter = csv.writer(csvfile)
  
  # header
  header = ["model", "race", "age"]
  for i in range(68):
    header.append('x' + str(i))
    header.append('y' + str(i))
  csvWriter.writerow(header)

  # store coefficients
  for name, coeff_arr in coeffs.items():
    row = [name]
    for c in coeff_arr:
      row.append(c)
    csvWriter.writerow(row)
