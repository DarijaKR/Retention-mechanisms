import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def getInputValues():
  d = dict()
  try:
    input = pd.read_csv('./input.csv')
    inputColumns = pd.DataFrame(input, columns=['xInput', 'yInput'])
    d['xInput'] = inputColumns.get('xInput')
    d['yInput'] = inputColumns.get('yInput')
  except Exception as err:
    print(err)
  return d

#todo: maxfev should be around 400 or at least less than 1000
input = getInputValues()
try:
  model_function = lambda x, a, b, c, d: a + b * x - c * np.log10( 1 + d * x)
  params, covariance = curve_fit(model_function, input['xInput'], input['yInput'], p0=(1, 1, 1, 1), maxfev = 1000000)
  print(params, covariance)

  y_fitted = model_function(input['xInput'], *params)
  residuals = input['yInput'] - y_fitted
  ss_res = np.sum(residuals**2)
  ss_tot = np.sum((input['yInput'] - np.mean(input['yInput']))**2)
  r_squared = 1 - (ss_res / ss_tot)

  print(y_fitted, r_squared)
  print ((0.434*params[2])/(params[1]-1/params[3]))

  coordinates_set1 = list(zip(input['xInput'], input['yInput']))
  coordinates_set2 = list(zip(input['xInput'], y_fitted))

  # Extract x and y values from the coordinate sets
  x_set1, y_set1 = zip(*coordinates_set1)
  x_set2, y_set2 = zip(*coordinates_set2)

  plt.title("R\u00b2 = " + str(r_squared))

  # Plot the coordinates
  plt.plot(x_set1, y_set1, label='Set 1', color='blue')
  plt.plot(x_set1, y_set1, 'go', color='blue')

  plt.plot(x_set2, y_set2, label='Set 2', color='red')
  plt.plot(x_set2, y_set2, 'go', color='red')
  plt.legend()

  # Show the plot
  plt.grid(True)
  plt.show()

except Exception as err:
  print(err)