import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def getInputValues():
  d = dict()
  try:
    input = pd.read_csv('./input_coeff.csv')
    inputColumns = pd.DataFrame(input, columns=['xInput', 'yInput1', 'yInput2'])
    d['xInput'] = inputColumns.get('xInput')
    d['yInput1'] = inputColumns.get('yInput1')
    d['yInput2'] = inputColumns.get('yInput2')
  except Exception as err:
    print(err)
  return d

input = getInputValues()
try:
  coordinates_set1 = list(zip(input['xInput'], input['yInput1']))
  coordinates_set2 = list(zip(input['xInput'], input['yInput2']))

  # Extract x and y values from the coordinate sets
  x_set1, y_set1 = zip(*coordinates_set1)
  x_set2, y_set2 = zip(*coordinates_set2)

  # Calculate the correlation coefficient between y values
  correlation_coefficient = np.corrcoef(y_set1, y_set2)[0, 1]

  plt.title("R\u00b2 = " + str(correlation_coefficient * correlation_coefficient))

  # Plot the coordinates
  plt.plot(x_set1, y_set1, label='Data', color='Red')
  plt.plot(x_set1, y_set1, 'go', color='Red')

  plt.plot(x_set2, y_set2, label='Model', color='Blue')
  plt.plot(x_set2, y_set2, 'go', color='Blue')

  plt.legend()

  # Show the plot
  plt.grid(True)
  plt.show()

except Exception as err:
  print(err)