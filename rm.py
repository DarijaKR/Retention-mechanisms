import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import math
import csv
from gekko import GEKKO

decimals = 4 # adjust if needed to change decimals used for calculation and graph presentation

def getPolynomialValue(x, coefficients, power):
  result = 0
  for i in range(power + 1):
    result += pow(x, power - i) * coefficients[i]
  return round(result, decimals)

def getPolynomialText(coefficients, power):
  result = ''
  for i in range(power + 1):
    coefficient = str(round(coefficients[i], decimals)) if coefficients[i] < 0 else '+' + str(round(coefficients[i], decimals))
    if (i == power):
      result += coefficient
    else:
      result += coefficient + 'x^' + str(power - i)

  result = result[1:] if result[0] == '+' else result
  result = result.replace('^1', '')
  result = result.replace('^2', '\u00b2')
  result = result.replace('^3', '\u00b3')
  return 'y=' + result

def getCorrelationText(value):
  valueString = str(round(value, decimals))
  return 'R\u00b2=' + valueString

def getMinValuesText(coefficients):
  fiMin = -coefficients[1] / (2 * coefficients[0])
  yMin = ((decimals * coefficients[0] * coefficients[2]) - math.pow(coefficients[1], 2)) / (decimals * coefficients[0])
  yMinAntilog = math.pow(10, yMin)
  return ', Fimin=' + str(round(fiMin, decimals)) + ', yMin=' + str(round(yMin, decimals)) + ', k\'=' + str(round(yMinAntilog, decimals))

def calculateY(xInput, power, polynomialCoefficients):
  result = []
  for i in range(len(xInput)):
    value = xInput[i]
    if (power > 0):
      result.append(getPolynomialValue(value, polynomialCoefficients, power))
    else:
      result = None
  return result

def showDualPlotAndWriteToCSV(title, xSet1, ySet1, xSet2, ySet2, params, fiMin):
  plt.title(title)
  subtitle = 'Params='
  for i in range(len(params)):
    subtitle += str(round(params[i], decimals)) + ', '
  plt.suptitle(subtitle + 'Fimin=' + str(fiMin))
  plt.plot(xSet1, ySet1, label='Set 1', color='blue')
  plt.plot(xSet1, ySet1, marker='o', color='blue')
  plt.plot(xSet2, ySet2, label='Set 2', color='red')
  plt.plot(xSet2, ySet2, marker='o', color='red')
  plt.legend()
  plt.grid(True)
  plt.show()
  with open('./output-dual-model.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['xInput', 'yInput', 'yFitted', title])
    for i in range(len(xSet1)):
      writer.writerow([xSet1[i], ySet1[i], str(round(ySet2[i], decimals))])

def showQuadraticPlotAndWriteToCSV(title, xInput, yInput, yResult):
  plt.title(title)
  plt.plot(xInput, yInput, 'go', label='Input values' )
  plt.plot(xInput, yResult, label='Function values' )
  plt.legend()
  plt.grid(True)
  plt.show()
  with open('./output-quadratic-model.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['xInput', 'yInput', 'yResult', title])
    for i in range(len(xInput)):
      writer.writerow([xInput[i], yInput[i], yResult[i]])

def showSubplotsAndWriteToCSV(title1, title2, xInput1, yInput1, yResult1, xInput2, yInput2, yResult2, isPartitionWinner, rightModel):
  fig, axs = plt.subplots(1, 2, constrained_layout=True)
  suptitlePrefix = 'RP-HPLC ' if rightModel else 'HILIC '
  fig.suptitle(suptitlePrefix + 'partition and adsorption model')
  axs[0].plot(xInput1, yInput1, 'go', label='Input values' )
  axs[0].plot(xInput1, yResult1, label='Function values' )
  axs[0].set_title(title1 + ' (winner)' if isPartitionWinner else title1)
  axs[1].plot(xInput2, yInput2, 'go', label='Input values' )
  axs[1].plot(xInput2, yResult2, label='Function values' )
  axs[1].set_title(title2 if isPartitionWinner else title2 + ' (winner)')
  plt.show()
  fileSuffix = '-rp-hplc' if rightModel else '-hilic'
  with open('./output' + fileSuffix + '.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['xInput1', 'yInput1', 'yResult1', 'xInput2', 'yInput2', 'yResult2', title1, title2])
    for i in range(len(xInput1)):
      writer.writerow([xInput1[i], yInput1[i], yResult1[i], xInput2[i], yInput2[i], yResult2[i]])

def calculateDualModel(xInput, yInput):    
  # Initialize GEKKO model
  m = GEKKO(remote=False)
  
  # Define parameters a, b, c, and d with initial values and bounds, and enable parameters to be optimized 
  a, b, c, d = [m.FV(value=0.5, lb=-10, ub=100) for _ in range(4)]
  for param in (a, b, c, d): param.STATUS = 1

  # Define input data in GEKKO
  x = m.Param(value=xInput)
  yActual = m.CV(value=yInput)  # Controlled variable for y_input data
  yFitted = m.Var()

  # Define the model equation
  m.Equation(yFitted == a + b * x - c * m.log(1 + d * x))
  # Objective: Minimize the sum of squared residuals
  yActual.FSTATUS = 1 # FSTATUS=1 to indicate this as the measured value
  m.Minimize((yActual - yFitted) ** 2)

  # Adjust solver options
  m.options.IMODE = 2  # Steady-state optimization
  m.options.MAX_ITER = 10000  # Increase max iterations for better convergence
  m.options.RTOL = 1e-6  # Relative tolerance for accuracy
  m.options.OTOL = 1e-6  # Absolute tolerance for accuracy
  m.solve(disp=False)

  # Extract optimized parameters
  params = [a.value[0], b.value[0], c.value[0], d.value[0]]
  # Calculate fitted y-values using the optimized parameters outside GEKKO
  yFitted = [params[0] + params[1] * x - params[2] * np.log(1 + params[3] * x) for x in xInput]
  
  residuals = np.array(yInput) - np.array(yFitted)
  ssRes = np.sum(residuals ** 2)
  ssTot = np.sum((yInput - np.mean(yInput)) ** 2)
  rSquared = 1 - (ssRes / ssTot)
  fiMin = round((params[2] * params[3] - params[1]) / (params[1] * params[3]), decimals)
  n = len(yInput)
  p = len(params)
  adjustedRSquared = 1 - (1 - rSquared) * ((n - 1) / (n - p - 1))

  title = getCorrelationText(rSquared) + ', Adjusted R²=' + str(round(adjustedRSquared, decimals))
  showDualPlotAndWriteToCSV(title, xInput, yInput, xInput, yFitted, params, fiMin)

def calculateQuadraticModel(xInput, yInput):
  power = 2
  polynomialCoefficients = np.polyfit(xInput, yInput, power)
  yResult = calculateY(xInput, power, polynomialCoefficients)
  correlationCoefficient = pd.Series(yInput).corr(pd.Series(yResult)) * pd.Series(yInput).corr(pd.Series(yResult)) # R^2
  title = getPolynomialText(polynomialCoefficients, power) + '\n' + getCorrelationText(correlationCoefficient) + getMinValuesText(polynomialCoefficients)
  showQuadraticPlotAndWriteToCSV(title, xInput, yInput, yResult)

def getSlicedModelXInput(xInput, resultLength, isPartition):
  if (isPartition):
    return xInput[0:resultLength]
  else:
    result = []
    for i in range(len(xInput)):
      if (i < resultLength):
        result.append(round(math.log10(xInput[i]), decimals))
    return result

def calculateSlicedModel(xInput, yInput, resultLength, isPartition):
  xInputNew = getSlicedModelXInput(xInput, resultLength, isPartition)
  power = 1
  yInputNew = yInput[0:resultLength]
  polynomialCoefficients = np.polyfit(xInputNew, yInputNew, power)
  yResult = calculateY(xInputNew, power, polynomialCoefficients)
  correlationCoefficient = pd.Series(yInputNew).corr(pd.Series(yResult)) * pd.Series(yInputNew).corr(pd.Series(yResult)) # R^2

  d = dict()
  d['correlationCoefficient'] = correlationCoefficient
  d['xInputNew'] = xInputNew
  d['yInputNew'] = yInputNew
  d['yResult'] = yResult
  d['title'] = getPolynomialText(polynomialCoefficients, power) + '\n' + getCorrelationText(correlationCoefficient)
  return d
  
def getSlicedModelValues(xInput, yInput, rangeStart, rangeEnd, isPartition):
  modelCoefficient = 0
  modelIndex = 0
  coefficientSum = 0
  for i in range(rangeStart, rangeEnd):
    result = calculateSlicedModel(xInput, yInput, i, isPartition)['correlationCoefficient']
    if (result > modelCoefficient):
      modelIndex = i
      modelCoefficient = result
    coefficientSum += result

  d = dict()
  d['modelCoefficient'] = modelCoefficient
  d['modelIndex'] = modelIndex
  d['coefficientSum'] = coefficientSum
  return d
  
def getBestSlicedModel(xInput, yInput, rightModel):
  # adjust these two values below if needed
  rangeStart = 5
  rangeEnd = 8
  if (rightModel):
    xInput = list(reversed(xInput[-rangeEnd:]))
    yInput = list(reversed(yInput[-rangeEnd:]))

  partitionModelValues = getSlicedModelValues(xInput, yInput, rangeStart, rangeEnd, True)
  adsorptionModelValues = getSlicedModelValues(xInput, yInput, rangeStart, rangeEnd, False)

  isPartitionWinner = partitionModelValues['coefficientSum'] > adsorptionModelValues['coefficientSum']
  winnerIndex = partitionModelValues['modelIndex'] if isPartitionWinner else adsorptionModelValues['modelIndex']
  result1 = calculateSlicedModel(xInput, yInput, winnerIndex, True)
  result2 = calculateSlicedModel(xInput, yInput, winnerIndex, False)
  showSubplotsAndWriteToCSV(result1['title'], result2['title'], result1['xInputNew'], result1['yInputNew'], result1['yResult'],
    result2['xInputNew'], result2['yInputNew'], result2['yResult'], isPartitionWinner, rightModel)

def getInputValues():
  d = dict()
  try:
    input = pd.read_csv('./input1.csv')
    inputColumns = pd.DataFrame(input, columns=['xInput', 'yInput'])
    d['xInput'] = inputColumns['xInput'].tolist()
    d['yInput'] = inputColumns['yInput'].tolist()
  except Exception as err:
    if (len(sys.argv) != 3):
      print('Please input valid X and Y values from CSV or command line arguments')
      d = None
    else:
      d['xInput'] = sys.argv[1].split(',')
      d['yInput'] = sys.argv[2].split(',')
  return d

try:
  input = getInputValues()
  if input:
    xInput = list(map(float, input['xInput']))
    yInput = list(map(float, input['yInput']))

    if len(xInput) == len(yInput):
      calculateDualModel(xInput, yInput)
      # calculateQuadraticModel(xInput, yInput)
      # getBestSlicedModel(xInput, yInput, False)
      # getBestSlicedModel(xInput, yInput, True)
    else:
      print('X and Y values are not the same length')
except Exception as err:
  print(err)
  print('Error occurred, please try with different input values')