import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

red_wine = pd.read_csv("Datasets/winequality-red.csv", delimiter=';')
print "Dataset loaded in."

X = red_wine[:-1]
Y = red_wine['quality']
