import sys, os
sys.path.append(r'c:/Users/samjo/Downloads/simulation/Jet-Engine-Failure-Predictor/src')
from data_loader import load_dataset

df = load_dataset('FD001','train')
print(df.head())
