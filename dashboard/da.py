import pandas as pd

df = pd.read_csv('D:\\Projects\\Sleep-Quality-Based-EEG-Signals\\dashboard\\results\\model_results.csv')
for i in range(len(df)):
    print(df.iloc[i])