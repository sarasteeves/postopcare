# question: can patient temperature measures be used to predict where patients will be moved following surgery?

import numpy as np
import pandas as pd
#import seaborn as sns
from pandas.tools.plotting import scatter_matrix

def featdecplots(dataframe, feature):
    feat_table = pd.crosstab(index=dataframe['decision'], columns=dataframe[feature])
    feat_table.plot(kind='bar', figsize=(8,8), title=feature, stacked=False)

if __name__ == "__main__":
    # load data
    # using the Postoperative Patient Data from UCI repository 
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/postoperative-patient-data/post-operative.data"
    names = ['L-CORE', 'L-SURF', 'L-O2', 'L-BP', 'SURF-STBL', 'CORE-STBL', 'BP-STBL', 'COMFORT', 'decision']
    data = pd.read_csv(url, names=names)

    # print dataset characteristics
    print data.shape # 90 samples, 9 columns
    print data.head() # first five rows
    print data.dtypes # gives the data type for each column
    description = data.describe() 
    print description
    
    #correct sample 4 with incorrect category name 'A '
    data['decision'][3] = "A"
    
    #print number of samples in each category
    print data['decision'].value_counts() 
    #data.decision.value_counts().plot(kind='bar')
    
    # plot features by decision
    featdecplots(data, 'L-CORE')
    featdecplots(data, 'L-SURF')
    featdecplots(data, 'L-O2')
    featdecplots(data, 'L-BP')
    featdecplots(data, 'SURF-STBL')
    featdecplots(data, 'CORE-STBL')
    featdecplots(data, 'BP-STBL')
    featdecplots(data, 'COMFORT')

    # split data into train and test sets
    train_set = data.sample(frac=0.8)
    test_set = data.drop(train_set.index)