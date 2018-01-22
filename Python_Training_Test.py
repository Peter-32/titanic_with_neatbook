import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from neatbook.neat import *
from sklearn.metrics import confusion_matrix
import pickle


##### IF YOU HAVE 1 DATASET UNCOMMENT THIS CODE: #####

df = pd.read_csv('train.csv') ## Edit: Your dataset
className = 'Survived' ## Edit: Replace class with the Y column name
trainX, testX, trainY, testY = train_test_split(df.drop([className], axis=1),
                                                    df[className], train_size=0.75, test_size=0.25)

#######################################################

##### IF YOU HAVE 2 DATASETS UNCOMMENT THIS CODE: #####

# trainDf = pd.read_csv('train_iris.csv') ## Edit: Your dataset
# testDf = pd.read_csv('test_iris.csv') ## Edit: Your dataset

# className = 'class' ## Edit: Replace class with the Y column name
# trainX = trainDf.drop([className], axis=1)
# trainY = trainDf[className]
# testX = testDf.drop([className], axis=1)
# testY = testDf[className]

#######################################################

################### Set Variables: ####################

indexColumns = ['PassengerId'] ## Edit: Optionally add column names
skipColumns = [] ## Edit: Optionally add column names

#######################################################

####################### Clean: ########################

# Clean training set
neat =  Neat(trainX, trainY, indexColumns, skipColumns)
cleanTrainX = neat.df
cleanTrainY = neat.getYAsNumber(trainY)

# Clean test set
neat.cleanNewData(testX)
cleanTestX = neat.df
cleanTestY = neat.getYAsNumber(testY)

#######################################################

###################### Pipeline: ######################

exported_pipeline = RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.4, min_samples_leaf=1, min_samples_split=14, n_estimators=100)

exported_pipeline.fit(cleanTrainX, cleanTrainY)
results = exported_pipeline.predict(cleanTestX)

#######################################################

################## Confusion Matrix: ##################

print("Confusion Matrix:")
print(confusion_matrix(cleanTestY, results))
print(accuracy_score(cleanTestY, results))

#######################################################

############ Create Python_Test.py File: ##############

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

save_object(neat, 'neat.pkl')
save_object(exported_pipeline, 'exported_pipeline.pkl')
save_object(indexColumns, 'indexColumns.pkl')
save_object(skipColumns, 'skipColumns.pkl')
save_object(className, 'className.pkl')


with open('Python_Test.py', 'w') as fileOut:
    fileOut.write("""
#################### Get Dataset: #####################

testDf = pd.read_csv('test_iris.csv') ## Edit: Your dataset
testX = testDf.drop([className], axis=1)

#######################################################

################### Set Variables: ####################

with open('neat.pkl', 'rb') as input:
    neat = pickle.load(input)
with open('exported_pipeline.pkl', 'rb') as input:
    exported_pipeline = pickle.load(input)
with open('indexColumns.pkl', 'rb') as input:
    indexColumns = pickle.load(input)
with open('skipColumns.pkl', 'rb') as input:
    skipColumns = pickle.load(input)
with open('className.pkl', 'rb') as input:
    className = pickle.load(input)

#######################################################

####################### Clean: ########################

neat.cleanNewData(testX)
cleanTestX = neat.df

#######################################################

###################### Predict: #######################

results = exported_pipeline.predict(cleanTestX)
resultsDf = pd.DataFrame(results)
submitDf = pd.concat([testDf, resultsDf], axis=1)
submitDf.to_csv('./submit.csv')
print("Done")
print(results)

#######################################################

#######################################################
""")
