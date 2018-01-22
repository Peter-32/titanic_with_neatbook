import pandas as pd
import pickle


#################### Get Dataset: #####################

testX = pd.read_csv('test.csv') ## Edit: Your dataset

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
submitDf = pd.concat([testX, resultsDf], axis=1)
submitDf.to_csv('./submit.csv')
print("Done")
print(results)

#######################################################

#######################################################
