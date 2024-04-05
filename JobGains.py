import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)



jobgains_file_path='MonthlyUSJobGainsFrom2018ToPresent.csv'
jobgains_data=pd.read_csv(jobgains_file_path)
#jobgains_data=jobgains_data.dropna(axis=0)
#print(jobgains_data)
#filtered_jobgains_data=jobgains_data.dropna(axis=0)

#FEATURES
jobgains_features=['UnemploymentRate','ConsumerPriceIndex','LaborForceParticipationRate','AverageHourlyEarnings']
X=jobgains_data[jobgains_features]

#TARGET VARIABLE
y=jobgains_data.NumberOfJobs

#FITTING
jobgains_model=DecisionTreeRegressor(random_state=1)
print(jobgains_model.fit(X,y))

#PREDICTING
print("Making predictions for the following 5 job gains:")
print(X.head())
print("The predictions are")
predicted_job_gains=jobgains_model.predict(X)
print(predicted_job_gains)
print(mean_absolute_error(y, predicted_job_gains))

#VALIDATING

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 104, train_size=0.8, test_size=0.2,shuffle=True)
jobgains_model.fit(train_X, train_y)
val_predictions=jobgains_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
# forest_model = RandomForestRegressor(random_state=1)
# forest_model.fit(train_X, train_y), 
# jobgains_preds = forest_model.predict(val_X)
# print(X)
#print(y)
#print(jobgains_preds)
#print(mean_absolute_error(val_y, jobgains_preds))

#FOREST MODEL
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
jobgains_preds = forest_model.predict(val_X)
print("WITH FOREST MODEL..")
print(mean_absolute_error(val_y, jobgains_preds))



