import tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from meta_ai_api import MetaAI
from GetInfoClass import getInfo
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)



jobgains_file_path='MonthlyJobGains20042024.csv'
jobgains_data=pd.read_csv(jobgains_file_path)
#jobgains_data=jobgains_data.dropna(axis=0)
#print(jobgains_data)
#filtered_jobgains_data=jobgains_data.dropna(axis=0)

#FEATURES
jobgains_features=['DowJonesClosing','UnemploymentRate','ConsumerPriceIndex','LaborForceParticipationRate']
X=jobgains_data[jobgains_features]

#TARGET VARIABLE
y=jobgains_data.JobGains

#FITTING
jobgains_model=DecisionTreeRegressor(random_state=1)
jobgains_model.fit(X,y)

#PREDICTING
#print("Making predictions for the following 5 job gains:")
#print(X.head())
print("The prediction :")
predicted_job_gains=jobgains_model.predict(X)
#print(predicted_job_gains)
#print(mean_absolute_error(y, predicted_job_gains))

#VALIDATING

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 104, train_size=0.8, test_size=0.2,shuffle=True)
jobgains_model.fit(train_X, train_y)
val_predictions=jobgains_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions * 1000))
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
print(list(jobgains_data["DowJonesClosing"])[-1])
jobgains_avg=numpy.mean(list(jobgains_data["JobGains"]))
jobgains_curr=list(jobgains_data["JobGains"])[-1]
dow_avg=numpy.mean(list(jobgains_data["DowJonesClosing"]))
dow_curr=list(jobgains_data["DowJonesClosing"])[-1]
unemployment_avg=numpy.mean(list(jobgains_data["UnemploymentRate"]))
unemployment_curr=list(jobgains_data["UnemploymentRate"])[-1]
CPI_avg=numpy.mean(list(jobgains_data["ConsumerPriceIndex"]))
CPI_curr=list(jobgains_data["ConsumerPriceIndex"])[-1]
labor_force_avg=numpy.mean(list(jobgains_data["LaborForceParticipationRate"]))
labor_force_curr=list(jobgains_data["LaborForceParticipationRate"])[-1]
avg_curr_data = {
    "Job Gains": [jobgains_avg * 1000, jobgains_curr * 1000],
    "Labor Force %": [labor_force_avg * 100, labor_force_curr * 100],
    "Unemployment %": [unemployment_avg * 100, unemployment_curr * 100],
    "Consumer Price Index": [CPI_avg, CPI_curr],
    "Dow Jones Closing": [dow_avg, dow_curr]

}
df_jobgains=pd.DataFrame(avg_curr_data, index=["Average", "Current"])
print(df_jobgains)
form=tkinter.Tk()
form.title("U S MONTHLY JOB GAINS PREDICTOR")
form.geometry("1000x1000")
lblBasis=tkinter.Label(form, text="DATA DATING FROM 2004 TO PRESENT", font=("Arial", 16, "bold"))
lblBasis.pack()
lblPredict= tkinter.Label(form, text="Predicted estimated job gains for current month:", font=("Arial", 12, "bold"))
lblPredict.pack()
entPredict=tkinter.Entry(form)
entPredict.insert(0,str(round(mean_absolute_error(val_y, val_predictions * 1000))))
entPredict.pack()
my_mae = get_mae(5000, train_X, val_X, train_y, val_y)
print(my_mae)
lblMAE=tkinter.Label(form,text="With Mean Absolute Error:",font=("Arial", 12, "bold"))
lblMAE.pack()
entMAE=tkinter.Entry(form)
entMAE.insert(0,str(round(get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y) * 1000)))
entMAE.pack()
lblRandFor=tkinter.Label(form,text="With Random Forest model:",font=("Arial", 12, "bold"))
lblRandFor.pack()
entRandFor=tkinter.Entry(form)
entRandFor.insert(0,str(round(mean_absolute_error(val_y, val_predictions)) * 1000))
entRandFor.pack()
lblAvgCurr=tkinter.Label(form,text="Average and Current Readings:",font=("Arial", 12, "bold"))
lblAvgCurr.pack()
txtAvgCurr=tkinter.Text(form, height=5, width=100)
txtAvgCurr.insert("1.0",df_jobgains.to_string())
txtAvgCurr.pack()
lblMinMax=tkinter.Label(form,text="Minimums and Maximums",font=("Arial", 12, "bold"))
lblMinMax.pack()
txtMinMax=tkinter.Text(form, height=5, width=100)
jobgains_min=numpy.min(list(jobgains_data["JobGains"]))
jobgains_minpos=list(jobgains_data["JobGains"]).index(jobgains_min)
jobgains_min_month=list(jobgains_data["Month"])[jobgains_minpos]
jobgains_max=numpy.max(list(jobgains_data["JobGains"]))
jobgains_maxpos=list(jobgains_data["JobGains"]).index(jobgains_max)
jobgains_max_month=list(jobgains_data["Month"])[jobgains_maxpos]
unemp_min=numpy.min(list(jobgains_data["UnemploymentRate"]))
unemp_minpos=list(jobgains_data["UnemploymentRate"]).index(unemp_min)
unemp_min_month=list(jobgains_data["Month"])[unemp_minpos]
unemp_max=numpy.max(list(jobgains_data["UnemploymentRate"]))
unemp_maxpos=list(jobgains_data["UnemploymentRate"]).index(unemp_max)
unemp_max_month=list(jobgains_data["Month"])[unemp_maxpos]
laborforce_min=numpy.min(list(jobgains_data["LaborForceParticipationRate"]))
laborforce_minpos=list(jobgains_data["LaborForceParticipationRate"]).index(laborforce_min)
laborforce_minmonth=list(jobgains_data["Month"])[laborforce_minpos]
laborforce_max=numpy.max(list(jobgains_data["LaborForceParticipationRate"]))
laborforce_maxpos=list(jobgains_data["LaborForceParticipationRate"]).index(laborforce_max)
laborforce_maxmonth=list(jobgains_data["Month"])[laborforce_maxpos]
CPI_min=numpy.min(list(jobgains_data["ConsumerPriceIndex"]))
CPI_minpos=list(jobgains_data["ConsumerPriceIndex"]).index(CPI_min)
CPI_minmonth=list(jobgains_data["Month"])[CPI_minpos]
CPI_max=numpy.max(list(jobgains_data["ConsumerPriceIndex"]))
CPI_maxpos=list(jobgains_data["ConsumerPriceIndex"]).index(CPI_max)
CPI_maxmonth=list(jobgains_data["Month"])[CPI_maxpos]

minmax_stats={
    "Job Gains": [jobgains_min * 1000, jobgains_min_month, jobgains_max * 1000, jobgains_max_month],
    "Unemployment %": [unemp_min * 100, unemp_min_month, unemp_max * 100, unemp_max_month],
    "Labor Force Participation Rate %": [laborforce_min * 100, laborforce_minmonth, laborforce_max * 100, laborforce_maxmonth],
    "CPI": [CPI_min, CPI_minmonth, CPI_max, CPI_maxmonth]
}

#"Labor Force %": [labor_force_avg * 100, labor_force_curr * 100],
#"Consumer Price Index": [CPI_avg, CPI_curr],
#"Dow Jones Closing": [dow_avg, dow_curr]
df_minmax=pd.DataFrame(minmax_stats, index=["Minimum","Min Month","Maximum", "Max Month"])
#print(list(jobgains_data["Month"]))
#print(f"MAX {jobgains_max} MONTHata {jobgains_max_month}")
print(df_minmax.to_string())
txtMinMax.insert("1.0",df_minmax.to_string())
txtMinMax.pack()
ai = MetaAI()
request_prompt="Meta, this is just a test.  Give a simple response."
response=ai.prompt(message=request_prompt)
mssg = ""
resp = getInfo(request_prompt, mssg)
mssg = response["message"]
print(mssg)
jobgains_past_year=list(jobgains_data["JobGains"])[-12:]
months_past_year=list(jobgains_data["Month"])[-12:]
unemp_rate_past_year=list(jobgains_data["UnemploymentRate"])[-12:]
laborforce_part_past_year=list(jobgains_data["LaborForceParticipationRate"])[-12:]
dowjones_past_year=list(jobgains_data["DowJonesClosing"])[-12:]
print("MONTHS:")
print(months_past_year)
print("JOB GAINS:")
print(jobgains_past_year)
print("UNEMPLOYMENT RATE:")
print(unemp_rate_past_year)
print("LABOR FORCE PARTICIPATION RATE:")
print(laborforce_part_past_year)
print("DOW JONES CLOSINGS:")
print(dowjones_past_year)
data_past_year = {
    "MONTHS": months_past_year,
    "JOB GAINS": jobgains_past_year,
    "UNEMPLOYMENT RATE": unemp_rate_past_year,
    "LABOR FORCE PARTICIPATION RATE": laborforce_part_past_year,
    "DOW JONES CLOSINGS": dowjones_past_year
}

df_past_year=pd.DataFrame(data_past_year)
print(df_past_year.to_string())
lblPastYear=tkinter.Label(form,text="READINGS FOR PAST YEAR:",font=("Arial", 14, "bold"))
lblPastYear.pack()
txtPastYear=tkinter.Text(form, height=15, width=100)
txtPastYear.insert("1.0",df_past_year.to_string())
txtPastYear.pack()
lblAnalysis=tkinter.Label(form,text="CURRENT ANALYSIS FOR THE PAST YEAR:",font=("Arial", 14, "bold"))
lblAnalysis.pack()
txtPastYear=tkinter.Text(form, height=20, width=100,font=("Arial", 10))
txtPastYear.pack()
request_prompt=f'Meta, please give a concise analysis of this data on the U S job gains for the last 12 months.  The numbers is the Job Gains column are in the thousands.\n{df_past_year.to_string()}'
response=ai.prompt(message=request_prompt)
mssg = ""
resp = getInfo(request_prompt, mssg)
mssg = response["message"]
#print(mssg)
txtPastYear.insert("1.0",mssg)
x_axis=numpy.array(months_past_year)
y_axis=numpy.array(jobgains_past_year)
plt.plot(x_axis, y_axis)
plt.show()
form.mainloop()




