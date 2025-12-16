import numpy as np
from ucimlrepo import fetch_ucirepo
import pandas as pd
import RegressionModel as model
import matplotlib.pyplot as plt

# fetch dataset
student_performance = fetch_ucirepo(id=320)

data = np.load("weights.npz", allow_pickle=True)
w = data['w']
b = data['b']
train_columns = data['train_columns']

features = student_performance.data.features.iloc[0]
g3 = student_performance.data.targets.values[:,2]*5



# Collect raw inputs from user
print("Warning: you must input answers exactly as shown or it will fail")

raw_data = {
    'sex' : input("sex (male/female): "),
    'age' : int(input("age (15-22): ")),
    'address' : input("address (rural/urban): "),
    'famsize' : input("famsize (<=3/>3): "),
    'Pstatus' : input("parent's cohabitation status (together/apart): "),
    'Medu' : input("mother's education (none, 4th grade, middle school, high school, university): "),
    'Fedu' : input("father's education (none, 4th grade, middle school, high school, university): "),
    'Mjob' : input("mothers's job (teacher, health, civil services, at home, other): "),
    'Fjob' : input("father's job (teacher, health, civil service, at home, other): "),
    'reason' : input("reason for choosing school (close to home, reputation, class preference, other): "),
    'guardian' : input("guardian (mother, father, other): "),
    'traveltime' : input("travel time to school in minutes (<15,15-30,30-60,>60): "),
    'studytime' : input("study time in hours (<2,2-5,5-10,>10): "),
    'failures' : int(input("number of past class failures (0,1,2,4=(>3)): ")),
    'schoolsup' : input("have you used school provided extra educational support? (yes, no): "),
    'famsup' : input("does your family support you educationally? (yes, no): "),
    'paid' : input("have you taken extra paid classes? (yes, no): "),
    'activities' : input("do you do extra-curricular activities? (yes, no): "),
    'nursery' : input("did you attend nursery school? (yes, no): "),
    'higher' : input("do you plan to have higher education? (yes, no): "),
    'internet' : input("do you have internet access? (yes, no): "),
    'romantic' : input("are you in a 'romantic' relationship? (yes, no): "),
    'famrel' : int(input("how would you rate your family relationships? (1:bad to 5:good): ")),
    'freetime' : int(input("how much free time do you have after school? (1:a little to 5:alot): ")),
    'goout' : int(input("how much do you go out with friends? (1:a little to 5:alot): ")),
    'Dalc' : int(input("weekday alcohol consumption? (1:a little to 5:alot): ")),
    'Walc' : int(input("weekend alcohol consumption? (1:a little to 5:alot): ")),
    'health' : int(input("how would you rate your health? (1:bad to 5:good): ")),
    'absences' : int(input("how many school absences do you have? (0-93): "))
}

binary_map = {'yes':0, 'no':1}
s_map = {'female':0, 'male':1}
Pstatus_map = {'together':0, 'apart':1}
address_map = {'urban':0, 'rural':1}
famsize_map = {'<=3':0, '>3':1}
Medu_Fedu_map = {'none':0, '4th grade':1, 'middle school':2, 'high school':3, 'university':4}
traveltime_map = {'<15':1, '15-30':2, '30-60':3, '>60':4}
studytime_map = {'<2':1, '2-5':2, '5-10':3, '>10':4}
Mjob_Fjob_map = {'teacher':'teacher','health':'health','civil services':'services','at home':'at_home','other':'other'}
reason_map = {'close to home':'home', 'reputation':'reputation', 'class preference':'course', 'other':'other'}

raw_data['sex'] = s_map[raw_data['sex']]
raw_data['Pstatus'] = Pstatus_map[raw_data['Pstatus']]
raw_data['address'] = address_map[raw_data['address']]
raw_data['famsize'] = famsize_map[raw_data['famsize']]
raw_data['schoolsup'] = binary_map[raw_data['schoolsup']]
raw_data['famsup'] = binary_map[raw_data['famsup']]
raw_data['paid'] = binary_map[raw_data['paid']]
raw_data['activities'] = binary_map[raw_data['activities']]
raw_data['nursery'] = binary_map[raw_data['nursery']]
raw_data['higher'] = binary_map[raw_data['higher']]
raw_data['internet'] = binary_map[raw_data['internet']]
raw_data['romantic'] = binary_map[raw_data['romantic']]
raw_data['Medu'] = Medu_Fedu_map[raw_data['Medu']]
raw_data['Fedu'] = Medu_Fedu_map[raw_data['Fedu']]
raw_data['traveltime'] = traveltime_map[raw_data['traveltime']]
raw_data['studytime'] = studytime_map[raw_data['studytime']]
raw_data['Mjob'] = Mjob_Fjob_map[raw_data['Mjob']]
raw_data['reason'] = reason_map[raw_data['reason']]
raw_data['Fjob'] = Mjob_Fjob_map[raw_data['Fjob']]

df_input = pd.DataFrame([raw_data])

for col in train_columns:
    if col not in df_input.columns:
        df_input[col] = 0

mjob_col = f"Mjob_{raw_data['Mjob']}"
fjob_col = f"Fjob_{raw_data['Fjob']}"
reason_col = f"reason_{raw_data['reason']}"
guardian_col = f"guardian_{raw_data['guardian']}"
df_input.drop(columns=['Mjob','Fjob','reason','guardian'], inplace=True)

for col_name in [mjob_col, fjob_col, reason_col, guardian_col]:
    if col_name in df_input.columns:
        df_input[col_name] = 1

X_input = np.array(df_input, dtype=float)
prediction = model.predict(X_input,w, b)
df_input = df_input[train_columns]
pred_val = prediction[0] if isinstance(prediction, np.ndarray) else prediction
print("Your final grade will be: {:.1f}/100 ±10 pts".format(pred_val*5))



