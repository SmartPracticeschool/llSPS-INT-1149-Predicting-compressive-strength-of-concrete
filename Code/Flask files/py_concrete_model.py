# Importing the libraries
import numpy as np
import pandas as pd
from scipy import stats
import pickle

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor

#import the dataset
data = pd.read_excel('Concrete_Data.xls')

#rename the columns
data = data.rename(columns={'Cement (component 1)(kg in a m^3 mixture)':"cement",
       'Blast Furnace Slag (component 2)(kg in a m^3 mixture)':"furnace_slag",
       'Fly Ash (component 3)(kg in a m^3 mixture)':"fly_ash",
       'Water  (component 4)(kg in a m^3 mixture)':"water",
       'Superplasticizer (component 5)(kg in a m^3 mixture)':"super_plasticizer",
       'Coarse Aggregate  (component 6)(kg in a m^3 mixture)':"coarse_agg",
       'Fine Aggregate (component 7)(kg in a m^3 mixture)':"fine_agg", 'Age (day)':"age",
       'Concrete compressive strength(MPa, megapascals) ':"compressive_strength"})

#Remove outliers
z = np.abs(stats.zscore(data))
z_scores = stats.zscore(data)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
dataset = data[filtered_entries]


#feature engineering
X = dataset.iloc[:,:8]
y = dataset.iloc[:,-1]


#Splitting Training and Test Set
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.15)

#create a regressor object for 'RandomForestRegressor'
GB_regressor = GradientBoostingRegressor()

#Fitting model with trainig data
GB_regressor.fit(x_train, y_train)

# Saving model to disk
pickle.dump(GB_regressor, open('concrete_model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('concrete_model.pkl','rb'))

#Evaluate the model's performance using unseen data
print(model.predict([[300.34,0.0,0.0,3.0,0.0,0.0,0.0,28]]))
print(model.predict([[350.6,0.0,0.0,162.0,0.0,930.0,550.89,28]]))
print(model.predict([[329.34,116.0,0.0,300.9,3.9,1000.0,543.0,100]]))

#Evaluate the model's performance on trained data
print(model.predict([[540.0,0.0,0.0,162.0,2.5,1040.0,676.0,28]]))
print(model.predict([[266.0,114.0,0.0,228.0,0.0,932.0,670.0,90]]))
print(model.predict([[380.0,95.0,0.0,228.0,0.0,932.0,594.0,28]]))



