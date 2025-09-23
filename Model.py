import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import seaborn as sns
<<<<<<< HEAD


=======
from sklearn.metrics import mean_absolute_error

    
>>>>>>> 0c63b40d7f9c3fcc698267e3a067b7c67c05e69e

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class Second_transformer(BaseEstimator, TransformerMixin):
    """
    Adds engineered features using numeric column *indices*:
    - rooms_per_household = total_rooms / households
    - population_per_household = population / households
    """
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        rooms_per_household = X[:, 3] / X[:, 6]       # total_rooms / households
        population_per_household = X[:, 5] / X[:, 6]  # population / households
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, 4] / X[:, 3]     # bedrooms / rooms
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        
    
class Third_transformer(BaseEstimator, TransformerMixin):
    """Handling categorical data: Wraps sklearn's 
    OneHotEncoder to fit the same custom transformer pattern."""

    def __init__(self):
        self.cat_encoder = OneHotEncoder(sparse_output=False)
    
    def fit(self, X, y=None):
        self.cat_encoder.fit(X)
        return self    
    
    def transform(self, X):
        return self.cat_encoder.transform(X)
    
# Load & split
df = pd.read_csv("housing.csv")

X = df.drop("median_house_value", axis=1)  #input
Y = df["median_house_value"].copy()  #target


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


# Tell ColumnTransformer which columns are numeric vs categorical
num_attribs = [
    "longitude", "latitude", "housing_median_age",
    "total_rooms", "total_bedrooms", "population",
    "households", "median_income"
]
cat_attribs = ["ocean_proximity"]


# Build the numeric pipeline (impute -> feature engineer -> scale)
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', Second_transformer(add_bedrooms_per_room=True)),
    ('std_scaler', StandardScaler()),
])

# Build the categorical pipeline (one-hot)
cat_pipeline = Pipeline(steps=[
    ("onehot", Third_transformer()),
])

# Combine them together  
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

X_train_prepared = full_pipeline.fit_transform(X_train)

#print(X_train_prepared.shape)



# Support Vector Machine Regressor 
np.random.seed(42)

# Svr linear classifier
svr_lin = SVR(kernel='linear', C=100, epsilon=0.5)
svr_lin.fit(X_train_prepared, y_train)


#Svr non linear classifier
svr_rbf = SVR(kernel='rbf', C=100, epsilon=0.5, gamma='scale')
svr_rbf.fit(X_train_prepared, y_train)

#Svr Polynomial
svr_poly = SVR(kernel='poly', degree=3, C=100, epsilon=0.5)
svr_poly.fit(X_train_prepared, y_train)

#Predictions for linear, rbf and poly.
lin_predictions = svr_lin.predict(X_train_prepared)
nonlinear_predictions = svr_rbf.predict(X_train_prepared)
poly_predictions = svr_poly.predict(X_train_prepared)

#Linear mean square error
lin_mse = mean_squared_error(y_train, lin_predictions)
lin_rmse = np.sqrt(lin_mse)

#Rbf mean square error
rbf_mse = mean_squared_error(y_train, nonlinear_predictions)
rbf_rmse = np.sqrt(rbf_mse)

#Polynomial mean square error
poly_mse = mean_squared_error(y_train, poly_predictions)
poly_rmse = np.sqrt(poly_mse)

# RMSE results
"""print("Linear prediction: ",lin_rmse)
print("Non linear prediction: ",rbf_rmse)
print("Polynomial prediction: ",poly_rmse)"""


# Fine Tuning with search grid
"""param_grid = [
    {"kernel": ["linear"],
     "C": [1, 10, 100],
     "epsilon": [0.1, 0.2, 0.5]}
]"""
"""param_grid = [
    {"kernel": ["rbf"],
     "C": [1, 10, 100],
     "epsilon": [0.1, 0.2, 0.5]}
]"""
"""param_grid = [
    {"kernel": ["poly"],
     "C": [1, 10, 100],
     "epsilon": [0.1, 0.2, 0.5]}
]"""

"""grid_search = GridSearchCV(SVR(), param_grid, cv=5,
                           scoring="neg_mean_squared_error",
                           verbose=2)
grid_search.fit(X_train_prepared, y_train)

print("Best params:", grid_search.best_params_)
print("Best RMSE:", np.sqrt(-grid_search.best_score_))"""

#Best params for rbf: {'C': 100, 'epsilon': 0.5, 'kernel': 'rbf'} =  97906.83682336772 
#Best params for linear: {'C': 100, 'epsilon': 0.1, 'kernel': 'linear'} = 70158.46280486112

#Best SVR predictor is linear with C = 100, e = 0.1




#Evaluating the model with Cross-Validation (k-fold of 10)
scores = cross_val_score(svr_poly, X_train_prepared, y_train,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

print("Scores:", tree_rmse_scores)
print("Mean:", tree_rmse_scores.mean())
print("Standard deviation:", tree_rmse_scores.std())


<<<<<<< HEAD
"""Mode	LR	DT	RF	SVM(l)	SVM(n)
mean	69104	71629	50266	70158	96268
std	2880.33	2914.03	2312	2004.15	1960
"""
=======
#Evaluating the model with Mean absolute error for rbf
absolute_prediction_error = mean_absolute_error(y_train, nonlinear_predictions)
print(absolute_prediction_error)

#Evaluating the model with Mean absolute error for linear
absolute_prediction_error = mean_absolute_error(y_train, lin_predictions)
print(absolute_prediction_error)

#Evaluating the model with Mean absolute error for polynomial
absolute_prediction_error = mean_absolute_error(y_train, poly_predictions)
print(absolute_prediction_error)

>>>>>>> 0c63b40d7f9c3fcc698267e3a067b7c67c05e69e


# Data (replace with your actual numbers)
models = ["LR", "DT", "SVM-l", "SVM-n", "RF", "SVM_p"]
means = [69104, 71629, 70158, 96268, 50266, 107249]
stds = [2880, 2914, 2004, 1960, 2312, 8355]

x = np.arange(len(models))

plt.figure(figsize=(8,5))
plt.bar(x, means, yerr=stds, capsize=7, color=["skyblue","orange","green","red","purple", "hotpink"])
plt.xticks(x, models)
plt.ylabel("RMSE")
plt.title("Comparison of Models (Mean RMSE ± Std)")
plt.show()