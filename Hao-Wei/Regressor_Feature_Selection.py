# from dictionaries import *;
import pandas as pd;
import numpy as np;
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.linear_model import Lasso

# Utility functions which sends from front end to back end
def front_to_back(fe, method, x_scaler, refit=False):
    '''
    Description:
    This is the function which construct the backend data (which directly feeds to CatBoostRegressor)
    given the frontend data.
    Input:
    fe: The frontend dataframe. Columns must be the same as those in version 6 of the housing data.
    method: The string which describes the regressor. Default = cat.
    Compatible values: "cat": CatBoostRegressor; "lm": Multilinear method (with lasso penalization)
    "svrl": Support vector regressor with linear kernel
    Output: The backend dataframe. Should be ready to "regressor.fit()".
    '''

    be = fe.copy();

    if method == "cat":
        be['ExterQualDisc']=be['ExterQual']-be['OverallQual'];
        be['OverallCondDisc']=be['OverallCond']-be['OverallQual'];
        be['KitchenQualDisc']=be['KitchenQual']-be['OverallQual'];
        be=be.drop(["SalePrice", 'ExterQual','OverallCond','KitchenQual'],axis=1);

        be = dummify(be, non_dummies, dummies);
    elif method in ["lm", "svrl"]:
        be.drop(columns = ['SalePrice'], axis =1, inplace = True);
        be['GrLivArea_log'] = np.log10(be['GrLivArea']);
        be['LotArea_log'] = np.log10(be['LotArea']);
        be['ExterQualDisc'] = be['ExterQual'] - be['OverallQual'];
        be['OverallCondDisc'] = be['OverallCond'] - be['OverallQual'];
        be['KitchenQualDisc'] = be['KitchenQual'] - be['OverallQual'];
        be = be.drop(['ExterQual','OverallCond','KitchenQual'], axis=1);

        be['BSMT_LowQual_bin'] = pd.cut(be['BSMT_LowQual'], [-1, 1, 500, 1000, 1500, 2500], labels = ['No basement', '0-500', '500-1000', '1000-1500', '1500+']);
        be['BSMT_HighQual_bin'] = pd.cut(be['BSMT_HighQual'], [-1, 1, 500, 1000, 1500, 2500], labels = ['No basement', '0-500', '500-1000', '1000-1500', '1500+']);
        be.drop(['BSMT_HighQual', 'BSMT_LowQual', 'GrLivArea', 'LotArea'], axis = 1, inplace = True);

        be = dummify(be, non_dummies_linear, dummies_linear);

        if method == "svrl":
            if refit: be = pd.DataFrame(x_scaler.fit_transform(be), columns = be.columns);
            else: be = pd.DataFrame(x_scaler.transform(be), columns = be.columns);
    elif method == "svrg":
        be = fe.copy();
        be.drop(columns = ['SalePrice'], axis =1, inplace = True);
        be['ExterQualDisc']=be['ExterQual']-be['OverallQual'];
        be['OverallCondDisc']=be['OverallCond']-be['OverallQual'];
        be['KitchenQualDisc']=be['KitchenQual']-be['OverallQual'];
        be=be.drop(['ExterQual','OverallCond','KitchenQual'],axis=1);

        be = dummify(be, non_dummies, dummies);

        be['GrLivArea_log'] = np.log10(be['GrLivArea']);
        be['LotArea_log'] = np.log10(be['LotArea']);
        be.drop(['GrLivArea', 'LotArea'], axis = 1, inplace = True);

        if refit: be = pd.DataFrame(x_scaler.fit_transform(be), columns = be.columns);
        else: be = pd.DataFrame(x_scaler.transform(be), columns = be.columns);

    return be;

def predictor_processing(y, method, y_scaler):
    if method == "cat": return y;
    if method == "lm": return np.log10(y);
    if method in ["svrl", "svrg"]: return y_scaler.fit_transform(np.array(np.log10(y)).reshape(-1,1)).ravel();

def predict_from_front(fe, method, instance, x_scaler, y_scaler):
    '''
    Description:
    Given a frontend data frame, and a string describing a regressor, predicts.
    Input:
    fe: The frontend dataframe. Columns must be the same as those in version 6 of the housing data.
    method: The string which describes the regressor. Default = "cat". Compatible values: See above.
    Output: A pd.Series predicting the output.
    '''
    if method == "cat":
        return instance.predict(front_to_back(fe, method, x_scaler));
    elif method == "lm": # The method is log scaled. Needs an exponentiation
        return 10 ** instance.predict(front_to_back(fe, method, x_scaler));
    else: # The method is log scaled then standardized.
        return 10 ** (y_scaler.inverse_transform(instance.predict(front_to_back(fe, method, x_scaler))));
    return;


### The Encapsulation class
# from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
class EncapsulatedModel(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, method="lin", instance = CatBoostRegressor(), feature = []):
        self.method = method;
        self.instance = instance;
        self.x_scaler = StandardScaler();
        self.y_scaler = StandardScaler();
        self.feature = feature; # This determines the feautures we are going to use.
        self.fitted = False;
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        back_end = front_to_back(X, self.method, self.x_scaler, True);
        y_proc = predictor_processing(y, self.method, self.y_scaler);
        self.instance.fit(back_end, y_proc);
        self.fitted = True;
        return self;
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        return predict_from_front(X, self.method, self.instance, self.x_scaler, self.y_scaler);