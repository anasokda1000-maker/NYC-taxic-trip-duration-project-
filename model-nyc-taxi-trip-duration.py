import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures, RobustScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBRegressor
import argparse
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()

parser.add_argument('--traindata', type=str, default="D:/project-nyc-taxi-trip-duration/feature engineered split/train.csv")
parser.add_argument('--valdata', type=str, default="D:/project-nyc-taxi-trip-duration/feature engineered split/val.csv")
parser.add_argument('--testdata', type=str, default="D:/project-nyc-taxi-trip-duration/feature engineered split/test.csv")

parser.add_argument('--preprocessing', type = int ,default = 2, help = '0 for no processing, 1 for min/max scaling,'
                                                                       ' 2 for StandardScaling and 3 for RobustScaler')

parser.add_argument('--lassoSelection', type=int , default = 1, help = '0 for no lasso selection')

parser.add_argument('--polynomialCross', type=int, default = 1, help = '0 for no polynomailFeatures')

parser.add_argument('--model', type=int, default =0, help = '0 for Ridge(alpha=1) model'
                                                                         '1 for xgboost model')

args = parser.parse_args()


def RobustScaling(training,validating,testing):
    processor = RobustScaler()
    strain_xRS = processor.fit_transform(training)
    sval_xRS = processor.transform(validating)
    stest_xRS = processor.transform(testing)
    return strain_xRS, sval_xRS, stest_xRS

def minMaxScaling(training,validating,testing):
    processor = MinMaxScaler()
    strain_x = processor.fit_transform(training)
    sval_x = processor.transform(validating)
    stest_x = processor.transform(testing)
    return strain_x, sval_x, stest_x

def standardScaling(training,validating,testing):
    processor = StandardScaler()
    strain_xsT = processor.fit_transform(training)
    sval_xsT = processor.transform(validating)
    stest_xsT = processor.transform(testing)
    return strain_xsT, sval_xsT, stest_xsT

def poly2(training,validating,testing):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    train_x_poly = poly.fit_transform(training)
    val_x_poly = poly.transform(validating)
    test_x_poly = poly.transform(testing)

    return train_x_poly, val_x_poly, test_x_poly

def lassoSelection(training,validating,trainT,valT):
    alphas = [.01,.01,.1,.2,.3,.4,.5,.6,.7,.8,.9,1,10]
    r2_scores = []
    masks = []

    for alpha in alphas:
        model = Lasso(fit_intercept=True, alpha=alpha, max_iter=1000,random_state=42)
        model.fit(training, trainT)

        selector = SelectFromModel(model, threshold=None, prefit=True)

        strain_xS = selector.transform(training)
        sval_xS = selector.transform(validating)

        if strain_xS.shape[1] == 0 :
            r2_scores.append(0)
            masks.append(0)
            continue

        model2 = Lasso(fit_intercept=True, alpha=alpha, max_iter=1000)
        model2.fit(strain_xS, trainT)

        predV = model2.predict(sval_xS)

        R2_score = r2_score(valT, predV)

        r2_scores.append(R2_score)
        masks.append(selector.get_support(indices=True))

    best_indices = masks[r2_scores.index(max(r2_scores))]

    strain_xB22 = training[:,best_indices]
    sval_xB22 = validating[:,best_indices]

    return strain_xB22, sval_xB22,best_indices

def crossRidge(training,validating,trainT,valT):
    hamada = {}
    hamada['alpha'] = np.array([1])
    hamada['fit_intercept'] = np.array([True,False])

    joe = GridSearchCV(Ridge(random_state=42),hamada,scoring='r2',cv=10)
    joe.fit(training,trainT)
    # print('All fold R²:',joe.cv_results_['mean_test_score'])
    # print('best CV R²:',joe.best_score_)

    model = Ridge(**joe.best_params_)
    model.fit(training,trainT)
    pred_val=model.predict(validating)

    r2_score_val = r2_score(valT, pred_val)

    return joe.best_params_, r2_score_val

if __name__ == "__main__":
    '''Reading the data'''
    train = pd.read_csv(args.traindata)
    val = pd.read_csv(args.valdata)
    test = pd.read_csv(args.testdata)
    '''separating features and targets'''
    train_x = train.iloc[:, :-1]
    train_t = train.iloc[:,-1]

    val_x = val.iloc[:, :-1]
    val_t = val.iloc[:, -1]

    test_x = test.iloc[:, :-1]
    test_t = test.iloc[:, -1]


    if args.model == 0 :
        '''Ridge(alpha=1) model'''

        '''different preprocessing'''
        if args.preprocessing == 1 :
            strain_x, sval_x, stest_x = minMaxScaling(train_x, val_x, test_x)
        elif args.preprocessing == 2 :
            strain_x, sval_x, stest_x = standardScaling(train_x, val_x, test_x)
        elif args.preprocessing == 0 :
            strain_x, sval_x, stest_x = train_x.copy(), val_x.copy(), test_x.copy()
        elif args.preprocessing == 3 :
            strain_x, sval_x, stest_x = RobustScaling(train_x, val_x, test_x)
        else :
            strain_x, sval_x, stest_x = minMaxScaling(train_x, val_x, test_x)


        '''polynomialFeaturing with degree(2)'''
        if args.polynomialCross == 0 :
            train_xp, val_xp, test_xp = strain_x.copy(), sval_x.copy(), stest_x.copy()
        else :
            train_xp, val_xp, test_xp = poly2(strain_x, sval_x, stest_x)

        '''selecting features with lasso for simplicity'''
        if args.lassoSelection == 0 :
            train_xB, val_xB,test_xB = train_xp.copy(), val_xp.copy(), test_xp.copy()
        else :
            train_xB, val_xB , best_indices= lassoSelection(train_xp,val_xp,train_t,val_t)
            test_xB = test_xp[:,best_indices]

        '''Ride model with and tuning parameters with GridSeachCV'''
        bestparams, r2_score_val = crossRidge(train_xB,val_xB,train_t,val_t)

        '''Ridge(alpha=1) and tuned intercept'''
        model = Ridge(**bestparams)
        model.fit(train_xB, train_t)
        test_prediction = model.predict(test_xB)

        R2_test = r2_score(test_t, test_prediction)
        print('Train\n', f"R² = {r2_score_val:.4f}")
        print('Test\n',f"R² = {R2_test:.4f}")

    elif args.model == 1 :
        '''xgboost model'''

        train_xs, val_xs, test_xs = RobustScaling(train_x,val_x,test_x)

        xgb_model = XGBRegressor(n_estimators=2000,
                                 max_depth=7,
                                 learning_rate=0.02,
                                 subsample=0.85,
                                 colsample_bytree=0.8,
                                 min_child_weight=4,
                                 reg_alpha=0.1,
                                 reg_lambda=1.2,
                                 random_state=42,
                                 tree_method='hist',
                                 early_stopping_rounds=100)

        xgb_model.fit(train_xs, train_t, eval_set=[(val_xs, val_t)], verbose=100)

        val_pred = xgb_model.predict(val_xs)
        test_pred = xgb_model.predict(test_xs)

        print(f"Train R²   : {r2_score(val_t, val_pred):.5f}")
        print(f"Test R²  : {r2_score(test_t, test_pred):.5f}")


