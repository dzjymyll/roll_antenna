import os
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pykrige.rk import RegressionKriging
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from RBF import RBF,RBFLayer
from keras.losses import binary_crossentropy
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# read data as array
def data_process(prj_path):

    data = pd.read_csv(prj_path ,header=None,sep=',')
    data = np.array(data)
    print(data.shape)
    X = data[:,:7]
    y = data[:,7:]
    scaler = StandardScaler()
    scaler.fit(X)
    #X = scaler.transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    data_path = r"C:\Users\yuqixiao2\Desktop\0_Phasecenter_data_Theta_plus-minus_90_degree"
    np.savetxt(os.path.join(data_path,"x_train.csv"), x_train, delimiter=",")
    np.savetxt(os.path.join(data_path,"x_test.csv"), x_test, delimiter=",")
    np.savetxt(os.path.join(data_path,"y_train.csv"), y_train, delimiter=",")
    np.savetxt(os.path.join(data_path,"y_test.csv"), y_test, delimiter=",")
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    np.save('x_test.npy',x_test)
    np.save('x_train',x_train)
    np.save('y_test.npy',y_test)
    np.save('y_train.npy',y_train)
    return x_train, x_test, y_train, y_test

    #return data

def svr_construct():
    svr_model = SVR(C=0.1, gamma="auto")
    svr_mor = MultiOutputRegressor(svr_model)

    return svr_mor

def rf_construct():
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_mor = MultiOutputRegressor(rf_model)

    return rf_mor

def rf_construct():
    lr_model = LinearRegression(normalize=True, copy_X=True, fit_intercept=False)
    lr_mor = MultiOutputRegressor(lr_model)
    return lr_mor

def nn_construct():
    model = Sequential()
    model.add(Dense(13, input_dim=7, kernel_initializer='normal', activation='relu'))
    #model.add(RBFLayer(100, 1))
    #model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(5, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    #model.compile(optimizer='rmsprop', loss=binary_crossentropy)
    return model

def mlp_construct():
    model_mlp = MLPRegressor(hidden_layer_sizes=(500,100),activation='relu',solver='adam',max_iter=100000)
    return model_mlp

def rbf_construct():
    rbf = RBF(7, 50, 5)
    #rbf.train(x, y)
    #z = rbf.test(x)
    return rbf


def result_evaluation(y_test,y_pred,model_name):
    mae_one = mean_absolute_error(y_test[:,0],y_pred[:,0])
    mae_two = mean_absolute_error(y_test[:,1],y_pred[:,1])
    mae_three = mean_absolute_error(y_test[:,2],y_pred[:,2])
    mae_four = mean_absolute_error(y_test[:,3],y_pred[:,3])
    mae_five = mean_absolute_error(y_test[:,4],y_pred[:,4])
    print(f'{model_name} MAE for first regressor: {mae_one} - second regressor: {mae_two} - third regressor: {mae_three} - forth regressor: {mae_four} - fifth regressor: {mae_five}')

if __name__ == "__main__":
   models = ["svr_model", "rf_model", "lr_model","neural_network","mlp","rbf"]
   prj_path = r"C:\Users\yuqixiao2\Desktop\0_Phasecenter_data_Theta_plus-minus_90_degree\whole.csv"
   x_train, x_test, y_train, y_test = data_process(prj_path)
   '''
   data = data_process(prj_path)
   x_train = data[:2000,:7]
   x_test = data[2000:,:7]
   y_train = data[:2000,7:]
   y_test = data[2000:,7:]
   '''
   print(x_train.shape,y_train.shape)
   svr_mor = svr_construct()
   rf_mor = rf_construct()
   lr_mor = rf_construct()
   nn = nn_construct()
   mlp = mlp_construct()
   rbf = rbf_construct()

   svr_mor.fit(x_train,y_train)
   rf_mor.fit(x_train,y_train)
   lr_mor.fit(x_train,y_train)
   nn.fit(x_train,y_train,batch_size = 10, epochs = 200)
   mlp.fit(x_train,y_train)
   rbf.train(x_train,y_train)
   svr_pred = svr_mor.predict(x_test)
   rf_pred = rf_mor.predict(x_test)
   lr_pred = lr_mor.predict(x_test)
   nn_pred = nn.predict(x_test)
   mlp_pred = mlp.predict(x_test)
   rbf_pred = rbf.test(x_test)

   results = [svr_pred,rf_pred, lr_pred,nn_pred,mlp_pred,rbf_pred]
   print(y_test.mean(axis=0))
   for i in range(len(models)):
       result_evaluation(y_test,results[i],models[i])




'''
for m in models:
    print("=" * 40)
    print("regression model:", m.__class__.__name__)
    m_rk = RegressionKriging(regression_model=m, n_closest_points=10)
    m_rk.fit(p_train, x_train, target_train)
    print("Regression Score: ", m_rk.regression_model.score(p_test, target_test))
    print("RK score: ", m_rk.score(p_test, x_test, target_test))
'''
