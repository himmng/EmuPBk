import keras as ks
from data.data_Powerspectrum.data_read import pk,params
from sklearn.model_selection import train_test_split

'''Splitting the data_Powerspectrum for the training and testing ANN'''

pk_train, pk_test , params_train, params_test = train_test_split(pk , params, shuffle=True, test_size=0.01)

'''securing the indices of the test data_Powerspectrum for MCMC analysis'''
index = np.argwhere(params = params_test)

'''=============================== ANN Structure=======================================

*   The ANN contains input layer + 6 hidden layers + output layer
*   The number of neurons in input and output layer are 3 & 7 respectively,
 these are corresponding to our 3 EoR parameters and  7 Powerspectrum values.
* The hidden layers have 3,300,600,300,150,70 neurons respectively.
* Exponential Linear Unit (elu) has been used as activation funtion.
'''
model = ks.models.Sequential({ks.layers.Dense(units=3, input_shape=[3, ], activation='elu'),
                              ks.layers.Dense(units=300, activation='relu'),
                              ks.layers.Dense(units=600, activation='elu'),
                              ks.layers.Dense(units=300, activation='elu'),
                              ks.layers.Dense(units=150, activation='elu'),
                              ks.layers.Dense(units=70, activation='relu'),
                              ks.layers.Dense(units=7)})


'''
* Mean Square Error (mse) as the loss function.
* ADAM as optimizer.
'''
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(params_train, pk_train, epochs=1000, batch_size=20,)

model.save('EMuPk22.h5')                         # Saving the ANN model.




