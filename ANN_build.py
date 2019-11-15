import keras as ks
import numpy as np
from data import pk,params,Nk_bins
from sklearn.model_selection import train_test_split


pk_train ,pk_test , params_train,params_test = train_test_split(pk,params, shuffle=True,test_size=0.8)


model = ks.models.Sequential({ks.layers.Dense(units=3, input_shape=[3, ], activation='elu'),
                              ks.layers.Dense(units=300, activation='relu'),
                             # ks.layers.Dense(units=300, activation='elu'),
                              ks.layers.Dense(units=600, activation='elu'),
                             # ks.layers.Dense(units=900, activation='elu'),
                              ks.layers.Dense(units=300, activation='elu'),
                              ks.layers.Dense(units=150, activation='elu'),
                              ks.layers.Dense(units=70, activation='relu'),
                              # ks.layers.Dense(units = 300, activation='elu'),

                              ks.layers.Dense(units=7)})

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(params_train, pk_train, epochs=1000, batch_size=20,)

plt.subplot(312)
plt.ylabel('freq.')
plt.xlabel('$R_{mfp}$')
plt.hist(R_mfp,bins=100,color='orange')
plt.axvline(R_mfp.mean(),color='green',lw=2)
plt.subplot(313)
plt.xlabel('$Mh_{min}$')
plt.ylabel('$freq$')

plt.hist(Mh,bins = 100,color='blue',alpha=0.5)
a=[]
a = np.c_[n_ion,R_mfp,Mh]
fig = corner.corner(a,bins=55,color='b',weights = -logL,labels = ['$\zeta$','$R_{mfp}$','$Mh_{min}$'])
fig.suptitle("$\zeta$ $h$,$R_{mfp}$,$Mh_{min}$  ")
fig.savefig('corner_plot.png')

data = pd.DataFrame({'$\zeta$':n_ion,'$R_{mfp}$':R_mfp,'$Mh_{min}$':Mh})
#print(data['$h$'].describe(),data['$\Omega_m$'].describe())



