"""
Train the regressor and plot the results
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from linear_model import LinearRegressionGD

def train_reg(df):
	X = df[['RM']].values
	y = df['MEDV'].values
	sc_x = StandardScaler()
	sc_y = StandardScaler()
	X_std = sc_x.fit_transform(X)
	y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
	lr = LinearRegressionGD()
	lr.fit(X_std, y_std)
	return lr

def cost_against_epochs(lr):
	plt.plot(range(1, lr.n_iter+1), lr.cost_)
	plt.ylabel('SSE')
	plt.xlabel('Epoch')
	# plt.show()
	plt.savefig('./figures/cost-against-epoch.png')
	plt.gcf().clear()

def lin_regplot(df, model):
	X = df[['RM']].values
	y = df['MEDV'].values
	sc_x = StandardScaler()
	sc_y = StandardScaler()
	X_std = sc_x.fit_transform(X)
	y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

	plt.scatter(X_std, y_std, c='blue')
	plt.plot(X_std, model.predict(X_std), color='red')
	plt.xlabel('Average number of rooms [RM] (standardized)')
	plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
	# plt.show()

	if not os.path.exists(os.path.join(os.getcwd(), 'figures')):
		os.mkdir('figures')

	plt.savefig('./figures/plotting-linear-reg.png')