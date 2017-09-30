# Loading the data and visualizing it.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
	df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
			header=None, sep='\s+')
	df.columns = ['CRIM','ZN','INDUS',
			'CHAS','NOX','RM',
			'AGE','DIS','RAD',
			'TAX','PTRATIO','B',
			'LSTAT','MEDV']
	return df	

def visualize_scatterplot(df):
	sns.set(style='whitegrid', context='notebook')
	cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
	sns.pairplot(df[cols], size=2.5);
	# plt.show()
	if not os.path.exists(os.path.join(os.getcwd(), 'figures')):
		os.mkdir('figures')
	
	plt.savefig('./figures/scatterplot.png')
	plt.gcf().clear()		# to ensure the canvas is clear for next plot

def corr_heatmap(df):
	cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
	cm = np.corrcoef(df[cols].values.T)
	sns.set(font_scale=1.5)
	hm = sns.heatmap(cm,
			cbar=True,
			annot=True,
			square=True,
			fmt='.2f',
			annot_kws={'size': 15},
			yticklabels=cols,
			xticklabels=cols)
	# plt.show()
	plt.savefig('./figures/correlation-heatmap.png')
	plt.gcf().clear()
