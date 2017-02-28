def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

def plot_ROC(actual, predictions):
	# plot the FPR vs TPR and AUC for a two class problem (0,1)
	import matplotlib.pyplot as plt
	from sklearn.metrics import roc_curve, auc

	false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
	roc_auc = auc(false_positive_rate, true_positive_rate)
	
	plt.title('Receiver Operating Characteristic')
	plt.plot(false_positive_rate, true_positive_rate, 'b',
	label='AUC = %0.2f'% roc_auc)
	plt.legend(loc='lower right')
	plt.plot([0,1],[0,1],'r--')
	plt.xlim([-0.1,1.2])
	plt.ylim([-0.1,1.2])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.show()