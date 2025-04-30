from scipy.stats import kde
import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf


class Visualizer():
    def __init__(self, analysis_files=[], figsize=None, nbins=40, cmap=plt.cm.jet):
        self.analysis_files = analysis_files
        self.figsize = figsize
        self.nbins = nbins
        self.cmap = cmap
        
        
    def make_visualization(self,lows):
        x, y = lows[:,0],lows[:,1]
        
        # Create a figure with 3 columns
        ncols=3
        nrows=math.ceil(len(self.analysis_files)/ncols)+1
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=self.figsize, sharex=True, sharey=True, squeeze=False)

        # Everything starts with a Scatterplot
        axes[0][0].set_title('Scatterplot')
        axes[0][0].scatter(x, y, s=0.1, cmap=self.cmap)

        # 2D Histogram
        hist2d = axes[0][1].set_title('2D Histogram')
        hist2d = axes[0][1].hist2d(x, y, bins=self.nbins, cmap=self.cmap)
        fig.colorbar(hist2d[3], ax=axes[0][2])

        # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
        k = kde.gaussian_kde((x,y))
        xi, yi = np.mgrid[x.min():x.max():self.nbins*1j, y.min():y.max():self.nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        # Gaussian kde with shading
        axes[0][2].set_title('2D Density with shading')
        axes[0][2].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=self.cmap)

        # Plot graphs based on specified analysis files
        if len(self.analysis_files) > 0:
            _analysis_files = []
            for file in self.analysis_files:
                data = np.loadtxt(file)
                _analysis_files.append((file, data))
                
            for row_i in range(nrows):
                for col_i in range(ncols):
                    file_ndx = (row_i*ncols)+col_i
                    if file_ndx >= len(_analysis_files):
                        break
                    axes[row_i+1][col_i].set_title(_analysis_files[file_ndx][0])
                    axes[row_i+1][col_i].scatter(x, y, s=0.1, c=_analysis_files[file_ndx][1])
                    

class VisualizeCallback(tf.keras.callbacks.Callback):
	def __init__(self,model=None,visualizer=None,freq=10,inputs=None,**kwargs):
		super().__init__()
		assert model
		self.mymodel = model
		self.inputs = inputs
		self.freq = freq
		if visualizer:
			self.visualizer = visualizer
		else:
			self.visualizer = Visualizer(**kwargs)

	def on_epoch_end(self,epoch,logs=None):
		if epoch % self.freq == self.freq - 1:
			lows = self.mymodel.call_enc(self.inputs).numpy()
			self.visualizer.make_visualization(lows)
			plt.pause(.01)

