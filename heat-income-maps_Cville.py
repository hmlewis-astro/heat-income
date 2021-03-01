import os
import glob

path = os.getcwd()

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox, TransformedBbox, blended_transform_factory
from mpl_toolkits.axes_grid1.inset_locator import BboxPatch, BboxConnector, BboxConnectorPatch
import numpy as np

plt.rcParams["font.sans-serif"] = "Tahoma"

def plot_grad(df_param, cmap, ax):

	'''Create a gradient colorbar
	
    df_param: pandas series
        Parameter to set the min/max values for the colorbar
    cmap: string
        Colormap to use in the colorbar
    ax: matplotlib.axes
		Axis to attach the colorbar to
    '''
	x_1, y_1 = (np.min(df_param),1.0)
	x_2, y_2 = (np.max(df_param),1.0)
    
	X = np.linspace(x_1, x_2, len(df_param))
	Xs = X[:-1]
	Xf = X[1:]
	Xpairs = zip(Xs, Xf)
    
	Y = np.linspace(y_1, y_2, len(df_param))
	Ys = Y[:-1]
	Yf = Y[1:]
	Ypairs = zip(Ys, Yf)
	
	C = np.linspace(0., 1., len(df_param))
	cmap = plt.get_cmap(cmap)
    
	for x, y, c in zip(Xpairs, Ypairs, C):
		ax.plot(x, y, c=cmap(c), linewidth=20.0)

def zoom_effect_box(source, dest, roi, color='k', linewidth=1.25, roiKwargs={}, arrowKwargs={}):

    '''Create a zoomed subplot outside the original subplot
    
    ax1: matplotlib.axes
        Source axis where locates the original chart
    ax2: matplotlib.axes
        Destination axis in which the zoomed chart will be plotted
    roi: list
        Region Of Interest is a rectangle defined by [xmin, ymin, xmax, ymax],
        all coordinates are expressed in the coordinate system of data
    roiKwargs: dict (optional)
        Properties for matplotlib.patches.Rectangle given by keywords
    arrowKwargs: dict (optional)
        Properties used to draw a FancyArrowPatch arrow in annotation
    '''
    roiKwargs = dict([("fill", False), ("linestyle", "-"),
                      ("color", color), ("linewidth", linewidth)]
                     + list(roiKwargs.items()))
    arrowKwargs = dict([("arrowstyle", "-"), ("color", color),
                        ("linewidth", linewidth)]
                       + list(arrowKwargs.items()))
    # Draw a rectangle on original chart
    source.add_patch(Rectangle([roi[0], roi[1]], roi[2]-roi[0], roi[3]-roi[1],
                            **roiKwargs))
                            
    dest.add_patch(Rectangle([roi[0]+0.0001, roi[1]+0.0001], roi[2]-roi[0]-0.00005, roi[3]-roi[1]-0.00005,
                            **roiKwargs))
                            
    # Get coordinates of corners
    srcCorners = [[roi[0], roi[1]], [roi[0], roi[3]],
                  [roi[2], roi[1]], [roi[2], roi[3]]]
    dstCorners = dest.get_position().corners()
    srcBB = source.get_position()
    dstBB = dest.get_position()

    # Find corners to be linked
    if (dstBB.min[0]>srcBB.max[0] and dstBB.max[1]<srcBB.min[1]) or (dstBB.max[0]<srcBB.min[0] and dstBB.min[1]>srcBB.max[1]):
        src = [0, 3]; dst = [0, 3]
    elif (dstBB.max[0]<srcBB.min[0] and dstBB.max[1]<srcBB.min[1]) or (dstBB.min[0]>srcBB.max[0] and dstBB.min[1]>srcBB.max[1]):
        src = [1, 2]; dst = [1, 2]
    elif dstBB.max[1] < srcBB.min[1]:
        src = [0, 2]; dst = [1, 3]
    elif dstBB.min[1] > srcBB.max[1]:
        src = [1, 3]; dst = [0, 2]
    elif dstBB.max[0] < srcBB.min[0]:
        src = [0, 1]; dst = [2, 3]
    elif dstBB.min[0] > srcBB.max[0]:
        src = [2, 3]; dst = [0, 1]
        
    # Plot 2 lines to link the region of interest and the zoomed chart
    for k in range(2):
        source.annotate('', xy=dstCorners[dst[k]], xytext=srcCorners[src[k]],
                     xycoords='figure fraction', textcoords='data', arrowprops=arrowKwargs)


maps_path = path + '/data/output/analysis_out/final/'
plot_path = path + '/data/output/analysis_out/final/plots/'
file_extn = '.geojson'

assert os.path.exists(maps_path), "Directory containing the final .geojson files missing!"

if not os.path.exists(plot_path):
    os.makedirs(plot_path)
    
# Get data for Charlottesville and Albemarle County
cville_map_files = glob.glob(maps_path + 'charlottesville' + file_extn)[0]
cville_landmark_files = glob.glob(maps_path + 'charlottesville' + '_landmarks.csv')[0]
albemr_map_files = glob.glob(maps_path + 'albemarle' + file_extn)[0]
albemr_landmark_files = glob.glob(maps_path + 'albemarle' + '_landmarks.csv')[0]


map_files = [cville_map_files, albemr_map_files]
landmark_files = [cville_landmark_files, albemr_landmark_files]

# Make plots for Charlottesville and Albemarle County separately
for i,m in enumerate(map_files):

    # Get map data
	df_map = gpd.read_file(m)
	df_landmarks = pd.read_csv(landmark_files[i], comment='#')
    
    # Format data of interest
	df_map['_median'] = df_map['_median'].astype(np.float64) # Median temp in census tract
	df_map['median_hou'] = df_map['median_hou'].astype(np.float64) # Median household income in census tract
	df_map['total_popu'] = df_map['total_popu'].astype(np.float64) # Total population in census tract
	df_map['white_popu'] = df_map['white_popu'].astype(np.float64) # White population in census tract
	df_map[df_map['median_hou'] < 0] = np.nan
    
    # Make plot
	fig, ax = plt.subplots(2,3, figsize=(20,10), gridspec_kw={'height_ratios': [20,1]})

    #Plot median temp
	df_map.plot(column=df_map._median,
                cmap='Spectral_r',
                ax=ax[0,0])
	ax[0,0].axis('off')
	ax[0,0].set_title('Surface Temperature', fontsize=24)

    # Colorbar and labels
	plot_grad(df_map._median, 'Spectral_r', ax[1,0])
	ax[1,0].axis('off')
	ax[1,0].text(np.min(df_map._median),0.75,
             '{0:.1f}$^\circ$\nbelow avg.'.format(np.min(df_map._median)-np.mean(df_map._median)),
             ha='center', fontsize=18)
	ax[1,0].text(np.max(df_map._median),0.75,
             '{0:.1f}$^\circ$\nabove avg.'.format(np.max(df_map._median)-np.mean(df_map._median)),
             ha='center', fontsize=18)

    # Plot household income
	df_map.plot(column=df_map.median_hou,
                  cmap='Greens',
                  ax=ax[0,1])
	df_landmarks.plot.scatter(x='long', y='lat', ax=ax[0,1], s=15, c='k')
	for j,lab in enumerate(df_landmarks.landmark):
		ax[0,1].annotate(lab,
						 xy=(df_landmarks.long[j], df_landmarks.lat[j]),
						 xytext=(df_landmarks.long[j],df_landmarks.lat[j]+0.00001),
						 ha='center', va='bottom',
						 fontsize=12)
	ax[0,1].axis('off')
	ax[0,1].set_title('Income', fontsize=24)

    # Colorbar and labels
	plot_grad(df_map.median_hou, 'Greens', ax[1,1])
	ax[1,1].axis('off')
	ax[1,1].text(np.min(df_map.median_hou),0.75,
             'Minimum\n\${0:.0f}K'.format(np.min(df_map.median_hou)/1000.0),
             ha='center', fontsize=18)
	ax[1,1].text(np.max(df_map.median_hou),0.75,
             'Maximum\n\${0:.0f}K'.format(np.max(df_map.median_hou)/1000.0),
             ha='center', fontsize=18)

    # Plot fraction of population
	vmax = round(np.max(1.0-(df_map.white_popu/df_map.total_popu)),1)
	df_map.plot(column=1.0-(df_map.white_popu/df_map.total_popu),
                  cmap='Blues', vmin=0.0, vmax=vmax,
                  ax=ax[0,2])
	ax[0,2].axis('off')
	ax[0,2].set_title('Percent Non-White Population', fontsize=24)

    # Colorbar and labels
	plot_grad(df_map.white_popu/df_map.total_popu, 'Blues', ax[1,2])
	ax[1,2].axis('off')
	ax[1,2].text(np.min(df_map.white_popu/df_map.total_popu),0.775,
             '0%',
             ha='center', fontsize=18)
	ax[1,2].text(np.max(df_map.white_popu/df_map.total_popu),0.775,
             '{:.0f}%'.format(vmax*100),
             ha='center', fontsize=18)
    
	fig.subplots_adjust(hspace=0, wspace=0.15)

    # Save figure
	plt.savefig(plot_path + '/' + m[len(maps_path):-len(file_extn)] + '.pdf', dpi=300)
	print('Plot for {} done!'.format(m[len(maps_path):-len(file_extn)]))
    
	plt.close()

# Aggregate Charlottesville and Albemarle County data
df_map_cville = gpd.read_file(cville_map_files)
df_map_albemr = gpd.read_file(albemr_map_files)
df_map_total = df_map_cville.append(df_map_albemr)

df_landmarks_cville = pd.read_csv(cville_landmark_files, comment='#')
df_landmarks_albemr = pd.read_csv(albemr_landmark_files, comment='#')

# Format data of interest
df_map_total['_median'] = df_map_total['_median'].astype(np.float64) # Median temp in census tract
df_map_total['median_hou'] = df_map_total['median_hou'].astype(np.float64) # Median household income in census tract
df_map_total['total_popu'] = df_map_total['total_popu'].astype(np.float64) # Total population in census tract
df_map_total['white_popu'] = df_map_total['white_popu'].astype(np.float64) # White population in census tract
df_map_total[df_map_total['median_hou'] < 0] = np.nan
    
# Make single plot for Charlottesville and Albemarle County data
fig, ax = plt.subplots(3,3, figsize=(20,10), gridspec_kw={'height_ratios': [20,20,2]})

#Plot median temp
df_map_total.plot(column=df_map_total._median,
            cmap='Spectral_r',
            ax=ax[0,0])
ax[0,0].axis('off')
ax[0,0].set_title('Surface Temperature', fontsize=24)

df_map_total.plot(column=df_map_total._median,
            cmap='Spectral_r',
            ax=ax[1,0])
ax[1,0].set_xlim(-78.54,-78.43)
ax[1,0].set_ylim(37.99,38.09)
ax[1,0].axis('off')

# Colorbar and labels
plot_grad(df_map_total._median, 'Spectral_r', ax[2,0])
ax[2,0].axis('off')
ax[2,0].text(np.min(df_map_total._median),0.75,
             '{0:.1f}$^\circ$\nbelow avg.'.format(np.min(df_map_total._median)-np.mean(df_map_total._median)),
             ha='center', fontsize=18)
ax[2,0].text(np.max(df_map_total._median),0.75,
             '{0:.1f}$^\circ$\nabove avg.'.format(np.max(df_map_total._median)-np.mean(df_map_total._median)),
             ha='center', fontsize=18)

zoom_effect_box(ax[0,0], ax[1,0], [-78.54, 37.99, -78.43, 38.09])

# Plot household income
df_map_total.plot(column=df_map_total.median_hou,
            cmap='Greens',
            ax=ax[0,1])
df_landmarks_albemr.plot.scatter(x='long', y='lat', ax=ax[0,1], s=15, c='k')
for j,lab in enumerate(df_landmarks_albemr.landmark):
	if j % 4 in [0,1]:
		offset = 0.00002
		va = 'bottom'
	else:
		offset = -0.00004
		va = 'top'
	ax[0,1].annotate(lab,
					 xy=(df_landmarks_albemr.long[j], df_landmarks_albemr.lat[j]),
					 xytext=(df_landmarks_albemr.long[j],df_landmarks_albemr.lat[j]+offset),
					 ha='center', va=va,
					 fontsize=12)
ax[0,1].axis('off')
ax[0,1].set_title('Income', fontsize=24)

df_map_total.plot(column=df_map_total.median_hou,
            cmap='Greens',
            ax=ax[1,1])
df_landmarks_cville.plot.scatter(x='long', y='lat', ax=ax[1,1], s=15, c='k')
for j,lab in enumerate(df_landmarks_cville.landmark):
	ax[1,1].annotate(lab,
					 xy=(df_landmarks_cville.long[j], df_landmarks_cville.lat[j]),
					 xytext=(df_landmarks_cville.long[j],df_landmarks_cville.lat[j]+0.00002),
					 ha='center', va='bottom',
					 fontsize=12)
ax[1,1].set_xlim(-78.54,-78.43)
ax[1,1].set_ylim(37.99,38.09)
ax[1,1].axis('off')

# Colorbar and labels
plot_grad(df_map_total.median_hou, 'Greens', ax[2,1])
ax[2,1].axis('off')
ax[2,1].text(np.min(df_map_total.median_hou),0.75,
             'Minimum\n\${0:.0f}K'.format(np.min(df_map_total.median_hou)/1000.0),
             ha='center', fontsize=18)
ax[2,1].text(np.max(df_map_total.median_hou),0.75,
             'Maximum\n\${0:.0f}K'.format(np.max(df_map_total.median_hou)/1000.0),
             ha='center', fontsize=18)
             
zoom_effect_box(ax[0,1], ax[1,1], [-78.54, 37.99, -78.43, 38.09])

# Plot fraction of population
df_map_total.plot(column=1.0-(df_map_total.white_popu/df_map_total.total_popu),
            cmap='Blues', vmin=0.0, vmax=1.0,
            ax=ax[0,2])
ax[0,2].axis('off')
ax[0,2].set_title('Percent Non-White Population', fontsize=24)

vmax = round(np.max(1.0-(df_map_total.white_popu/df_map_total.total_popu)),1)
df_map_total.plot(column=1.0-(df_map_total.white_popu/df_map_total.total_popu),
            cmap='Blues', vmin=0.0, vmax=vmax,
            ax=ax[1,2])
            
ax[1,2].set_xlim(-78.54,-78.43)
ax[1,2].set_ylim(37.99,38.09)
ax[1,2].axis('off')

# Colorbar and labels
plot_grad(df_map_total.white_popu/df_map_total.total_popu, 'Blues', ax[2,2])
ax[2,2].axis('off')
ax[2,2].text(np.min(df_map_total.white_popu/df_map_total.total_popu),0.775,
             '0%',
             ha='center', fontsize=18)
ax[2,2].text(np.max(df_map_total.white_popu/df_map_total.total_popu),0.775,
             '{:.0f}%'.format(vmax*100),
             ha='center', fontsize=18)
             
zoom_effect_box(ax[0,2], ax[1,2], [-78.54, 37.99, -78.43, 38.09])
             
fig.subplots_adjust(hspace=0, wspace=0.15)

# Save figure
plt.savefig(plot_path + '/charlottesville_albemarle.pdf', dpi=300)
print('Plot for charlottesville + albemarle done!'.format(m[len(maps_path):-len(file_extn)]))
    
plt.close()
    
