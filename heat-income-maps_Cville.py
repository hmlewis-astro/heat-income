import os
import glob

path = os.getcwd()

import geopandas as gpd
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox, TransformedBbox, blended_transform_factory
from mpl_toolkits.axes_grid1.inset_locator import BboxPatch, BboxConnector, BboxConnectorPatch
import numpy as np
from sklearn.linear_model import LinearRegression

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

def proc_map_files(map_file):
	df = gpd.read_file(map_file)
	df['median'] = df['median'].astype(np.float64) # Median temp in census tract
	df['median_hou'] = df['median_hou'].astype(np.float64) # Median household income in census tract
	df['total_popu'] = df['total_popu'].astype(np.float64) # Total population in census tract
	df['white_popu'] = df['white_popu'].astype(np.float64) # White population in census tract
	df[df['median_hou'] < 0] = np.nan
	df['nonwhite_popu'] = 1.0-(df['white_popu']/df['total_popu'])
	
	return df
	
def plot_map_income(df, ax0, ax1):
	df.plot(column=df['median_hou'],
				cmap='Greens',
				ax=ax0)
	ax0.axis('off')
	ax0.set_title('Median Household Income', fontsize=24)

    # Colorbar and labels
	plot_grad(df['median_hou'], 'Greens', ax1)
	ax1.axis('off')
	ax1.text(np.min(df['median_hou']),0.75,
				 'Minimum\n\${0:.0f}K'.format(np.min(df['median_hou'])/1000.0),
				 ha='center', fontsize=18)
	ax1.text(np.max(df['median_hou']),0.75,
				 'Maximum\n\${0:.0f}K'.format(np.max(df['median_hou'])/1000.0),
				 ha='center', fontsize=18)

def plot_map_temp(df, df_landmark, ax0, ax1):
	df.plot(column=df['median'],
                cmap='Spectral_r',
                ax=ax0)
	df_landmark.plot.scatter(x='long', y='lat', ax=ax0, s=15, c='k')
	for j,lab in enumerate(df_landmark['landmark']):
		if j % 4 in [0,1]:
			offset = 0.00002
			va = 'bottom'
		else:
			offset = -0.0001
			va = 'top'
		ax0.annotate(lab,
						 xy=(df_landmark['long'][j], df_landmark['lat'][j]),
						 xytext=(df_landmark['long'][j],df_landmark['lat'][j]+offset),
						 ha='center', va=va,
						 fontsize=12)

	ax0.axis('off')
	ax0.set_title('Surface Temperature', fontsize=24)

    # Colorbar and labels
	plot_grad(df['median'], 'Spectral_r', ax1)
	ax1.axis('off')
	ax1.text(np.min(df['median']),0.75,
				 '{0:.1f}$^\circ$\nbelow avg.'.format(np.min(df['median'])-np.mean(df['median'])),
				 ha='center', fontsize=18)
	ax1.text(np.max(df['median']),0.75,
				 '{0:.1f}$^\circ$\nabove avg.'.format(np.max(df['median'])-np.mean(df['median'])),
				 ha='center', fontsize=18)
		
def plot_map_popstats(df, ax0, ax1):
	vmax = round(np.max(df['nonwhite_popu']),1)
	df.plot(column=df['nonwhite_popu'],
				cmap='Blues', vmin=0.0, vmax=vmax,
				ax=ax[0,2])
	ax0.axis('off')
	ax0.set_title('Percent Population of Color', fontsize=24)

    # Colorbar and labels
	plot_grad(df['nonwhite_popu'], 'Blues', ax1)
	ax1.axis('off')
	ax1.text(np.min(df['nonwhite_popu']),0.775,
				 '0%',
				 ha='center', fontsize=18)
	ax1.text(np.max(df['nonwhite_popu']),0.775,
				 '{:.0f}%'.format(vmax*100),
				 ha='center', fontsize=18)

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
	df_map = proc_map_files(m)
	df_landmarks = pd.read_csv(landmark_files[i], comment='#')
    
    # Make plot
	fig, ax = plt.subplots(2,3, figsize=(20,10), gridspec_kw={'height_ratios': [20,1]})
	
	# Plot household income
	plot_map_income(df_map, ax[0,0], ax[1,0])

    #Plot median temp
	plot_map_temp(df_map, df_landmarks, ax[0,1], ax[1,1])

    # Plot population stats
	plot_map_popstats(df_map, ax[0,2], ax[1,2])
    
	fig.subplots_adjust(hspace=0, wspace=0.15)

    # Save figure
	plt.savefig(plot_path + '/' + m[len(maps_path):-len(file_extn)] + '.pdf', dpi=300)
	print('Plot for {} done!'.format(m[len(maps_path):-len(file_extn)]))
    
	plt.close()

# Aggregate Charlottesville and Albemarle County map data
df_map_cville = proc_map_files(cville_map_files)
df_map_albemr = proc_map_files(albemr_map_files)
df_map_total = df_map_cville.append(df_map_albemr)
    
df_landmarks_cville = pd.read_csv(cville_landmark_files, comment='#')
df_landmarks_albemr = pd.read_csv(albemr_landmark_files, comment='#')

# Make single plot for Charlottesville and Albemarle County data
fig, ax = plt.subplots(3,3, figsize=(20,10), gridspec_kw={'height_ratios': [20,20,2]})

# Plot household income
plot_map_income(df_map_total, ax[0,0], ax[2,0])

df_map_total.plot(column=df_map_total['median_hou'],
				  cmap='Greens',
				  ax=ax[1,0])
ax[1,0].set_xlim(-78.54,-78.43)
ax[1,0].set_ylim(37.99,38.09)
ax[1,0].axis('off')

zoom_effect_box(ax[0,0], ax[1,0], [-78.54, 37.99, -78.43, 38.09])

#Plot median temp
plot_map_temp(df_map_total, df_landmarks_albemr, ax[0,1], ax[2,1])

df_map_total.plot(column=df_map_total['median'],
				  cmap='Spectral_r',
				  ax=ax[1,1])
df_landmarks_cville.plot.scatter(x='long', y='lat', ax=ax[1,1], s=15, c='k')
for j,lab in enumerate(df_landmarks_cville['landmark']):
	ax[1,1].annotate(lab,
					 xy=(df_landmarks_cville['long'][j], df_landmarks_cville['lat'][j]),
					 xytext=(df_landmarks_cville['long'][j],df_landmarks_cville['lat'][j]+0.00002),
					 ha='center', va='bottom',
					 fontsize=12)
ax[1,1].set_xlim(-78.54,-78.43)
ax[1,1].set_ylim(37.99,38.09)
ax[1,1].axis('off')

zoom_effect_box(ax[0,1], ax[1,1], [-78.54, 37.99, -78.43, 38.09])

# Plot fraction of population
plot_map_popstats(df_map_total, ax[0,2], ax[2,2])

vmax = round(np.max(df_map_total['nonwhite_popu']),1)
df_map_total.plot(column=df_map_total['nonwhite_popu'],
				  cmap='Blues', vmin=0.0, vmax=vmax,
				  ax=ax[1,2])
ax[1,2].set_xlim(-78.54,-78.43)
ax[1,2].set_ylim(37.99,38.09)
ax[1,2].axis('off')

zoom_effect_box(ax[0,2], ax[1,2], [-78.54, 37.99, -78.43, 38.09])
             
fig.subplots_adjust(hspace=0, wspace=0.15)

# Save figure
plt.savefig(plot_path + '/charlottesville_albemarle.pdf', dpi=300)
plt.savefig(plot_path + '/charlottesville_albemarle.png', dpi=300)

plt.close()

# Make single correlation plot for Charlottesville and Albemarle County data
fig, ax = plt.subplots(1,2, figsize=(20,7), sharex=True)

cmap = plt.get_cmap('Spectral_r')
#Plot median temp vs. income
ax[0].scatter(df_map_albemr['median']-np.mean(df_map_albemr['median']), df_map_albemr['median_hou'], color=cmap(0.8), marker='^', s=100, label='Albemarle Co block groups')
ax[0].scatter(df_map_cville['median']-np.mean(df_map_total['median']), df_map_cville['median_hou'], color=cmap(0.1), s=100, label='City of Cville block groups')

plot_x = np.arange(-12,12)

X_albemr = (df_map_albemr['median']-np.mean(df_map_total['median']))
y_albemr = df_map_albemr['median_hou']
mask = ~np.isnan(X_albemr) & ~np.isnan(y_albemr)
X_albemr = np.asarray(X_albemr[mask]).reshape(-1, 1)
y_albemr = np.asarray(y_albemr[mask])
reg_income_albemr = LinearRegression().fit(X_albemr, y_albemr)
ax[0].plot(plot_x, reg_income_albemr.intercept_ + reg_income_albemr.coef_[0]*plot_x, color=cmap(0.75), ls=':', lw=3.0)
print('\n')
print('In Albemarle Co, for an increase of 1 degree in surface temperature, we see a{0} {1} of ${2:.0f} in median income.'.format('n' if reg_income_albemr.coef_[0] > 0.0 else '','increase' if reg_income_albemr.coef_[0] > 0.0 else 'decrease', np.abs(reg_income_albemr.coef_[0])))
factor = 1 if reg_income_albemr.coef_[0] > 0.0 else -1
print('R score = {:.3f}'.format(factor*np.sqrt(reg_income_albemr.score(X_albemr, y_albemr))))
print('\n')

X_cville = (df_map_cville['median']-np.mean(df_map_total['median']))
y_cville = df_map_cville['median_hou']
mask = ~np.isnan(X_cville) & ~np.isnan(y_cville)
X_cville = np.asarray(X_cville[mask]).reshape(-1, 1)
y_cville = np.asarray(y_cville[mask])
reg_income_cville = LinearRegression().fit(X_cville, y_cville)
ax[0].plot(plot_x, reg_income_cville.intercept_ + reg_income_cville.coef_[0]*plot_x, color=cmap(0.05), ls='--', lw=3.0)
print('\n')
print('In the City of Cville, for an increase of 1 degree in surface temperature, we see a{0} {1} of ${2:.0f} in median income.'.format('n' if reg_income_cville.coef_[0] > 0.0 else '','increase' if reg_income_cville.coef_[0] > 0.0 else 'decrease', np.abs(reg_income_cville.coef_[0])))
factor = 1 if reg_income_cville.coef_[0] > 0.0 else -1
print('R score = {:.3f}'.format(factor*np.sqrt(reg_income_cville.score(X_cville, y_cville))))
print('\n')

ax[0].set_xlabel('Surface Temperature', fontsize=24)
ax[0].set_ylabel('Median Household Income', fontsize=24)
ax[0].tick_params(axis='both', labelsize=18)
ax[0].set_aspect(1.0/ax[0].get_data_ratio(), adjustable='box')

#Plot median temp vs. non-white pct
ax[1].scatter(df_map_albemr['median']-np.mean(df_map_albemr['median']), df_map_albemr['nonwhite_popu'], color=cmap(0.8), marker='^', s=100, label='Albemarle Co block groups')
ax[1].scatter(df_map_cville['median']-np.mean(df_map_total['median']), df_map_cville['nonwhite_popu'], color=cmap(0.1), s=100, label='City of Cville block groups')

X_albemr = (df_map_albemr['median']-np.mean(df_map_total['median']))
y_albemr = df_map_albemr['nonwhite_popu']
mask = ~np.isnan(X_albemr) & ~np.isnan(y_albemr)
X_albemr = np.asarray(X_albemr[mask]).reshape(-1, 1)
y_albemr = np.asarray(y_albemr[mask])
reg_popu_albemr = LinearRegression().fit(X_albemr, y_albemr)
ax[1].plot(plot_x, reg_popu_albemr.intercept_ + reg_popu_albemr.coef_[0]*plot_x, color=cmap(0.75), ls=':', lw=3.0)
print('\n')
print('In Albemarle Co, for an increase of 1 degree in surface temperature, we see a{0} {1} of {2:.0f}% in percent population of color.'.format('n' if reg_popu_albemr.coef_[0] > 0.0 else '','increase' if reg_popu_albemr.coef_[0] > 0.0 else 'decrease', np.abs(reg_popu_albemr.coef_[0])*100))
factor = 1 if reg_popu_albemr.coef_[0] > 0.0 else -1
print('R score = {:.3f}'.format(factor*np.sqrt(reg_popu_albemr.score(X_albemr, y_albemr))))
print('\n')

X_cville = (df_map_cville['median']-np.mean(df_map_total['median']))
y_cville = df_map_cville['nonwhite_popu']
mask = ~np.isnan(X_cville) & ~np.isnan(y_cville)
X_cville = np.asarray(X_cville[mask]).reshape(-1, 1)
y_cville = np.asarray(y_cville[mask])
reg_popu_cville = LinearRegression().fit(X_cville, y_cville)
ax[1].plot(plot_x, reg_popu_cville.intercept_ + reg_popu_cville.coef_[0]*plot_x, color=cmap(0.05), ls='--', lw=3.0)
print('\n')
print('In the City of Cville, for an increase of 1 degree in surface temperature, we see a{0} {1} of {2:.0f}% in percent population of color.'.format('n' if reg_popu_cville.coef_[0] > 0.0 else '','increase' if reg_popu_cville.coef_[0] > 0.0 else 'decrease', np.abs(reg_popu_cville.coef_[0])*100))
factor = 1 if reg_popu_cville.coef_[0] > 0.0 else -1
print('R score = {:.3f}'.format(factor*np.sqrt(reg_popu_cville.score(X_cville, y_cville))))
print('\n')

ax[0].set_xlim(math.floor(np.min(df_map_total['median'])-np.mean(df_map_total['median'])-2.0), math.ceil(np.max(df_map_total['median'])-np.mean(df_map_total['median'])+2.5))
ax[0].set_ylim(-5000, math.ceil(np.max(df_map_total['median_hou'])+20000))
ax[1].set_ylim(-0.05, round(np.max(df_map_total['nonwhite_popu'])+0.1,1))

def major_x_formatter(x, pos):
    return '{0:.1f}$^\circ$'.format(x)
ax[0].xaxis.set_major_formatter(plt.FuncFormatter(major_x_formatter))

def major_income_formatter(x, pos):
    return '\${0:.0f}K'.format(x/1000.0)
ax[0].yaxis.set_major_formatter(plt.FuncFormatter(major_income_formatter))

def major_popu_formatter(x, pos):
    return '{:.0f}%'.format(x*100.0)
ax[1].yaxis.set_major_formatter(plt.FuncFormatter(major_popu_formatter))

ax[0].set_aspect(1.0/ax[0].get_data_ratio(), adjustable='box')
ax[1].set_aspect(1.0/ax[1].get_data_ratio(), adjustable='box')

ax[1].set_xlabel('Surface Temperature', fontsize=24)
ax[1].set_ylabel('Percent Population of Color', fontsize=24)
ax[1].tick_params(axis='both', labelsize=18)

h1, l1 = ax[0].get_legend_handles_labels()

box = ax[0].get_position()
ax[0].set_position([box.x0, box.y0 + box.height * 0.15,
                 box.width, box.height * 0.95])
box = ax[1].get_position()
ax[1].set_position([box.x0, box.y0 + box.height * 0.15,
                 box.width, box.height * 0.95])
                 
fig.legend(h1, l1,  loc=8, #bbox_to_anchor=(0,-.05, 2.0,-0.10), loc=9,
           ncol=2, fontsize=18)

# Save figure
plt.savefig(plot_path + '/charlottesville_albemarle_R.pdf', dpi=300)
plt.savefig(plot_path + '/charlottesville_albemarle_R.png', dpi=300)
print('Plot for charlottesville + albemarle done!'.format(m[len(maps_path):-len(file_extn)]))
    
plt.close()
