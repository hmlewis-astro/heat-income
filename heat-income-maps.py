import os
import glob

path = os.getcwd()

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = "Tahoma"

def plot_grad(df_param, cmap, ax):
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

def plot_map_temp(df, ax0, ax1):
	df.plot(column=df['median'],
                cmap='Spectral_r',
                ax=ax0)
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
    
map_files = glob.glob(maps_path + '/*' + file_extn)

for m in map_files:
    # Get map data
    df_map = proc_map_files(m)
    
    # Make plot
    fig, ax = plt.subplots(2,3, figsize=(20,10), gridspec_kw={'height_ratios': [20,1]})

	# Plot household income
    plot_map_income(df_map, ax[0,0], ax[1,0])

    #Plot median temp
    plot_map_temp(df_map, ax[0,1], ax[1,1])

    # Plot population stats
    plot_map_popstats(df_map, ax[0,2], ax[1,2])
    
    fig.subplots_adjust(hspace=0, wspace=0.15)
        
    # Save figure
    plt.savefig(plot_path + '/' + m[len(maps_path):-len(file_extn)] + '.pdf', dpi=300)
    print('Plot for {} done!'.format(m[len(maps_path):-len(file_extn)]))
    
    plt.close()
