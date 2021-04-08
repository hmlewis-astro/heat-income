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
        

maps_path = path + '/data/output/analysis_out/final/'
plot_path = path + '/data/output/analysis_out/final/plots/'
file_extn = '.geojson'

assert os.path.exists(maps_path), "Directory containing the final .geojson files missing!"

if not os.path.exists(plot_path):
    os.makedirs(plot_path)
    

map_files = glob.glob(maps_path + '/*' + file_extn)

for m in map_files:
    # Get map data
    df_map = gpd.read_file(m)

    # Format data of interest
    df_map['median'] = df_map['median'].astype(np.float64) # Median temp in census tract
    df_map['median_hou'] = df_map['median_hou'].astype(np.float64) # Median household income in census tract
    df_map['total_popu'] = df_map['total_popu'].astype(np.float64) # Total population in census tract
    df_map['white_popu'] = df_map['white_popu'].astype(np.float64) # White population in census tract
    df_map[df_map['median_hou'] < 0] = np.nan
    df_map['nonwhite_popu'] = 1.0-(df_map['white_popu']/df_map['total_popu'])
    
    # Make plot
    fig, ax = plt.subplots(2,3, figsize=(20,10), gridspec_kw={'height_ratios': [20,1]})

    #Plot median temp
    df_map.plot(column=df_map['median'],
                cmap='Reds',
                ax=ax[0,0])
    ax[0,0].axis('off')
    ax[0,0].set_title('Surface Temperature', fontsize=24)

    # Colorbar and labels
    plot_grad(df_map['median'], 'Reds', ax[1,0])
    ax[1,0].axis('off')
    ax[1,0].text(np.min(df_map['median']),0.75,
				 '{0:.1f}$^\circ$\nbelow avg.'.format(np.min(df_map['median'])-np.mean(df_map['median'])),
				 ha='center', fontsize=18)
    ax[1,0].text(np.max(df_map['median']),0.75,
				 '{0:.1f}$^\circ$\nabove avg.'.format(np.max(df_map['median'])-np.mean(df_map['median'])),
				 ha='center', fontsize=18)

    # Plot household income
    df_map.plot(column=df_map['median_hou'],
				cmap='Greens',
				ax=ax[0,1])
    ax[0,1].axis('off')
    ax[0,1].set_title('Income', fontsize=24)

    # Colorbar and labels
    plot_grad(df_map['median_hou'], 'Greens', ax[1,1])
    ax[1,1].axis('off')
    ax[1,1].text(np.min(df_map['median_hou']),0.75,
				 'Minimum\n\${0:.0f}K'.format(np.min(df_map['median_hou'])/1000.0),
				 ha='center', fontsize=18)
    ax[1,1].text(np.max(df_map['median_hou']),0.75,
				 'Maximum\n\${0:.0f}K'.format(np.max(df_map['median_hou'])/1000.0),
				 ha='center', fontsize=18)

    # Plot fraction of population
    vmax = round(np.max(df_map['nonwhite_popu']),1)
    df_map.plot(column=df_map['nonwhite_popu'],
				cmap='Blues', vmin=0.0, vmax=vmax,
				ax=ax[0,2])
    ax[0,2].axis('off')
    ax[0,2].set_title('Percent Non-White Population', fontsize=24)

    # Colorbar and labels
    plot_grad(df_map['nonwhite_popu'], 'Blues', ax[1,2])
    ax[1,2].axis('off')
    ax[1,2].text(np.min(df_map['nonwhite_popu']),0.775,
				 '0%',
				 ha='center', fontsize=18)
    ax[1,2].text(np.max(df_map['nonwhite_popu']),0.775,
				 '{:.0f}%'.format(vmax*100),
				 ha='center', fontsize=18)
    
    # Save figure
    plt.savefig(plot_path + '/' + m[len(maps_path):-len(file_extn)] + '.pdf', dpi=300)
    print('Plot for {} done!'.format(m[len(maps_path):-len(file_extn)]))
    
    plt.close()
