import sys
import getopt

import json
import geopandas as gpd
from rasterstats import zonal_stats


########## start __main__ function ##########
   

def main(argv):


	shapefile = ''
	rasterfile = ''
	outfile = ''
   
	try:
		opts, args = getopt.getopt(argv,"hs:r:o:",["sfile=", "rfile=", "ofile="])
	except getopt.GetoptError:
		print('rio_zonalstats_sub.py -s <shapefile> -r <rasterfile> -o <outputfile>')
		sys.exit(2)
		
	for opt, arg in opts:
		if opt == '-h':
			print('rio_zonalstats_sub.py -s <shapefile> -r <rasterfile> -o <outputfile>')
			sys.exit()
		elif opt in ("-s", "--sfile"):
			shapefile = arg
		elif opt in ("-r", "--rfile"):
			rasterfile = arg
		elif opt in ("-o", "--ofile"):
			outfile = arg
	
	stats = zonal_stats(shapefile, rasterfile, stats=['median'], geojson_out=True, nodata=-9999)
	geostats = gpd.GeoDataFrame.from_features(stats)
	
	geostats.to_file(outfile, driver='GeoJSON')
	

if __name__ == "__main__":
   main(sys.argv[1:])
