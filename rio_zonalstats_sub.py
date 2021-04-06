import sys
import getopt

import json
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
			
	#print(shapefile)
	#print(rasterfile)
	#print(outfile)
	
	stats = zonal_stats(shapefile, rasterfile, stats=['median'], geojson_out=True, nodata=-9999)
	
	#str_front = "{" + "'type': 'FeatureCollection', 'features': "
	#str_back = "}"
	
	#old_med = "('median'"
	#new_med = "('_median'"
	
	#stats = str_front + str(stats).replace(old_str,new_str) + str_back
	#stats = str(stats).replace(old_med,new_med)
	with open(outfile, 'w') as ofile:
		json.dump(stats,ofile)
	

if __name__ == "__main__":
   main(sys.argv[1:])
