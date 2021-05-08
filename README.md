# NPR Heat and Poverty Analysis

_Updated code by Hannah Lewis_

**As part of ongoing climate justice work for the University of Virginia Equity Center and University of Virginia Sustainability Partnership, I have reproduced the analysis originally laid out in the NPR article ["As Rising Heat Bakes U.S. Cities, The Poor Often Feel It Most"](https://www.npr.org/2019/09/03/754044732/as-rising-heat-bakes-u-s-cities-the-poor-often-feel-it-most) for the City of Charlottesville and Albemarle County, Virginia.**

**With the addition of these two regions, I've also opted to use block group-level data from the U.S. Census, rather than tract-level data, since population statistics tend to vary widely across tracts in these particular areas.**

**For the City of Charlottesville, there is a strong negative correlation between heat and income (i.e., as heat increases in a block group, median household income decreases), whereas in Albemarle County, there is no correlation observed. In both the City of Charlottesville and Albemarle County there are moderate positive correlations between heat and percent population of color (i.e., as heat increases in a block group, the percentage of the population made up by people of color also increases), as illustrated here:**
![Charlottesville Albemarle map](https://github.com/hmlewis-astro/heat-income/blob/master/data/output/analysis_out/final/plots/charlottesville_albemarle_stitch.png)

_Original code by Sean McMinn and Nick Underwood; additional reporting by Meg Anderson and Nora Eckert_

To determine the link between heat and income in U.S. cities, NPR used NASA satellite imagery and U.S. Census American Community Survey data. An open-source computer program developed by NPR downloaded median household income data for census tracts in the 100 most populated American cities, as well as geographic boundaries for census tracts. NPR combined these data with TIGER/Line shapefiles of the cities.

The software also downloaded thermal imagery for each city from NASA's Landsat 8 satellite, looking for days since 2011 in June, July and August when there was less than 4 percent cloud cover. NPR reviewed each of the satellite images and removed images that contained clouds or other obscuring features over the city of interest. In cases when there were multiple clear images of a city, we used the thermal reading that showed a greater contrast between the warm and cool parts of the area of interest. In cases where there were no acceptable images, we manually searched for additional satellite images, and found acceptable images from Landsat 8 for every city except for Hialeah and Miami, Fla., and Honolulu, which are frequently covered by clouds.

For each city, NPR aligned the satellite surface temperature data with the census tracts. For each census tract, the software trimmed the geography to only what is contained within the city of interest's boundaries, then removed any lakes, rivers, ocean, etc. It calculated a median temperature reading for each census tract. When all the tracts in a city were completed, it calculated a correlation coefficient (R) of the tracts to find the relationship between income and heat.

The satellite data measures temperature at a surface, like the ground or a rooftop. We used this measurement rather than ambient temperature, which measures the air about two meters above the ground. Measuring air is a more accurate measure of how people experience heat, but satellite data is more widely available than air temperature data. Using it allowed us to provide a more complete snapshot of temperature trends across many cities.

## Instructions To Reproduce Analysis

- `virtualenv heat-income`
- `cd heat-income`
- `. bin/activate`
- `pip install -r requirements.txt`
- Set your Census API key as an environment variable in `bin/activate`
- Images not previously downloaded, should be downloaded manually from EarthExplorer. **This repository already includes GEOTIFFs for most cities (based on list of approved images in `good_images.json`). Images are stored in `data/output/images/state-city`.**
- `sh mkfile.sh`
	- Details on each process in `sh mkfile.sh`:
		- `. bin/activate`
		- `sh download_water.sh` &mdash; downloads a 10m raster of oceans and US bodies of water
		- `python download_data.py` &mdash; downloads US Census block group shapefiles and income, race, and population data within each block group, merges block group shapefile and data, and downloads satellite images (if they aren't already available locally)
		- `sh parallel_process.sh` &mdash; runs geoanalysis (`single_process.sh`) on each city in parallel and calculates correlation statistics (median household income vs. surface temperature, percent population of color vs. surface temperature)
			- `sh single_process.sh` &mdash; removes oceans and bodies of water, calculates zonalstats (median spectral radiance in block group), converts spectral radiance to degrees Kelvin, calculates correlation statistics, and produces simplified output for mapping on the web
		- `python heat-income-maps.py` &mdash; creates median household income, surface temperature, and percent population of color maps for each city, maps saved in `data/output/analysis_out/final/plots/`
		- `python heat-income-maps_Cville.py` &mdash; creates more detailed maps (including landmarks) for the City of Charlottesville and Albemarle County, VA

**Added: Python script as part of `mkfile.sh` to create maps, similar to those shown in the NPR article, as well as additional maps for the City of Charlottesville and Albemarle County, Virginia.**

## Data files

Completed data files for each city are saved as .geojson files in `data/output/analysis_out/final/`.

Correlations for each city are listed in `good_images_w_r.json`.

**Added: Maps for each city are saved as PDFs in `data/output/analysis_out/final/plots/`.**

## Simplified files

Also in the `final` directory is a directory called `simpl`. This has .geojson files with simplified polygons for mapping on the web. These are used in NPR's web maps and were simplified using [mapshaper](https://github.com/mbloch/mapshaper).

## Caveats
- There is a difference between poverty and low-income.
- More detailed Census geography yields larger margins of error.
- Satellite imagery provide surface temperature, not ambient (air) temperature.
- Results include analysis of only one day of data per city.
- Not every US city is included.
- Some cities split satellite scene paths/rows, so we took the image from the scene that contained most of the city. You'll need to filter out tracts without heat data in any data analysis/mapping you do.
- For satellite images that are not defined manually, the script will grab recent images. As more scenes are captured by NASA, results may vary slightly and/or good_images may need to be redefined.
