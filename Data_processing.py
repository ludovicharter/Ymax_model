# Import libraries
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray
from shapely.geometry import mapping
import rasterio
from shapely.geometry import Point
import rasterio as rio
from rasterstats import zonal_stats
import rioxarray
from rasterio.crs import CRS
import netCDF4 as nc

#%% Read the NUTS shapefile (from https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/administrative-units-statistical-units/nuts)
nuts = gpd.read_file('/Users/ludovicharter/Desktop/PhD_Sorbonne/Data/NUTS_RG_01M_2021_4326/NUTS_RG_01M_2021_4326.shp', crs="epsg:4326")

# Extract the 127 GU used to compute Ymax from Billen et al. (2023)
data = pd.read_csv('/Users/ludovicharter/Desktop/PhD_Sorbonne/Data/SM4 GRAFS Europe Ref and scenarios rev/REF 2014-2019-Tableau 1.csv', on_bad_lines='skip', sep=';')
id = data.iloc[95]  # extract Ymax values
id = pd.DataFrame(id)
id['NUTS_ID'] = data.iloc[0]  # extract GU NUTS identification
id = id.reset_index(drop=True)
id = id.iloc[5:132, :]  # remove nan and non-float values
id = id.reset_index(drop=True)
# Rename columns
id = id.rename(columns={95: 'Ymax_cropland'})
id = id.rename(columns={0: 'NUTS_ID'})
# Replace Ymax values to float
id['Ymax_cropland'] = id['Ymax_cropland'].str.replace(',', '.').astype(float)
id = id.dropna()  # No Ymax value for BE1

# Merge the 127 nuts shapes GU and Ymax into one file
nuts = pd.merge(nuts, id, on='NUTS_ID')

#%% ERA5 atmospheric reanalysis (https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5)

# ERA5 land-sea and inland water mask
mask = xarray.open_dataarray('/Users/ludovicharter/Desktop/PhD_Sorbonne/Data/land-sea_mask/adaptor.mars.internal-1701166990.3851645-17722-9-1509f163-3bae-4d18-9bf0-34ab9f41b6b7.nc')
# Change longitude to -180 180
mask = mask.assign_coords(longitude=(((mask.longitude + 180) % 360) - 180)).sortby('longitude')
mask.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
# Select grids covered by less than 50% water
mask = mask.where(mask > 0.5, 0)

# Compile ERA5 variables
list = ['radiation', '2m_temperature', 'evaporation', 'potential_evaporation', 'total_precipitation',
        'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3']

for i in list:
    # import ERA5 data
    era = xarray.open_dataarray('/Users/ludovicharter/Desktop/PhD_Sorbonne/Data/ERA5/era5_%s_monthly_2014_2019.nc' % i)
    # Change longitude to -180 180
    era = era.assign_coords(longitude=(((era.longitude + 180) % 360) - 180)).sortby('longitude')
    era.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
    era.rio.write_crs("epsg:4326", inplace=True)

    # Time-averaging
    yearly_mean = era.mean(axis=0)

    # Apply mask
    yearly_mean = yearly_mean.where(mask > 0)  # (!) Malta is removed

    # Compute the mean for each NUTS region
    means = []
    for j in nuts['NUTS_ID']:
        n = nuts.where(nuts['NUTS_ID'] == j)
        n = n.dropna()
        clipped = yearly_mean.rio.clip(n.geometry.apply(mapping), n.crs, drop=False, all_touched=True)
        m = float(clipped.mean())
        means.append(m)

    nuts['%s' % i] = means
    nuts = nuts.dropna()

# Convert units
nuts['radiation'] = (nuts['radiation'] / 3600) / 1000  # J.m-2 to kW.m-2
nuts['2m_temperature'] = nuts['2m_temperature'] - 273.15  # K to °C
nuts['evaporation'] = nuts['evaporation'] * 1000  # m to mm
nuts['potential_evaporation'] = nuts['potential_evaporation'] * 1000  # m to mm
nuts['total_precipitation'] = nuts['total_precipitation'] * 1000  # m to mm
# Add Precipitations - Evaporation and Precipitation - Potential Evaporation
nuts['P-EV'] = nuts['total_precipitation'] + nuts['evaporation']
nuts['P-PEV'] = nuts['total_precipitation'] + nuts['potential_evaporation']

#%% LUCAS topsoil properties 2018 survey (https://esdac.jrc.ec.europa.eu/projects/lucas)
lucas = pd.read_csv('/Users/ludovicharter/Desktop/PhD_Sorbonne/Data/LUCAS-SOIL-2018-data-report-readme-v2/LUCAS-SOIL-2018-v2/LUCAS-SOIL-2018.csv')

# Choose only cropland samples
lucas = lucas.where(lucas['LC0_Desc'] == 'Cropland')
lucas = lucas[lucas['TH_LAT'].notna()]
lucas = lucas.reset_index(drop=True)

# Replace limit of detection (LOD) values to 0
lucas = lucas.replace('<  LOD', 0)
lucas = lucas.replace('< LOD', 0)
lucas = lucas.replace('<0.0', 0)

variables = ['EC', 'OC', 'CaCO3', 'P', 'N', 'K', 'pH_CaCl2', 'pH_H2O']

# Convert values to float
for v in variables:
    lucas['%s' % v] = lucas['%s' % v].astype(float)

# Extract sample locations
samples = []
for i in lucas.index:
    samples.append(Point((lucas['TH_LONG'][i], lucas['TH_LAT'][i])))
# Coordinate reference system
gdf_samples = gpd.GeoSeries(samples, crs={'init': 'epsg:4326'})

# Plot sample locations
fig, ax = plt.subplots(figsize=(15, 15))
nuts.plot(ax=ax, color='gray')
nuts.boundary.plot(color='k', ax=ax)
gdf_samples.plot(ax=ax, c='r', markersize=5)
ax.tick_params(axis='both', labelsize=20)
# Save figure
#plt.savefig('/Users/ludovicharter/Desktop/PhD_Sorbonne/figures/LUCAS_sample_locations.pdf', format='pdf', dpi=600)
plt.show()

# Averaging at NUTS scale for each variable
for v in variables:

    M = []
    # Iterate for each region
    for j in nuts.index:
        mean = lucas.where(lucas['NUTS_%s' % nuts['LEVL_CODE'][j]] == nuts['NUTS_ID'][j])
        mean = mean[mean['TH_LAT'].notna()]
        mean = mean.reset_index(drop=True)
        M.append(mean['%s' % v].mean())

    # Add to NUTS file
    nuts['%s' % v] = M

#%% TERRACLIMATE (https://www.climatologylab.org/terraclimate.html)

variables = ['aet', 'def', 'PDSI', 'pet', 'soil', 'srad', 'ppt']

# Iterate each variable
for v in variables:

    # Read data each year from 2014 to 2019
    xraet = xarray.open_mfdataset('/Users/ludovicharter/Desktop/PhD_Sorbonne/Data/TerraClimate/TerraClimate_%s_*.nc' % v)
    # Calculate the 6 years average
    dataset = xraet.mean(dim='time')
    # Coordinate reference system
    dataset.rio.write_crs("epsg:4326", inplace=True)
    # Transform into array
    arr = np.array(dataset['%s' % v])
    # Extract the CRS affine transformation
    affine = rio.open('/Users/ludovicharter/Desktop/PhD_Sorbonne/Data/TerraClimate/TerraClimate_%s_2014.nc' % v).transform
    # Calculate the zonal statistics
    stats = zonal_stats(nuts, arr, affine=affine, stats=['mean'], all_touched=True)
    # Create a float list and store mean values
    mean = []
    for i in range(125):
        mean.append(stats[i]['mean'])
    nuts['%s' % v] = mean

#%% ESDAC topsoil physical properties (https://esdac.jrc.ec.europa.eu/content/topsoil-physical-properties-europe-based-lucas-topsoil-data)

variables = ['AWC', 'Silt', 'Clay', 'Sand', 'Bulk_density', 'Coarse_fragments']

# Iterate each variable
for v in variables:

    # Read datasets
    path = '/Users/ludovicharter/Desktop/PhD_Sorbonne/Data/LUCAS_topsoilproperties/%s_Extra' % v
    dataset = rasterio.open(path + '/%s.tif' % v)
    # Extract the CRS affine transformation
    affine = dataset.transform
    # Transform into array
    arr = dataset.read(1)
    # Replace nodata value by NaN
    arr[arr == dataset.nodata] = np.nan
    # Set same CRS for both files
    nuts = nuts.to_crs(dataset.crs)
    # Calculate the zonal statistics
    stats = zonal_stats(nuts, arr, affine=affine, stats=['mean'], all_touched=True)
    # Create a float list and store mean values
    mean = []
    for i in range(125):
        mean.append(stats[i]['mean'])
    nuts['%s' % v] = mean

# Set original CRS
nuts = nuts.to_crs('epsg:4326')

#%% European Soil Database Derived data (https://esdac.jrc.ec.europa.eu/content/european-soil-database-derived-data)

variables = ['STU_EU_DEPTH_ROOTS', 'STU_EU_T_OC', 'STU_EU_S_OC', 'SMU_EU_T_TAWC', 'SMU_EU_S_TAWC', 'STU_EU_T_TAWC',
             'STU_EU_S_TAWC']

# Iterate each variable
for v in variables:

    # Read datasets
    with rasterio.open('/Users/ludovicharter/Desktop/PhD_Sorbonne/Data/STU_EU_Layers/%s.rst' % v, 'r+') as rds:
        # Coordinate reference system
        rds.crs = CRS.from_epsg(3035)
        # Extract the CRS affine transformation
        affine = rds.transform
        # Transform into array
        arr = rds.read(1)
        # Set same CRS for both files
        nuts = nuts.to_crs(rds.crs)
    # Replace nodata value by NaN
    arr = arr.astype(float)
    arr[arr == 0.0] = np.nan
    # Calculate the zonal statistics
    stats = zonal_stats(nuts, arr, affine=affine, stats=['mean'], all_touched=True)
    # Create a float list and store mean values
    mean = []
    for i in range(125):
        mean.append(stats[i]['mean'])
    nuts['%s' % v] = mean

# Set original CRS
nuts = nuts.to_crs('epsg:4326')

#%% Calculating soil water content

# Create an empty DataFrame to store monthly values
df = pd.DataFrame()

# Initialisation from TerraClimate Soil Moisture
ini = xarray.open_dataset('/Users/ludovicharter/Desktop/PhD_Sorbonne/Data/TerraClimate/TerraClimate_soil_2013.nc')
# Select Soil Moisture value for December 2013
ini = ini['soil'][11, :, :]
# Coordinate reference system
ini.rio.write_crs("epsg:4326", inplace=True)
# Transform into array
arr = np.array(ini)
# Extract the CRS affine transformation
affine = rio.open('/Users/ludovicharter/Desktop/PhD_Sorbonne/Data/TerraClimate/TerraClimate_soil_2013.nc').transform
# Calculate the zonal statistics
stats = zonal_stats(nuts, arr, affine=affine, stats=['mean'], all_touched=True)
# Create a float list and store mean values
ini = []
for i in range(125):
    ini.append(stats[i]['mean'])

# Calculating monthly precipitation and evapotranspiration

# Read evapotranspiration data
aet = xarray.open_mfdataset('/Users/ludovicharter/Desktop/PhD_Sorbonne/Data/TerraClimate/TerraClimate_aet_*.nc')
# Iterate for each month between 2014 and 2019
for m in range(72):
    aetm = aet['aet'][m, :, :]
    # Coordinate reference system
    aetm.rio.write_crs("epsg:4326", inplace=True)
    # Transform into array
    arr = np.array(aetm)
    # Extract the CRS affine transformation
    affine = rio.open('/Users/ludovicharter/Desktop/PhD_Sorbonne/Data/TerraClimate/TerraClimate_soil_2013.nc').transform
    # Calculate the zonal statistics
    stats = zonal_stats(nuts, arr, affine=affine, stats=['mean'], all_touched=True)
    # Create a float list and store mean values
    mean = []
    for i in range(125):
        mean.append(stats[i]['mean'])
    df['aet_%s' % m] = mean

# Read precipitation data
ppt = xarray.open_mfdataset('/Users/ludovicharter/Desktop/PhD_Sorbonne/Data/TerraClimate/TerraClimate_ppt_*.nc')
# Iterate for each month between 2014 and 2019
for m in range(72):
    pptm = ppt['ppt'][m, :, :]
    # Coordinate reference system
    pptm.rio.write_crs("epsg:4326", inplace=True)
    # Transform into array
    arr = np.array(pptm)
    # Extract the CRS affine transformation
    affine = rio.open('/Users/ludovicharter/Desktop/PhD_Sorbonne/Data/TerraClimate/TerraClimate_soil_2013.nc').transform
    # Calculate the zonal statistics
    stats = zonal_stats(nuts, arr, affine=affine, stats=['mean'], all_touched=True)
    # Create a float list and store mean values
    mean = []
    for i in range(125):
        mean.append(stats[i]['mean'])
    df['ppt_%s' % m] = mean

# Calculating P - ETP
for m in range(72):
    df['P-ETP_%s' % m] = df['ppt_%s' % m] - df['aet_%s' % m]

# Calculating soil water content
nuts = nuts.reset_index(drop=True)
for m in range(72):
    sw = []
    for i in range(125):
        sw.append(min((ini[i] + df['P-ETP_%s' % m][i]), nuts['SMU_EU_T_TAWC'][i]))
    df['soil_water_%s' % m] = sw
    ini = sw


sw = []
for i in range(72):
    sw.append(df['soil_water_%s' % i][68])
plt.plot(sw)
plt.show()

#%% ERA5 hourly temperature and precipitation data
# Read hourly temperature data
temp = xarray.open_mfdataset('/Users/ludovicharter/Desktop/PhD_Sorbonne/Data/ERA5_hourly/era5_2m_temperature_hourly_2014_2019.nc')
T = temp['time'].to_numpy()

t2m = np.zeros((len(T), len(nuts)))

for t in range(len(T)):
    dataset = temp['t2m'][t, :, :]
    # Coordinate reference system
    dataset.rio.write_crs("epsg:4326", inplace=True)
    # Transform into array
    arr = np.array(dataset)
    # Extract the CRS affine transformation
    affine = rio.open('/Users/ludovicharter/Desktop/PhD_Sorbonne/Data/ERA5_hourly/era5_2m_temperature_hourly_2014_2019.nc').transform
    # Calculate the zonal statistics
    stats = zonal_stats(nuts, arr, affine=affine, stats=['mean'], all_touched=True)
    # Create a float list and store mean values
    mean = []
    for i in range(125):
        mean.append(stats[i]['mean'])
    t2m[t, :] = mean


# Save data
#nuts.to_file('/Users/ludovicharter/Desktop/PhD_Sorbonne/Data/era5_all.shp', driver='ESRI Shapefile')


#%%
'''
#_______________________________________________________________________________________________________________________
# LUCAS topsoil properties 2015 survey (https://esdac.jrc.ec.europa.eu/projects/lucas)
lucas15 = pd.read_csv('/Users/ludovicharter/Desktop/PhD_Sorbonne/Data/LUCAS2015_topsoildata_20200323/LUCAS_Topsoil_2015_20200323.csv')
# Select same cropland samples from the 2018 survey
lucas15 = lucas15.loc(lucas15['Point_ID'].isin(lucas['POINTID']))
lucas15 = lucas15.reset_index(drop=True)

# Average at GU scale
M_coarse = []
M_sand = []
M_clay = []
M_silt = []

for j in nuts.index:
    mean = lucas15.where(lucas15['NUTS_%s' % nuts['LEVL_CODE'][j]] == nuts['NUTS_ID'][j])
    mean = mean.reset_index(drop=True)
    M_coarse.append(mean['Coarse'].mean())
    M_sand.append(mean['Sand'].mean())
    M_clay.append(mean['Clay'].mean())
    M_silt.append(mean['Silt'].mean())

# Add to the GU file
nuts['coarse'] = M_coarse
nuts['sand'] = M_sand
nuts['clay'] = M_clay
nuts['silt'] = M_silt

# Save data
#nuts.to_file('/Users/ludovicharter/Desktop/PhD_Sorbonne/Data/era5_all.shp', driver='ESRI Shapefile')


# Plot data
plot = pd.DataFrame()
plot['id'] = ['Ymax_cropland', 'radiation', '2m_temperature', 'evaporation', 'potential_evaporation', 'total_precipitation',
        'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3', 'P-EV', 'P-PEV']
plot['cmap'] = ['YlGn', 'Oranges', 'Reds', 'Blues', 'Purples', 'GnBu', 'Greys', 'Greys', 'Greys', 'YlOrBr', 'YlOrBr']
plot['label'] = ['Ymax cropland 2014-2019 (kgN/ha/yr)', 'Clear-sky surface solar radiation 2014-2019 (kW/m2)',
                 'Surface temperature 2014-2019 (°C)', '1-hour accumulated evaporation 2014-2019 (mm)',
                 '1-hour accumulated potential evaporation 2014-2019 (mm)',
                 '1-hour accumulated precipitation 2014-2019 (mm)', 'Volumetric soil water (layer 1) 2014-2019 (m3/m3)',
                 'Volumetric soil water (layer 2) 2014-2019 (m3/m3)', 'Volumetric soil water (layer 3) 2014-2019 (m3/m3)',
                 'P - EV (mm)', 'P - PEV (mm)']

for i in plot.index:
    # Plot
    f, ax = plt.subplots(1, figsize=(15, 20))
    nuts.plot(plot['id'][i], ax=ax, colormap=plot['cmap'][i])
    nuts.boundary.plot(color='k', ax=ax)
    vmin = min(nuts[plot['id'][i]])
    vmax = max(nuts[plot['id'][i]])
    sm = plt.cm.ScalarMappable(cmap=plot['cmap'][i], norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, orientation="horizontal")
    cbar.set_label(plot['label'][i], fontsize=30)
    cbar.ax.tick_params(labelsize=25)
    ax.tick_params(axis='both', labelsize=25)
    #plt.savefig('/Users/ludovicharter/Desktop/PhD_Sorbonne/figures/NUTS_%s.pdf' % plot['id'][i], format='pdf', dpi=600)
    plt.show()


# ERA5 data
era = xarray.open_dataarray('/Users/ludovicharter/Desktop/PhD_Sorbonne/Data/temperature/adaptor.mars.internal-1701270260.0772028-12542-7-02fba47d-de87-4bda-a87c-20858bf96c25.nc')
era = era.assign_coords(longitude=(((era.longitude + 180) % 360) - 180)).sortby('longitude')
era.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
era.rio.write_crs("epsg:4326", inplace=True)

# Compute the time mean
yearly_mean = era.mean(axis=0)


# Land-sea and inland water mask
mask = xarray.open_dataarray('/Users/ludovicharter/Desktop/PhD_Sorbonne/Data/land-sea_mask/adaptor.mars.internal-1701166990.3851645-17722-9-1509f163-3bae-4d18-9bf0-34ab9f41b6b7.nc')
mask = mask.assign_coords(longitude=(((mask.longitude + 180) % 360) - 180)).sortby('longitude')
mask.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
mask = mask.where(mask > 0.5, 0)
yearly_mean = yearly_mean.where(mask > 0) # mask region with a water surface


# Compute the mean for each NUTS region
means = []
for i in nuts['NUTS_ID']:
    n = nuts.where(nuts['NUTS_ID'] == i)
    n = n.dropna()
    clipped = yearly_mean.rio.clip(n.geometry.apply(mapping), n.crs, drop=False)
    m = float(clipped.mean())
    means.append(m)

nuts['mean_radiation'] = means

# Plot
ax = nuts.plot(figsize=(20, 20))
nuts.plot('mean_radiation', figsize=(20, 20), ax=ax, colormap='coolwarm')
nuts.boundary.plot(color='k', ax=ax)
vmin = min(nuts['mean_radiation'])
vmax = max(nuts['mean_radiation'])
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=vmin, vmax=vmax))
cbar = plt.colorbar(sm, ax=ax, shrink=0.5, orientation="horizontal")
cbar.set_label('2m temperature (K)', fontsize=30)
cbar.ax.tick_params(labelsize=25)
ax.tick_params(axis='both', labelsize=25)
plt.title('Regional distribution of surface temperature', fontsize=30, pad=30, fontweight='bold')
plt.savefig('/Users/ludovicharter/Desktop/PhD_Sorbonne/figures/NUTS_temperature.pdf', format='pdf', dpi=600)
plt.show()

# Plot scatter
nuts = nuts.dropna()
plt.scatter(nuts['mean_radiation'], nuts['Ymax_cropland'])
plt.ylabel('Ymax (KgN/ha/yr)', fontsize=15)
plt.xlabel('Temperature (K)', fontsize=15)
# Linear regression
res = linregress(nuts['mean_radiation'], nuts['Ymax_cropland'])
plt.axline(xy1=(0, res.intercept), slope=res.slope, linestyle="--", color="k")
p = pearsonr(nuts['mean_radiation'], nuts['Ymax_cropland'])
plt.text(272, 700, 'R2 = %s' % np.round(p[0]**2, 2), fontsize=15, c='k')
# Polynomial regression
coefficients = np.polyfit(nuts['mean_radiation'], nuts['Ymax_cropland'], 2)
xmod = np.linspace(min(nuts['mean_radiation']), max(nuts['mean_radiation']), 100)
modele = [coefficients[2] + coefficients[1] * val + coefficients[0] * val**2 for val in xmod]
nuts['y_pred'] = (coefficients[2] + (coefficients[1] * nuts['mean_radiation']) + (coefficients[0] * (nuts['mean_radiation']**2)))
plt.plot(xmod, modele, c="red", linestyle="--")
p = pearsonr(nuts['Ymax_cropland'], nuts['y_pred'])
plt.text(272, 800, 'R2 = %s' % np.round(p[0]**2, 2), fontsize=15, c='red')
plt.xlim(270, 295)
plt.ylim(0, 900)
plt.savefig('/Users/ludovicharter/Desktop/PhD_Sorbonne/figures/scatter_temperature.pdf', format='pdf', dpi=600)
plt.show()
'''

