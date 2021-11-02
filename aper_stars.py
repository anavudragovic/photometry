#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:11:56 2020

@author: ana
"""
#import time
import os
#import sys
import glob
import fnmatch
import numpy as np

from astropy.wcs import WCS
from astropy.io import fits
from astropy.visualization import simple_norm
#from astropy.stats import gaussian_sigma_to_fwhm
from astropy.table import QTable

from photutils import aperture_photometry
from astropy.stats import sigma_clipped_stats
from photutils import CircularAperture, CircularAnnulus
from photutils.centroids import centroid_sources, centroid_2dg
#from photutils.detection import IRAFStarFinder
#rom photutils.background import MMMBackground, MADStdBackgroundRMS
#from photutils.centroids import centroid_2dg

from matplotlib import pyplot as plt
import re
# Celestial coordinates of stars in the single Landolt field
#field="SA35_SF1" 
field="GD2"
field_file = '/home/ana/vidojevica/31_10_2021/'+field+'_field.txt'
print(field_file)
radec = np.genfromtxt(field_file, names="name,raj2000,decj2000",delimiter=" ", 
                      usecols=(0,1,2), dtype=None, encoding=None) 

path = "/home/ana/vidojevica/31_10_2021/"
#star_names="bdf_wcskal*fit"
star_names="bdf_wcskal_GD2-00*.fit"
os.chdir(path)


for filename in glob.glob(star_names):
    for filter_name in {'B','V','R','I'}:
    # B filter in the desired Landolt field
      if fnmatch.fnmatch(filename, 'bdf_wcskal_'+field+'-00[0-9][0-9]_'+filter_name+'.fit'):
        result=re.search('-(.*)_', filename)
        num_file=result.group(1)
        print(field+' '+filter_name+'     ****************************************************************')
        hdul = fits.open(filename)
        image = hdul[0]
        science = image.data
        hdu = image.header        
        wcs = WCS(hdu)
        hdul.close()

        # Match Landolt stars with the image
        #pixel_coo = wcs.wcs_world2pix(radec['raj2000'], radec['decj2000'], 0)
        #positions = np.transpose(pixel_coo)
        #print(positions)
        pixel_coo = wcs.wcs_world2pix(radec['raj2000'], radec['decj2000'], 0)
        x_init, y_init = pixel_coo
        x, y = centroid_sources(science, x_init, y_init, box_size=21,
                        centroid_func=centroid_2dg)
        #norm_science = simple_norm(science_cps, 'sqrt', percent=99)
        pixel_coo = x, y
        positions = np.transpose(pixel_coo)
        #print(positions)


        # Overplot the sources on the image
        norm_science = simple_norm(science, 'sqrt', percent=99.97)
        #positions = np.transpose((sources_science['xcentroid'], sources_science['ycentroid']))
        apertures = CircularAperture(positions, r=12)
        annulus_apertures = CircularAnnulus(positions, r_in=20, r_out=30)
        fig = plt.figure(figsize=(8,8))
        plt.xlabel("NAXIS1 [pix]")
        plt.ylabel("NAXIS2 [pix]")
        plt.title(field+'   '+filter_name+'-band'+' ('+num_file+')')
        #plt.text(np.transpose(positions),radec['name'])
        i=0
        for x,y in positions:

            label = "{:s}".format(radec['name'][i].replace(field,'').replace('_','').replace('-',''))
            plt.annotate(label, # this is the text
                         (x,y), # this is the point to label
                         textcoords="offset points", # how to position the text
                         color='blue', 
                         xytext=(-10,5), ha='center') # horizontal alignment 
            i=i+1
        plt.imshow(science, norm=norm_science, cmap = 'Greys')
        apertures.plot(color='yellow', lw=1.5, alpha=0.7)
        annulus_apertures.plot(color='blue', lw=1.5, alpha=0.5)
        plt.gca().invert_yaxis()
        plt.savefig('stars_positions_'+field+'_'+filter_name+'_'+num_file+'.png', format='png',pdi=300)
        """ IRAF definitions of mag and mag_err
        flux = sum - area * msky
        mag = zmag - 2.5 * log10 (flux) + 2.5 * log10 (itime)
        merr = 1.0857 * err / flux
        err = sqrt (flux / gain + area * stdev**2 + area**2 * stdev**2 / nsky)
        """
        # Implement a mask that track the bad pixels and/or saturated pixels 
        # from the dq array, in order to exclude those regions from the data.
        #tic = time.perf_counter()
        bkg_median = []
        bkg_stddev = []
        annulus_masks = annulus_apertures.to_mask(method='center')
        for mask in annulus_masks:
            annulus_data = mask.multiply(science)
            annulus_data_1d = annulus_data[mask.data > 0]
            _, median_sigclip, annulus_stddev = sigma_clipped_stats(annulus_data_1d)
        bkg_median.append(median_sigclip)
        bkg_stddev.append(annulus_stddev)
        bkg_median = np.array(bkg_median)
        bkg_stddev = np.array(bkg_stddev)
        phot = aperture_photometry(science, apertures)
        phot['flux'] = phot['aperture_sum'] - bkg_median * apertures.area
        """ gain > 1 ? phot['flux'] -> phot['flux'] / gain """
        phot['flux_err'] = np.sqrt(phot['flux'] + apertures.area * bkg_stddev**2 * (1+apertures.area/annulus_apertures.area))
        phot['mag'] = -2.5*np.log10(phot['flux'])
        phot['mag_err'] = 1.0857 * phot['flux_err'] / phot['flux']
        for col in phot.colnames:
            phot[col].info.format = '%.8g'  # for consistent table output
        print(phot)

        # World coordinates
        # world = wcs.wcs_pix2world(sources_science['xcentroid'], sources_science['ycentroid'], 0)
        # ra = world[:1] dec = world[1:]
        #skycoo = QTable(world,names=('RAJ2000','DECJ2000'))
        #skycoo = QTable([radec['raj2000'], radec['decj2000']],names=('RAJ2000','DECJ2000'))        
        # Write output table with both aperture and psf photometry, including pixel and sky coo
        phot.add_column(radec['raj2000'], name='RAJ2000')
        phot.add_column(radec['decj2000'], name='DECJ2000')
        
        new_order=['RAJ2000','DECJ2000','flux','flux_err', 'mag', 'mag_err']
        phot[new_order].write('phot_'+field+'_'+num_file+'_'+filter_name+'.txt', format='ascii', overwrite=True)

        # =============================================================================
        # import imexam
        # ds9=imexam.connect(target="",path="/Applications/SAOImageDS9.app/Contents/MacOS/ds9",viewer="ds9",wait_time=10)
        # ds9.load_fits('/Users/anavudragovic/monika/05_08_2019/wcsbdfSA35_SF1_B10s.fit')
        # ds9.scale()
        # ds9.imexam()
        # ds9.close()
        # 
        # =============================================================================
