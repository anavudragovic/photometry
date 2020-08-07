#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:11:56 2020

@author: ana
"""
import time

import numpy as np

from astropy.wcs import WCS
from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.table import Table, QTable, hstack

from photutils import aperture_photometry
from astropy.stats import sigma_clipped_stats
from photutils import CircularAperture, CircularAnnulus
#from photutils import EPSFBuilder, find_peaks
from photutils.detection import IRAFStarFinder
#from photutils.psf import DAOGroup, IntegratedGaussianPRF, extract_stars, IterativelySubtractedPSFPhotometry
from photutils.background import MMMBackground, MADStdBackgroundRMS
#from photutils.centroids import centroid_2dg

#import matplotlib as plt
from matplotlib import style, pyplot as plt

galaxy = "/Volumes/Storage/Vidojevica/dark_swarp_all_cut.fits"

hdu = fits.open(galaxy)[0]
science = hdu.data
np.dtype('f8')
wcs = WCS(hdu.header)

# Implement a mask that track the bad pixels and/or saturated pixels 
# from the dq array, in order to exclude those regions from the data.
tic = time.perf_counter()

# Keeping a high threshold (10 sigma) in order to use only the bright stars in the FOV.
bkgrms = MADStdBackgroundRMS()
mmm_bkg = MMMBackground()

sigma_psf = 4.5/gaussian_sigma_to_fwhm  # fwhm=4.5
std_science = bkgrms(science)
bkg_science = mmm_bkg(science)
psf_stars_science = IRAFStarFinder(threshold=10*std_science + bkg_science,
                          fwhm=sigma_psf*gaussian_sigma_to_fwhm,
                          minsep_fwhm=10, roundhi=1.0, roundlo=-1.0,
                          sharplo=0.30, sharphi=1.40, peakmax=60000, 
                          exclude_border = True)


sources_science = psf_stars_science(science)
# sources_science.info()
toc = time.perf_counter()
print("Elapsed Time:", toc - tic, "number of sources detected for galaxy:", 
      len(sources_science))

# Overplot the sources on the image
norm_science = simple_norm(science, 'sqrt', percent=99.)
positions = np.transpose((sources_science['xcentroid'], sources_science['ycentroid']))
apertures = CircularAperture(positions, r=10)
annulus_apertures = CircularAnnulus(positions, r_in=20, r_out=30)
""" IRAF definitions of mag and mag_err
flux = sum - area * msky
mag = zmag - 2.5 * log10 (flux) + 2.5 * log10 (itime)
merr = 1.0857 * err / flux
err = sqrt (flux / gain + area * stdev**2 + area**2 * stdev**2 / nsky)
"""
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
world = wcs.wcs_pix2world(sources_science['xcentroid'], sources_science['ycentroid'], 0)
# ra = world[:1] dec = world[1:]
skycoo = QTable(world,names=('RAJ2000','DECJ2000'))
           
# Write output table with both aperture and psf photometry, including pixel and sky coo
#ascii.write(phot,, 'values.dat', names=['x', 'y'], overwrite=True)
phot.write('dark_swarp_all_cut_phot.txt', format='ascii')
sources_science.write('dark_swarp_all_cut_psf_phot.txt', format='ascii')
skycoo.write('dark_swarp_all_cut_radec.txt', format='ascii')
fig = plt.figure(figsize=(10,10))
#fig.add_subplot(111, projection=wcs)
#plt.imshow(science, origin='lower', norm=norm, cmap=plt.cm.viridis)
plt.xlabel("X [px]")
plt.ylabel("Y [px]")
plt.imshow(science, norm=norm_science, cmap = 'Greys')
apertures.plot(color='blue', lw=1.5, alpha=0.5)
plt.savefig('stars_positions.png', format='png',pdi=300)