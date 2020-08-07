# photometry
Photometry of galaxies

# aper_stars.py
Stellar aperture photometry done with aperture = 1.7*fwhm. Stars are found using IRAFStarFinder. 
Sky in each annulus is a median value obtained after 3-sigma clipping, along with stddev. 
This sky value is subtracted from each aperture sum, and stddev was used for error calculation. 
Magnitude errors are done as in Iraf, although there was no output info on number of pixels in sky annulus. 
So, the formula has been rewritten as: 
mag_err = 1.0857 * sqrt (flux / gain + aperture_area * annulus_stddev2 + aperture_area2 * annulus_stddev**2 / annnulus_area)/flux, where gain=1.
