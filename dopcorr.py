#!/usr/bin/env pythonimport numpy as np

import numpy as np

from astropy import time
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, FK5

def doppler_corrections(ra, dec, mean_time, obs_lat=38.433056, obs_lon=-79.839722, obs_alt=807.):
    """
    computes the projected velocity of the telescope wrt four coordinate systems: 
    geo, helio, bary, lsr.
    To correct a velocity axis to the LSR do:
    vcorr = doppler_corrections(ra, dec, mean_time)
    rv = rv + vcorr[3] + rv * vcorr[3] / consts.c
    where rv is the observed radial velocity.
    
    :param ra: right ascension in degrees, J2000 frame.
    :param dec: declination in degrees, J2000 frame.
    :param mean_time: mean time of observation in isot format. Example "2017-01-15T01:59:58.99"
    :param obs_lon: East-longitude of observatory in degrees.
    :param obs_lat: Latitude of observatory in degrees.
    :param obs_alt: Altitude of the observatory in meters.
    """

    # Initialize ra and dec of source.
    src = SkyCoord(ra, dec, frame='icrs', unit=u.deg)
    
    # Local properties.
    mytime = time.Time(mean_time, format='isot', scale='utc')
    location = EarthLocation.from_geodetic(lat=obs_lat*u.deg, lon=obs_lon*u.deg, height=obs_alt*u.m)
 
    # Orbital velocity of Earth with respect to the Sun.
    # helio = for source projected velocity of earth orbit with respect to the Sun center.
    # bary = for source projected velocity of earth + moon orbit with respect to the Sun center.
    barycorr = src.radial_velocity_correction(obstime=mytime, location=location)  
    barycorr = barycorr.to(u.km/u.s)
    heliocorr = src.radial_velocity_correction('heliocentric', obstime=mytime, location=location)  
    heliocorr = heliocorr.to(u.km/u.s)
        
    # Earth rotational velocity
    # Taken from chdoppler.pro, "Spherical Astronomy" R. Green p.270 
    lst = mytime.sidereal_time('apparent', obs_lon)
    obs_lat = obs_lat * u.deg
    obs_lon = obs_lon * u.deg
    hour_angle = lst - src.ra
    v_spin = -0.465 * np.cos( obs_lat ) * np.cos( src.dec ) * np.sin( hour_angle )

    # LSR defined as: Sun moves at 20.0 km/s toward RA=18.0h and dec=30.0d in 1900J coords
    # WRT objects near to us in Milky Way (not sun's rotation WRT to galactic center!)
    lsr_coord = SkyCoord( '18h', '30d', frame='fk5', equinox='J1900')
    lsr_coord = lsr_coord.transform_to(FK5(equinox='J2000'))

    lsr_comp = np.array([ np.cos(lsr_coord.dec.rad) * np.cos(lsr_coord.ra.rad), \
                          np.cos(lsr_coord.dec.rad) * np.sin(lsr_coord.ra.rad), \
                          np.sin(lsr_coord.dec.rad) ])

    src_comp = np.array([ np.cos(src.dec.rad) * np.cos(src.ra.rad), \
                          np.cos(src.dec.rad) * np.sin(src.ra.rad), \
                          np.sin(src.dec.rad) ])

    k = np.array( [lsr_comp[0]*src_comp[0], lsr_comp[1]*src_comp[1], lsr_comp[2]*src_comp[2]] )
    v_lsr = 20. * np.sum(k, axis=0) * u.km/u.s
    
    geo = - v_spin
    helio = heliocorr
    bary = barycorr
    lsr = barycorr + v_lsr
    vtotal = [geo, helio, bary, lsr]
    
    return vtotal
