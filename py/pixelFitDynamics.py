import os, os.path
import sys
import math
import numpy
from scipy import optimize
import cPickle as pickle
from optparse import OptionParser
import multi
from galpy.util import bovy_coords, bovy_plot, save_pickles
import bovy_mcmc
from segueSelect import read_gdwarfs, read_kdwarfs, _GDWARFFILE, _KDWARFFILE
from fitSigz import _ZSUN
from pixelFitDens import pixelAfeFeh
_REFR0= 8. #kpc
_REFV0= 220. #km/s
def pixelFitDynamics(options,args):
    return None
def get_potparams(p,options,npops):
    """Function that returns the set of potential parameters for these options"""
    startindx= 0
    if options.fitro: startindx+= 1
    if options.fitvsun: startindx+= 3
    ndfparams= get_ndfparams(options)
    startindx+= ndfparams*npops
    if options.potential.lower == 'flatlog':
        return (p[startindx],p[startindx+1])

def get_dfparams(p,indx,options):
    """Function that returns the set of DF parameters for population indx for these options"""
    startindx= 0
    if options.fitro: startindx+= 1
    if options.fitvsun: startindx+= 3
    ndfparams= get_ndfparams(options)
    if options.dfmodel.lower() == 'qdf':
        return (p[startindx],p[startindx+1],p[startindx+2],p[startindx+3],
                p[startindx+4])

def get_ndfparams(options):
    """Function that returns the number of DF parameters for a single population"""
    if options.dfmodel.lower() == 'qdf':
        return 5

def get_ro(p,options):
    """Function that returns R0 for these options"""
    if options.fitro:
        return p[0]
    else:
        return 1.

def get_vsun(p,options):
    """Function to return motion of the Sun in the Galactocentric reference frame"""
    if options.fitvsun:
        return (p[1],p[2],p[3])
    else:
        return (-11.1/_REFV0,245./_REFV0,7.25/_REFV0) #BOVY:ADJUST?

def get_options():
    usage = "usage: %prog [options] <savefile>\n\nsavefile= name of the file that the fits will be saved to"
    parser = OptionParser(usage=usage)
    #Data options
    parser.add_option("--sample",dest='sample',default='g',
                      help="Use 'G' or 'K' dwarf sample")
    parser.add_option("--select",dest='select',default='all',
                      help="Select 'all' or 'program' stars")
    parser.add_option("--dfeh",dest='dfeh',default=0.05,type='float',
                      help="FeH bin size")   
    parser.add_option("--dafe",dest='dafe',default=0.05,type='float',
                      help="[a/Fe] bin size")   
    parser.add_option("--minndata",dest='minndata',default=100,type='int',
                      help="Minimum number of objects in a bin to perform a fit")   
    parser.add_option("--bmin",dest='bmin',type='float',
                      default=None,
                      help="Minimum Galactic latitude")
    parser.add_option("--rmin",dest='rmin',type='float',
                      default=None,
                      help="Minimum radius")
    parser.add_option("--rmax",dest='rmax',type='float',
                      default=None,
                      help="Maximum radius")
    parser.add_option("--snmin",dest='snmin',type='float',
                      default=15.,
                      help="Minimum S/N")
    #Potential model
    parser.add_option("--potential",dest='potential',default='flatlog',
                      help="Potential model to fit")
    #DF model
    parser.add_option("--dfmodel",dest='dfmodel',default='qdf',#Quasi-isothermal
                      help="DF model to fit")
    #Fit options
    parser.add_option("--fitro",action="store_true", dest="fitro",
                      default=False,
                      help="If set, fit for R_0")
    #Sample?
    parser.add_option("--mcsample",action="store_true", dest="mcsample",
                      default=False,
                      help="If set, sample around the best fit, save in args[1]")
    parser.add_option("--nsamples",dest='nsamples',default=1000,type='int',
                      help="Number of MCMC samples to obtain")
    return parser
  
if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    pixelFitDynamics(options,args)

