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
def pixelFitDynamics(options,args):
    return None
def get_options():
    usage = "usage: %prog [options] <savefile>\n\nsavefile= name of the file that the fits will be saved to"
    parser = OptionParser(usage=usage)
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
    parser.add_option("--potential",dest='model',default='flatlog',
                      help="Potential Model to fit")
#    parser.add_option("-o","--plotfile",dest='plotfile',default=None,
#                      help="Name of the file for plot")
#    parser.add_option("-t","--type",dest='type',default='sz',
#                      help="Quantity to plot ('sz', 'hs', 'afe', 'feh'")
#    parser.add_option("--plot",action="store_true", dest="plot",
#                      default=False,
#                      help="If set, plot, otherwise, fit")
#    parser.add_option("--tighten",action="store_true", dest="tighten",
#                      default=False,
#                      help="If set, tighten axes")
    parser.add_option("--mcsample",action="store_true", dest="mcsample",
                      default=False,
                      help="If set, sample around the best fit, save in args[1]")
    parser.add_option("--nsamples",dest='nsamples',default=1000,type='int',
                      help="Number of MCMC samples to obtain")
    parser.add_option("--rmin",dest='rmin',type='float',
                      default=None,
                      help="Minimum radius")
    parser.add_option("--rmax",dest='rmax',type='float',
                      default=None,
                      help="Maximum radius")
    parser.add_option("--snmin",dest='snmin',type='float',
                      default=15.,
                      help="Minimum S/N")
    return parser
  
if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    pixelFitDynamics(options,args)

