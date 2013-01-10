import os, os.path
import sys
import copy
import tempfile
import time
import math
import numpy
from scipy import optimize, interpolate, linalg, special
from scipy.maxentropy import logsumexp
import cPickle as pickle
from optparse import OptionParser
import multi
import multiprocessing
from galpy.util import bovy_coords, bovy_plot, save_pickles
from galpy import potential
from galpy.actionAngle import actionAngleAdiabatic
from galpy.actionAngle import actionAngleAdiabaticGrid
from galpy.actionAngle import actionAngleStaeckel
from galpy.actionAngle import actionAngleStaeckelGrid
from galpy.df_src.quasiisothermaldf import quasiisothermaldf
from galpy.util import save_pickles
import monoAbundanceMW
from segueSelect import read_gdwarfs, read_kdwarfs, _GDWARFFILE, _KDWARFFILE, \
    segueSelect, _mr_gi, _gi_gr, _ERASESTR, _append_field_recarray, \
    ivezic_dist_gr
from fitDensz import cb, _ZSUN, DistSpline, _ivezic_dist, _NDS
from pixelFitDens import pixelAfeFeh
from pixelFitDF import *
from pixelFitDF import _REFR0, _REFV0
def plotFidDF(options,args):
    #Load potential parameters
    if not options.init is None and os.path.exists(options.init):
        #Load initial parameters from file
        print "Loading parameters for file "+options.init
        savefile= open(options.init,'rb')
        params= pickle.load(savefile)
        print params
        savefile.close()
    else:
        raise IOError("--init with potential parameters needs to be set")
    try:
        pot= setup_potential(params,options,1)#Assume that the potential parameters come from a file with a single set of df parameters first
    except RuntimeError: #if this set of parameters gives a nonsense potential
        raise
    ro= get_ro(params,options)
    vo= get_vo(params,options,1)
    aA= setup_aA(pot,options)
    #Setup DF
    qdf= quasiisothermaldf(3./8.,0.19,0.125,0.875,0.875,aA=aA,pot=pot,cutcounter=True)
    if options.type.lower() == 'lzjr':
        njs= 201
        jrs= numpy.linspace(0.,500.,njs)/ro/vo/_REFR0/_REFV0
        lzs= numpy.linspace(0.,3600.,njs)/ro/vo/_REFR0/_REFV0
        plotthis= qdf((numpy.tile(jrs,(njs,1)),
                       numpy.tile(lzs,(njs,1)).T,
                       numpy.zeros((njs,njs))))
        bovy_plot.bovy_print()
        bovy_plot.bovy_dens2d(plotthis.T,origin='lower',cmap='gist_yarg',
                              xlabel=r'$L_z\ [\mathrm{km\,s}^{-1}\,\mathrm{kpc}]$',
                              ylabel=r'$J_R\ [\mathrm{km\,s}^{-1}\,\mathrm{kpc}]$',
                              xrange=[0.,3600.],
                              yrange=[0.,500.],
                              onedhists=True,
                              interpolation='nearest',
                              cntrmass=True,contours=True,
                              levels= special.erf(0.5*numpy.arange(1,4)))
    elif options.type.lower() == 'jrjz':
        njs= 201
        jrs= numpy.linspace(0.,500.,njs)/ro/vo/_REFR0/_REFV0
        jzs= numpy.linspace(0.,250.,njs)/ro/vo/_REFR0/_REFV0
        plotthis= qdf((numpy.tile(jrs,(njs,1)).T,
                       numpy.ones((njs,njs)),
                       numpy.tile(jzs,(njs,1))))
        bovy_plot.bovy_print()
        bovy_plot.bovy_dens2d(plotthis.T,origin='lower',cmap='gist_yarg',
                              xlabel=r'$J_R\ [\mathrm{km\,s}^{-1}\,\mathrm{kpc}]$',
                              ylabel=r'$J_Z\ [\mathrm{km\,s}^{-1}\,\mathrm{kpc}]$',
                              xrange=[0.,500.],
                              yrange=[0.,250.],
                              onedhists=True,
                              interpolation='nearest',
                              cntrmass=True,contours=True,
                              levels= special.erf(0.5*numpy.arange(1,4)))
    bovy_plot.bovy_end_print(options.outfilename)


if __name__ == '__main__':
    (options,args)= get_options().parse_args()
    plotFidDF(options,args)
