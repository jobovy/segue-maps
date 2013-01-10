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
def plotActionData(options,args):
    #Read the data
    print "Reading the data ..."
    raw= read_rawdata(options)
    #Setup error mc integration
    options.nmcerr= 1#in case this isn't set correctly
    raw, errstuff= setup_err_mc(raw,options)
    #Bin the data
    binned= pixelAfeFeh(raw,dfeh=options.dfeh,dafe=options.dafe)
    #Map the bins with ndata > minndata in 1D
    fehs, afes= [], []
    for ii in range(len(binned.fehedges)-1):
        for jj in range(len(binned.afeedges)-1):
            data= binned(binned.feh(ii),binned.afe(jj))
            if len(data) < options.minndata:
                continue
            #print binned.feh(ii), binned.afe(jj), len(data)
            fehs.append(binned.feh(ii))
            afes.append(binned.afe(jj))
    nabundancebins= len(fehs)
    fehs= numpy.array(fehs)
    afes= numpy.array(afes)
    #print numpy.argmin((fehs+0.25)**2.+(afes-0.175)**2.)
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
    jr, lz, jz= [], [], []
    #Get data ready
    for indx in range(len(fehs)):
        R,vR,vT,z,vz= prepare_coordinates(params,indx,fehs,afes,binned,
                                          errstuff,
                                          options,1)
        tjr, tlz, tjz= aA(R.flatten(),vR.flatten(),vT.flatten(),z.flatten(),vz.flatten())
        jr.extend(list(tjr))
        lz.extend(list(tlz))
        jz.extend(list(tjz))
    #Now plot
    jr= numpy.array(jr)*ro*vo*_REFR0*_REFV0
    lz= numpy.array(lz)*ro*vo*_REFR0*_REFV0
    jz= numpy.array(jz)*ro*vo*_REFR0*_REFV0
    bovy_plot.bovy_print()
    levels= special.erf(0.5*numpy.arange(1,4))
    levels= list(levels)
    levels.append(1.01)
    print len(jr)
    if options.type.lower() == 'lzjr':
        bovy_plot.scatterplot(lz,jr,',',
                              xlabel=r'$L_z\ [\mathrm{km\,s}^{-1}\,\mathrm{kpc}]$',
                              ylabel=r'$J_R\ [\mathrm{km\,s}^{-1}\,\mathrm{kpc}]$',
                              xrange=[0.,3600.],
                              yrange=[0.,500.],
                              onedhists=True,
                              levels=levels)
    elif options.type.lower() == 'jrjz':
        bovy_plot.scatterplot(jr,jz,color='k',marker=',',ls='none',
                              xlabel=r'$J_R\ [\mathrm{km\,s}^{-1}\,\mathrm{kpc}]$',
                              ylabel=r'$J_Z\ [\mathrm{km\,s}^{-1}\,\mathrm{kpc}]$',
                              xrange=[0.,500.],
                              yrange=[0.,250.],
                              onedhists=True,
                              levels=levels)
    if options.sample == 'g':
        bovy_plot.bovy_text(r'$\mathrm{G\!-\!type\ dwarfs}$',top_right=True,size=16.)
    elif options.sample == 'k':
        bovy_plot.bovy_text(r'$\mathrm{K\!-\!type\ dwarfs}$',top_right=True,size=16.)
    bovy_plot.bovy_end_print(options.outfilename)
    return None

if __name__ == '__main__':
    (options,args)= get_options().parse_args()
    plotActionData(options,args)
