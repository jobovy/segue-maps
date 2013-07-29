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
from matplotlib import pyplot
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
    #Setup potential
    params= numpy.array([-1.33663190049,0.998420232634,-3.49031638164,0.31949840593,-1.63965169376])
    try:
        pot= setup_potential(params,options,0)#Assume that the potential parameters come from a file with a single set of df parameters first
    except RuntimeError: #if this set of parameters gives a nonsense potential
        raise
    ro= 1.
    vo= params[1]
    aA= setup_aA(pot,options)
    jr, lz, jz= [], [], []
    #Get data ready
    params= [None,None,None,None,None,None,params[0],params[1],params[2],params[3],params[4]]
    for indx in range(len(fehs)):
        if numpy.log(monoAbundanceMW.hr(fehs[indx],afes[indx],k=(options.sample.lower() == 'k'))/8.) > -0.5 :
            continue
        R,vR,vT,z,vz,e,ee,eee= prepare_coordinates(params,indx,fehs,afes,binned,
                                          errstuff,
                                          options,1)
        tjr, tlz, tjz= aA(R.flatten(),vR.flatten(),vT.flatten(),z.flatten(),vz.flatten())
        jr.extend(list(tjr))
        lz.extend(list(tlz))
        jz.extend(list(tjz))
    #Now plot
    jr= numpy.array(jr).flatten()*ro*vo*_REFR0*_REFV0
    lz= numpy.array(lz).flatten()*ro*vo*_REFR0*_REFV0
    jz= numpy.array(jz).flatten()*ro*vo*_REFR0*_REFV0
    bovy_plot.bovy_print()
    levels= special.erf(numpy.sqrt(0.5)*numpy.arange(1,3))
    levels= list(levels)
    levels.append(1.01)
    print len(jr)
    if options.type.lower() == 'lzjr':
        axScatter, axHistx,axHisty= bovy_plot.scatterplot(lz/220.,jr/220.,',',
                              xlabel=r'$L_z\ (220\,\mathrm{km\,s}^{-1}\,\mathrm{kpc})$',
                              ylabel=r'$J_R\ (220\,\mathrm{km\,s}^{-1}\,\mathrm{kpc})$',
                              xrange=[0.,3600./220.],
                              yrange=[0.,500./220.],
                              onedhists=True,
                              bins=41,
                              levels=levels,retAxes=True)
        axScatter.set_xlim(0.,3600./220.)
        axScatter.set_ylim(0.,500./220.)
        axHistx.set_xlim( axScatter.get_xlim() )
        axHisty.set_ylim( axScatter.get_ylim() )
        #Calculate locus of 6 kpc pericenter
        nlzs= 1001
        plzs= numpy.linspace(0.,6./8.,nlzs)
        pjrs= numpy.zeros(nlzs)
        for ii in range(nlzs):
            pjrs[ii]= aA(6./8.,0.,plzs[ii]/6.*8.,0.,0.)[0]*ro*vo*_REFR0*_REFV0
        bovy_plot.bovy_plot(plzs*ro*vo*_REFR0*_REFV0/220.,
                            pjrs/220.,'k--',overplot=True)
        plzs= numpy.linspace(11./8.,2.,nlzs)
        pjrs= numpy.zeros(nlzs)
        for ii in range(nlzs):
            pjrs[ii]= aA(11./8.,0.,plzs[ii]/11.*8.,0.,0.)[0]*ro*vo*_REFR0*_REFV0
        bovy_plot.bovy_plot(plzs*ro*vo*_REFR0*_REFV0/220.,
                            pjrs/220.,'k--',overplot=True)
    elif options.type.lower() == 'jrjz':
        bovy_plot.scatterplot(jr/220.,jz/220.,color='k',marker=',',ls='none',
                              xlabel=r'$J_R\ (220\,\mathrm{km\,s}^{-1}\,\mathrm{kpc})$',
                              ylabel=r'$J_Z\ (220\,\mathrm{km\,s}^{-1}\,\mathrm{kpc})$',
                              xrange=[0.,500./220.],
                              yrange=[0.,250./220.],
                              bins=41,
                              onedhists=True,
                              levels=levels)
    #if options.sample == 'g':
    #    bovy_plot.bovy_text(r'$\mathrm{G\!-\!type\ dwarfs}$',top_right=True,size=16.)
    #elif options.sample == 'k':
    #    bovy_plot.bovy_text(r'$\mathrm{K\!-\!type\ dwarfs}$',top_right=True,size=16.)
    bovy_plot.bovy_end_print(options.outfilename)
    return None

if __name__ == '__main__':
    (options,args)= get_options().parse_args()
    plotActionData(options,args)
