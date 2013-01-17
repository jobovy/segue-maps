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
from matplotlib import pyplot
from matplotlib.ticker import NullFormatter
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
    options.aAmethod='adiabatic'
    aA= setup_aA(pot,options)
    options.aAmethod='staeckel'
    aAS= setup_aA(pot,options)
    #Setup DF
    qdf= quasiisothermaldf(3./8.,0.19,0.125,0.875,0.875,aA=aAS,pot=pot,cutcounter=True)
    qdfa= quasiisothermaldf(3./8.,0.19,0.125,0.875,0.875,aA=aA,pot=pot,cutcounter=True)
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
    elif options.type.lower() == 'tilt':
        zs= numpy.linspace(0.,5.,101)
        tilt= numpy.array([qdf.tilt(1.,z/ro/_REFR0,gl=True) for z in zs])
        bovy_plot.bovy_print()
        line1= bovy_plot.bovy_plot(zs,tilt,'k-',
                                   xlabel=r'$Z\ [\mathrm{kpc}]$',
                                   ylabel=r'$\mathrm{tilt\ of\ the\ velocity\ ellipsoid}\ [\mathrm{deg}]$',
                                   xrange=[0.,5.],
                                   yrange=[-5.,30.])
        line2= bovy_plot.bovy_plot(zs,zs*0.,'k--',overplot=True)
        pyplot.legend((line1[0],line2[0]),
                      (r'$\mathrm{St\ddot{a}ckel\ actions}$',
                       r'$\mathrm{Adiabatic\ actions}$'),
                      loc='upper left',#bbox_to_anchor=(.91,.375),
                      numpoints=2,
                      prop={'size':16},
                      frameon=False)
        bovy_plot.bovy_plot(zs,numpy.arctan(zs/ro/_REFR0)/numpy.pi*180.,
                            '-',color='0.65',
                            overplot=True)
        pyplot.errorbar(numpy.array([1.]),
                        numpy.array([7.3]),
                        yerr=numpy.array([1.8,1.8]).reshape((2,1)),
                        color='k',fmt='o',ms=8)
        bovy_plot.bovy_text(.55,8.2,r'$\mathrm{S08}$',fontsize=14.)
    elif options.type.lower() == 'sigz':
        zs= numpy.linspace(0.,5.,101)
        sigz2= numpy.array([qdf.sigmaz2(1.,z/ro/_REFR0,gl=True) for z in zs])
        sigz2a= numpy.array([qdfa.sigmaz2(1.,z/ro/_REFR0,gl=True) for z in zs])
        bovy_plot.bovy_print()
        line1= bovy_plot.bovy_plot(zs,numpy.sqrt(sigz2)*vo*_REFV0,'k-',
                                   xlabel=r'$Z\ [\mathrm{kpc}]$',
                                   ylabel=r'$\sigma_Z(Z)\ [\mathrm{km\,s}^{-1}]$',
                                   xrange=[0.,5.],
                                   yrange=[0.,60.])
        line2= bovy_plot.bovy_plot(zs,numpy.sqrt(sigz2a)*vo*_REFV0,
                                   'k--',overplot=True)
        #pyplot.legend((line1[0],line2[0]),
        #              (r'$\mathrm{St\ddot{a}ckel\ actions}$',
        #               r'$\mathrm{Adiabatic\ actions}$'),
        #              loc='lower left',#bbox_to_anchor=(.91,.375),
        #              numpoints=2,
        #              prop={'size':16},
        #              frameon=False)
    elif options.type.lower() == 'densz':
        zs= numpy.linspace(0.,5.,101)
        densz= numpy.array([qdf.surfacemass(1.,z/ro/_REFR0,gl=True) for z in zs])
        densza= numpy.array([qdfa.surfacemass(1.,z/ro/_REFR0,gl=True) for z in zs])
        bovy_plot.bovy_print()
        line1= bovy_plot.bovy_plot(zs,densz/densz[0],'k-',
                                   xlabel=r'$Z\ [\mathrm{kpc}]$',
                                   ylabel=r'$\nu_*(R_0,Z)/\nu_*(R_0,0)$',
                                   xrange=[0.,5.],
                                   semilogy=True)
        line2= bovy_plot.bovy_plot(zs,densza/densza[0],'k--',overplot=True)
        pyplot.legend((line1[0],line2[0]),
                      (r'$\mathrm{St\ddot{a}ckel\ actions}$',
                       r'$\mathrm{Adiabatic\ actions}$'),
                      loc='upper right',#bbox_to_anchor=(.91,.375),
                      numpoints=2,
                      prop={'size':16},
                      frameon=False)
        #Create inset with profile at different R
        denszr12= numpy.array([qdf.surfacemass(11./8.,z/ro/_REFR0,gl=True) for z in zs])
        denszr4= numpy.array([qdf.surfacemass(5./8.,z/ro/_REFR0,gl=True) for z in zs])
        insetAxes= pyplot.axes([0.15,0.12,0.3,0.4])
        line1= insetAxes.semilogy(zs,densz/densz[0],'k-')
        line2= insetAxes.semilogy(zs,denszr12/denszr12[0],'k:')
        line3= insetAxes.semilogy(zs,denszr4/denszr4[0],'k--')
        nullfmt   = NullFormatter()         # no labels
        insetAxes.xaxis.set_major_formatter(nullfmt)
        insetAxes.yaxis.set_major_formatter(nullfmt)
        pyplot.legend((line3[0],line1[0],line2[0]),
                      (r'$R = 5\,\mathrm{kpc}$',
                       r'$R = 8\,\mathrm{kpc}$',
                       r'$R = 11\,\mathrm{kpc}$'),
                      loc='lower left',#bbox_to_anchor=(.91,.375),
                      numpoints=2,
                      prop={'size':10},
                      frameon=False)
    elif options.type.lower() == 'densr':
        rs= numpy.linspace(4.,15.,101)
        densr= numpy.array([qdf.surfacemass(r/ro/_REFR0,1./ro/_REFR0,gl=True) for r in rs])
        densra= numpy.array([qdfa.surfacemass(r/ro/_REFR0,1./ro/_REFR0,gl=True) for r in rs])
        bovy_plot.bovy_print()
        line1= bovy_plot.bovy_plot(rs,densr/densr[numpy.argmin((rs-8.)**2.)],'k-',
                                   xlabel=r'$R\ [\mathrm{kpc}]$',
                                   ylabel=r'$\nu_*(R,1\,\mathrm{kpc})/\nu_*(R_0,1\,\mathrm{kpc})$',
                                   xrange=[4.,15.],
                                   semilogy=True)
        line2= bovy_plot.bovy_plot(rs,densra/densra[numpy.argmin((rs-8.)**2.)],'k--',overplot=True)
        line3= bovy_plot.bovy_plot(rs,numpy.exp(-(rs-8.)/3.),'-',
                                   overplot=True,
                                   color='0.65') 
        #pyplot.legend((line1[0],line2[0]),
        #              (r'$\mathrm{St\ddot{a}ckel\ actions}$',
        #               r'$\mathrm{Adiabatic\ actions}$'),
        #              loc='lower left',#bbox_to_anchor=(.91,.375),
        #              numpoints=2,
        #              prop={'size':16},
        #              frameon=False)
    bovy_plot.bovy_end_print(options.outfilename)

if __name__ == '__main__':
    (options,args)= get_options().parse_args()
    plotFidDF(options,args)
