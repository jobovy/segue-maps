import os, os.path
import sys
import math
import numpy
import pickle
import asciitable
from scipy import maxentropy, integrate, special
from galpy import potential
from galpy.util import bovy_plot, save_pickles
import monoAbundanceMW
import AnDistance
from pixelFitDF import _REFV0, _REFR0, setup_potential, read_rawdata
from pixelFitDens import pixelAfeFeh
from calcDFResults import setup_options
from matplotlib import pyplot
_RDFAC= 1.10
def plot_vcdvc_mstar(plotfilename):
    #Read pizagno data
    piz= asciitable.read("../data/pizagno05/table1.dat",
                         readme="../data/pizagno05/ReadMe")
    vcdvc= numpy.zeros(len(piz))
    for ii in range(len(piz)):
        zh= 0.05/2.2/piz['Rd'][ii]
        pot= potential.DoubleExponentialDiskPotential(hr=1./_RDFAC/2.2,
                                                      hz=zh,
                                                      normalize=1.)
        ro= 2.2*piz['Rd'][ii]/8.
        vo= piz['V2.2'][ii]/_REFV0
        rhod= pot.dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.
        massdisk= rhod*2.*zh*numpy.exp(_RDFAC*2.2)*(1./_RDFAC/2.2)**2.*2.*numpy.pi*(ro*_REFR0)**3./10.
        frac= piz['Mass'][ii]/massdisk
        vcdvc[ii]= numpy.sqrt(frac)
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(piz['Mass']*10**10.,vcdvc,'ko',
                        ms=5.,
                        semilogx=True,
                        yrange=[0.,1.1],
                        xrange=[10.**9.,10.**12.],
                        xlabel=r'$\mathrm{stellar\ disk\ mass}\,(M_\odot)$',
                        ylabel=r'$\mathrm{disk\ maximality}\equiv V_{c,\mathrm{disk}}/V_c$')
    pyplot.errorbar([4.6*10.**10.],[0.83],xerr=[0.3],yerr=[0.04],color='r',marker='d',ms=10.,mec='none')
    bovy_plot.bovy_end_print(plotfilename)
        
def plot_vcdvc_surfstar(plotfilename):
    #Read pizagno data
    piz= asciitable.read("../data/pizagno05/table1.dat",
                         readme="../data/pizagno05/ReadMe")
    vcdvc= numpy.zeros(len(piz))
    for ii in range(len(piz)):
        zh= 0.05/2.2/piz['Rd'][ii]
        pot= potential.DoubleExponentialDiskPotential(hr=1./_RDFAC/2.2,
                                                      hz=zh,
                                                      normalize=1.)
        ro= 2.2*piz['Rd'][ii]/8.
        vo= piz['V2.2'][ii]/_REFV0
        rhod= pot.dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.
        massdisk= rhod*2.*zh*numpy.exp(_RDFAC*2.2)*(1./_RDFAC/2.2)**2.*2.*numpy.pi*(ro*_REFR0)**3./10.
        frac= piz['Mass'][ii]/massdisk
        vcdvc[ii]= numpy.sqrt(frac)
    bovy_plot.bovy_print(fig_width=6.)
    plotrd= piz['Rd']/_RDFAC
    plotrd[plotrd > 6.]= 6.
    bovy_plot.bovy_plot(piz['Mass']*10**4./(piz['Rd']/_RDFAC)**2.,vcdvc,
                               c=plotrd,
                               s=25.,
                               edgecolor='none',
                               vmin=1.,vmax=6.,
                               clabel=r'$\mathrm{stellar\ disk\ scale\ length}\,(\mathrm{kpc})$',
                               scatter=True,
                               colorbar=True,
                               semilogx=True,
                               yrange=[0.,1.1],
                               xrange=[10.**2,10.**4.5],
                               xlabel=r'$\mathrm{stellar\ surface\ density}\,(M_\odot\,\mathrm{pc}^{-2})$',
                               ylabel=r'$\mathrm{disk\ maximality}\equiv V_{c,\mathrm{disk}}/V_c$')
    pyplot.errorbar([4.6*10.**4./2.15**2.],[0.83],yerr=[0.04],color='k',marker='d',ms=10.,mec='none')
    bovy_plot.bovy_text(300.,0.05,r'$\mathrm{data\ from\ Pizagno\ et\ al.\ (2005)}$'
                        +'\n'+
                        r'$\mathrm{Milky\!-\!Way\ (this\ paper)}$',
                        horizontalalignment='left',
                        size=16.)
    pyplot.errorbar([200.],[0.07],yerr=[0.032],color='k',marker='d',
                    ms=10.,mec='none')  
    bovy_plot.bovy_end_print(plotfilename)
        
if __name__ == '__main__':
    if sys.argv[1].lower() == 'vcdvc_mstar':
        plot_vcdvc_mstar(sys.argv[2])
    elif sys.argv[1].lower() == 'vcdvc_surfstar':
        plot_vcdvc_surfstar(sys.argv[2])

