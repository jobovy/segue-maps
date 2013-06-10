import os, os.path
import sys
import numpy
import pickle
from scipy import maxentropy, integrate
from galpy import potential
from galpy.util import bovy_plot, save_pickles
import monoAbundanceMW
from pixelFitDF import _REFV0, _REFR0, setup_potential
from calcDFResults import setup_options
from matplotlib import pyplot
def plotMAPMassScaleLength(plotfilename):
    #First calculate the mass-weighted scale length
    fehs= monoAbundanceMW.fehs()
    afes= monoAbundanceMW.afes()
    hrs= numpy.zeros_like(fehs)
    mass= numpy.zeros_like(fehs)
    rs= numpy.linspace(2.,11.5,101)
    rds= numpy.zeros_like(rs)
    for ii in range(len(fehs)):
        mass[ii]= monoAbundanceMW.abundanceDist(fehs[ii],afes[ii])
        hrs[ii]= monoAbundanceMW.hr(fehs[ii],afes[ii])
    hrs[hrs > 3.5]= 3.5
    hrs*= 0.92
    for ii in range(len(rs)):
        rds[ii]= numpy.sum(hrs*mass*numpy.exp(-(rs[ii]-8.)/hrs))/numpy.sum(mass*numpy.exp(-(rs[ii]-8.)/hrs))
    #Now plot
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(rs,rds,'k-',
                        xrange=[0.,12.],
                        yrange=[0.,4.],
                        xlabel=r'$R\ (\mathrm{kpc})$',
                        ylabel=r'$\mathrm{effective\ disk\ scale\ length\,(kpc)}$')
    pyplot.errorbar([7.],[2.3],
                    xerr=[2.],
                    yerr=[0.2],
                    elinewidth=1.,capsize=3,
                    linestyle='none',zorder=5,
                    marker='o',color='k')
    bovy_plot.bovy_end_print(plotfilename)
                
if __name__ == '__main__':
    plotMAPMassScaleLength(sys.argv[1])
