import os, os.path
import sys
import copy
import math
import numpy
from scipy import optimize, interpolate, linalg, special
from scipy.maxentropy import logsumexp
import cPickle as pickle
from optparse import OptionParser
from galpy.util import bovy_coords, bovy_plot, save_pickles
from galpy import potential
from galpy.df_src.quasiisothermaldf import quasiisothermaldf
from galpy.util import save_pickles
from segueSelect import read_gdwarfs, read_kdwarfs, _GDWARFFILE, _KDWARFFILE, \
    segueSelect, _mr_gi, _gi_gr, _ERASESTR, _append_field_recarray, \
    ivezic_dist_gr
from fitDensz import cb, _ZSUN, DistSpline, _ivezic_dist, _NDS
from pixelFitDens import pixelAfeFeh
from pixelFitDF import *
from pixelFitDF import _REFR0, _REFV0
from matplotlib import pyplot
from matplotlib.ticker import NullFormatter
def mn_zhexp(mp,z,dz):
    return dz/(numpy.log(mp.dens(1.,z-dz/2.))-numpy.log(mp.dens(1.,z+dz/2.)))
def plotMNDblExp(options,args):
    if options.type.lower() == 'zh':
        bs= numpy.linspace(0.1/8.,1.1/8.,11)
        zs= numpy.linspace(0.,2./8.,1001)
        bovy_plot.bovy_print()
        overplot=False
        for ii in range(len(bs)):
            mp= potential.MiyamotoNagaiPotential(a=0.4,b=bs[ii])
            f= mp.dens(1.,zs)
            df= (f-numpy.roll(f,1))/(zs[1]-zs[0])
            if not options.relative:
                y= -f/df/bs[ii]
                ylabel=r'$z_h / b$'
                yrange= [0.,2.]
            else:
                y= mn_zhexp(mp,zs,dz=1./8.)#-f/df
                ylabel=r'$z_h$'
                yrange= [0.,.1]
            bovy_plot.bovy_plot(zs*8.,y,
                                '-',color='%f' % (float(ii)/(len(bs)-1)*0.8),
                                xlabel=r'$Z\ [\mathrm{kpc}]$',
                                ylabel=ylabel,
                                yrange=yrange,
                                overplot=overplot)
            overplot= True
    elif options.type.lower() == 'rd':
        aas= numpy.linspace(.5/8.,8/8.,16)
        rs= numpy.linspace(0.5,1.5,1001)
        bovy_plot.bovy_print()
        overplot=False
        for ii in range(len(aas)):
            mp= potential.MiyamotoNagaiPotential(a=aas[ii],b=0.05)
            f= mp.dens(rs,1./8.)
            df= (f-numpy.roll(f,1))/(rs[1]-rs[0])
            if not options.relative:
                y= -f/df/aas[ii] 
                ylabel=r'$R_d / a$'
                yrange= [0.,2.]
            else:
                y= -f/df
                ylabel=r'$R_d$'
                yrange= [0.,1.0]
            bovy_plot.bovy_plot(rs*8.,y,
                                '-',color='%f' % (float(ii)/(len(aas)-1)*0.8),
                                xlabel=r'$R\ [\mathrm{kpc}]$',
                                ylabel=ylabel,
                                yrange=yrange,
                                overplot=overplot)
            overplot= True
    bovy_plot.bovy_end_print(options.outfilename)
    return None
        
if __name__ == '__main__':
    (options,args)= get_options().parse_args()
    plotMNDblExp(options,args)
