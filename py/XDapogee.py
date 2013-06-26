import os, os.path
import sys
import math
import numpy
import pickle
from scipy import optimize, integrate, special
from optparse import OptionParser
import extreme_deconvolution
from galpy import potential
from galpy.util import save_pickles, bovy_plot, multi
import multiprocessing
from matplotlib import pyplot
from pixelFitDF import _REFR0, _REFV0, setup_potential, logprior_dlnvcdlnr
from fitDensz import cb
from calcDFResults import setup_options
import readTerminalData
from plotOverview import labels, ranges
from fitSurfwPot import get_options
_APOGEEREFV0= 235.
def XDapogee(options,args):
    #First load the chains
    savefile= open(args[0],'rb')
    thesesamples= pickle.load(savefile)
    savefile.close()
    vcs= numpy.array([s[0] for s in thesesamples])*_APOGEEREFV0/_REFV0
    dvcdrs= numpy.array([s[6] for s in thesesamples])*30. #To be consistent with this project's dlnvcdlnr 
    print numpy.mean(vcs)
    print numpy.mean(dvcdrs)
    #Now fit XD to the 2D PDFs
    ydata= numpy.zeros((len(vcs),2))
    ycovar= numpy.zeros((len(vcs),2))
    ydata[:,0]= numpy.log(vcs)
    ydata[:,1]= dvcdrs
    vcxamp= numpy.ones(options.g)/options.g
    vcxmean= numpy.zeros((options.g,2))
    vcxcovar= numpy.zeros((options.g,2,2))
    for ii in range(options.g):
        vcxmean[ii,:]= numpy.mean(ydata,axis=0)+numpy.std(ydata,axis=0)*numpy.random.normal(size=(2))/4.
        vcxcovar[ii,0,0]= numpy.var(ydata[:,0])
        vcxcovar[ii,1,1]= numpy.var(ydata[:,1])
    extreme_deconvolution.extreme_deconvolution(ydata,ycovar,
                                                vcxamp,vcxmean,vcxcovar)
    save_pickles(options.plotfile,
                 vcxamp,vcxmean,vcxcovar)
    print vcxamp
    print vcxmean[:,0]
    print vcxmean[:,1]
    return None

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    numpy.random.seed(options.seed)
    XDapogee(options,args)
