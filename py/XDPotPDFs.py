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
def XDPotPDFs(options,args):
    #First load the chains
    savefile= open(args[0],'rb')
    thesesamples= pickle.load(savefile)
    savefile.close()
    if not options.derivedfile is None:
        if os.path.exists(options.derivedfile):
            derivedfile= open(options.derivedfile,'rb')
            derivedsamples= pickle.load(derivedfile)
            derivedfile.close()
        else:
            raise IOError("--derivedfile given but does not exist ...")
    samples= {}
    scaleDict= {}
    paramnames= ['rd','vc','zh','fh','dlnvcdlnr','usun','vsun']
    scale= [_REFR0,_REFV0,1000.*_REFR0,1.,1./30.*_REFV0/_REFR0,_REFV0,_REFV0]
    if len(thesesamples[0]) == 5:
        paramnames.pop()
        paramnames.pop()
        scale.pop()
        scale.pop()
    if not options.derivedfile is None:
        paramnames.extend(['surfz','surfzdisk','rhodm',
                           'rhoo','massdisk','plhalo','vcdvc'])
        scale.extend([1.,1.,1.,1.,1.,1.,1.])
    for kk in range(len(thesesamples[0])):
        xs= numpy.array([s[kk] for s in thesesamples])
        if paramnames[kk] == 'rd' or paramnames[kk] == 'zh':
            xs= numpy.exp(xs)
        samples[paramnames[kk]]= xs
        scaleDict[paramnames[kk]]= scale[kk]
    if not options.derivedfile is None:
        for ll in range(len(thesesamples[0]),
                        len(thesesamples[0])+7):#len(derivedsamples[0])):
            kk= ll-len(thesesamples[0])
            xs= numpy.array([s[kk] for s in derivedsamples])
            samples[paramnames[ll]]= xs
            scaleDict[paramnames[ll]]= scale[ll]
    #Now fit XD to the three 2D PDFs
    #1) Vd/v vs. Rd
    ydata= numpy.zeros((len(samples['vcdvc']),2))
    ycovar= numpy.zeros((len(samples['vcdvc']),2))
    ydata[:,0]= numpy.log(samples['rd'])
    ydata[:,1]= special.logit(samples['vcdvc'])
    vcdxamp= numpy.ones(options.g)/options.g
    vcdxmean= numpy.zeros((options.g,2))
    vcdxcovar= numpy.zeros((options.g,2,2))
    for ii in range(options.g):
        vcdxmean[ii,:]= numpy.mean(ydata,axis=0)+numpy.std(ydata,axis=0)*numpy.random.normal(size=(2))/4.
        vcdxcovar[ii,0,0]= numpy.var(ydata[:,0])
        vcdxcovar[ii,1,1]= numpy.var(ydata[:,1])
    extreme_deconvolution.extreme_deconvolution(ydata,ycovar,
                                                vcdxamp,vcdxmean,vcdxcovar)
    #2) alpha_dm vs. rho_dm
    ydata= numpy.zeros((len(samples['rhodm']),2))
    ycovar= numpy.zeros((len(samples['rhodm']),2))
    ydata[:,0]= numpy.log(samples['rhodm'])
    ydata[:,1]= special.logit(samples['plhalo']/3.)
    rhodmxamp= numpy.ones(options.g)/options.g
    rhodmxmean= numpy.zeros((options.g,2))
    rhodmxcovar= numpy.zeros((options.g,2,2))
    for ii in range(options.g):
        rhodmxmean[ii,:]= numpy.mean(ydata,axis=0)+numpy.std(ydata,axis=0)*numpy.random.normal(size=(2))/4.
        rhodmxcovar[ii,0,0]= numpy.var(ydata[:,0])
        rhodmxcovar[ii,1,1]= numpy.var(ydata[:,1])
    extreme_deconvolution.extreme_deconvolution(ydata,ycovar,
                                                rhodmxamp,rhodmxmean,rhodmxcovar)
    #3) dlnvcdlnr vs. vc
    ydata= numpy.zeros((len(samples['vc']),2))
    ycovar= numpy.zeros((len(samples['vc']),2))
    ydata[:,0]= numpy.log(samples['vc'])
    ydata[:,1]= samples['dlnvcdlnr']
    vcxamp= numpy.ones(options.g)/options.g
    vcxmean= numpy.zeros((options.g,2))
    vcxcovar= numpy.zeros((options.g,2,2))
    for ii in range(options.g):
        vcxmean[ii,:]= numpy.mean(ydata,axis=0)+numpy.std(ydata,axis=0)*numpy.random.normal(size=(2))/4.
        vcxcovar[ii,0,0]= numpy.var(ydata[:,0])
        vcxcovar[ii,1,1]= numpy.var(ydata[:,1])
    extreme_deconvolution.extreme_deconvolution(ydata,ycovar,
                                                vcxamp,vcxmean,vcxcovar)
    save_pickles(options.plotfile,vcdxamp,vcdxmean,vcdxcovar,
                 rhodmxamp,rhodmxmean,rhodmxcovar,
                 vcxamp,vcxmean,vcxcovar)
    return None

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    numpy.random.seed(options.seed)
    XDPotPDFs(options,args)
