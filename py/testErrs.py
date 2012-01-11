import sys
import os, os.path
import cPickle as pickle
from optparse import OptionParser
import numpy
from scipy import special, maxentropy
from monoAbundanceMW import *
from galpy.util import bovy_plot, save_pickles
from segueSelect import _ERASESTR
_DFEH= 0.1
_DAFE= 0.05
_SQRTTWO= numpy.sqrt(2.)
#Pre-define
sig2zs= numpy.array([sigmaz(results['feh'][ii],results['afe'][ii])**2. for ii in range(len(results))])
fehs= results['feh']
afes= results['afe']
fehmax= fehs+_DFEH/2.
fehmin= fehs-_DFEH/2.
afemax= afes+_DAFE/2.
afemin= afes-_DAFE/2.
nbins= len(results)
def sigmazObs(z,feh,afe,dfeh,dafe):
    if isinstance(z,(list,numpy.ndarray)):
        pab= numpy.zeros((len(z),len(results)))
        integrated= _integrateFehAfeDist(feh,afe,dfeh,dafe)
        for ii in range(len(z)):
            for jj in range(len(results)):
                pab[ii,jj]= abundanceDist(results['feh'][jj],
                                          results['afe'][jj],z=z[ii])
            pab[ii,:]*= integrated
            pab[ii,:]/= numpy.sum(pab[ii,:])
            pab[ii,:]*= sig2zs
        return numpy.sqrt(numpy.sum(pab,axis=1))
    else:
        pab= numpy.array([abundanceDist(results['feh'][ii],
                                        results['afe'][ii],z=z) for ii in range(len(results))])
        pab*= _integrateFehAfeDist(feh,afe,dfeh,dafe)
        pab/= numpy.sum(pab)
        return numpy.sqrt(numpy.sum(pab*sig2zs))
        
def _integrateFehAfeDist(feh,afe,dfeh,dafe):
    return 0.25*(special.erf((fehmax-feh)/_SQRTTWO/dfeh)\
                     -special.erf((fehmin-feh)/_SQRTTWO/dfeh))\
                     *(special.erf((afemax-afe)/_SQRTTWO/dafe)\
                           -special.erf((afemin-afe)/_SQRTTWO/dafe))

def calcSlope(feh,afe,dfeh,dafe,options):
    if True: #So we can change this with a boolean option
        return (sigmazObs(2000.,feh,afe,dfeh,dafe)-sigmazObs(500.,feh,afe,dfeh,dafe))/1.5

def errsLogLike(dfeh,dafe,options):
    #For each bin, calculate the expected slope, and compare to measured slope
    loglike= 0.
    for ii in range(nbins):
        tslope= calcSlope(fehs[ii],afes[ii],dfeh,dafe,options)
        dslope, dslope_err= sigmazSlope(fehs[ii],afes[ii],err=True)
        loglike+= -0.5*(tslope-dslope)/dslope_err**2.
    return loglike

def testErrs(options,args):
    ndfehs, ndafes= 21, 21
    dfehs= numpy.linspace(0.01,0.4,ndfehs)
    dafes= numpy.linspace(0.01,0.3,ndafes)
    if os.path.exists(args[0]):
        savefile= open(args[0],'rb')
        loglike= pickle.load(savefile)
        ii= pickle.load(savefile)
        jj= pickle.load(savefile)
        savefile.close()
    else:
        loglike= numpy.zeros((ndfehs,ndafes))
        ii, jj= 0, 0
    while ii < ndfehs:
        while jj < ndafes:
            sys.stdout.write('\r'+"Working on %i / %i" %(ii*ndafes+jj+1,ndafes*ndfehs))
            sys.stdout.flush()
            loglike[ii,jj]= errsLogLike(dfehs[ii],dafes[jj],options)
            jj+= 1
            save_pickles(args[0],loglike,ii,jj)
        ii+= 1
        jj= 0
        save_pickles(args[0],loglike,ii,jj)
    save_pickles(args[0],loglike,ii,jj)
    sys.stdout.write('\r'+_ERASESTR+'\r')
    sys.stdout.flush()
    loglike/= maxentropy.logsumexp(loglike)
    loglike= numpy.exp(loglike)
    loglike/= numpy.sum(loglike)*(dfehs[1]-dfehs[0])*(dafes[1]-dafes[0])
    #Plot
    bovy_plot.bovy_print()
    bovy_plot.bovy_dens2d(loglike.T,origin='lower',
                          cmap='gist_yarg',
                          xlabel=r'\sigma_{[\mathrm{Fe/H}]}',
                          ylabel=r'\sigma_{[\alpha/\mathrm{Fe}]}',
                          xrange=[dfehs[0],dfehs[-1]],
                          yrange=[dafes[0],dafes[-1]],
                          contours=True,
                          cntrmass=True,
                          onedhists=True,
                          levels= special.erf(0.5*numpy.arange(1,4)))
    bovy_plot.bovy_end_print(options.plotfile)

def plotIllustrative(plotfilename):
    #Make an illustrative plot of the effect of uncertainties on sigmaz
    bovy_plot.bovy_print()
    feh, afe= -0.15,0.125
    nzs= 1001
    zs= numpy.linspace(500.,2000.,nzs)
    sigs= numpy.zeros(nzs)+sigmaz(feh,afe)
    bovy_plot.bovy_plot(zs,sigs,'k-',
                        xlabel=r'$|Z|\ [\mathrm{pc}]$',
                        ylabel=r'$\sigma_z(Z)\ [\mathrm{km\ s}^{-1}]$',
                        xrange=[0.,2700.],
                        yrange=[0.,60.])
    dfeh, dafe= 0.2, 0.1
    sigs= sigmazObs(zs,feh,afe,dfeh,dafe)
    bovy_plot.bovy_plot(zs,sigs,'k--',overplot=True)
    dfeh, dafe= .4,.2
    sigs= sigmazObs(zs,feh,afe,dfeh,dafe)
    bovy_plot.bovy_plot(zs,sigs,'k-.',overplot=True)
    feh, afe= -0.65,0.375
    nzs= 1001
    zs= numpy.linspace(500.,2000.,nzs)
    sigs= numpy.zeros(nzs)+sigmaz(feh,afe)
    bovy_plot.bovy_plot(zs,sigs,'k-',overplot=True)
    dfeh, dafe= 0.2, 0.1
    sigs= sigmazObs(zs,feh,afe,dfeh,dafe)
    bovy_plot.bovy_plot(zs,sigs,'k--',overplot=True)
    dfeh, dafe= 0.4,0.2
    sigs= sigmazObs(zs,feh,afe,dfeh,dafe)
    bovy_plot.bovy_plot(zs,sigs,'k-.',overplot=True)
    bovy_plot.bovy_end_print(plotfilename)

def get_options():
    usage = "usage: %prog [options] <savefilename>\n\nsavefilename= name of the file that the loglikes will be saved to"
    parser = OptionParser(usage=usage)
    parser.add_option("-o",dest='plotfile',
                      help="Name of file for plot")
    parser.add_option("--plotillustrative",action="store_true",
                      dest="plotillustrative",
                      default=False,
                      help="Make an illustrative plot")
    return parser

if __name__ == '__main__':
    parser= get_options()
    (options,args)= parser.parse_args()
    if options.plotillustrative:
        plotIllustrative(options.plotfile)
    else:
        testErrs(options,args)
