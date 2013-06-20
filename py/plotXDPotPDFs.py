import os, os.path
import sys
import math
import numpy
import pickle
from scipy import optimize, integrate, special
from scipy.maxentropy import logsumexp
from optparse import OptionParser
import extreme_deconvolution
from galpy import potential
from galpy.util import save_pickles, bovy_plot, multi
import multiprocessing
from matplotlib import pyplot
from matplotlib.lines import Line2D
from pixelFitDF import _REFR0, _REFV0, setup_potential, logprior_dlnvcdlnr
from plotOverview import labels, ranges
from fitSurfwPot import get_options
_SQRTTWOPI= -0.5*numpy.log(2.*numpy.pi)
def plotXDPotPDFs_RD(options,args):
    #Read all XDs
    if os.path.exists(args[0]):
        savefile= open(args[0],'rb')
        vcdxamp= pickle.load(savefile)
        vcdxmean= pickle.load(savefile)
        vcdxcovar= pickle.load(savefile)
        savefile.close()
    else:
        raise IOError("At least one input file has to exist ...")
    #Plot vcd vs. rd
    nx= 101
    ny= 101
    xs= numpy.linspace(1.,5.,nx)
    ys= numpy.linspace(0.00001,0.999999,ny)
    XDxs= numpy.log(xs/_REFR0)
    XDys= special.logit(ys)
    pdf= _eval_gauss_grid(XDxs,XDys,vcdxamp,vcdxmean,vcdxcovar)
    pdf-= numpy.amax(pdf)
    pdf= numpy.exp(pdf)
    #Multiply in Jacobian
    for ii in range(nx):
        pdf[ii,:]*= 1./xs[ii]
    for jj in range(ny):
        pdf[:,jj]*= (1./ys[jj]+1./(1.-ys[jj]))
    #Plot
    nlevels= 2
    bovy_plot.bovy_print()
    c1= bovy_plot.bovy_dens2d(pdf.T,origin='lower',cmap='gist_yarg',
                          xrange=[xs[0]-(xs[1]-xs[0])/2.,xs[-1]+(xs[1]-xs[0])/2.],
                          yrange=[ys[0]-(ys[1]-ys[0])/2.,ys[-1]+(ys[1]-ys[0])/2.],
                          xlabel=r'$\mathrm{stellar\ disk\ scale\ length\,(kpc)}$',
                          ylabel=r'$\mathrm{disk\ maximality}\equiv V_{c,\mathrm{disk}}/V_c$',
                          contours=True,
                          levels= special.erf(numpy.arange(1,nlevels+1)/numpy.sqrt(2.)),
                          justcontours=True,cntrmass=True,
                          cntrcolors='r',
                          cntrlw=2.,retCont=True)
    if os.path.exists(args[1]):
        savefile= open(args[1],'rb')
        vcdxamp= pickle.load(savefile)
        vcdxmean= pickle.load(savefile)
        vcdxcovar= pickle.load(savefile)
        savefile.close()
    else:
        raise IOError("At least one input file has to exist ...")
    pdf= _eval_gauss_grid(XDxs,XDys,vcdxamp,vcdxmean,vcdxcovar)
    pdf-= numpy.amax(pdf)
    pdf= numpy.exp(pdf)
    #Multiply in Jacobian
    for ii in range(nx):
        pdf[ii,:]*= 1./xs[ii]
    for jj in range(ny):
        pdf[:,jj]*= (1./ys[jj]+1./(1.-ys[jj]))
    c2= bovy_plot.bovy_dens2d(pdf.T,origin='lower',cmap='gist_yarg',
                          xrange=[xs[0]-(xs[1]-xs[0])/2.,xs[-1]+(xs[1]-xs[0])/2.],
                          yrange=[ys[0]-(ys[1]-ys[0])/2.,ys[-1]+(ys[1]-ys[0])/2.],
                          overplot=True,
                          contours=True,
                          levels= special.erf(numpy.arange(1,nlevels+1)/numpy.sqrt(2.)),
                          justcontours=True,cntrmass=True,cntrcolors='y',
                          cntrlw=2.,retCont=True)
    if os.path.exists(args[2]):
        savefile= open(args[2],'rb')
        vcdxamp= pickle.load(savefile)
        vcdxmean= pickle.load(savefile)
        vcdxcovar= pickle.load(savefile)
        savefile.close()
    else:
        raise IOError("At least one input file has to exist ...")
    pdf= _eval_gauss_grid(XDxs,XDys,vcdxamp,vcdxmean,vcdxcovar)
    pdf-= numpy.amax(pdf)
    pdf= numpy.exp(pdf)
    #Multiply in Jacobian
    for ii in range(nx):
        pdf[ii,:]*= 1./xs[ii]
    for jj in range(ny):
        pdf[:,jj]*= (1./ys[jj]+1./(1.-ys[jj]))
    c3= bovy_plot.bovy_dens2d(pdf.T,origin='lower',cmap='gist_yarg',
                          xrange=[xs[0]-(xs[1]-xs[0])/2.,xs[-1]+(xs[1]-xs[0])/2.],
                          yrange=[ys[0]-(ys[1]-ys[0])/2.,ys[-1]+(ys[1]-ys[0])/2.],
                          overplot=True,
                          contours=True,
                          levels= special.erf(numpy.arange(1,nlevels+1)/numpy.sqrt(2.)),
                          justcontours=True,cntrmass=True,
                          cntrlw=2.,retCont=True)
    #Proxies
    c1= Line2D([0.],[0.],ls='-',lw=2.,color='r')
    c2= Line2D([0.],[0.],ls='-',lw=2.,color='y')
    c3= Line2D([0.],[0.],ls='-',lw=2.,color='k')
    pyplot.legend((c1,c2,c3),
                  (r'$\Sigma_{1.1}(R)\ \mathrm{only}$',
                   r'$V_\mathrm{term}, \Sigma(R_0,Z), \&\ \mathrm{d} \ln V_c / \mathrm{d} \ln R$',
                   r'$\mathrm{Combined}$'),
                  loc='lower left',#bbox_to_anchor=(.91,.375),
                  numpoints=2,
                  prop={'size':14},
                  frameon=False)
    bovy_plot.bovy_end_print(options.plotfile)
    return None

def _eval_gauss_grid(x,y,xamp,xmean,xcovar):
    nx= len(x)
    ny= len(y)
    out= numpy.zeros((nx,ny))
    ngauss= len(xamp)
    dim= xmean.shape[1]
    loglike= numpy.zeros(ngauss)
    for ii in range(nx):
        for jj in range(ny):
            a= numpy.array([x[ii],y[jj]])
            for kk in range(ngauss):
                if xamp[kk] == 0.:
                    loglike[kk]= numpy.finfo(numpy.dtype(numpy.float64)).min
                    continue
                tinv= numpy.linalg.inv(xcovar[kk,:,:])
                delta= a-xmean[kk,:]
                loglike[kk]= numpy.log(xamp[kk])+0.5*numpy.log(numpy.linalg.det(tinv))\
                    -0.5*numpy.dot(delta,numpy.dot(tinv,delta))+\
                    dim*_SQRTTWOPI
            out[ii,jj]= logsumexp(loglike)
    return out

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    numpy.random.seed(options.seed)
    if options.type.lower() == 'rd':
        plotXDPotPDFs_RD(options,args)
