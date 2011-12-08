plotSigmas= False
plotVars= False
plotSkews= True

import os, os.path
import cPickle as pickle
import numpy
from scipy import optimize
from matplotlib import pyplot
from galpy.df import dehnendf
from galpy.util import bovy_plot, save_pickles
from skewnormal import skewnormal, logskewnormal, \
    multiskewnormal,logmultiskewnormal, alphaskew

if plotSigmas:
    def optm(l,x):
        return -numpy.sum(logskewnormal(x,l[0],numpy.exp(l[1]),
                                        l[2]))
    savefilename= 'testSkewSigmas.sav'
    if os.path.exists(savefilename):
        savefile= open(savefilename,'rb')
        vts01= pickle.load(savefile)
        vts02= pickle.load(savefile)
        vts04= pickle.load(savefile)
        savefile.close()
    else:
        dfc= dehnendf(beta=0.,correct=False,
                      profileParams=(1./3.,1.,0.1))
        vs= dfc.sampleVRVT(1.,n=100000,nsigma=5)
        vts01= vs[:,1].flatten()
        dfc= dehnendf(beta=0.,correct=False,
                      profileParams=(1./3.,1.,0.2))
        vs= dfc.sampleVRVT(1.,n=100000,nsigma=5)
        vts02= vs[:,1].flatten()
        dfc= dehnendf(beta=0.,correct=False,
                      profileParams=(1./3.,1.,0.4))
        vs= dfc.sampleVRVT(1.,n=100000,nsigma=5)
        vts04= vs[:,1].flatten()
        save_pickles(savefilename,vts01,vts02,vts04)
    xs= numpy.linspace(-1.,2.,1001)
    bf= optimize.fmin(optm,[.8,0.4,-0.3],(vts04,))
    ys= skewnormal(xs,bf[0],numpy.exp(bf[1]),bf[2])
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(xs,ys,'k-',zorder=2,
                        xlabel=r'$v_T / v_0$',
                        ylabel=r'$\mathrm{normalized\ distribution}$',
                        xrange=[-.5,1.5],
                        yrange=[0.,6.])
    bovy_plot.bovy_hist(vts04,bins=101,normed=True,
                        overplot=True,
                        histtype='step',
                        color='k')
    bf= optimize.fmin(optm,[9.,0.2,-0.1],(vts02,))
    ys= skewnormal(xs,bf[0],numpy.exp(bf[1]),bf[2])
    bovy_plot.bovy_hist(vts02,bins=101,normed=True,
                        histtype='step',
                        overplot=True,
                        color='k')
    bovy_plot.bovy_plot(xs,ys,'k-',overplot=True)
    bf= optimize.fmin(optm,[1.,0.1,0.1],(vts01,))
    ys= skewnormal(xs,bf[0],numpy.exp(bf[1]),bf[2])
    bovy_plot.bovy_hist(vts01,bins=101,normed=True,
                        histtype='step',
                        overplot=True,
                        color='k')
    bovy_plot.bovy_plot(xs,ys,'k-',overplot=True)
    bovy_plot.bovy_text(-0.3,4.,r'$\sigma_R = 0.1\ v_0$',size=14)
    bovy_plot.bovy_text(-0.3,2.,r'$\sigma_R = 0.2\ v_0$',size=14)
    bovy_plot.bovy_text(-0.3,1.,r'$\sigma_R = 0.4\ v_0$',size=14)
    bovy_plot.bovy_end_print('../tex-vel/testSkewSigmas.ps')

if plotVars:
    def optm(l,x):
        return -numpy.sum(logskewnormal(x,l[0],numpy.exp(l[1]),
                                        l[2]))
    savefilename= 'testSkewVars.sav'
    if os.path.exists(savefilename):
        savefile= open(savefilename,'rb')
        vtsfid= pickle.load(savefile)
        vtshR05= pickle.load(savefile)
        vtshR025= pickle.load(savefile)
        vtshs06= pickle.load(savefile)
        vtsbeta02= pickle.load(savefile)
        vtsbetam02= pickle.load(savefile)
        savefile.close()
    else:
        dfc= dehnendf(beta=0.,correct=False,
                      profileParams=(1./3.,1.,0.2))
        vs= dfc.sampleVRVT(1.,n=100000,nsigma=5)
        vtsfid= vs[:,1].flatten()
        dfc= dehnendf(beta=0.,correct=False,
                      profileParams=(1./2.,1.,0.2))
        vs= dfc.sampleVRVT(1.,n=100000,nsigma=5)
        vtshR05= vs[:,1].flatten()
        dfc= dehnendf(beta=0.,correct=False,
                      profileParams=(1./4.,1.,0.2))
        vs= dfc.sampleVRVT(1.,n=100000,nsigma=5)
        vtshR025= vs[:,1].flatten()
        dfc= dehnendf(beta=0.,correct=False,
                      profileParams=(1./3.,2./3.,0.2))
        vs= dfc.sampleVRVT(1.,n=100000,nsigma=5)
        vtshs06= vs[:,1].flatten()
        dfc= dehnendf(beta=0.2,correct=False,
                      profileParams=(1./3.,1.,0.2))
        vs= dfc.sampleVRVT(1.,n=100000,nsigma=5)
        vtsbeta02= vs[:,1].flatten()
        dfc= dehnendf(beta=-0.2,correct=False,
                      profileParams=(1./3.,1.,0.2))
        vs= dfc.sampleVRVT(1.,n=100000,nsigma=5)
        vtsbetam02= vs[:,1].flatten()
        save_pickles(savefilename,vtsfid,vtshR05,vtshR025,vtshs06,
                     vtsbeta02,vtsbetam02)
    xs= numpy.linspace(0.,1.5,1001)
    bf= optimize.fmin(optm,[1.,0.2,-0.2],(vtsfid,))
    ys= skewnormal(xs,bf[0],numpy.exp(bf[1]),bf[2])
    bovy_plot.bovy_print()
    pyplot.subplot(321)
    bovy_plot.bovy_plot(xs,ys,'k-',zorder=2,
                        xlabel=r'$v_T / v_0$',
                        ylabel=r'$\mathrm{normalized\ distribution}$',
                        xrange=[0.,1.5],
                        yrange=[0.,3.3],
                        overplot=True)
    bovy_plot.bovy_hist(vtsfid,bins=101,normed=True,
                        overplot=True,
                        histtype='step',
                        color='k')
    pyplot.xlim(0.,1.5)
    pyplot.ylim(0.,3.3)
    bf= optimize.fmin(optm,[1.,0.2,-0.2],(vtshs06,))
    ys= skewnormal(xs,bf[0],numpy.exp(bf[1]),bf[2])
    bovy_plot.bovy_text(r'$h_R = R_0 / 3$'+
                        '\n'+
                        r'$h_\sigma = 3\ h_R$'+
                        '\n'+
                        r'$\sigma_R = 0.2\ v_0$'+
                        '\n'+
                        r'$\beta = 0.0$',top_left=True)
    pyplot.subplot(322)
    bovy_plot.bovy_plot(xs,ys,'k-',zorder=2,
                        overplot=True)
    bovy_plot.bovy_hist(vtshs06,bins=101,normed=True,
                        overplot=True,
                        histtype='step',
                        color='k')
    pyplot.xlim(0.,1.5)
    pyplot.ylim(0.,3.3)
    bovy_plot.bovy_text(r'$h_\sigma = 2\ h_R$',top_left=True)
    bf= optimize.fmin(optm,[1.,0.2,-0.2],(vtshR05,))
    ys= skewnormal(xs,bf[0],numpy.exp(bf[1]),bf[2])
    pyplot.subplot(323)
    bovy_plot.bovy_plot(xs,ys,'k-',zorder=2,
                        overplot=True)
    bovy_plot.bovy_hist(vtshR05,bins=101,normed=True,
                        overplot=True,
                        histtype='step',
                        color='k')
    pyplot.xlim(0.,1.5)
    pyplot.ylim(0.,3.3)
    pyplot.ylabel(r'$\mathrm{normalized\ distribution}$')
    bovy_plot.bovy_text(r'$h_R = R_0 / 2$',top_left=True)
    bf= optimize.fmin(optm,[1.,0.2,-0.2],(vtshR025,))
    ys= skewnormal(xs,bf[0],numpy.exp(bf[1]),bf[2])
    pyplot.subplot(324)
    bovy_plot.bovy_plot(xs,ys,'k-',zorder=2,
                        overplot=True)
    bovy_plot.bovy_hist(vtshR025,bins=101,normed=True,
                        overplot=True,
                        histtype='step',
                        color='k')
    pyplot.xlim(0.,1.5)
    pyplot.ylim(0.,3.3)
    bovy_plot.bovy_text(r'$h_R = R_0 / 4$',top_left=True)
    bf= optimize.fmin(optm,[1.,0.2,-0.2],(vtsbeta02,))
    ys= skewnormal(xs,bf[0],numpy.exp(bf[1]),bf[2])
    pyplot.subplot(325)
    bovy_plot.bovy_plot(xs,ys,'k-',zorder=2,
                        overplot=True)
    bovy_plot.bovy_hist(vtsbeta02,bins=101,normed=True,
                        overplot=True,
                        histtype='step',
                        color='k')
    pyplot.xlim(0.,1.5)
    pyplot.ylim(0.,3.3)
    pyplot.xlabel(r'$v_T / v_0$',)
    bovy_plot.bovy_text(r'$\beta = 0.2$',top_left=True)
    bf= optimize.fmin(optm,[1.,0.2,-0.2],(vtsbetam02,))
    ys= skewnormal(xs,bf[0],numpy.exp(bf[1]),bf[2])
    pyplot.subplot(326)
    bovy_plot.bovy_plot(xs,ys,'k-',zorder=2,
                        overplot=True)
    bovy_plot.bovy_hist(vtsbetam02,bins=101,normed=True,
                        overplot=True,
                        histtype='step',
                        color='k')
    pyplot.xlabel(r'$v_T / v_0$',)
    pyplot.xlim(0.,1.5)
    pyplot.ylim(0.,3.3)
    bovy_plot.bovy_text(r'$\beta = -0.2$',top_left=True)
    bovy_plot.bovy_end_print('../tex-vel/testSkewVars.ps')
    
if plotSkews:
    dfc1= dehnendf(beta=0.,correct=False,
                   profileParams=(1./3.,1.,0.1))
    dfc2= dehnendf(beta=0.,correct=False,
                   profileParams=(1./3.,1.,0.2))
    dfc4= dehnendf(beta=0.,correct=False,
                   profileParams=(1./3.,1.,0.4))
    nrs= 101
    rs= numpy.linspace(0.1,2.2,nrs)
    savefilename= 'testSkewAlphas.sav'
    if os.path.exists(savefilename):
        savefile= open(savefilename,'rb')
        alphas1= pickle.load(savefile)
        alphas2= pickle.load(savefile)
        alphas4= pickle.load(savefile)
        savefile.close()
    else:
        alphas1= numpy.zeros(nrs)
        alphas2= numpy.zeros(nrs)
        alphas4= numpy.zeros(nrs)
        for ii in range(nrs):
            alphas1[ii]= alphaskew(dfc1.skewvT(rs[ii]))
            alphas2[ii]= alphaskew(dfc2.skewvT(rs[ii]))
            alphas4[ii]= alphaskew(dfc4.skewvT(rs[ii]))
            print alphas1[ii], alphas2[ii], alphas4[ii]
        save_pickles(savefilename,alphas1,alphas2,alphas4)
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(rs,alphas4,'k-',
                        xlabel=r'$R / R_0$',
                        ylabel=r'$\mathrm{shape}\ \alpha$',
                        xrange=[0.,2.3],
                        yrange=[-1.,1.])
    bovy_plot.bovy_plot(rs,alphas2,'k-',
                        overplot=True)
    bovy_plot.bovy_plot(rs,alphas1,'k-',
                        overplot=True)
    bovy_plot.bovy_text(1.4,-.45,r'$\sigma_R(R_0) = 0.1\ v_0$',size=14)
    bovy_plot.bovy_text(1.4,-.7,r'$\sigma_R(R_0) = 0.2\ v_0$',size=14)
    bovy_plot.bovy_text(1.4,-.93,r'$\sigma_R(R_0) = 0.4\ v_0$',size=14)
    bovy_plot.bovy_end_print('../tex-vel/testSkewAlphas.ps')
