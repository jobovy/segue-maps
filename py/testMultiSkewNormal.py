plots2= True

import os, os.path
import sys
import cPickle as pickle
import math
import numpy
from scipy import optimize, special
from matplotlib import pyplot
from galpy.df import dehnendf
from galpy.util import bovy_plot, save_pickles
from skewnormal import skewnormal, logskewnormal, \
    multiskewnormal,logmultiskewnormal
from velocity_field import read_output
OUTDIR=os.path.join(os.getenv('DATADIR'),'bovy','nonaximw','elliptical')

if plots2:
    def optm(l,x):
        m= numpy.array([l[0],l[1]])
        V= numpy.array([[l[2],l[4]],[l[4],l[3]]])
        a= numpy.array([l[5],l[6]])
        return -numpy.sum(logmultiskewnormal(x,m,V,a))
    def gaussoptm(l,x):
        m= numpy.array([l[0],l[1]])
        V= numpy.array([[l[2],l[4]],[l[4],l[3]]])
        return -numpy.sum(logmultiskewnormal(x,m,V,[0.,0.]))
    savefilename= 'testMultiSkews2.sav'
    if os.path.exists(savefilename):
        savefile= open(savefilename,'rb')
        vs= pickle.load(savefile)
        savefile.close()
    else:
        dfstr= 'dehnen'
        s= numpy.exp(numpy.linspace(numpy.log(0.0125),numpy.log(0.2),21))
        so= s[-2]
        #so= 0.2
        elsavefilename= os.path.join(OUTDIR,
                                     'el_rect_so_%.6f_res_%i_grid_%i_tform_%.5f_tsteady_%.5f_cp_%.5f_nsigma_%.1f' % (so,101,101,-150.,125.,0.05,5) + 
                                     '_'+dfstr+'_%.6f_%.6f_%.6f.sav'  % (1./3.,1.,0.))
        surfmass,meanvr,meanvt,sigmar2,sigmat2,sigmart,vertexdev,surfmass_init,meanvt_init,sigmar2_init,sigmat2_init,ii,jj,grid = read_output(elsavefilename)
        indx= int(long(numpy.floor(45./180.*101)))
        grid= grid[indx]
        #Set up everything up for sampling
        sortindx= numpy.argsort(grid.df.flatten())[::-1]
        cumul= numpy.cumsum(numpy.sort(grid.df.flatten())[::-1])/numpy.sum(grid.df.flatten())
        vrs= numpy.zeros(grid.df.shape)
        vts= numpy.zeros(grid.df.shape)
        for ii in range(grid.df.shape[1]):
            vrs[:,ii]= grid.vRgrid.reshape((101,1))
        for ii in range(grid.df.shape[0]):
            vts[ii,:]= grid.vTgrid.reshape((101,1))
        vrs= vrs.flatten()
        vts= vts.flatten()
        vrs= numpy.take(vrs,sortindx)
        vts= numpy.take(vts,sortindx)
        #Now sample
        nsamples= 10000
        samples= []
        for ii in range(nsamples):
            kk= 0
            #generate random
            r= numpy.random.uniform()
            while cumul[kk] < r and kk < (len(vrs)-1): kk+= 1
            samples.append([vrs[kk],vts[kk]])
        vs= numpy.array(samples).T
        print numpy.mean(vs,axis=1)
        print vs.shape
        #Resample
        vs[0,:]+= 0.5*(2.*numpy.random.uniform(size=vs.shape[1])-1.)*(grid.vRgrid[1]-grid.vRgrid[0])
        vs[1,:]+= 0.5*(2.*numpy.random.uniform(size=vs.shape[1])-1.)*(grid.vTgrid[1]-grid.vTgrid[0])
        save_pickles(savefilename,vs)
    xs= numpy.linspace(-1.,1.,101)
    ys= numpy.linspace(0.,1.5,101)
    zs= numpy.zeros((2,len(xs)*len(ys)))
    for ii in range(len(xs)):
        zs[0,ii*len(ys):(ii+1)*len(ys)]= xs[ii]
        zs[1,ii*len(ys):(ii+1)*len(ys)]= ys
    m0= 0.
    m1= 1.
    V00= 0.2**2.
    V01= -0.0036
    V11= 0.2**2./2.
    a0= 0.
    a1= 0.1
    l= optimize.fmin(optm,[m0,m1,V00,V11,V01,a0,a1],(vs,))
    g= optimize.fmin(gaussoptm,[m0,m1,V00,V11,V01],(vs,))
    m= numpy.array([l[0],l[1]])
    V= numpy.array([[l[2],l[4]],[l[4],l[3]]])
    a= numpy.array([l[5],l[6]])
    X= multiskewnormal(zs,m,V,a)
    X= numpy.reshape(X,(len(xs),len(ys)))
    #Also cumulative for contouring
    sortindx= numpy.argsort(X.flatten())[::-1]
    cumul= numpy.cumsum(numpy.sort(X.flatten())[::-1])/numpy.sum(X.flatten())
    cntrThis= numpy.zeros(numpy.prod(X.shape))
    cntrThis[sortindx]= cumul
    cntrThis= numpy.reshape(cntrThis,X.shape)
    print "vertex deviation", -0.5*math.atan(2*V[0,1]/(V[0,0]-V[1,1]))/math.pi*180.
    print m
    print a
    print numpy.sqrt([V[0,0],V[1,1]])
    bovy_plot.bovy_print()
    axScatter, axHistx,axHisty= bovy_plot.scatterplot(vs[0,:],vs[1,:],
                                                      'k,',zorder=2,
                                                      xlabel=r'$v_R / v_0$',
                                                      ylabel=r'$v_T / v_0$',
                                                      yrange=[.0,1.5],
                                                      xrange=[-1.,1.],bins=51,
                                                      onedhists=True,
                                                      retAxes=True,
                                                      onedhistxnormed=True,
                                                      onedhistynormed=True)
    pyplot.contour(xs,ys,cntrThis.T,special.erf(0.5*numpy.arange(1,4)),
                   colors='r',linestyles='dashed')
    X/= numpy.sum(X)*(xs[1]-xs[0])*(ys[1]-ys[0])
    axHistx.plot(xs,numpy.sum(X,axis=1)*(ys[1]-ys[0]),'k-')
    axHisty.plot(numpy.sum(X,axis=0)*(xs[1]-xs[0]),ys,'k-')
    #Gaussian
    m= numpy.array([g[0],g[1]])
    V= numpy.array([[g[2],g[4]],[g[4],g[3]]])
    a= numpy.array([0.,0.])
    print "vertex deviation", -0.5*math.atan(2*V[0,1]/(V[0,0]-V[1,1]))/math.pi*180.
    print m
    print a
    print numpy.sqrt([V[0,0],V[1,1]])
    Y= multiskewnormal(zs,m,V,a)
    Y= numpy.reshape(Y,(len(xs),len(ys)))
    #Also cumulative for contouring
    sortindx= numpy.argsort(Y.flatten())[::-1]
    cumul= numpy.cumsum(numpy.sort(Y.flatten())[::-1])/numpy.sum(Y.flatten())
    cntrThis= numpy.zeros(numpy.prod(Y.shape))
    cntrThis[sortindx]= cumul
    cntrThis= numpy.reshape(cntrThis,Y.shape)
    pyplot.contour(xs,ys,cntrThis.T,special.erf(0.5*numpy.arange(1,4)),
                   colors='b',linestyles='dashed')
    Y/= numpy.sum(Y)*(xs[1]-xs[0])*(ys[1]-ys[0])
    axHistx.plot(xs,numpy.sum(Y,axis=1)*(ys[1]-ys[0]),'b-')
    axHisty.plot(numpy.sum(Y,axis=0)*(xs[1]-xs[0]),ys,'b-')
    bovy_plot.bovy_end_print('testMultiSkew.png')
