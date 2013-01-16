import os, os.path
import sys
import math
import numpy
from scipy import optimize
import cPickle as pickle
from optparse import OptionParser
from galpy.util import bovy_coords, bovy_plot, save_pickles
import bovy_mcmc
from segueSelect import read_gdwarfs, read_kdwarfs, _GDWARFFILE, _KDWARFFILE
from fitSigz import _IsothermLikeMinus, _HWRLikeMinus, _ZSUN, \
    _HWRLike, _IsothermLike, _HWRRZLike, _HWRRZLikeMinus
from pixelFitDens import pixelAfeFeh
from pixelFitDF import get_options, _VRSUN, _VZSUN, _DEGTORAD
def plot_distsystematic(options,args):
    if options.sample.lower() == 'g':
        if options.select.lower() == 'program':
            raw= read_gdwarfs(_GDWARFFILE,logg=True,ebv=True,sn=options.snmin)
        else:
            raw= read_gdwarfs(logg=True,ebv=True,sn=options.snmin)
    elif options.sample.lower() == 'k':
        if options.select.lower() == 'program':
            raw= read_kdwarfs(_KDWARFFILE,logg=True,ebv=True,sn=options.snmin)
        else:
            raw= read_kdwarfs(logg=True,ebv=True,sn=options.snmin)
    if not options.bmin is None:
        #Cut on |b|
        raw= raw[(numpy.fabs(raw.b) > options.bmin)]
    #Bin the data
    binned= pixelAfeFeh(raw,dfeh=options.dfeh,dafe=options.dafe)
    if options.tighten:
        tightbinned= pixelAfeFeh(raw,dfeh=options.dfeh,dafe=options.dafe,
                                 fehmin=-1.6,fehmax=0.5,afemin=-0.05,
                                 afemax=0.55)
    else:
        tightbinned= binned
    plotthis= numpy.zeros((tightbinned.npixfeh(),tightbinned.npixafe()))+numpy.nan
    #Run through the bins
    for ii in range(tightbinned.npixfeh()):
        for jj in range(tightbinned.npixafe()):
            data= binned(tightbinned.feh(ii),tightbinned.afe(jj))
            if len(data) < options.minndata:
                jj+= 1
                if jj == len(binned.afeedges)-1: 
                    jj= 0
                    ii+= 1
                    break
                continue               
            #Create XYZ and R, vxvyvz, cov_vxvyvz
            R= ((8.-data.xc)**2.+data.yc**2.)**0.5
            #Confine to R-range?
            if not options.rmin is None and not options.rmax is None:
                dataindx= (R >= options.rmin)*\
                    (R < options.rmax)
                data= data[dataindx]
                R= R[dataindx]
            XYZ= numpy.zeros((len(data),3))
            XYZ[:,0]= data.xc
            XYZ[:,1]= data.yc
            XYZ[:,2]= data.zc+_ZSUN
            vxvyvz= numpy.zeros((len(data),3))
            vxvyvz[:,0]= data.vxc
            vxvyvz[:,1]= data.vyc
            vxvyvz[:,2]= data.vzc
            """
            cov_vxvyvz= numpy.zeros((len(data),3,3))
            cov_vxvyvz[:,0,0]= data.vxc_err**2.
            cov_vxvyvz[:,1,1]= data.vyc_err**2.
            cov_vxvyvz[:,2,2]= data.vzc_err**2.
            cov_vxvyvz[:,0,1]= data.vxvyc_rho*data.vxc_err*data.vyc_err
            cov_vxvyvz[:,0,2]= data.vxvzc_rho*data.vxc_err*data.vzc_err
            cov_vxvyvz[:,1,2]= data.vyvzc_rho*data.vyc_err*data.vzc_err
            """
            cosphi= (8.-XYZ[:,0])/R
            sinphi= XYZ[:,1]/R
            sinbeta= XYZ[:,2]/numpy.sqrt(R*R+XYZ[:,2]*XYZ[:,2])
            cosbeta= R/numpy.sqrt(R*R+XYZ[:,2]*XYZ[:,2])
            ndata= len(data.ra)
            cov_pmradec= numpy.zeros((ndata,2,2))
            cov_pmradec[:,0,0]= data.pmra_err**2.
            cov_pmradec[:,1,1]= data.pmdec_err**2.
            cov_pmllbb= bovy_coords.cov_pmrapmdec_to_pmllpmbb(cov_pmradec,data.ra,
                                                              data.dec,degree=True)
            """
            vR= -vxvyvz[:,0]*cosphi+vxvyvz[:,1]*sinphi
            vT= vxvyvz[:,0]*sinphi+vxvyvz[:,1]*cosphi
            vz= vxvyvz[:,2]
            vxvyvz[:,0]= vR               
            vxvyvz[:,1]= vT
            for rr in range(len(XYZ[:,0])):
                rot= numpy.array([[cosphi[rr],sinphi[rr]],
                                  [-sinphi[rr],cosphi[rr]]])
                sxy= cov_vxvyvz[rr,0:2,0:2]
                sRT= numpy.dot(rot,numpy.dot(sxy,rot.T))
                cov_vxvyvz[rr,0:2,0:2]= sRT
            """
            #calculate x and y
            lb= bovy_coords.radec_to_lb(data.ra,data.dec,degree=True)
            lb*= _DEGTORAD
            tuu= 1.-numpy.cos(lb[:,1])**2.*numpy.cos(lb[:,0])**2.
            tuv= -0.5*numpy.cos(lb[:,1])**2.*numpy.sin(2.*lb[:,0])
            tuw= -0.5*numpy.cos(lb[:,0])*numpy.sin(2.*lb[:,1])
            tvv= 1.-numpy.cos(lb[:,1])**2.*numpy.sin(lb[:,0])**2.
            tvw= -0.5*numpy.sin(2.*lb[:,1])*numpy.sin(lb[:,0])
            tww= numpy.cos(lb[:,1])**2.
            #x= tuu*_VRSUN+tuv*vxvyvz[:,1]+tuw*vxvyvz[:,2]
            #y= -tww*_VZSUN+tuw*vxvyvz[:,0]+tvw*vxvyvz[:,1]
            x= -tuu*numpy.mean(vxvyvz[:,0])+tuv*vxvyvz[:,1]+tuw*vxvyvz[:,2]
            y= -tww*numpy.mean(vxvyvz[:,2])+tuw*vxvyvz[:,0]+tvw*vxvyvz[:,1]
            if options.type.lower() == 'u':
                corcorr=0.
                plotthis[ii,jj]= (numpy.mean(vxvyvz[:,0]*x)-numpy.mean(vxvyvz[:,0])*numpy.mean(x))/(numpy.var(x)+numpy.mean(tuv**2.+tuw**2.)*numpy.var(vxvyvz[:,0]))
            elif options.type.lower() == 'meanu':
                plotthis[ii,jj]= numpy.mean(vxvyvz[:,0])
            elif options.type.lower() == 'meanw':
                plotthis[ii,jj]= numpy.mean(vxvyvz[:,2])
            else:
                corcorr= 0.25*numpy.mean(2.*sinbeta*cosbeta*numpy.sin(2.*lb[:,1])*numpy.cos(lb[:,0])*cosphi)*(numpy.var(vxvyvz[:,0])-numpy.var(vxvyvz[:,2]))\
                    -0.25*numpy.mean(2.*sinbeta*cosbeta*numpy.sin(2.*lb[:,1])*numpy.sin(lb[:,0])*sinphi)*(numpy.var(vxvyvz[:,0])-numpy.var(vxvyvz[:,2]))\
                    +0.25*numpy.mean(numpy.sin(lb[:,1])**2.*(cov_pmllbb[:,1,1]*data.dist**2.*4.74**2.-data.vr_err**2.))
                plotthis[ii,jj]= (numpy.mean(vxvyvz[:,2]*y)-numpy.mean(vxvyvz[:,2])*numpy.mean(y)-corcorr)/(numpy.var(y)+numpy.mean(tvw**2.+tuw**2.)*numpy.var(vxvyvz[:,2]))
            #print ii, jj, plotthis[ii,jj], corcorr, numpy.mean(vxvyvz[:,2]*y)-numpy.mean(vxvyvz[:,2])*numpy.mean(y)-corcorr
            jj+= 1
            if jj == len(binned.afeedges)-1: 
                jj= 0
                ii+= 1
            if jj == 0: #this means we've reset the counter 
                break
    #print plotthis
    #Set up plot
    if options.type.lower() == 'meanu':
        vmin, vmax= -20.,20.
        zlabel=r'$\mathrm{mean}\ U$'
    elif options.type.lower() == 'meanw':
        vmin, vmax= -20.,20.
        zlabel=r'$\mathrm{mean}\ W$'
    else:
        vmin, vmax= -0.2,0.2
        zlabel=r'$\mathrm{fractional\ distance\ overestimate}$'
    if options.tighten:
        xrange=[-1.6,0.5]
        yrange=[-0.05,0.55]
    else:
        xrange=[-2.,0.5]
        yrange=[-0.2,0.6]
    bovy_plot.bovy_print()
    bovy_plot.bovy_dens2d(plotthis.T,origin='lower',cmap='jet',
                          interpolation='nearest',
                          xlabel=r'$[\mathrm{Fe/H}]$',
                          ylabel=r'$[\alpha/\mathrm{Fe}]$',
                          zlabel=zlabel,
                          xrange=xrange,yrange=yrange,
                          vmin=vmin,vmax=vmax,
                          contours=False,
                          colorbar=True,shrink=0.78)
    bovy_plot.bovy_text(r'$\mathrm{median} = %.2f$' % (numpy.median(plotthis[numpy.isfinite(plotthis)])),
                        bottom_left=True,size=14.)
    bovy_plot.bovy_end_print(options.outfilename)
    return None

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    plot_distsystematic(options,args)
