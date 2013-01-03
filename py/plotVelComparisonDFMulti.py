import os, os.path
import copy
import numpy
from scipy import ndimage
import cPickle as pickle
from optparse import OptionParser
from galpy.util import bovy_plot, bovy_coords
import segueSelect
import compareDataModel
from segueSelect import read_gdwarfs, read_kdwarfs, _GDWARFFILE, _KDWARFFILE, \
    segueSelect, _ERASESTR
from fitDensz import cb, _ZSUN, DistSpline, _ivezic_dist, _NDS
from pixelFitDens import pixelAfeFeh
from pixelFitDF import *
from pixelFitDF import _SURFNRS, _SURFNZS, _PRECALCVSAMPLES, _REFR0, _REFV0, _VRSUN, _VTSUN, _VZSUN
from plotDensComparisonDFMulti import getMultiComparisonBins, get_median_potential
_legendsize= 16
def plotVelComparisonDFMulti(options,args):
    #Read data etc.
    print "Reading the data ..."
    raw= read_rawdata(options)
    #Bin the data
    binned= pixelAfeFeh(raw,dfeh=options.dfeh,dafe=options.dafe)
    #Map the bins with ndata > minndata in 1D
    fehs, afes= [], []
    for ii in range(len(binned.fehedges)-1):
        for jj in range(len(binned.afeedges)-1):
            data= binned(binned.feh(ii),binned.afe(jj))
            if len(data) < options.minndata:
                continue
            #print binned.feh(ii), binned.afe(jj), len(data)
            fehs.append(binned.feh(ii))
            afes.append(binned.afe(jj))
    nabundancebins= len(fehs)
    fehs= numpy.array(fehs)
    afes= numpy.array(afes)
    gafes, gfehs, left_legend= getMultiComparisonBins(options)
    if options.usemedianpotential:
        potparams= get_median_potential(options,nabundancebins)
        print "Median potential parameters: ", potparams
    M= len(gfehs)
    #Check whether fits exist, if not, pop
    removeBins= numpy.ones(M,dtype='bool')
    for jj in range(M):
        #Find pop corresponding to this bin
        pop= numpy.argmin((gfehs[jj]-fehs)**2./0.1+(gafes[jj]-afes)**2./0.0025)
        #Load savefile
        if not options.init is None:
            #Load initial parameters from file
            savename= options.init
            spl= savename.split('.')
            newname= ''
            for ll in range(len(spl)-1):
                newname+= spl[ll]
                if not ll == len(spl)-2: newname+= '.'
            newname+= '_%i.' % pop
            newname+= spl[-1]
            if not os.path.exists(newname):
                removeBins[jj]= False
        else:
            raise IOError("base filename not specified ...")
    if numpy.sum(removeBins) == 0:
        raise IOError("None of the group bins have been fit ...")
    elif numpy.sum(removeBins) < M:
        #Some bins have not been fit yet, and have to be remove
        gfehs= list((numpy.array(gfehs))[removeBins])
        gafes= list((numpy.array(gafes))[removeBins])
        print "Only using %i bins out of %i ..." % (numpy.sum(removeBins),M)
        M= len(gfehs)
    data= []
    zs= []
    velps= numpy.zeros((len(binned.data),options.nv))
    velps[:,:]= numpy.nan
    velps2= numpy.zeros((len(binned.data),options.nv))
    velps2[:,:]= numpy.nan
    velps3= numpy.zeros((len(binned.data),options.nv))
    velps3[:,:]= numpy.nan
    cumulndata= 0
    if options.type.lower() == 'vz':
        if options.group == 'aenhanced':
            vs= numpy.linspace(-180.,180.,options.nv)
            xrange=[-180.,180.]
            bins= 39
        else: # options.group == 'aenhanced':
            vs= numpy.linspace(-120.,120.,options.nv)
            xrange=[-120.,120.]
            bins= 26
        xlabel=r'$v_Z\ [\mathrm{km\,s}^{-1}]$'
    elif options.type.lower() == 'vr':
        if options.group == 'aenhanced':
            vs= numpy.linspace(-220.,220.,options.nv)
            xrange=[-220.,220.]
            bins= 39
        else: # options.group == 'aenhanced':
            vs= numpy.linspace(-150.,150.,options.nv)
            xrange=[-150.,150.]
            bins= 26
        xlabel=r'$v_R\ [\mathrm{km\,s}^{-1}]$'
    elif options.type.lower() == 'vt':
        if options.group == 'aenhanced':
            vs= numpy.linspace(0.01,450.,options.nv)
            xrange=[0.,450.]
            bins= 39
        else: # options.group == 'aenhanced':
            vs= numpy.linspace(0.01,250.,options.nv)
            xrange=[0.,250.]
            bins= 39
        xlabel=r'$v_T\ [\mathrm{km\,s}^{-1}]$'
    for jj in range(M):
        print "Working on group %i / %i ..." % (jj+1,M)
        #Find pop corresponding to this bin
        pop= numpy.argmin((gfehs[jj]-fehs)**2./0.1+(gafes[jj]-afes)**2./0.0025)
        #Load savefile
        if not options.init is None:
            #Load initial parameters from file
            savename= options.init
            spl= savename.split('.')
            newname= ''
            for ll in range(len(spl)-1):
                newname+= spl[ll]
                if not ll == len(spl)-2: newname+= '.'
            newname+= '_%i.' % pop
            newname+= spl[-1]
            savefile= open(newname,'rb')
            tparams= pickle.load(savefile)
            savefile.close()
        else:
            raise IOError("base filename not specified ...")
        if options.usemedianpotential:
            tparams= set_potparams(potparams,tparams,options,1)
        print tparams
        thisdata= binned(fehs[pop],afes[pop])
        velps[cumulndata:cumulndata+len(thisdata),:]= calc_model(tparams,options,thisdata,vs)
        alts= True
        if alts:
            if not options.usemedianpotential:
                potparams= get_potparams(tparams,options,1)
            if options.potential.lower() == 'flatlog':
                tparams= set_potparams([potparams[0]*1.05,potparams[1]],
                                       tparams,options,1)
                velps2[cumulndata:cumulndata+len(thisdata),:]= calc_model(tparams,options,thisdata,vs)
                tparams= set_potparams([potparams[0]*0.95,potparams[1]],
                                       tparams,options,1)
                velps3[cumulndata:cumulndata+len(thisdata),:]= calc_model(tparams,options,thisdata,vs)
        #Also add the correct data
        vo= get_vo(tparams,options,1)
        ro= get_ro(tparams,options)
        if 'vr' in options.type.lower() or 'vt' in options.type.lower():
            R= ((ro*_REFR0-thisdata.xc)**2.+thisdata.yc**2.)**(0.5)
            cosphi= (_REFR0*ro-thisdata.xc)/R
            sinphi= thisdata.yc/R
            vR= -(thisdata.vxc-_VRSUN)*cosphi+(thisdata.vyc+_VTSUN)*sinphi
            vT= (thisdata.vxc-_VRSUN)*sinphi+(thisdata.vyc+_VTSUN)*cosphi
        if options.type.lower() == 'vz':
            data.extend(thisdata.vzc+_VZSUN)
        elif options.type.lower() == 'vr':
            data.extend(vR)
        elif options.type.lower() == 'vt':
            data.extend(vT)
        zs.extend(thisdata.zc)
        cumulndata+= len(thisdata)
    bovy_plot.bovy_print()
    bovy_plot.bovy_hist(data,bins=26,normed=True,color='k',
                        histtype='step',
                        xrange=xrange,xlabel=xlabel)
    plotp= numpy.nansum(velps,axis=0)/cumulndata
    print numpy.sum(plotp)*(vs[1]-vs[0])
    bovy_plot.bovy_plot(vs,plotp,'k-',overplot=True)
    if alts:
        plotp= numpy.nansum(velps2,axis=0)/cumulndata
        bovy_plot.bovy_plot(vs,plotp,'k--',overplot=True)
        plotp= numpy.nansum(velps3,axis=0)/cumulndata
        bovy_plot.bovy_plot(vs,plotp,'k:',overplot=True)
    if not left_legend is None:
        bovy_plot.bovy_text(left_legend,top_left=True,size=_legendsize)
    bovy_plot.bovy_text(r'$\mathrm{full\ subsample}$'
                        +'\n'+
                        '$%i \ \ \mathrm{stars}$' % 
                        len(data),top_right=True,
                        size=_legendsize)
    bovy_plot.bovy_end_print(args[0]+'model_data_g_'+options.group+'_'+options.type+'dist_all.'+options.ext)
    if options.all: return None
    #Plot zranges
    zranges= [0.5,1.,1.5,2.,3.,4.]
    nzranges= len(zranges)-1
    zs= numpy.array(zs)
    data= numpy.array(data)
    for ii in range(nzranges):
        indx= (numpy.fabs(zs) >= zranges[ii])*(numpy.fabs(zs) < zranges[ii+1])
        bovy_plot.bovy_print()
        bovy_plot.bovy_hist(data[indx],bins=26,normed=True,color='k',
                            histtype='step',
                            xrange=xrange,xlabel=xlabel)
        plotp= numpy.nansum(velps[indx,:],axis=0)/numpy.sum(indx)
        bovy_plot.bovy_plot(vs,plotp,'k-',overplot=True)
        if alts:
            plotp= numpy.nansum(velps2[indx,:],axis=0)/numpy.sum(indx)
            bovy_plot.bovy_plot(vs,plotp,'k--',overplot=True)
            plotp= numpy.nansum(velps3[indx,:],axis=0)/numpy.sum(indx)
            bovy_plot.bovy_plot(vs,plotp,'k:',overplot=True)
        bovy_plot.bovy_text(r'$ %i\ \mathrm{pc} \leq |Z| < %i\ \mathrm{pc}$' % (int(1000*zranges[ii]),int(1000*zranges[ii+1]))
                            +'\n'+
                            '$%i \ \ \mathrm{stars}$' % 
                            (numpy.sum(indx)),top_right=True,
                            size=_legendsize)
        bovy_plot.bovy_end_print(args[0]+'model_data_g_'+options.group+'_'+options.type+'dist_z%.1f_z%.1f.' % (zranges[ii],zranges[ii+1])+options.ext)
    return None

def calc_model(params,options,data,vs):
    out= numpy.zeros((len(data),options.nv))
    #Model
    vo= get_vo(params,options,1)
    ro= get_ro(params,options)
    pot= setup_potential(params,options,1)
    aA= setup_aA(pot,options)
    dfparams= get_dfparams(params,0,options,log=False)
    if options.dfmodel.lower() == 'qdf':
        #Normalize
        hr= dfparams[0]/ro
        sr= dfparams[1]/vo
        sz= dfparams[2]/vo
        hsr= dfparams[3]/ro
        hsz= dfparams[4]/ro
        #Setup
        qdf= quasiisothermaldf(hr,sr,sz,hsr,hsz,pot=pot,aA=aA,cutcounter=True)
    #Get coordinates
    R= ((ro*_REFR0-data.xc)**2.+data.yc**2.)**(0.5)/ro/_REFR0
    z= (data.zc+_ZSUN)/ro/_REFR0
    if 'vr' in options.type.lower() or 'vt' in options.type.lower():
        cov_vxvyvz= numpy.zeros((len(data),3,3))
        cov_vxvyvz[:,0,0]= data.vxc_err**2.
        cov_vxvyvz[:,1,1]= data.vyc_err**2.
        cov_vxvyvz[:,2,2]= data.vzc_err**2.
        cov_vxvyvz[:,0,1]= data.vxvyc_rho*data.vxc_err*data.vyc_err
        cov_vxvyvz[:,0,2]= data.vxvzc_rho*data.vxc_err*data.vzc_err
        cov_vxvyvz[:,1,2]= data.vyvzc_rho*data.vyc_err*data.vzc_err
        #Rotate vxvyvz to vRvTvz
        cosphi= (_REFR0*ro-data.xc)/R
        sinphi= data.yc/R
        for rr in range(len(data.xc)):
            rot= numpy.array([[cosphi[rr],sinphi[rr]],
                              [-sinphi[rr],cosphi[rr]]])
            sxy= cov_vxvyvz[rr,0:2,0:2]
            sRT= numpy.dot(rot,numpy.dot(sxy,rot.T))
            cov_vxvyvz[rr,0:2,0:2]= sRT
    for ii in range(len(data)):
        if options.type.lower() == 'vz':
            thisp= numpy.array([qdf.pvz(v/_REFV0/vo,R[ii],z[ii],ngl=options.ngl,gl=True) for v in vs])
            ndimage.filters.gaussian_filter1d(thisp,
                                              data.vzc_err[ii]/(vs[1]-vs[0]),
                                              output=thisp)
        elif options.type.lower() == 'vr':
            thisp= numpy.array([qdf.pvR(v/_REFV0/vo,R[ii],z[ii],ngl=options.ngl,gl=True) for v in vs])
            ndimage.filters.gaussian_filter1d(thisp,
                                              numpy.sqrt(cov_vxvyvz[ii,0,0])/(vs[1]-vs[0]),
                                              output=thisp)
        elif options.type.lower() == 'vt':
            thisp= numpy.array([qdf.pvT(v/_REFV0/vo,R[ii],z[ii],ngl=options.ngl,gl=True) for v in vs])
            ndimage.filters.gaussian_filter1d(thisp,
                                              numpy.sqrt(cov_vxvyvz[ii,1,1])/(vs[1]-vs[0]),
                                              output=thisp)
        out[ii,:]= thisp/numpy.sum(thisp)/(vs[1]-vs[0])
    return out

if __name__ == '__main__':
    (options,args)= get_options().parse_args()
    plotVelComparisonDFMulti(options,args)

