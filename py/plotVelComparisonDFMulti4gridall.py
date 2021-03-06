import os, os.path
import copy
import numpy
from scipy import ndimage
import cPickle as pickle
from optparse import OptionParser
from galpy.util import bovy_plot, bovy_coords, multi
from matplotlib import pyplot
import segueSelect
import compareDataModel
from segueSelect import read_gdwarfs, read_kdwarfs, _GDWARFFILE, _KDWARFFILE, \
    segueSelect, _ERASESTR
from fitDensz import cb, _ZSUN, DistSpline, _ivezic_dist, _NDS
from pixelFitDens import pixelAfeFeh
from pixelFitDF import *
from pixelFitDF import _SURFNRS, _SURFNZS, _PRECALCVSAMPLES, _REFR0, _REFV0, _VRSUN, _VTSUN, _VZSUN
from plotDensComparisonDFMulti4gridall import getMultiComparisonBins
_legendsize= 16
_NOTDONEYET= True
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
    M= len(gfehs)
    if options.andistances:
        distancefacs= numpy.zeros_like(fehs)
        gdistancefacs= numpy.zeros_like(gfehs)
        for jj in range(M):
            #Find pop corresponding to this bin
            ii= numpy.argmin((gfehs[jj]-fehs)**2./0.1+(gafes[jj]-afes)**2./0.0025)
            print ii
            #Get the relevant data
            data= binned(fehs[ii],afes[ii])
            distancefacs[ii]= AnDistance.AnDistance(data.dered_g-data.dered_r,
                                                    data.feh)
            gdistancefacs[jj]= distancefacs[ii]
            options.fixdm= numpy.log10(distancefacs[ii])*5.
            #Apply distance factor to the data
            newraw= read_rawdata(options)
            newbinned= pixelAfeFeh(newraw,dfeh=options.dfeh,dafe=options.dafe)
            thisdataIndx= binned.callIndx(fehs[ii],afes[ii])
            binned.data.xc[thisdataIndx]= newbinned.data.xc[thisdataIndx]
            binned.data.yc[thisdataIndx]= newbinned.data.yc[thisdataIndx]
            binned.data.zc[thisdataIndx]= newbinned.data.zc[thisdataIndx]
            binned.data.plate[thisdataIndx]= newbinned.data.plate[thisdataIndx]
            binned.data.dered_r[thisdataIndx]= newbinned.data.dered_r[thisdataIndx]
    else:
        distancefacs=numpy.ones_like(fehs)
        gdistancefacs=numpy.ones_like(gfehs)
    ##########POTENTIAL PARAMETERS####################
    potparams1= numpy.array([numpy.log(2.5/8.),options.fixvc/220.,
                             numpy.log(400./8000.),0.2,0.])
    if options.group.lower() == 'aenhanced':
        potparams2= numpy.array([numpy.log(2.8/8.),options.fixvc/220.,
                                 numpy.log(400./8000.),0.266666666,0.])
        potparams3= numpy.array([numpy.log(2.8/8.),options.fixvc/220.,
                                 numpy.log(400./8000.),0.8,0.])
    elif options.group.lower() == 'aintermediate':
        potparams2= numpy.array([numpy.log(3.0/8.),options.fixvc/220.,
                                 numpy.log(400./8000.),0.3333333333333,0.])
        potparams3= numpy.array([numpy.log(3.0/8.),options.fixvc/220.,
                                 numpy.log(400./8000.),0.933333333333,0.])
    elif options.group.lower() == 'apoor':
        potparams2= numpy.array([numpy.log(2.6/8.),options.fixvc/220.,
                                 numpy.log(400./8000.),0.4,0.])
        potparams3= numpy.array([numpy.log(2.6/8.),options.fixvc/220.,
                                 numpy.log(400./8000.),1.0,0.])
    options.potential=  'dpdiskplhalofixbulgeflatwgasalt'
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
        elif options.group == 'aintermediate':
            vs= numpy.linspace(-150.,150.,options.nv)
            xrange=[-150.,150.]
            bins= 33
        else: # options.group == 'aenhanced':
            vs= numpy.linspace(-120.,120.,options.nv)
            xrange=[-140.,140.]
            bins= 26
        xlabel=r'$V_Z\ (\mathrm{km\,s}^{-1})$'
    elif options.type.lower() == 'vr':
        if options.group == 'aenhanced':
            vs= numpy.linspace(-220.,220.,options.nv)
            xrange=[-220.,220.]
            bins= 39
        else: # options.group == 'aenhanced':
            vs= numpy.linspace(-150.,150.,options.nv)
            xrange=[-150.,150.]
            bins= 26
        xlabel=r'$V_R\ (\mathrm{km\,s}^{-1})$'
    elif options.type.lower() == 'vt':
        if options.group == 'aenhanced':
            vs= numpy.linspace(0.01,350.,options.nv)
            xrange=[0.,350.]
            bins= 39
        else: # options.group == 'aenhanced':
            vs= numpy.linspace(0.01,350.,options.nv)
            xrange=[0.,350.]
            bins= 39
        xlabel=r'$V_T\ (\mathrm{km\,s}^{-1})$'
    alts= True
    if not options.multi is None:
        #Generate list of temporary files
        tmpfiles= []
        for jj in range(M): 
            tfile, tmpfile= tempfile.mkstemp()
            os.close(tfile) #Close because it's open
            tmpfiles.append(tmpfile)
        try:
            dummy= multi.parallel_map((lambda x: run_calc_model_multi(x,M,gfehs,gafes,fehs,afes,options,
                                                                      vs,
                                                                      potparams1,potparams2,potparams3,
                                                                      distancefacs,
                                                                      binned,alts,True,tmpfiles)),
                                      range(M),
                                      numcores=numpy.amin([M,
                                                           multiprocessing.cpu_count(),
                                                           options.multi]))
            #Now read all of the temporary files
            for jj in range(M):
                tmpfile= open(tmpfiles[jj],'rb')
                tvelps= pickle.load(tmpfile)
                if tvelps is None:
                    continue
                tvelps2= pickle.load(tmpfile)
                tvelps3= pickle.load(tmpfile)
                data.extend(pickle.load(tmpfile))
                zs.extend(pickle.load(tmpfile))
                tndata= pickle.load(tmpfile)
                velps[cumulndata:cumulndata+tndata,:]= tvelps
                velps2[cumulndata:cumulndata+tndata,:]= tvelps2
                velps3[cumulndata:cumulndata+tndata,:]= tvelps3
                cumulndata+= tndata
                tmpfile.close()
        finally:
            for jj in range(M):
                os.remove(tmpfiles[jj])
    else:
        for jj in range(M):
            try:
                tvelps, tvelps2, tvelps3, tdata, tzs, tndata= run_calc_model_multi(jj,M,gfehs,gafes,fehs,afes,options,
                                                                                   vs,
                                                                                                                                potparams1,potparams2,potparams3,
                                                                      distancefacs,
                                                                                                                                binned,alts,
                                                                                                                                False,None)
            except TypeError:
                continue
            velps[cumulndata:cumulndata+tndata,:]= tvelps
            velps2[cumulndata:cumulndata+tndata,:]= tvelps2
            velps3[cumulndata:cumulndata+tndata,:]= tvelps3
            data.extend(tdata)
            zs.extend(tzs)
            cumulndata+= tndata
    bovy_plot.bovy_print()
    bovy_plot.bovy_hist(data,bins=26,normed=True,color='k',
                        histtype='step',
                        xrange=xrange,xlabel=xlabel)
    plotp= numpy.nansum(velps[:cumulndata,:],axis=0)/cumulndata
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
    #First determine the ranges that have nstars in them
    rranges_nstars= 1000
    zs= numpy.array(zs)
    data= numpy.array(data)
    tdata_z= sorted(numpy.fabs(zs))
    nbins= numpy.ceil(len(tdata_z)/float(rranges_nstars))
    rranges_nstars= int(numpy.floor(float(len(tdata_z))/nbins))
    accum= rranges_nstars
    zranges= [0.0]
    while accum < len(tdata_z):
        zranges.append(tdata_z[accum])
        accum+= rranges_nstars
    zranges.append(5.0)
    print zranges
    #zranges= [0.5,1.,1.5,2.,3.,4.]
    nzranges= len(zranges)-1
    sigzsd= numpy.empty(nzranges)
    esigzsd= numpy.empty(nzranges)
    sigzs1= numpy.empty(nzranges)
    sigzs2= numpy.empty(nzranges)
    sigzs3= numpy.empty(nzranges)
    for ii in range(nzranges):
        indx= (numpy.fabs(zs) >= zranges[ii])*(numpy.fabs(zs) < zranges[ii+1])
        plotp= numpy.nansum(velps[indx,:],axis=0)/numpy.sum(indx)
        yrange= [0.,1.35*numpy.nanmax(plotp)]
        bovy_plot.bovy_print()
        bovy_plot.bovy_hist(data[indx],bins=26,normed=True,color='k',
                            histtype='step',
                            yrange=yrange,
                            xrange=xrange,xlabel=xlabel)
        sigzsd[ii]= numpy.std(data[indx][(numpy.fabs(data[indx]) < 100.)])
        esigzsd[ii]= sigzsd[ii]/numpy.sqrt(float(len(data[indx][(numpy.fabs(data[indx]) < 100.)])))
        sigzs1[ii]= numpy.sqrt(numpy.sum(vs**2.*plotp)/numpy.sum(plotp)-(numpy.sum(vs*plotp)/numpy.sum(plotp))**2.)
        bovy_plot.bovy_plot(vs,plotp,'k-',overplot=True)
        if alts:
            plotp= numpy.nansum(velps2[indx,:],axis=0)/numpy.sum(indx)
            sigzs2[ii]= numpy.sqrt(numpy.sum(vs**2.*plotp)/numpy.sum(plotp)-(numpy.sum(vs*plotp)/numpy.sum(plotp))**2.)
            bovy_plot.bovy_plot(vs,plotp,'k--',overplot=True)
            plotp= numpy.nansum(velps3[indx,:],axis=0)/numpy.sum(indx)
            sigzs3[ii]= numpy.sqrt(numpy.sum(vs**2.*plotp)/numpy.sum(plotp)-(numpy.sum(vs*plotp)/numpy.sum(plotp))**2.)
            bovy_plot.bovy_plot(vs,plotp,'k:',overplot=True)
        bovy_plot.bovy_text(r'$ %i\ \mathrm{pc} \leq |Z| < %i\ \mathrm{pc}$' % (int(1000*zranges[ii]),int(1000*zranges[ii+1])),
#                            +'\n'+
#                            '$%i \ \ \mathrm{stars}$' % (numpy.sum(indx)),
                            top_right=True,
                            size=_legendsize)
        bovy_plot.bovy_end_print(args[0]+'model_data_g_'+options.group+'_'+options.type+'dist_z%.1f_z%.1f.' % (zranges[ii],zranges[ii+1])+options.ext)
    #Plot velocity dispersion as a function of |Z|
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot((((numpy.roll(zranges,-1)+zranges)/2.)[:-1]),sigzsd,
                        'ko',
                        xlabel=r'$|Z|\ (\mathrm{kpc})$',
                        ylabel=r'$\sigma_z\ (\mathrm{km\,s}^{-1})$',
                        xrange=[0.,4.],
                        yrange=[0.,60.])
    pyplot.errorbar(((numpy.roll(zranges,-1)+zranges)/2.)[:-1],sigzsd,
                    yerr=esigzsd,
                    marker='o',color='k',linestyle='none')
    bovy_plot.bovy_plot((((numpy.roll(zranges,-1)+zranges)/2.)[:-1]),sigzs1,
                        'r+',overplot=True,ms=10.)
    bovy_plot.bovy_plot((((numpy.roll(zranges,-1)+zranges)/2.)[:-1]),sigzs2,
                        'cx',overplot=True,ms=10.)
    bovy_plot.bovy_plot((((numpy.roll(zranges,-1)+zranges)/2.)[:-1]),sigzs3,
                        'gd',overplot=True,ms=10.)
    bovy_plot.bovy_end_print(args[0]+'model_data_g_'+options.group+'_'+options.type+'dist_szvsz.' +options.ext)
    return None

def run_calc_model_multi(jj,M,gfehs,gafes,fehs,afes,options,
                         vs,
                         potparams1,potparams2,potparams3,distancefacs,
                         binned,alts,savetopickle,tmpfiles):
    print "Working on group %i / %i ..." % (jj+1,M)
    #Find pop corresponding to this bin
    pop= numpy.argmin((gfehs[jj]-fehs)**2./0.1+(gafes[jj]-afes)**2./0.0025)
    #Apply distance factor
    toptions= copy.copy(options)
    if options.andistances:
        toptions.fixdm= numpy.log10(distancefacs[pop])*5.
    #Load initial parameters from file
    if not options.init is None:
        savename= options.init
        spl= savename.split('.')
        newname= ''
        for ll in range(len(spl)-1):
            newname+= spl[ll]
            if not ll == len(spl)-2: newname+= '.'
        newname+= '_%i.' % pop
        newname+= spl[-1]
        savefile= open(newname,'rb')
        try:
            if not _NOTDONEYET:
                params= pickle.load(savefile)
                mlogl= pickle.load(savefile)
            logl= pickle.load(savefile)
        except:
            if savetopickle:
                    save_pickles(tmpfiles[jj],None)
                    return None
            else:
                return None
        finally:
            savefile.close()
    else:
        raise IOError("base filename not specified ...")
    #First model is best-fit for this particular bin
    marglogl= numpy.zeros((8,16))
    for ll in range(8):
        for kk in range(16):
            marglogl[ll,kk]= logsumexp(logl[ll,0,0,kk,:,:,:,0])
    indx= numpy.unravel_index(numpy.nanargmax(marglogl),(8,16))
    print "Maximum for %i at %i,%i" % (pop,indx[0],indx[1])
    rds= numpy.linspace(2.0,3.4,options.nrds)/_REFR0
    rds= numpy.log(rds)
    fhs= numpy.linspace(0.,1.,options.nfhs)
    potparams1[0]= rds[indx[0]]
    potparams1[3]= fhs[indx[1]]
    #######DF PARAMETER RANGES###########
    hrs, srs, szs=  setup_dfgrid([gfehs[jj]],[gafes[jj]],toptions)
    dfindx= numpy.unravel_index(numpy.nanargmax(logl[indx[0],0,0,indx[1],:,:,:,0]),
                                (8,8,16))
    print "Maximum for %i at %i,%i,%i" % (pop,dfindx[0],dfindx[1],dfindx[2])
    tparams= initialize(toptions,[gfehs[jj]],[gafes[jj]])
    startindx= 0
    if options.fitdvt: startindx+= 1
    tparams[startindx]= hrs[dfindx[0]]
    tparams[startindx+4]= srs[dfindx[1]]
    tparams[startindx+2]= szs[dfindx[2]]
    tparams[startindx+5]= 0. #outlier fraction
    tparams= set_potparams(potparams1,tparams,toptions,1)
    thisdata= binned(fehs[pop],afes[pop])
    velps= calc_model(tparams,options,thisdata,vs)
    if alts:
        indx0= numpy.argmin((potparams2[0]-rds)**2.)
        indx1= numpy.argmin((potparams2[3]-fhs)**2.)
        #indx0= indx[0]
        #indx1= indx[1]
        dfindx= numpy.unravel_index(numpy.argmax(logl[indx0,0,0,indx1,:,:,:,0]),
                                    (8,8,16))
        tparams[startindx]= hrs[dfindx[0]]
        tparams[startindx+4]= srs[dfindx[1]]
        tparams[startindx+2]= szs[dfindx[2]]
        #print "BOVY: YOU HAVE MESSED WITH MODEL 2"
        tparams= set_potparams(potparams2,tparams,toptions,1)
        print "Working on model 2 ..."
        velps2=  calc_model(tparams,options,thisdata,vs)
        indx0= numpy.argmin((potparams3[0]-rds)**2.)
        indx1= numpy.argmin((potparams3[3]-fhs)**2.)
        dfindx= numpy.unravel_index(numpy.argmax(logl[indx0,0,0,indx1,:,:,:,0]),
                                    (8,8,16))
        tparams[startindx]= hrs[dfindx[0]]
        tparams[startindx+4]= srs[dfindx[1]]
        tparams[startindx+2]= szs[dfindx[2]]
        tparams= set_potparams(potparams3,tparams,toptions,1)
        print "Working on model 3 ..."
        velps3= calc_model(tparams,options,thisdata,vs)
    else:
        velps2= None
        velps3= None
    #Also add the correct data
    ro= get_ro(tparams,options)
    if 'vr' in options.type.lower() or 'vt' in options.type.lower():
        R= ((ro*_REFR0-thisdata.xc)**2.+thisdata.yc**2.)**(0.5)
        cosphi= (_REFR0*ro-thisdata.xc)/R
        sinphi= thisdata.yc/R
        vR= -(thisdata.vxc-_VRSUN)*cosphi+(thisdata.vyc+_VTSUN)*sinphi
        vT= (thisdata.vxc-_VRSUN)*sinphi+(thisdata.vyc+_VTSUN)*cosphi
    if options.type.lower() == 'vz':
        data= thisdata.vzc+_VZSUN
    elif options.type.lower() == 'vr':
        data= vR
    elif options.type.lower() == 'vt':
        data= vT
    zs= thisdata.zc
    ndata= len(thisdata)
    if savetopickle:
        save_pickles(tmpfiles[jj],velps,velps2,velps3,
                     data,zs,ndata)
        return None
    else:
        return (velps,velps2,velps3,
                data,zs,ndata)

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
        cosphi= (_REFR0*ro-data.xc)/R/ro/_REFR0
        sinphi= data.yc/R/ro/_REFR0
        for rr in range(len(data.xc)):
            rot= numpy.array([[cosphi[rr],sinphi[rr]],
                              [-sinphi[rr],cosphi[rr]]])
            sxy= cov_vxvyvz[rr,0:2,0:2]
            sRT= numpy.dot(rot,numpy.dot(sxy,rot.T))
            cov_vxvyvz[rr,0:2,0:2]= sRT
    for ii in range(len(data)):
        if options.type.lower() == 'vz':
            thisp= qdf.pvz(vs/_REFV0/vo,R[ii]+numpy.zeros(len(vs)),z[ii]+numpy.zeros(len(vs)),ngl=options.ngl,gl=True)
            ndimage.filters.gaussian_filter1d(thisp,
                                              data.vzc_err[ii]/(vs[1]-vs[0]),
                                              output=thisp)
        elif options.type.lower() == 'vr':
            thisp= numpy.array([qdf.pvR(v/_REFV0/vo,R[ii],z[ii],ngl=options.ngl,gl=True) for v in vs])
            ndimage.filters.gaussian_filter1d(thisp,
                                              numpy.sqrt(cov_vxvyvz[ii,0,0])/(vs[1]-vs[0]),
                                              output=thisp)
        elif options.type.lower() == 'vt':
            if options.fitdvt:
                dvt= get_dvt(params,options)
            else:
                dvt= 0.
            thisp= numpy.array([qdf.pvT(v/_REFV0/vo+dvt/vo,R[ii],z[ii],ngl=options.ngl,gl=True) for v in vs])
            ndimage.filters.gaussian_filter1d(thisp,
                                              numpy.sqrt(cov_vxvyvz[ii,1,1])/(vs[1]-vs[0]),
                                              output=thisp)
        out[ii,:]= thisp/numpy.sum(thisp)/(vs[1]-vs[0])
    return out

if __name__ == '__main__':
    (options,args)= get_options().parse_args()
    plotVelComparisonDFMulti(options,args)

