import os, os.path
import numpy
from scipy import ndimage
import cPickle as pickle
from galpy.util import bovy_plot, multi
from matplotlib import pyplot
from segueSelect import read_gdwarfs, read_kdwarfs, _GDWARFFILE, _KDWARFFILE, \
    segueSelect
from fitDensz import _ZSUN
from pixelFitDens import pixelAfeFeh
from pixelFitDF import *
from pixelFitDF import _REFR0, _REFV0, _VRSUN, _VTSUN, _VZSUN, _SZHALO
from plotDensComparisonDFMulti import getMultiComparisonBins, get_median_potential
_legendsize= 16
_NOTDONEYET= True
_VARYHSZ= True
def plotVelComparisonDF(options,args):
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
    if not options.singlefeh is None:
        if True: #Just to keep indentation the same
            #Set up single feh
            indx= binned.callIndx(options.singlefeh,options.singleafe)
            if numpy.sum(indx) == 0:
                raise IOError("Bin corresponding to singlefeh and singleafe is empty ...")
            allraw= copy.copy(raw)
            raw= copy.copy(binned.data[indx])
            #newerrstuff= []
            #for ii in range(len(binned.data)):
            #    if indx[ii]: newerrstuff.append(errstuff[ii])
            #errstuff= newerrstuff
            print "Using %i data points ..." % (numpy.sum(indx))
            #Bin again
            binned= pixelAfeFeh(raw,dfeh=options.dfeh,dafe=options.dafe)
            fehs, afes= [], []
            for ii in range(len(binned.fehedges)-1):
                for jj in range(len(binned.afeedges)-1):
                    data= binned(binned.feh(ii),binned.afe(jj))
                    if len(data) < options.minndata:
                        continue
                    fehs.append(binned.feh(ii))
                    afes.append(binned.afe(jj))
            nabundancebins= len(fehs)
            fehs= numpy.array(fehs)
            afes= numpy.array(afes)
    if options.singles:
        run_abundance_singles_plotdens(options,args,fehs,afes)
        return None
    normintstuff= setup_normintstuff(options,raw,binned,fehs,afes,allraw)
    ##########POTENTIAL PARAMETERS####################
    potparams1= numpy.array([numpy.log(2.6/8.),230./220.,numpy.log(400./8000.),0.266666666,0.])
    potparams2= numpy.array([numpy.log(2.8/8.),230./220,numpy.log(400./8000.),0.266666666666,0.])
    #potparams2= numpy.array([numpy.log(2.5/8.),1.,numpy.log(400./8000.),0.466666,0.,2.])
    potparams3= numpy.array([numpy.log(2.6/8.),230./220.,
                             numpy.log(400./8000.),0.5333333,0.])
    pop= 0 #assume first population
    #Load savefile
    if not options.init is None:
        #Load initial parameters from file
        savename= options.init
#        spl= savename.split('.')
#        newname= ''
#        for ll in range(len(spl)-1):
#            newname+= spl[ll]
#            if not ll == len(spl)-2: newname+= '.'
#        newname+= '_%i.' % pop
#        newname+= spl[-1]
        savefile= open(savename,'rb')
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
    logl[numpy.isnan(logl)]= -numpy.finfo(numpy.dtype(numpy.float64)).max   
    #Set DF parameters as the maximum at R_d=2.4, f_h=0.4
    #######DF PARAMETER RANGES###########
    lnhr, lnsr, lnsz, rehr, resr, resz= approxFitResult(fehs[0],afes[0],
                                                        relerr=True)
    #if rehr > 0.3: rehr= 0.3 #regularize
    if True: rehr= 0.3 #regularize
    #if resr > 0.3: resr= 0.3
    #if resz > 0.3: resz= 0.3
    if True: resr= 0.3
    if True: resz= 0.3
    hrs= numpy.linspace(-1.85714286,0.9,options.nhrs)
    #hrs= numpy.linspace(lnhr-1.5*rehr,lnhr+1.5*rehr,options.nhrs)
    if _VARYHSZ:
        srs= numpy.linspace(numpy.log(0.5),numpy.log(2.),options.nsrs)#hsz now
    else:
        srs= numpy.linspace(lnsr-0.6*resz,lnsr+0.6*resz,options.nsrs)#USE ESZ
    szs= numpy.linspace(lnsz-0.6*resz,lnsz+0.6*resz,options.nszs)
    #hrs= numpy.linspace(lnhr-0.3,lnhr+0.3,options.nhrs)
    #srs= numpy.linspace(lnsr-0.1,lnsr+0.1,options.nsrs)
    #szs= numpy.linspace(lnsz-0.1,lnsz+0.1,options.nszs)
    dvts= numpy.linspace(-0.35,0.05,options.ndvts)
    #dvts= numpy.linspace(-0.05,0.05,options.ndvts)
    pouts= numpy.linspace(10.**-5.,.5,options.npouts)
    #indx= numpy.unravel_index(numpy.argmax(logl[3,0,0,3,:,:,:,:,:,0,0]),
    indx= numpy.unravel_index(numpy.argmax(logl[3,0,0,4,:,:,:,0]),
                              logl[3,0,0,4,:,:,:,0].shape)
    #tparams= numpy.array([dvts[indx[3]],hrs[indx[0]],
    if _VARYHSZ:
        tparams= numpy.array([0.,hrs[indx[0]],
                              #srs[indx[1]-2.*(indx[1] != 0)],
                              #szs[indx[2]-2.*(indx[2] != 0)],
                              lnsr,
                              szs[indx[2]],
                              numpy.log(8./_REFR0),
                              srs[indx[1]],
                              0.0,#logl[3,0,0,10,indx[0],indx[1],indx[2],2],#pouts[indx[4]],
                              0.,0.,0.,0.,0.])
    else:
        tparams= numpy.array([0.,hrs[indx[0]],
                              #srs[indx[1]-2.*(indx[1] != 0)],
                              #szs[indx[2]-2.*(indx[2] != 0)],
                              srs[indx[1]],
                              szs[indx[2]],
                              numpy.log(8./_REFR0),
                              numpy.log(7./_REFR0),0.,#pouts[indx[4]],
                              0.,0.,0.,0.,0.])
    options.potential=  'dpdiskplhalofixbulgeflatwgasalt'
    tparams= set_potparams(potparams1,tparams,options,1)
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
        xlabel=r'$V_Z\ [\mathrm{km\,s}^{-1}]$'
    elif options.type.lower() == 'vr':
        if options.group == 'aenhanced':
            vs= numpy.linspace(-220.,220.,options.nv)
            xrange=[-220.,220.]
            bins= 39
        else: # options.group == 'aenhanced':
            vs= numpy.linspace(-150.,150.,options.nv)
            xrange=[-150.,150.]
            bins= 26
        xlabel=r'$V_R\ [\mathrm{km\,s}^{-1}]$'
    elif options.type.lower() == 'vt':
        if options.group == 'aenhanced':
            vs= numpy.linspace(0.01,350.,options.nv)
            xrange=[0.,350.]
            bins= 39
        else: # options.group == 'aenhanced':
            vs= numpy.linspace(0.01,350.,options.nv)
            xrange=[0.,350.]
            bins= 39
        xlabel=r'$V_T\ [\mathrm{km\,s}^{-1}]$'
    thisdata= binned(fehs[pop],afes[pop])
    velps[cumulndata:cumulndata+len(thisdata),:]= calc_model(tparams,options,thisdata,vs,normintstuff=normintstuff)
    alts= True
    if alts:
        indx= numpy.unravel_index(numpy.argmax(logl[4,0,0,4,:,:,:,0]),
                                  logl[4,0,0,4,:,:,:,0].shape)
        #tparams= numpy.array([dvts[indx[3]],hrs[indx[0]],
        if not _VARYHSZ:
            tparams= numpy.array([0.,hrs[indx[0]],
                                  #srs[indx[1]-1.*(indx[1] != 0)],
                                  #szs[indx[2]-1.*(indx[2] != 0)],
                                  srs[indx[1]],
                                  szs[indx[2]],
                                  numpy.log(8./_REFR0),
                                  numpy.log(7./_REFR0),0.,#pouts[indx[4]],
                                  0.,0.,0.,0.,0.,0.])
        else:
            tparams= numpy.array([0.,hrs[indx[0]],
                                  #srs[indx[1]-1.*(indx[1] != 0)],
                                  #szs[indx[2]-1.*(indx[2] != 0)],
                                  lnsr,
                                  szs[indx[2]],
                                  numpy.log(8./_REFR0),
                                  srs[indx[1]],0.,#pouts[indx[4]],
                                  0.,0.,0.,0.,0.,0.])
        #options.potential= 'dpdiskplhalodarkdiskfixbulgeflatwgasalt'
        options.potential= 'dpdiskplhalofixbulgeflatwgasalt'
        tparams= set_potparams(potparams2,tparams,options,1)
        print "Working on model 2 ..."
        velps2[cumulndata:cumulndata+len(thisdata),:]= calc_model(tparams,options,thisdata,vs)
        indx= numpy.unravel_index(numpy.argmax(logl[3,0,0,8,:,:,:,0]),
                                  logl[3,0,0,8,:,:,:,0].shape)
        #tparams= numpy.array([dvts[indx[3]],hrs[indx[0]],
        if _VARYHSZ:
            tparams= numpy.array([0.,hrs[indx[0]],
                                  #srs[indx[1]-2.*(indx[1] != 0)],
                                  #szs[indx[2]-2.*(indx[2] != 0)],
                                  lnsr,
                                  szs[indx[2]],
                                  numpy.log(8./_REFR0),
                                  srs[indx[1]],0.,#pouts[indx[4]],
                                  0.,0.,0.,0.,0.])
        else:
            tparams= numpy.array([0.,hrs[indx[0]],
                                  #srs[indx[1]-2.*(indx[1] != 0)],
                                  #szs[indx[2]-2.*(indx[2] != 0)],
                                  srs[indx[1]],
                                  szs[indx[2]],
                                  numpy.log(8./_REFR0),
                                  numpy.log(7./_REFR0),0.,#pouts[indx[4]],
                                  0.,0.,0.,0.,0.])
        options.potential= 'dpdiskplhalofixbulgeflatwgasalt'
        tparams= set_potparams(potparams3,tparams,options,1)
        print "Working on model 3 ..."
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
    bovy_plot.bovy_text(r'$\mathrm{full\ subsample}$'
                        +'\n'+
                        '$%i \ \ \mathrm{stars}$' % 
                        len(data),top_right=True,
                        size=_legendsize)
    bovy_plot.bovy_end_print(args[0]+'model_data_g_'+options.type+'dist_all.'+options.ext)
    if options.all: return None
    #Plot zranges
    zranges= numpy.array([0.5,1.,1.5,2.,3.,4.])
    nzranges= len(zranges)-1
    zs= numpy.array(zs)
    data= numpy.array(data)
    sigzsd= numpy.empty(nzranges)
    esigzsd= numpy.empty(nzranges)
    sigzs1= numpy.empty(nzranges)
    sigzs2= numpy.empty(nzranges)
    sigzs3= numpy.empty(nzranges)
    for ii in range(nzranges):
        indx= (numpy.fabs(zs) >= zranges[ii])*(numpy.fabs(zs) < zranges[ii+1])
        bovy_plot.bovy_print()
        bovy_plot.bovy_hist(data[indx],bins=26,normed=True,color='k',
                            histtype='step',
                            xrange=xrange,xlabel=xlabel)
        sigzsd[ii]= numpy.std(data[indx][(numpy.fabs(data[indx]) < 100.)])
        esigzsd[ii]= sigzsd[ii]/numpy.sqrt(float(len(data[indx][(numpy.fabs(data[indx]) < 100.)])))
        plotp= numpy.nansum(velps[indx,:],axis=0)/numpy.sum(indx)
        sigzs1[ii]= numpy.sqrt(numpy.sum(vs**2.*plotp)/numpy.sum(plotp)-(numpy.sum(vs*plotp)/numpy.sum(plotp))**2.)
        bovy_plot.bovy_plot(vs,plotp,'k-',overplot=True)
        if alts:
            plotp= numpy.nansum(velps2[indx,:],axis=0)/numpy.sum(indx)
            sigzs2[ii]= numpy.sqrt(numpy.sum(vs**2.*plotp)/numpy.sum(plotp)-(numpy.sum(vs*plotp)/numpy.sum(plotp))**2.)
            bovy_plot.bovy_plot(vs,plotp,'k--',overplot=True)
            plotp= numpy.nansum(velps3[indx,:],axis=0)/numpy.sum(indx)
            sigzs3[ii]= numpy.sqrt(numpy.sum(vs**2.*plotp)/numpy.sum(plotp)-(numpy.sum(vs*plotp)/numpy.sum(plotp))**2.)
            bovy_plot.bovy_plot(vs,plotp,'k:',overplot=True)
        bovy_plot.bovy_text(r'$ %i\ \mathrm{pc} \leq |Z| < %i\ \mathrm{pc}$' % (int(1000*zranges[ii]),int(1000*zranges[ii+1]))
                            +'\n'+
                            '$%i \ \ \mathrm{stars}$' % 
                            (numpy.sum(indx)),top_right=True,
                            size=_legendsize)
        bovy_plot.bovy_end_print(args[0]+'model_data_g_'+options.type+'dist_z%.1f_z%.1f.' % (zranges[ii],zranges[ii+1])+options.ext)
    #Plot velocity dispersion as a function of |Z|
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot((((numpy.roll(zranges,-1)+zranges)/2.)[:5]),sigzsd,
                        'ko',
                        xlabel=r'$|Z|\ (\mathrm{kpc})$',
                        ylabel=r'$\sigma_z\ (\mathrm{km\,s}^{-1})$',
                        xrange=[0.,4.],
                        yrange=[0.,60.])
    pyplot.errorbar(((numpy.roll(zranges,-1)+zranges)/2.)[:5],sigzsd,
                    yerr=esigzsd,
                    marker='o',color='k',linestyle='none')
    bovy_plot.bovy_plot((((numpy.roll(zranges,-1)+zranges)/2.)[:5]),sigzs1,
                        'r+',overplot=True,ms=10.)
    bovy_plot.bovy_plot((((numpy.roll(zranges,-1)+zranges)/2.)[:5]),sigzs2,
                        'cx',overplot=True,ms=10.)
    bovy_plot.bovy_plot((((numpy.roll(zranges,-1)+zranges)/2.)[:5]),sigzs3,
                        'gd',overplot=True,ms=10.)
    bovy_plot.bovy_end_print(args[0]+'model_data_g_'+options.type+'dist_szvsz.'+options.ext)
    return None

def calc_model(params,options,data,vs,normintstuff=None):
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
    if not params[6] == 0.:
        print "Calculating normalization of qdf ..."
        normalization_qdf= calc_normint(qdf,0,normintstuff,params,1,options,
                                        -numpy.finfo(numpy.dtype(numpy.float64)).max)
        print "Calculating normalization of outliers ..."
        normalization_out= calc_normint(qdf,0,normintstuff,params,1,options,
                                        0.,fqdf=0.)   
    else:
        normalization_qdf= 0.
        normalization_out= 1.
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
    else:
        cov_vxvyvz= None # FOR MULTI
    if not options.multi is None:
        multOut= multi.parallel_map((lambda x: _calc_model_one(x,R,z,vs,qdf,options,data,params,cov_vxvyvz,vo,norm=normalization_qdf/normalization_out/12.)),
                                    range(len(data)),
                                    numcores=numpy.amin([len(data),
                                                         multiprocessing.cpu_count(),
                                                         options.multi]))
        for ii in range(len(data)):
            out[ii,:]= multOut[ii]
    else:
        for ii in range(len(data)):
            if options.type.lower() == 'vz':
                #thisp= numpy.array([qdf.pvz(v/_REFV0/vo,R[ii],z[ii],ngl=options.ngl,gl=True) for v in vs])
                thisp= qdf.pvz(vs/_REFV0/vo,R[ii]+numpy.zeros(len(vs)),z[ii]+numpy.zeros(len(vs)),ngl=options.ngl,gl=True)
                if not params[6] == 0.:
                    thisp+= params[6]*normalization_qdf/normalization_out/12./_SZHALO*_REFV0*vo*numpy.exp(-0.5*vs**2./_SZHALO**2.)*vo**2./numpy.sqrt(2.*math.pi)
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

def _calc_model_one(ii,R,z,vs,qdf,options,data,params,cov_vxvyvz,vo,norm=0.):
    if options.type.lower() == 'vz':
        thisp= qdf.pvz(vs/_REFV0/vo,R[ii]+numpy.zeros(len(vs)),z[ii]+numpy.zeros(len(vs)),ngl=options.ngl,gl=True)
        if not params[6] == 0.:
            thisp+= params[6]*norm/_SZHALO*_REFV0*vo*numpy.exp(-0.5*vs**2./_SZHALO**2.)*vo**2./numpy.sqrt(2.*math.pi)
        #thisp= numpy.array([qdf.pvz(v/_REFV0/vo,R[ii],z[ii],ngl=options.ngl,gl=True) for v in vs])
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
    out= thisp/numpy.sum(thisp)/(vs[1]-vs[0])
    return out

if __name__ == '__main__':
    (options,args)= get_options().parse_args()
    plotVelComparisonDF(options,args)

