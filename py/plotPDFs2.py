import sys
import os, os.path
import cPickle as pickle
import numpy
from scipy import maxentropy
import multiprocessing
from galpy.util import bovy_plot, multi
import monoAbundanceMW
from segueSelect import _ERASESTR
from pixelFitDF import get_options, approxFitResult, _REFV0, _REFR0
_NOTDONEYET= True
def plotRdfh(options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= 30
    if not options.multi is None:
        dummy= multi.parallel_map((lambda x: plotRdfh_single(x,options,args)),
                                  range(npops),
                                  numcores=numpy.amin([options.multi,
                                                       npops,
                                                       multiprocessing.cpu_count()]))
    else:
        for ii in range(npops):
            plotRdfh_single(ii,options,args)

def plotRdfh_single(ii,options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    npops= len(fehs)
    savefile.close()
    if True:
        if _NOTDONEYET:
            spl= options.restart.split('.')
        else:
            spl= args[0].split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        savefile= open(newname,'rb')
        try:
            if not _NOTDONEYET:
                params= pickle.load(savefile)
                mlogl= pickle.load(savefile)
            logl= pickle.load(savefile)
        except:
            return None
        finally:
            savefile.close()
        if _NOTDONEYET:
            logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        logl[numpy.isnan(logl)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        marglogl= numpy.zeros((logl.shape[0],logl.shape[3]))
        condfh= numpy.zeros((logl.shape[0]))
        condlogp= numpy.zeros(logl.shape[0])
        #Get hR range
        lnhr, lnsr, lnsz, rehr, resr, resz= approxFitResult(fehs[ii],afes[ii],
                                                            relerr=True)
        if True: rehr= 0.3 #regularize
        hrs= numpy.linspace(lnhr-6.*rehr,lnhr+6.*rehr,options.nhrs)
        if not numpy.sum(hrs <= 1.5) == 0:
            logl[:,:,:,:,(hrs > 1.5),:,:,:,:,:,:]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        else:
            logl[:,:,:,:,1:,:,:,:,:,:,:]= -numpy.finfo(numpy.dtype(numpy.float64)).max
            logl[:,:,:,:,3,:,:,:,:,:,:]= logl[:,:,:,:,0,:,:,:,:,:,:] #BOVY: HACK
        #logl[:,:,:,:,:,:,:,0,:,:,:]= -numpy.finfo(numpy.dtype(numpy.float64)).max   
        for jj in range(marglogl.shape[0]):
            for kk in range(marglogl.shape[1]):
                marglogl[jj,kk]= maxentropy.logsumexp(logl[jj,0,0,kk,3:,:,:,:,:,:,:].flatten())
            condlogp[jj]= maxentropy.logsumexp(marglogl[jj,:])
            condlogl= marglogl[jj,:]-maxentropy.logsumexp(marglogl[jj,:])
            condfh[jj]= numpy.sum(numpy.exp(condlogl)*numpy.linspace(0.,1.,logl.shape[3]))/numpy.sum(numpy.exp(condlogl))
#        if monoAbundanceMW.hr(fehs[ii],afes[ii]) < 3.5 \
#                and numpy.amax(logl) < 0.: #latter removes ridiculous bins
#            allmarglogl[:,:,ii]= marglogl
        #Normalize
        alogl= marglogl-numpy.amax(marglogl)
        bovy_plot.bovy_print()
        bovy_plot.bovy_dens2d(numpy.exp(alogl).T,
                              origin='lower',cmap='gist_yarg',
                              interpolation='nearest',
                              xrange=[1.9,3.5],yrange=[-1./32.,1.+1./32.],
                              xlabel=r'$R_d\ (\mathrm{kpc})$',ylabel=r'$f_h$')
        s= 2.*condlogp
        s-= numpy.amax(s)
        s+= 16.
        s*= 3.
        bovy_plot.bovy_plot(numpy.linspace(2.0,3.4,logl.shape[0]),
                            condfh,color='0.75',ls='-',
                            overplot=True,zorder=2)
        bovy_plot.bovy_plot(numpy.linspace(2.,3.4,logl.shape[0]),
                            condfh,color='0.75',marker='o',
                            s=s,scatter=True,overplot=True,zorder=10)
        maxindx= numpy.argmax(s)
        bovy_plot.bovy_plot(numpy.linspace(2.0,3.4,logl.shape[0])[maxindx],
                            condfh[maxindx],color='blue',marker='o',
                            ls='none',
                            ms=8.,mec='none',
                            overplot=True,zorder=13)
        #Plotname
        spl= options.outfilename.split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        bovy_plot.bovy_end_print(newname)
    return None
    #Now plot combined
    alogl= numpy.sum(allmarglogl,axis=2)\
        -numpy.amax(numpy.sum(allmarglogl,axis=2))
    bovy_plot.bovy_print()
    bovy_plot.bovy_dens2d(numpy.exp(alogl).T,
                          origin='lower',cmap='gist_yarg',
                          interpolation='nearest',
                          xrange=[1.9,3.5],
                          yrange=[-1./32.,1.+1./32.],
                          xlabel=r'$R_d\ (\mathrm{kpc})$',ylabel=r'$f_h$')
    bovy_plot.bovy_end_print(options.outfilename)

def plotRdhr(options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= 30
    if not options.multi is None:
        dummy= multi.parallel_map((lambda x: plotRdhr_single(x,options,args)),
                                  range(npops),
                                  numcores=numpy.amin([options.multi,
                                                       npops,
                                                       multiprocessing.cpu_count()]))
    else:
        for ii in range(npops):
            plotRdhr_single(ii,options,args)

def plotRdhr_single(ii,options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    #Go through all of the bins
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= 30
    if True:
        if _NOTDONEYET:
            spl= options.restart.split('.')
        else:
            spl= args[0].split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        savefile= open(newname,'rb')
        try:
            if not _NOTDONEYET:
                params= pickle.load(savefile)
                mlogl= pickle.load(savefile)
            logl= pickle.load(savefile)
        except:
            return None
        finally:
            savefile.close()
        logl[numpy.isnan(logl)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        if _NOTDONEYET:
            logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        marglogl= numpy.zeros((logl.shape[0],logl.shape[4]))
        for jj in range(marglogl.shape[0]):
            for kk in range(marglogl.shape[1]):
                if options.conditional:
                    marglogl[jj,kk]= maxentropy.logsumexp(logl[jj,0,0,:,kk,:,:,:,:,:,:])-maxentropy.logsumexp(logl[:,0,0,:,kk,:,:,:,:,:,:])
                else:
                    marglogl[jj,kk]= maxentropy.logsumexp(logl[jj,0,0,:,kk,:,:,:,:,:,:])
        #Normalize
        alogl= marglogl-numpy.amax(marglogl)
        #Get hR range
        lnhr, lnsr, lnsz, rehr, resr, resz= approxFitResult(fehs[ii],afes[ii],
                                                            relerr=True)
        if True: rehr= 0.3 #regularize
        hrs= numpy.linspace(lnhr-6.*rehr,lnhr+6.*rehr,options.nhrs)
        bovy_plot.bovy_print()
        bovy_plot.bovy_dens2d(numpy.exp(alogl).T,
                              origin='lower',cmap='gist_yarg',
                              interpolation='nearest',
                              xrange=[1.9,3.5],
                              yrange=[hrs[0]-(hrs[1]-hrs[0])/2.,
                                      hrs[-1]+(hrs[1]-hrs[0])/2.],
                              xlabel=r'$R_d$',
                              ylabel=r'$\ln h_R / 8\,\mathrm{kpc}$')
        #Plotname
        spl= options.outfilename.split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        bovy_plot.bovy_end_print(newname)

def plotRdPout(options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= 30
    if not options.multi is None:
        dummy= multi.parallel_map((lambda x: plotRdPout_single(x,options,args)),
                                  range(npops),
                                  numcores=numpy.amin([options.multi,
                                                       npops,
                                                       multiprocessing.cpu_count()]))
    else:
        for ii in range(npops):
            plotRdPout_single(ii,options,args)

def plotRdPout_single(ii,options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= 30
    if True:
        if _NOTDONEYET:
            spl= options.restart.split('.')
        else:
            spl= args[0].split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        savefile= open(newname,'rb')
        try:
            if not _NOTDONEYET:
                params= pickle.load(savefile)
                mlogl= pickle.load(savefile)
            logl= pickle.load(savefile)
        except:
            return None
        finally:
            savefile.close()
        logl[numpy.isnan(logl)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        if _NOTDONEYET:
            logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        marglogl= numpy.zeros((logl.shape[0],logl.shape[8]))
        for jj in range(marglogl.shape[0]):
            for kk in range(marglogl.shape[1]):
                if options.conditional:
                    marglogl[jj,kk]= maxentropy.logsumexp(logl[jj,0,0,:,3:,:,:,:,kk,:,:].flatten())-maxentropy.logsumexp(logl[:,0,0,:,3:,:,:,:,kk,:,:].flatten())
                else:
                    marglogl[jj,kk]= maxentropy.logsumexp(logl[jj,0,0,:,3:,:,:,:,kk,:,:].flatten())
        #Normalize
        alogl= marglogl-numpy.amax(marglogl)
        bovy_plot.bovy_print()
        bovy_plot.bovy_dens2d(numpy.exp(alogl).T,
                              origin='lower',cmap='gist_yarg',
                              interpolation='nearest',
                              xrange=[1.9,3.5],
                              yrange=[-0.01,.51],
                              xlabel=r'$R_d\ (\mathrm{kpc})$',
                              ylabel=r'$P_{\mathrm{out}}$')
        #Plotname
        spl= options.outfilename.split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        bovy_plot.bovy_end_print(newname)

def plotfhPout(options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= 30
    if not options.multi is None:
        dummy= multi.parallel_map((lambda x: plotfhPout_single(x,options,args)),
                                  range(npops),
                                  numcores=numpy.amin([options.multi,
                                                       npops,
                                                       multiprocessing.cpu_count()]))
    else:
        for ii in range(npops):
            plotfhPout_single(ii,options,args)

def plotfhPout_single(ii,options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= 30
    if True:
        if _NOTDONEYET:
            spl= options.restart.split('.')
        else:
            spl= args[0].split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        savefile= open(newname,'rb')
        try:
            if not _NOTDONEYET:
                params= pickle.load(savefile)
                mlogl= pickle.load(savefile)
            logl= pickle.load(savefile)
        except:
            return None
        finally:
            savefile.close()
        logl[numpy.isnan(logl)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        if _NOTDONEYET:
            logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        marglogl= numpy.zeros((logl.shape[8],logl.shape[3]))
        for jj in range(marglogl.shape[0]):
            for kk in range(marglogl.shape[1]):
                if options.conditional:
                    marglogl[jj,kk]= maxentropy.logsumexp(logl[:,0,0,kk,3:,:,:,:,jj,:,:].flatten())-maxentropy.logsumexp(logl[:,0,0,:,3:,:,:,:,jj,:,:].flatten())
                else:
                    marglogl[jj,kk]= maxentropy.logsumexp(logl[:,0,0,kk,3:,:,:,:,jj,:,:].flatten())
        #Normalize
        alogl= marglogl-numpy.amax(marglogl)
        bovy_plot.bovy_print()
        bovy_plot.bovy_dens2d(numpy.exp(alogl).T,
                              origin='lower',cmap='gist_yarg',
                              interpolation='nearest',
                              yrange=[-1./32.,1.+1./32.],
                              xrange=[-0.01,.51],
                              ylabel=r'$f_h$',
                              xlabel=r'$P_{\mathrm{out}}$')
        #Plotname
        spl= options.outfilename.split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        bovy_plot.bovy_end_print(newname)

def plotRddvt(options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= 30
    if not options.multi is None:
        dummy= multi.parallel_map((lambda x: plotRddvt_single(x,options,args)),
                                  range(npops),
                                  numcores=numpy.amin([options.multi,
                                                       npops,
                                                       multiprocessing.cpu_count()]))
    else:
        for ii in range(npops):
            plotRddvt_single(ii,options,args)

def plotRddvt_single(ii,options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= 30
    if True:
        if _NOTDONEYET:
            spl= options.restart.split('.')
        else:
            spl= args[0].split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        savefile= open(newname,'rb')
        try:
            if not _NOTDONEYET:
                params= pickle.load(savefile)
                mlogl= pickle.load(savefile)
            logl= pickle.load(savefile)
        except:
            return None
        finally:
            savefile.close()
        if _NOTDONEYET:
            logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        logl[numpy.isnan(logl)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        marglogl= numpy.zeros((logl.shape[0],logl.shape[7]))
        for jj in range(marglogl.shape[0]):
            for kk in range(marglogl.shape[1]):
                    marglogl[jj,kk]= maxentropy.logsumexp(logl[jj,0,0,:,3:,:,:,kk,:,:,:].flatten())
        #Normalize
        alogl= marglogl-numpy.amax(marglogl)
        bovy_plot.bovy_print()
        bovy_plot.bovy_dens2d(numpy.exp(alogl).T,
                              origin='lower',cmap='gist_yarg',
                              interpolation='nearest',
                              xrange=[1.9,3.5],
                              yrange=[-80.-1./3.,14.+1./3.],
                              xlabel=r'$R_d\ (\mathrm{kpc})$',
                              ylabel=r'$\Delta \bar{V}_T\ (\mathrm{km\,s}^{-1})$')
        #Plotname
        spl= options.outfilename.split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        bovy_plot.bovy_end_print(newname)

def plotsrsz(options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= 30
    if not options.multi is None:
        dummy= multi.parallel_map((lambda x: plotsrsz_single(x,options,args)),
                                  range(npops),
                                  numcores=numpy.amin([options.multi,
                                                       npops,
                                                       multiprocessing.cpu_count()]))
    else:
        for ii in range(npops):
            plotsrsz_single(ii,options,args)

def plotsrsz_single(ii,options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    #Go through all of the bins
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= 30
    if True:
        if _NOTDONEYET:
            spl= options.restart.split('.')
        else:
            spl= args[0].split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        savefile= open(newname,'rb')
        try:
            if not _NOTDONEYET:
                params= pickle.load(savefile)
                mlogl= pickle.load(savefile)
            logl= pickle.load(savefile)
        except:
            return None
        finally:
            savefile.close()
        if _NOTDONEYET:
            logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        logl[numpy.isnan(logl)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        marglogl= numpy.zeros((logl.shape[5],logl.shape[6]))
        for jj in range(marglogl.shape[0]):
            for kk in range(marglogl.shape[1]):
                if options.conditional:
                    marglogl[jj,kk]= maxentropy.logsumexp(logl[:,0,0,:,3:,jj,kk,:,:,:,:].flatten())-maxentropy.logsumexp(logl[:,0,0,:,3:,jj,:,:,:,:,:].flatten())
                else:
                    marglogl[jj,kk]= maxentropy.logsumexp(logl[:,0,0,:,3:,jj,kk,:,:,:,:].flatten())
        #Normalize
        alogl= marglogl-numpy.amax(marglogl)
        #Get ranges
        lnhr, lnsr, lnsz, rehr, resr, resz= approxFitResult(fehs[ii],afes[ii],
                                                            relerr=True)
        if True: resr= 0.3
        if True: resz= 0.3
        srs= numpy.linspace(lnsr-0.66*resz,lnsr+0.66*resz,options.nsrs)#USE ESZ
        szs= numpy.linspace(lnsz-0.66*resz,lnsz+0.66*resz,options.nszs)
        bovy_plot.bovy_print()
        bovy_plot.bovy_dens2d(numpy.exp(alogl).T,
                              origin='lower',cmap='gist_yarg',
                              interpolation='nearest',
                              xrange=[srs[0]-(srs[1]-srs[0])/2.,
                                      srs[-1]+(srs[1]-srs[0])/2.],
                              yrange=[szs[0]-(szs[1]-szs[0])/2.,
                                      szs[-1]+(szs[1]-szs[0])/2.],
                              xlabel=r'$\ln \sigma_R / 220\ \mathrm{km\,s}^{-1}$',
                              ylabel=r'$\ln \sigma_Z / 220\ \mathrm{km\,s}^{-1}$')
        bovy_plot.bovy_plot([-3.,0.],[-3.,0.],'k--',overplot=True)
        #Plotname
        spl= options.outfilename.split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        bovy_plot.bovy_end_print(newname)

def plotRdsz(options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= 30
    if not options.multi is None:
        dummy= multi.parallel_map((lambda x: plotRdsz_single(x,options,args)),
                                  range(npops),
                                  numcores=numpy.amin([options.multi,
                                                       npops,
                                                       multiprocessing.cpu_count()]))
    else:
        for ii in range(npops):
            plotRdsz_single(ii,options,args)

def plotRdsz_single(ii,options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    if True:
        if _NOTDONEYET:
            spl= options.restart.split('.')
        else:
            spl= args[0].split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        if not os.path.exists(newname): return None
        savefile= open(newname,'rb')
        try:
            if not _NOTDONEYET:
                params= pickle.load(savefile)
                mlogl= pickle.load(savefile)
            logl= pickle.load(savefile)
        except:
            return None
        finally:
            savefile.close()
        logl[numpy.isnan(logl)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        if _NOTDONEYET:
            logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        marglogl= numpy.zeros((logl.shape[0],logl.shape[6]))
        for jj in range(marglogl.shape[0]):
            for kk in range(marglogl.shape[1]):
                marglogl[jj,kk]= maxentropy.logsumexp(logl[jj,0,0,:,3:,:,kk,:,:,:,:].flatten())
        #Normalize
        alogl= marglogl-numpy.amax(marglogl)
        #Get range
        lnhr, lnsr, lnsz, rehr, resr, resz= approxFitResult(fehs[ii],afes[ii],
                                                            relerr=True)
        if True: resz= 0.3
        szs= numpy.linspace(lnsz-0.66*resz,lnsz+0.66*resz,options.nszs)
        bovy_plot.bovy_print()
        bovy_plot.bovy_dens2d(numpy.exp(alogl).T,
                              origin='lower',cmap='gist_yarg',
                              interpolation='nearest',
                              xrange=[1.9,3.5],
                              yrange=[szs[0]-(szs[1]-szs[0])/2.,
                                      szs[-1]+(szs[1]-szs[0])/2.],
                              xlabel=r'$R_d\ (\mathrm{kpc})$',
                              ylabel=r'$\ln \sigma_Z / 220\ \mathrm{km\,s}^{-1}$')
        #Plotname
        spl= options.outfilename.split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        bovy_plot.bovy_end_print(newname)
    return None

def plotPout(options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= 30
    if not options.multi is None:
        dummy= multi.parallel_map((lambda x: plotPout_single(x,options,args)),
                                  range(npops),
                                  numcores=numpy.amin([options.multi,
                                                       npops,
                                                       multiprocessing.cpu_count()]))
    else:
        for ii in range(npops):
            plotPout_single(ii,options,args)

def plotPout_single(ii,options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= 30
    if True:
        if _NOTDONEYET:
            spl= options.restart.split('.')
        else:
            spl= args[0].split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        savefile= open(newname,'rb')
        try:
            if not _NOTDONEYET:
                params= pickle.load(savefile)
                mlogl= pickle.load(savefile)
            logl= pickle.load(savefile)
        except:
            return None
        finally:
            savefile.close()
        if _NOTDONEYET:
            logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        logl[numpy.isnan(logl)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        marglogl= numpy.zeros((logl.shape[8]))
        for jj in range(marglogl.shape[0]):
            marglogl[jj]= maxentropy.logsumexp(logl[:,0,0,:,3:,:,:,:,jj,:,:].flatten())
        #Normalize
        alogl= marglogl-numpy.nanmax(marglogl)
        bovy_plot.bovy_print()
        bovy_plot.bovy_plot(numpy.linspace(10.**-5.,0.5,logl.shape[8]),
                            numpy.exp(alogl).T,'k-',
                            xrange=[0.0,0.5],
                            xlabel='$P_{\mathrm{out}}$',
                            yrange=[0.,1.1])
        #Plotname
        spl= options.outfilename.split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        bovy_plot.bovy_end_print(newname)

def plotdvt(options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= 30
    for ii in range(npops):
        if _NOTDONEYET:
            spl= options.restart.split('.')
        else:
            spl= args[0].split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        savefile= open(newname,'rb')
        try:
            if not _NOTDONEYET:
                params= pickle.load(savefile)
                mlogl= pickle.load(savefile)
            logl= pickle.load(savefile)
        except:
            continue
        finally:
            savefile.close()
        logl[numpy.isnan(logl)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        if _NOTDONEYET:
            logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        marglogl= numpy.zeros((logl.shape[7]))
        if ii == 0:
            allmarglogl= numpy.zeros((logl.shape[7],npops))
        for jj in range(marglogl.shape[0]):
            indx= True-numpy.isnan(logl[:,0,0,:,:,:,:,jj,:,:,:].flatten())
            if numpy.sum(indx) > 0:
                marglogl[jj]= maxentropy.logsumexp(logl[:,0,0,:,:,:,:,jj,:,:,:].flatten()[indx])
            else:
                marglogl[jj]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        allmarglogl[:,ii]= marglogl
        #Normalize
        alogl= marglogl-numpy.nanmax(marglogl)
        bovy_plot.bovy_print()
        bovy_plot.bovy_plot(numpy.linspace(-0.35,0.05,logl.shape[7])*220.,
                            numpy.exp(alogl).T,'k-',
                            xrange=[-85.,20.],
                            xlabel=r'$\Delta \bar{V}_T\ (\mathrm{km\,s}^{-1})$',
                            yrange=[0.,1.1])
        #Plotname
        spl= options.outfilename.split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        bovy_plot.bovy_end_print(newname)
    bovy_plot.bovy_end_print(options.outfilename)

def plotloglhist(options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= 30
    if not options.multi is None:
        dummy= multi.parallel_map((lambda x: plotloglhist_single(x,options,args)),
                                  range(npops),
                                  numcores=numpy.amin([options.multi,
                                                       npops,
                                                       multiprocessing.cpu_count()]))
    else:
        for ii in range(npops):
            plotloglhist_single(ii,options,args)

def plotloglhist_single(ii,options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= 30
    if True:
        if _NOTDONEYET:
            spl= options.restart.split('.')
        else:
            spl= args[0].split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        savefile= open(newname,'rb')
        try:
            if not _NOTDONEYET:
                params= pickle.load(savefile)
                mlogl= pickle.load(savefile)
            logl= pickle.load(savefile)
        except:
            return None
        finally:
            savefile.close()
        logl[numpy.isnan(logl)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        logl[:,:,:,:3,:,:,:,:,:,:]= -100000000000000.
        if _NOTDONEYET:
            logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        bovy_plot.bovy_print()
        bovy_plot.bovy_hist(logl.flatten()[(logl.flatten() >-1000000.)],
                            xrange=[numpy.nanmax(logl)-50.,numpy.nanmax(logl)],
                            xlabel=r'$\log \mathcal{L}$',
                            histtype='step',color='k')
        #Plotname
        spl= options.outfilename.split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        bovy_plot.bovy_end_print(newname)
    return None

def plotprops(options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        npops= 62
        savefile= open('binmapping_g.sav','rb')
    elif options.sample.lower() == 'k':
        npops= 30
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    ndatas= pickle.load(savefile)
    savefile.close()
    for ii in range(npops):
        bovy_plot.bovy_print()
        bovy_plot.bovy_text(r'$[\mathrm{Fe/H}] = %.2f$' % (fehs[ii])
                            +'\n'
                            r'$[\alpha/\mathrm{Fe}] = %.3f$' % (afes[ii])
                            +'\n'
                            r'$N_{\mathrm{data}} = %i$' % (ndatas[ii])
                            +'\n'
                            r'$\ln h_R / 8\,\mathrm{kpc} = %.1f$' % (numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii])/8.)) 
                            +'\n'
                            +r'$\ln \sigma_R / 220\,\mathrm{km\,s}^{-1} = %.1f$' % (numpy.log(monoAbundanceMW.sigmar(fehs[ii],afes[ii])/220.))
                            +'\n'
                            +r'$\ln \sigma_Z / 220\,\mathrm{km\,s}^{-1} = %.1f$' % (numpy.log(monoAbundanceMW.sigmaz(fehs[ii],afes[ii])/220.)),
                            top_left=True,size=16.)
        #Plotname
        spl= options.outfilename.split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        bovy_plot.bovy_end_print(newname)

def plotDF4fidpot(options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    npops= len(fehs)
    savefile.close()
    plotthis= numpy.zeros(npops)+numpy.nan
    for ii in range(npops):
        sys.stdout.write('\r'+"Working on bin %i / %i ..." % (ii+1,npops))
        sys.stdout.flush()
        if _NOTDONEYET:
            spl= options.restart.split('.')
        else:
            spl= args[0].split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        savefile= open(newname,'rb')
        try:
            if not _NOTDONEYET:
                params= pickle.load(savefile)
                mlogl= pickle.load(savefile)
            logl= pickle.load(savefile)
        except:
            continue
        finally:
            savefile.close()
        if _NOTDONEYET:
            logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        hrs= numpy.log(numpy.linspace(1.5,5.,logl.shape[4])/_REFR0)
        srs= numpy.log(numpy.linspace(25.,70.,logl.shape[5])/_REFV0)
        szs= numpy.log(numpy.linspace(15.,60.,logl.shape[6])/_REFV0)
        if options.restrictdvt:
            logl= logl[:,:,:,:,:,:,:,1:4,:,:,:]
        if options.restrictdf:
            lnhrin, lnsrin, lnszin= approxFitResult(fehs[ii],afes[ii])
            hrindx= numpy.argmin((hrs-lnhrin)**2.)
            srindx= numpy.argmin((srs-lnsrin)**2.)
            szindx= numpy.argmin((szs-lnszin)**2.)
            minhrindx= hrindx-1
            maxhrindx= hrindx+1
            if minhrindx < 0: 
                minhrindx+= 1
                maxhrindx+= 1
            elif maxhrindx >= logl.shape[4]: 
                minhrindx-= 1
                maxhrindx-= 1
            minsrindx= srindx-1
            maxsrindx= srindx+1
            if minsrindx < 0: 
                minsrindx+= 1
                maxsrindx+= 1
            elif maxsrindx >= logl.shape[5]: 
                minsrindx-= 1
                maxsrindx-= 1
            minszindx= szindx-1
            maxszindx= szindx+1
            if minszindx < 0: 
                minszindx+= 1
                maxszindx+= 1
            elif maxszindx >= logl.shape[6]: 
                minszindx-= 1
                maxszindx-= 1
            logl= logl[:,:,:,:,minhrindx:maxhrindx+1,minsrindx:maxsrindx+1,
                               minszindx:maxszindx+1,:,:,:,:]
        #Find best-fit df for the fiducial potential
        indx= numpy.unravel_index(options.index,(logl.shape[0],logl.shape[3]))
        logl= logl[indx[0],0,0,indx[1],:,:,:,:,:,:,:]
        maxindx= numpy.unravel_index(numpy.argmax(logl),(logl.shape))
        if options.subtype.lower() == 'hr':
            plotthis[ii]= numpy.exp(hrs[maxindx[0]])*_REFR0
        elif options.subtype.lower() == 'sr':
            plotthis[ii]= numpy.exp(srs[maxindx[1]])*_REFV0
        elif options.subtype.lower() == 'sz':
            plotthis[ii]= numpy.exp(szs[maxindx[2]])*_REFV0
    sys.stdout.write('\r'+_ERASESTR+'\r')
    sys.stdout.flush()
    #Now plot
    if options.subtype.lower() == 'hr':
        vmin, vmax= 1.35,4.5
        zlabel=r'$\mathrm{Input\ radial\ scale\ length\ [kpc]}$'
    elif options.subtype.lower() == 'sr':
        vmin, vmax= 30., 60.
        zlabel= r'$\mathrm{Input\ radial\ velocity\ dispersion\ [km\,s}^{-1}]$'
    elif options.subtype.lower() == 'sz':
        vmin, vmax= 10., 80.
        zlabel= r'$\mathrm{Input\ vertical\ velocity\ dispersion\ [km\,s}^{-1}]$'
    bovy_plot.bovy_print()
    monoAbundanceMW.plotPixelFunc(fehs,afes,plotthis,
                                  vmin=vmin,vmax=vmax,
                                  zlabel=zlabel)
    bovy_plot.bovy_end_print(options.outfilename)

def plotbestpot(options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    npops= len(fehs)
    savefile.close()
    plotthis_rd= numpy.zeros(npops)+numpy.nan
    plotthis_fh= numpy.zeros(npops)+numpy.nan
    for ii in range(npops):
        sys.stdout.write('\r'+"Working on bin %i / %i ..." % (ii+1,npops))
        sys.stdout.flush()
        if _NOTDONEYET:
            spl= options.restart.split('.')
        else:
            spl= args[0].split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        savefile= open(newname,'rb')
        try:
            if not _NOTDONEYET:
                params= pickle.load(savefile)
                mlogl= pickle.load(savefile)
            logl= pickle.load(savefile)
        except:
            continue
        finally:
            savefile.close()
        logl[numpy.isnan(logl)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        if _NOTDONEYET:
            logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        rds= numpy.linspace(1.5,4.5,logl.shape[0])
        fhs= numpy.linspace(0.,1.,logl.shape[3])
        if options.restrictdvt:
            logl= logl[:,:,:,:,:,:,:,1:4,:,:,:]
        hrs= numpy.log(numpy.linspace(1.5,5.,logl.shape[4])/_REFR0)
        srs= numpy.log(numpy.linspace(25.,70.,logl.shape[5])/_REFV0)
        szs= numpy.log(numpy.linspace(15.,60.,logl.shape[6])/_REFV0)
        if options.restrictdf:
            lnhrin, lnsrin, lnszin= approxFitResult(fehs[ii],afes[ii])
            hrindx= numpy.argmin((hrs-lnhrin)**2.)
            srindx= numpy.argmin((srs-lnsrin)**2.)
            szindx= numpy.argmin((szs-lnszin)**2.)
            minhrindx= hrindx-1
            maxhrindx= hrindx+1
            if minhrindx < 0: 
                minhrindx+= 1
                maxhrindx+= 1
            elif maxhrindx >= logl.shape[4]: 
                minhrindx-= 1
                maxhrindx-= 1
            minsrindx= srindx-1
            maxsrindx= srindx+1
            if minsrindx < 0: 
                minsrindx+= 1
                maxsrindx+= 1
            elif maxsrindx >= logl.shape[5]: 
                minsrindx-= 1
                maxsrindx-= 1
            minszindx= szindx-1
            maxszindx= szindx+1
            if minszindx < 0: 
                minszindx+= 1
                maxszindx+= 1
            elif maxszindx >= logl.shape[6]: 
                minszindx-= 1
                maxszindx-= 1
            logl= logl[:,:,:,:,minhrindx:maxhrindx+1,minsrindx:maxsrindx+1,
                               minszindx:maxszindx+1,:,:,:,:]
        #Find best-fit potential
        maxindx= numpy.unravel_index(numpy.argmax(logl),(logl.shape))
        plotthis_rd[ii]= rds[maxindx[0]]
        plotthis_fh[ii]= fhs[maxindx[3]]
    sys.stdout.write('\r'+_ERASESTR+'\r')
    sys.stdout.flush()
    #Now plot
    bovy_plot.bovy_print()
    monoAbundanceMW.plotPixelFunc(fehs,afes,plotthis_rd,
                                  vmin=1.5,vmax=4.5,
                                  zlabel=r'$R_d\ [\mathrm{kpc}]$')
    #Plotname
    spl= options.outfilename.split('.')
    newname= ''
    for jj in range(len(spl)-1):
        newname+= spl[jj]
        if not jj == len(spl)-2: newname+= '.'
    newname+= '_rd.'
    newname+= spl[-1]
    bovy_plot.bovy_end_print(newname)
    #fh
    bovy_plot.bovy_print()
    monoAbundanceMW.plotPixelFunc(fehs,afes,plotthis_fh,
                                  vmin=0.,vmax=1.,
                                  zlabel=r'$f_h$')
    #Plotname
    spl= options.outfilename.split('.')
    newname= ''
    for jj in range(len(spl)-1):
        newname+= spl[jj]
        if not jj == len(spl)-2: newname+= '.'
    newname+= '_fh.'
    newname+= spl[-1]
    bovy_plot.bovy_end_print(newname)

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    if options.type.lower() == 'rdfh':
        plotRdfh(options,args)
    elif options.type.lower() == 'pout':
        plotPout(options,args)
    elif options.type.lower() == 'srsz':
        plotsrsz(options,args)
    elif options.type.lower() == 'dvt':
        plotdvt(options,args)
    elif options.type.lower() == 'rdhr':
        plotRdhr(options,args)
    elif options.type.lower() == 'rdsz':
        plotRdsz(options,args)
    elif options.type.lower() == 'rdpout':
        plotRdPout(options,args)
    elif options.type.lower() == 'fhpout':
        plotfhPout(options,args)
    elif options.type.lower() == 'rddvt':
        plotRddvt(options,args)
    elif options.type.lower() == 'loglhist':
        plotloglhist(options,args)
    elif options.type.lower() == 'props':
        plotprops(options,args)
    elif options.type.lower() == 'df4fidpot':
        plotDF4fidpot(options,args)
    elif options.type.lower() == 'bestpot':
        plotbestpot(options,args)
                            
