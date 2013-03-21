import sys
import os, os.path
import cPickle as pickle
import numpy
from scipy import maxentropy
from galpy.util import bovy_plot
import monoAbundanceMW
from segueSelect import _ERASESTR
from pixelFitDF import get_options, approxFitResult, _REFV0, _REFR0
_NOTDONEYET= True
def plotRdfh(options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    npops= len(fehs)
    savefile.close()
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
        if _NOTDONEYET:
            logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        if options.restrictdvt:
            logl= logl[:,:,:,:,:,:,:,1:4,:,:,:]
        if options.restrictdf:
            hrs= numpy.log(numpy.linspace(1.5,5.,logl.shape[4])/_REFR0)
            srs= numpy.log(numpy.linspace(25.,70.,logl.shape[5])/_REFV0)
            szs= numpy.log(numpy.linspace(15.,60.,logl.shape[6])/_REFV0)
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
        marglogl= numpy.zeros((logl.shape[0],logl.shape[3]))
        marglogldvt0= numpy.zeros((logl.shape[0],logl.shape[3]))
        condfh= numpy.zeros((logl.shape[3]))
        condlogp= numpy.zeros(logl.shape[0])
        condfhdvt0= numpy.zeros((logl.shape[3]))
        condlogpdvt0= numpy.zeros(logl.shape[0])
        if ii == 0:
            allmarglogl= numpy.zeros((logl.shape[0],logl.shape[3],npops))
        for jj in range(marglogl.shape[0]):
            for kk in range(marglogl.shape[1]):
                indx= True-numpy.isnan(logl[jj,0,0,kk,:,:,:,:,:,:,:].flatten())
                indxdvt0= True-numpy.isnan(logl[jj,0,0,kk,:,:,:,2,:,:,:].flatten())
                if numpy.sum(indx) > 0:
                    marglogl[jj,kk]= maxentropy.logsumexp(logl[jj,0,0,kk,:,:,:,:,:,:,:].flatten()[indx])
                else:
                    marglogl[jj,kk]= -numpy.finfo(numpy.dtype(numpy.float64)).max
                if numpy.sum(indxdvt0) > 0:
                    marglogldvt0[jj,kk]= maxentropy.logsumexp(logl[jj,0,0,kk,:,:,:,2,:,:,:].flatten()[indxdvt0])
                else:
                    marglogldvt0[jj,kk]= -numpy.finfo(numpy.dtype(numpy.float64)).max
            condlogp[jj]= maxentropy.logsumexp(marglogl[jj,:])
            condlogl= marglogl[jj,:]-maxentropy.logsumexp(marglogl[jj,:])
            condfh[jj]= numpy.sum(numpy.exp(condlogl)*numpy.linspace(0.,1.,logl.shape[3]))/numpy.sum(numpy.exp(condlogl))
            condlogpdvt0[jj]= maxentropy.logsumexp(marglogldvt0[jj,:])
            condlogldvt0= marglogldvt0[jj,:]-maxentropy.logsumexp(marglogldvt0[jj,:])
            condfhdvt0[jj]= numpy.sum(numpy.exp(condlogldvt0)*numpy.linspace(0.,1.,logl.shape[3]))/numpy.sum(numpy.exp(condlogldvt0))
        if monoAbundanceMW.hr(fehs[ii],afes[ii]) < 3.5 \
                and numpy.amax(logl) < 0.: #latter removes ridiculous bins
            allmarglogl[:,:,ii]= marglogl
        #Normalize
        alogl= marglogl-numpy.amax(marglogl)
        bovy_plot.bovy_print()
        bovy_plot.bovy_dens2d(numpy.exp(alogl).T,
                              origin='lower',cmap='gist_yarg',
                              interpolation='nearest',
                              xrange=[1.5,4.5],yrange=[0.,1.],
                              xlabel=r'$R_d$',ylabel=r'$f_h$')
        s= 2.*condlogp
        s-= numpy.amax(s)
        s+= 16.
        s*= 3.
        bovy_plot.bovy_plot(numpy.linspace(1.5,4.5,logl.shape[0]),
                            condfh,color='0.75',ls='-',
                            overplot=True,zorder=2)
        bovy_plot.bovy_plot(numpy.linspace(1.5,4.5,logl.shape[0]),
                            condfh,color='0.75',marker='o',
                            s=s,scatter=True,overplot=True,zorder=10)
        maxindx= numpy.argmax(s)
        bovy_plot.bovy_plot(numpy.linspace(1.5,4.5,logl.shape[0])[maxindx],
                            condfh[maxindx],color='blue',marker='o',
                            ls='none',
                            ms=8.,mec='none',
                            overplot=True,zorder=13)
        #dvt0
        s= 2.*condlogpdvt0
        s-= numpy.amax(s)
        s+= 16.
        s*= 3.
        if not options.restrictdvt:
            bovy_plot.bovy_plot(numpy.linspace(1.5,4.5,logl.shape[0]),
                                condfhdvt0,color='0.25',ls='-',
                                overplot=True,zorder=1)
            bovy_plot.bovy_plot(numpy.linspace(1.5,4.5,logl.shape[0]),
                                condfhdvt0,color='0.25',marker='o',
                                s=s,scatter=True,overplot=True,zorder=11)
            maxindx= numpy.argmax(s)
            bovy_plot.bovy_plot(numpy.linspace(1.5,4.5,logl.shape[0])[maxindx],
                                condfhdvt0[maxindx],color='red',marker='o',
                                ls='none',
                                ms=8.,mec='none',
                                overplot=True,zorder=12)
        #Plotname
        spl= options.outfilename.split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        if options.restrictdf and not options.restrictdvt:
            bovy_plot.bovy_text(r'$\mathrm{Restricted\ DF\ parameters}$',
                                top_right=True,size=14.)
        elif options.restrictdvt and not options.restrictdf:
            bovy_plot.bovy_text(r'$\mathrm{Restricted}\ \Delta \bar{V}_T\ \mathrm{range}$',
                                top_right=True,size=14.)
        elif options.restrictdvt and options.restrictdf:
            bovy_plot.bovy_text(r'$\mathrm{Restricted\ DF\ parameters}$'
                                +'\n'
                                +r'$\mathrm{Restricted}\ \Delta \bar{V}_T\ \mathrm{range}$',
                                top_right=True,size=14.)
        bovy_plot.bovy_end_print(newname)
    #Now plot combined
    alogl= numpy.sum(allmarglogl,axis=2)\
        -numpy.amax(numpy.sum(allmarglogl,axis=2))
    bovy_plot.bovy_print()
    bovy_plot.bovy_dens2d(numpy.exp(alogl).T,
                              origin='lower',cmap='gist_yarg',
                              interpolation='nearest',
                              xrange=[1.5,4.5],yrange=[0.,1.],
                              xlabel=r'$R_d$',ylabel=r'$f_h$')
    bovy_plot.bovy_end_print(options.outfilename)

def plotRdhr(options,args):
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
        if _NOTDONEYET:
            logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        marglogl= numpy.zeros((logl.shape[0],logl.shape[4]))
        if ii == 0:
            allmarglogl= numpy.zeros((logl.shape[0],logl.shape[4],npops))
        for jj in range(marglogl.shape[0]):
            for kk in range(marglogl.shape[1]):
                indx= True-numpy.isnan(logl[jj,0,0,:,kk,:,:,:,:,:,:].flatten())
                if numpy.sum(indx) > 0:
                    marglogl[jj,kk]= maxentropy.logsumexp(logl[jj,0,0,:,kk,:,:,:,:,:,:].flatten()[indx])
                else:
                    marglogl[jj,kk]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        allmarglogl[:,:,ii]= marglogl
        #Normalize
        alogl= marglogl-numpy.amax(marglogl)
        bovy_plot.bovy_print()
        bovy_plot.bovy_dens2d(numpy.exp(alogl).T,
                              origin='lower',cmap='gist_yarg',
                              interpolation='nearest',
                              xrange=[1.5,4.5],
                              yrange=[numpy.log(1.5/8.),numpy.log(5./8.)],
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
    #Now plot combined
    alogl= numpy.sum(allmarglogl,axis=2)\
        -numpy.amax(numpy.sum(allmarglogl,axis=2))
    bovy_plot.bovy_print()
    bovy_plot.bovy_dens2d(numpy.exp(alogl).T,
                          origin='lower',cmap='gist_yarg',
                          interpolation='nearest',
                          xrange=[1.5,4.5],
                          yrange=[numpy.log(1.5/8.),numpy.log(5./8.)],
                          xlabel=r'$R_d$',
                          ylabel=r'$\ln h_R / 8\,\mathrm{kpc}$')
    bovy_plot.bovy_end_print(options.outfilename)

def plotRdPout(options,args):
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
        if _NOTDONEYET:
            logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        marglogl= numpy.zeros((logl.shape[0],logl.shape[8]))
        if ii == 0:
            allmarglogl= numpy.zeros((logl.shape[0],logl.shape[8],npops))
        for jj in range(marglogl.shape[0]):
            for kk in range(marglogl.shape[1]):
                indx= True-numpy.isnan(logl[jj,0,0,:,:,:,:,:,kk,:,:].flatten())
                if numpy.sum(indx) > 0:
                    marglogl[jj,kk]= maxentropy.logsumexp(logl[jj,0,0,:,:,:,:,:,kk,:,:].flatten()[indx])
                else:
                    marglogl[jj,kk]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        allmarglogl[:,:,ii]= marglogl
        #Normalize
        alogl= marglogl-numpy.amax(marglogl)
        bovy_plot.bovy_print()
        bovy_plot.bovy_dens2d(numpy.exp(alogl).T,
                              origin='lower',cmap='gist_yarg',
                              interpolation='nearest',
                              xrange=[1.5,4.5],
                              yrange=[0.,.3],
                              xlabel=r'$R_d$',
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
    #Now plot combined
    alogl= numpy.sum(allmarglogl,axis=2)\
        -numpy.amax(numpy.sum(allmarglogl,axis=2))
    bovy_plot.bovy_print()
    bovy_plot.bovy_dens2d(numpy.exp(alogl).T,
                          origin='lower',cmap='gist_yarg',
                          interpolation='nearest',
                          xrange=[1.5,4.5],
                          yrange=[0.,.3],
                          xlabel=r'$R_d$',
                          ylabel=r'$P_{\mathrm{out}}$')
    bovy_plot.bovy_end_print(options.outfilename)

def plotRddvt(options,args):
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
        if _NOTDONEYET:
            logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        marglogl= numpy.zeros((logl.shape[0],logl.shape[7]))
        if ii == 0:
            allmarglogl= numpy.zeros((logl.shape[0],logl.shape[7],npops))
        for jj in range(marglogl.shape[0]):
            for kk in range(marglogl.shape[1]):
                indx= True-numpy.isnan(logl[jj,0,0,:,:,:,:,kk,:,:,:].flatten())
                if numpy.sum(indx) > 0:
                    marglogl[jj,kk]= maxentropy.logsumexp(logl[jj,0,0,:,:,:,:,kk,:,:,:].flatten()[indx])
                else:
                    marglogl[jj,kk]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        allmarglogl[:,:,ii]= marglogl
        #Normalize
        alogl= marglogl-numpy.amax(marglogl)
        bovy_plot.bovy_print()
        bovy_plot.bovy_dens2d(numpy.exp(alogl).T,
                              origin='lower',cmap='gist_yarg',
                              interpolation='nearest',
                              xrange=[1.5,4.5],
                              yrange=[-22.,22.],
                              xlabel=r'$R_d$',
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
    #Now plot combined
    alogl= numpy.sum(allmarglogl,axis=2)\
        -numpy.amax(numpy.sum(allmarglogl,axis=2))
    bovy_plot.bovy_print()
    bovy_plot.bovy_dens2d(numpy.exp(alogl).T,
                          origin='lower',cmap='gist_yarg',
                          interpolation='nearest',
                          xrange=[1.5,4.5],
                          yrange=[-22.,22.],
                          xlabel=r'$R_d$',
                          ylabel=r'$\Delta \bar{V}_T\ (\mathrm{km\,s}^{-1})$')
    bovy_plot.bovy_end_print(options.outfilename)

def plotsrsz(options,args):
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
        if _NOTDONEYET:
            logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        marglogl= numpy.zeros((logl.shape[5],logl.shape[6]))
        if ii == 0:
            allmarglogl= numpy.zeros((logl.shape[5],logl.shape[6],npops))
        for jj in range(marglogl.shape[0]):
            for kk in range(marglogl.shape[1]):
                indx= True-numpy.isnan(logl[:,0,0,:,:,jj,kk,:,:,:,:].flatten())
                if numpy.sum(indx) > 0:
                    marglogl[jj,kk]= maxentropy.logsumexp(logl[:,0,0,:,:,jj,kk,:,:,:,:].flatten()[indx])
                else:
                    marglogl[jj,kk]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        allmarglogl[:,:,ii]= marglogl
        #Normalize
        alogl= marglogl-numpy.amax(marglogl)
        bovy_plot.bovy_print()
        bovy_plot.bovy_dens2d(numpy.exp(alogl).T,
                              origin='lower',cmap='gist_yarg',
                              interpolation='nearest',
                              xrange=[numpy.log(25./220.),numpy.log(70./220.)],
                              yrange=[numpy.log(15./220.),numpy.log(60./220.)],
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
    #Now plot combined
    alogl= numpy.sum(allmarglogl,axis=2)\
        -numpy.amax(numpy.sum(allmarglogl,axis=2))
    bovy_plot.bovy_print()
    bovy_plot.bovy_dens2d(numpy.exp(alogl).T,
                          origin='lower',cmap='gist_yarg',
                          interpolation='nearest',
                          xrange=[numpy.log(25./220.),numpy.log(70./220.)],
                          yrange=[numpy.log(15./220.),numpy.log(60./220.)],
                          xlabel=r'$\ln \sigma_R / 220\ \mathrm{km\,s}^{-1}$',
                          ylabel=r'$\ln \sigma_Z / 220\ \mathrm{km\,s}^{-1}$')
    bovy_plot.bovy_plot([-3.,0.],[-3.,0.],'k--',overplot=True)
    bovy_plot.bovy_end_print(options.outfilename)

def plotPout(options,args):
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
        if _NOTDONEYET:
            logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        marglogl= numpy.zeros((logl.shape[8]))
        if ii == 0:
            allmarglogl= numpy.zeros((logl.shape[8],npops))
        for jj in range(marglogl.shape[0]):
            indx= True-numpy.isnan(logl[:,0,0,:,:,:,:,:,jj,:,:].flatten())
            if numpy.sum(indx) > 0:
                marglogl[jj]= maxentropy.logsumexp(logl[:,0,0,:,:,:,:,:,jj,:,:].flatten()[indx])
            else:
                marglogl[jj]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        allmarglogl[:,ii]= marglogl
        #Normalize
        alogl= marglogl-numpy.nanmax(marglogl)
        bovy_plot.bovy_print()
        bovy_plot.bovy_plot(numpy.linspace(10.**-5.,0.3,logl.shape[8]),
                            numpy.exp(alogl).T,'k-',
                            xrange=[0.,0.3],
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
    #Now plot combined
    alogl= numpy.sum(allmarglogl,axis=1)\
        -numpy.amax(numpy.sum(allmarglogl,axis=1))
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(numpy.linspace(10.**-5.,0.3,logl.shape[8]),
                        numpy.exp(alogl).T,'k-',
                        xrange=[0.,0.3],
                        xlabel='$P_{\mathrm{out}}$',
                        yrange=[0.,1.1])
    bovy_plot.bovy_end_print(options.outfilename)

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
        bovy_plot.bovy_plot(numpy.linspace(-0.1,0.1,logl.shape[7])*220.,
                            numpy.exp(alogl).T,'k-',
                            xrange=[-25.,25.],
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
    #Now plot combined
    alogl= numpy.sum(allmarglogl,axis=1)\
        -numpy.amax(numpy.sum(allmarglogl,axis=1))
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(numpy.linspace(-0.1,0.1,logl.shape[7])*220.,
                        numpy.exp(alogl).T,'k-',
                        xrange=[-25.,25.],
                        xlabel=r'$\Delta \bar{V}_T\ (\mathrm{km\,s}^{-1})$',
                        yrange=[0.,1.1])
    bovy_plot.bovy_end_print(options.outfilename)

def plotloglhist(options,args):
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
    elif options.type.lower() == 'rdpout':
        plotRdPout(options,args)
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
                            
