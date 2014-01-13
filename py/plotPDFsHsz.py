import sys
import os, os.path
import cPickle as pickle
import numpy
from scipy import misc, integrate
import multiprocessing
from galpy.util import bovy_plot, multi
from galpy.df_src.quasiisothermaldf import quasiisothermaldf
from galpy import potential
import monoAbundanceMW
from segueSelect import _ERASESTR
from pixelFitDF import get_options, approxFitResult, _REFV0, _REFR0, \
    setup_potential, setup_aA, setup_dfgrid, nnsmooth
_NOTDONEYET= True
_FIXOUTLIERS= False
_NKPOPS= 54
def plotRdfh(options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= _NKPOPS
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
    if numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii],
                                    k=(options.sample.lower() == 'k'))/8.) > -0.5:
        return None
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
        for jj in range(marglogl.shape[0]):
            for kk in range(marglogl.shape[1]):
                marglogl[jj,kk]= misc.logsumexp(logl[jj,0,0,kk,:,:,:,0].flatten())
            condlogp[jj]= misc.logsumexp(marglogl[jj,:])
            condlogl= marglogl[jj,:]-misc.logsumexp(marglogl[jj,:])
            condfh[jj]= numpy.sum(numpy.exp(condlogl)*numpy.linspace(0.,1.,logl.shape[3]))/numpy.sum(numpy.exp(condlogl))
#        if monoAbundanceMW.hr(fehs[ii],afes[ii]) < 3.5 \
#                and numpy.amax(logl) < 0.: #latter removes ridiculous bins
#            allmarglogl[:,:,ii]= marglogl
        #Normalize
        if _FIXOUTLIERS:
            #marglogl[(numpy.fabs(marglogl-numpy.median(marglogl[True-numpy.isnan(marglogl)])) > 8)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
            indx= marglogl < -1000000000000.
            nns, dev= nnsmooth(marglogl)
            marglogl= nns
            marglogl[indx]= -1000000000000.
        alogl= marglogl-numpy.nanmax(marglogl)
        bovy_plot.bovy_print(text_fontsize=20.,
                             legend_fontsize=24.,
                             xtick_labelsize=18.,
                             ytick_labelsize=18.,
                             axes_labelsize=24.)
        bovy_plot.bovy_dens2d(numpy.exp(alogl).T,
                              origin='lower',cmap='gist_yarg',
                              interpolation='nearest',
                              xrange=[1.9,3.5],yrange=[-1./32.,1.+1./32.],
                              xlabel=r'$\mathrm{disk\ scale\ length}\,(\mathrm{kpc})$',
                              ylabel=r'$\mathrm{relative\ halo\ contribution\ to}\ V^2_c$')
#                              xlabel=r'$R_d\ (\mathrm{kpc})$',ylabel=r'$f_h$')
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
        bovy_plot.bovy_text(r'$[\mathrm{Fe/H}] = %.2f$' % (fehs[ii])
                            +'\n'
                            r'$[\alpha/\mathrm{Fe}] = %.3f$' % (afes[ii]),
#                            +'\n'
#                            r'$N_{\mathrm{data}} = %i$' % (ndatas[ii]),
                            size=16.,bottom_right=True)
#        bovy_plot.bovy_plot(numpy.linspace(2.0,3.4,logl.shape[0])[maxindx],
#                            condfh[maxindx],color='blue',marker='o',
#                            ls='none',
#                            ms=8.,mec='none',
#                            overplot=True,zorder=13)
        #Add bin
        
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

def plotRdhr(options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= _NKPOPS
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
        npops= _NKPOPS
    if numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii],
                                    k=(options.sample.lower() == 'k'))/8.) > -0.5:
        return None
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
                    marglogl[jj,kk]= misc.logsumexp(logl[jj,0,0,:,kk,:,:,0])-misc.logsumexp(logl[:,0,0,:,kk,:,:,0])
                else:
                    marglogl[jj,kk]= misc.logsumexp(logl[jj,0,0,:,kk,:,:,0])
        #Normalize
        alogl= marglogl-numpy.amax(marglogl)
        #Get hR range
        hrs, srs, szs=  setup_dfgrid(fehs,afes,options)
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
        npops= _NKPOPS
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
        npops= _NKPOPS
    if numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii],
                                    k=(options.sample.lower() == 'k'))/8.) > -0.5:
        return None
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
                    marglogl[jj,kk]= misc.logsumexp(logl[jj,0,0,:,3:,:,:,:,kk,:,:].flatten())-misc.logsumexp(logl[:,0,0,:,3:,:,:,:,kk,:,:].flatten())
                else:
                    marglogl[jj,kk]= misc.logsumexp(logl[jj,0,0,:,3:,:,:,:,kk,:,:].flatten())
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
        npops= _NKPOPS
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
        npops= _NKPOPS
    if options.sample.lower() == 'g' \
            and numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii]) /8.) > -0.5:
        return None
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
                    marglogl[jj,kk]= misc.logsumexp(logl[:,0,0,kk,3:,:,:,:,jj,:,:].flatten())-misc.logsumexp(logl[:,0,0,:,3:,:,:,:,jj,:,:].flatten())
                else:
                    marglogl[jj,kk]= misc.logsumexp(logl[:,0,0,kk,3:,:,:,:,jj,:,:].flatten())
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
        npops= _NKPOPS
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
        npops= _NKPOPS
    if options.sample.lower() == 'g' \
            and numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii]) /8.) > -0.5:
        return None
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
                    marglogl[jj,kk]= misc.logsumexp(logl[jj,0,0,:,3:,:,:,kk,:,:,:].flatten())
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

def plothszsz(options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= _NKPOPS
    if not options.multi is None:
        dummy= multi.parallel_map((lambda x: plothszsz_single(x,options,args)),
                                  range(npops),
                                  numcores=numpy.amin([options.multi,
                                                       npops,
                                                       multiprocessing.cpu_count()]))
    else:
        for ii in range(npops):
            plothszsz_single(ii,options,args)

def plothszsz_single(ii,options,args):
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
        npops= _NKPOPS
    if numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii],
                                    k=(options.sample.lower() == 'k'))/8.) > -0.5:
        return None
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
                    marglogl[jj,kk]= misc.logsumexp(logl[:,0,0,:,:,jj,kk,0].flatten())-misc.logsumexp(logl[:,0,0,:,:,:,kk,0].flatten())
                else:
                    marglogl[jj,kk]= misc.logsumexp(logl[:,0,0,:,:,jj,kk,0].flatten())
        #Normalize
        alogl= marglogl-numpy.amax(marglogl)
        #Get ranges
        hrs, srs, szs=  setup_dfgrid([fehs[ii]],[afes[ii]],options)
        bovy_plot.bovy_print(text_fontsize=20.,
                             legend_fontsize=24.,
                             xtick_labelsize=18.,
                             ytick_labelsize=18.,
                             axes_labelsize=24.)
        bovy_plot.bovy_dens2d(numpy.exp(alogl).T,
                              origin='lower',cmap='gist_yarg',
                              interpolation='nearest',
                              xrange=[srs[0]-(srs[1]-srs[0])/2.,
                                      srs[-1]+(srs[1]-srs[0])/2.],
                              yrange=[szs[0]-(szs[1]-szs[0])/2.,
                                      szs[-1]+(szs[1]-szs[0])/2.],
                              xlabel=r'$\ln h_{\sigma_Z} / 8\ \mathrm{kpc}$',
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

def plotRdsz(options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= _NKPOPS
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
    savefile.close()
    if numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii],
                                    k=(options.sample.lower() == 'k'))/8.) > -0.5:
        return None
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
                marglogl[jj,kk]= misc.logsumexp(logl[jj,0,0,:,:,:,kk,0].flatten())
        #Normalize
        alogl= marglogl-numpy.amax(marglogl)
        #Get range
        hrs, srs, szs=  setup_dfgrid(fehs,afes,options)
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
        npops= _NKPOPS
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
        npops= _NKPOPS
    #Go through all of the bins
    if options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    savefile.close()
    if numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii],
                                    k=(options.sample.lower() == 'k'))/8.) > -0.5:
        return None
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
        bfpouts= numpy.zeros((logl.shape[0],logl.shape[3]))+numpy.nan
        for jj in range(bfpouts.shape[0]):
            for kk in range(bfpouts.shape[1]):
                if misc.logsumexp(logl[jj,0,0,kk,:,:,:,0]) > -10000000000000.:
                    hrindx, srindx, szindx= numpy.unravel_index(numpy.argmax(logl[jj,0,0,kk,:,:,:,0]),(options.nhrs,options.nsrs,options.nszs))
                    bfpouts[jj,kk]= logl[jj,0,0,kk,hrindx,srindx,szindx,2]
        #Normalize
        bovy_plot.bovy_print()
        bovy_plot.bovy_dens2d(bfpouts.T,origin='lower',cmap='jet',
                              interpolation='nearest',
                              xrange=[1.9,3.5],yrange=[-1./32.,1.+1./32.],
                              xlabel=r'$R_d\ (\mathrm{kpc})$',ylabel=r'$f_h$',
                              zlabel=r'$f_{out}$',
                              colorbar=True,
                              vmin=0.,vmax=.5)
        #Plotname
        spl= options.outfilename.split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        bovy_plot.bovy_end_print(newname)

def plothrreal(options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= _NKPOPS
    if not options.multi is None:
        dummy= multi.parallel_map((lambda x: plothrreal_single(x,options,args)),
                                  range(npops),
                                  numcores=numpy.amin([options.multi,
                                                       npops,
                                                       multiprocessing.cpu_count()]))
    else:
        for ii in range(npops):
            plothrreal_single(ii,options,args)

def plothrreal_single(ii,options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= _NKPOPS
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    if numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii],
                                    k=(options.sample.lower() == 'k'))/8.) > -0.5:
        return None
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
        #Get hR range
        hrs, srs, szs=  setup_dfgrid(fehs,afes,options)
        rds= numpy.linspace(2.,3.4,8)
        fhs= numpy.linspace(0.,1.,16)
        bfhrs= numpy.zeros((logl.shape[0],logl.shape[3]))+numpy.nan
        vo= options.fixvc/220.
        options.fitdvt= False #Just to make sure what follows works
        for jj in range(bfhrs.shape[0]):
            print "Rd %f" % (rds[jj])
            for kk in range(bfhrs.shape[1]):
                print "fh %i" % kk
                if misc.logsumexp(logl[jj,0,0,kk,:,:,:,0]) > -10000000000000.:
                    hrindx, srindx, szindx= numpy.unravel_index(numpy.argmax(logl[jj,0,0,kk,:,:,:,0]),(options.nhrs,options.nsrs,options.nszs))
                    #Setup potential
                    potparams= numpy.array([numpy.log(rds[jj]/8.),vo,numpy.log(options.fixzh/8000.),fhs[kk],options.dlnvcdlnr])
                    pot= setup_potential(potparams,options,0)
                    aA= setup_aA(pot,options)
                    qdf= quasiisothermaldf(numpy.exp(hrs[hrindx]),
                                           numpy.exp(srs[srindx])/vo,
                                           numpy.exp(szs[szindx])/vo,
                                           1.,7./8.,
                                           pot=pot,aA=aA,cutcounter=True)
                    bfhrs[jj,kk]= qdf.estimate_hr(1.,z=0.8/8.,dR=0.33,gl=True)*8.
                    print bfhrs[jj,kk], numpy.exp(hrs[hrindx])*8.
        #Normalize
        bovy_plot.bovy_print()
        bovy_plot.bovy_dens2d(bfhrs.T,origin='lower',cmap='jet',
                              interpolation='nearest',
                              xrange=[1.9,3.5],yrange=[-1./32.,1.+1./32.],
                              xlabel=r'$R_d\ (\mathrm{kpc})$',ylabel=r'$f_h$',
                              zlabel=r'$h_R^{\mathrm{phys}}\ (\mathrm{kpc})$',
                              colorbar=True,
                              vmin=0.5,vmax=4.5)
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
        npops= _NKPOPS
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
                marglogl[jj]= misc.logsumexp(logl[:,0,0,:,:,:,:,jj,:,:,:].flatten()[indx])
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
        npops= _NKPOPS
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
        npops= _NKPOPS
    #Go through all of the bins
    if options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    savefile.close()
    if numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii],
                                    k=(options.sample.lower() == 'k'))/8.) > -0.5:
        return None
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
        #logl[:,:,:,:3,:,:,:,:,:,:]= -100000000000000.
        if _NOTDONEYET:
            logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        bovy_plot.bovy_print()
        bovy_plot.bovy_hist(logl[:,:,:,:,:,:,:,0].flatten()[(logl[:,:,:,:,:,:,:,0].flatten() >-1000000.)],
                            xrange=[numpy.nanmax(logl[:,:,:,:,:,:,:,0])-50.,numpy.nanmax(logl[:,:,:,:,:,:,:,0])],
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

def plotderived(options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= _NKPOPS
    if not options.multi is None:
        dummy= multi.parallel_map((lambda x: plotderived_single(x,options,args)),
                                  range(npops),
                                  numcores=numpy.amin([options.multi,
                                                       npops,
                                                       multiprocessing.cpu_count()]))
    else:
        for ii in range(npops):
            plotderived_single(ii,options,args)

def plotderived_single(ii,options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    npops= len(fehs)
    savefile.close()
    if numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii],
                                    k=(options.sample.lower() == 'k'))/8.) > -0.5:
        return None
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
        nderived= 8
        dmarglogl= numpy.zeros((logl.shape[0],logl.shape[3],nderived))
        rds= numpy.linspace(2.,3.4,8)
        fhs= numpy.linspace(0.,1.,16)
        ro= 1.
        vo= options.fixvc/220.
        options.fitdvt= False #Just to make sure what follows works
        for jj in range(marglogl.shape[0]):
            for kk in range(marglogl.shape[1]):
                marglogl[jj,kk]= misc.logsumexp(logl[jj,0,0,kk,:,:,:,0].flatten())
                #Setup potential to calculate stuff
                potparams= numpy.array([numpy.log(rds[jj]/8.),vo,numpy.log(options.fixzh/8000.),fhs[kk],options.dlnvcdlnr])
                try:
                    pot= setup_potential(potparams,options,0,returnrawpot=True)
                except RuntimeError:
                    continue
                #First up, total surface density
                surfz= 2.*integrate.quad((lambda zz: potential.evaluateDensities(1.,zz,pot)),0.,options.height/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro
                dmarglogl[jj,kk,0]= surfz
                #Disk density
                surfzdisk= 2.*pot[0].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.*400./8000.*ro*_REFR0*1000.
                dmarglogl[jj,kk,1]= surfzdisk
                #halo density
                rhodm= pot[1].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.
                dmarglogl[jj,kk,2]= rhodm
                #total density
                rhoo= potential.evaluateDensities(1.,0.,pot)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.
                dmarglogl[jj,kk,3]= rhoo
                #mass of the disk
                rhod= pot[0].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.
                massdisk= rhod*2.*options.fixzh/8000.*numpy.exp(8./rds[jj])*rds[jj]**2./8**2.*2.*numpy.pi*(ro*_REFR0)**3./10.
                dmarglogl[jj,kk,4]= massdisk
                #pl halo
                dmarglogl[jj,kk,5]= pot[1].alpha
                #vcdvc
                vcdvc= pot[0].vcirc(2.2*rds[jj]/8.)/potential.vcirc(pot,2.2*rds[jj]/8.)
                dmarglogl[jj,kk,6]= vcdvc
                #rd
                dmarglogl[jj,kk,7]= rds[jj]
        #Calculate mean and stddv
        alogl= marglogl-misc.logsumexp(marglogl.flatten())
        margp= numpy.exp(alogl)
        mean_surfz= numpy.sum(dmarglogl[:,:,0]*margp)
        std_surfz= numpy.sqrt(numpy.sum(dmarglogl[:,:,0]**2.*margp)-mean_surfz**2.)
        mean_surfzdisk= numpy.sum(dmarglogl[:,:,1]*margp)
        std_surfzdisk= numpy.sqrt(numpy.sum(dmarglogl[:,:,1]**2.*margp)-mean_surfzdisk**2.)
        mean_rhodm= numpy.sum(dmarglogl[:,:,2]*margp)
        std_rhodm= numpy.sqrt(numpy.sum(dmarglogl[:,:,2]**2.*margp)-mean_rhodm**2.)
        mean_rhoo= numpy.sum(dmarglogl[:,:,3]*margp)
        std_rhoo= numpy.sqrt(numpy.sum(dmarglogl[:,:,3]**2.*margp)-mean_rhoo**2.)
        mean_massdisk= numpy.sum(dmarglogl[:,:,4]*margp)
        std_massdisk= numpy.sqrt(numpy.sum(dmarglogl[:,:,4]**2.*margp)-mean_massdisk**2.)
        mean_plhalo= numpy.sum(dmarglogl[:,:,5]*margp)
        std_plhalo= numpy.sqrt(numpy.sum(dmarglogl[:,:,5]**2.*margp)-mean_plhalo**2.)
        mean_vcdvc= numpy.sum(dmarglogl[:,:,6]*margp)
        std_vcdvc= numpy.sqrt(numpy.sum(dmarglogl[:,:,6]**2.*margp)-mean_vcdvc**2.)
        mean_rd= numpy.sum(dmarglogl[:,:,7]*margp)
        std_rd= numpy.sqrt(numpy.sum(dmarglogl[:,:,7]**2.*margp)-mean_rd**2.)
        bovy_plot.bovy_print()
        bovy_plot.bovy_text(r'$\Sigma(R_0,|Z| < 1.1\,\mathrm{kpc}) = %.1f \pm %.1f\,M_\odot\,\mathrm{pc}^{-2}$' % (mean_surfz,std_surfz)
                            +'\n'
                            +r'$\Sigma_{\mathrm{disk}}(R_0) = %.1f \pm %.1f\,M_\odot\,\mathrm{pc}^{-2}$' % (mean_surfzdisk,std_surfzdisk)
                            +'\n'
                            +r'$\rho_{\mathrm{total}}(R_0,Z=0) = %.3f \pm %.3f\,M_\odot\,\mathrm{pc}^{-3}$' % (mean_rhoo,std_rhoo)
                            +'\n'
                            +r'$M_{\mathrm{disk}} = %.3f \pm %.3f\, \times 10^{10}\,M_{\odot}$' % (mean_massdisk,std_massdisk)
                            +'\n'
                            +r'$R_d = %.1f \pm %.1f\, \mathrm{kpc}$' % (mean_rd,std_rd)
                            +'\n'
                            +r'$V_{c,\mathrm{disk}}/V_c\,(2.2\,R_d) = %.2f \pm %.2f$' % (mean_vcdvc,std_vcdvc)
                            +'\n'
                            +r'$\rho_{\mathrm{DM}}(R_0,Z=0) = %.4f \pm %.4f\,M_\odot\,\mathrm{pc}^{-3}$' % (mean_rhodm,std_rhodm)
                            +'\n'
                            +r'$\alpha_{h}= %.2f \pm %.2f$' % (mean_plhalo,std_plhalo),
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
    return None

def plotprops(options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        npops= 62
        savefile= open('binmapping_g.sav','rb')
    elif options.sample.lower() == 'k':
        npops= _NKPOPS
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
                            r'$\ln h_R / 8\,\mathrm{kpc} = %.1f$' % (numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii],k=(options.sample.lower() == 'k'))/8.)) 
                            +'\n'
                            +r'$\ln \sigma_R / 220\,\mathrm{km\,s}^{-1} = %.1f$' % (numpy.log(monoAbundanceMW.sigmar(fehs[ii],afes[ii],k=(options.sample.lower() == 'k'))/220.))
                            +'\n'
                            +r'$\ln \sigma_Z / 220\,\mathrm{km\,s}^{-1} = %.1f$' % (numpy.log(monoAbundanceMW.sigmaz(fehs[ii],afes[ii],k=(options.sample.lower() == 'k'))/220.)),
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
    elif options.type.lower() == 'hszsz':
        plothszsz(options,args)
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
    elif options.type.lower() == 'derived':
        plotderived(options,args)
    elif options.type.lower() == 'df4fidpot':
        plotDF4fidpot(options,args)
    elif options.type.lower() == 'bestpot':
        plotbestpot(options,args)
    elif options.type.lower() == 'hrreal':
        plothrreal(options,args)
                            
