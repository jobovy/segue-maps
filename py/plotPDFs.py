import os, os.path
import cPickle as pickle
import numpy
from scipy import maxentropy
from galpy.util import bovy_plot
from pixelFitDF import get_options
_NOTDONEYET= True
def plotRdfh(options,args):
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
        if not _NOTDONEYET:
            params= pickle.load(savefile)
            mlogl= pickle.load(savefile)
        logl= pickle.load(savefile)
        savefile.close()
        if _NOTDONEYET:
            logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        marglogl= numpy.zeros((logl.shape[0],logl.shape[3]))
        if ii == 0:
            allmarglogl= numpy.zeros((logl.shape[0],logl.shape[3],npops))
        for jj in range(marglogl.shape[0]):
            for kk in range(marglogl.shape[1]):
                marglogl[jj,kk]= maxentropy.logsumexp(logl[jj,0,0,kk,:,:,:,:,:,:,:].flatten())
        allmarglogl[:,:,ii]= marglogl
        #Normalize
        alogl= marglogl-numpy.amax(marglogl)
        bovy_plot.bovy_print()
        bovy_plot.bovy_dens2d(numpy.exp(alogl).T,
                              origin='lower',cmap='gist_yarg',
                              interpolation='nearest',
                              xrange=[1.5,4.5],yrange=[0.,1.],
                              xlabel='$R_d$',ylabel='$f_h$')
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
                              xrange=[1.5,4.5],yrange=[0.,1.],
                              xlabel='$R_d$',ylabel='$f_h$')
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
        if not _NOTDONEYET:
            params= pickle.load(savefile)
            mlogl= pickle.load(savefile)
        logl= pickle.load(savefile)
        savefile.close()
        if _NOTDONEYET:
            logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        marglogl= numpy.zeros((logl.shape[5],logl.shape[6]))
        if ii == 0:
            allmarglogl= numpy.zeros((logl.shape[5],logl.shape[6],npops))
        for jj in range(marglogl.shape[0]):
            for kk in range(marglogl.shape[1]):
                marglogl[jj,kk]= maxentropy.logsumexp(logl[:,0,0,:,:,jj,kk,:,:,:,:].flatten())
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
        if not _NOTDONEYET:
            params= pickle.load(savefile)
            mlogl= pickle.load(savefile)
        logl= pickle.load(savefile)
        savefile.close()
        if _NOTDONEYET:
            logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        marglogl= numpy.zeros((logl.shape[8]))
        if ii == 0:
            allmarglogl= numpy.zeros((logl.shape[8],npops))
        for jj in range(marglogl.shape[0]):
            marglogl[jj]= maxentropy.logsumexp(logl[:,0,0,:,:,:,:,:,jj,:,:].flatten())
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
        if not _NOTDONEYET:
            params= pickle.load(savefile)
            mlogl= pickle.load(savefile)
        logl= pickle.load(savefile)
        savefile.close()
        if _NOTDONEYET:
            logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        marglogl= numpy.zeros((logl.shape[7]))
        if ii == 0:
            allmarglogl= numpy.zeros((logl.shape[7],npops))
        for jj in range(marglogl.shape[0]):
            marglogl[jj]= maxentropy.logsumexp(logl[:,0,0,:,:,:,:,jj,:,:,:].flatten())
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
                            
