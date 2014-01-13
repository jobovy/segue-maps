import sys
import os, os.path
import cPickle as pickle
import numpy
from scipy import misc, integrate, optimize
import multiprocessing
from matplotlib.patches import Ellipse
from matplotlib import pyplot, cm
from galpy.util import bovy_plot, multi, save_pickles
from galpy.df_src.quasiisothermaldf import quasiisothermaldf
from galpy import potential
import bovy_mcmc
import monoAbundanceMW
from segueSelect import _ERASESTR
from pixelFitDF import get_options, approxFitResult, _REFV0, _REFR0, \
    setup_potential, setup_aA, setup_dfgrid, nnsmooth, read_rawdata
from calcDerivProps import rawDerived, calcSurfErr, calcSurfRdCorr, \
    calcSurfRdCorrZ, calcSurfErrZ, calcDerivProps
from plotDensComparisonDFMulti4gridall import getMultiComparisonBins
from selectFigs import _squeeze
from pixelFitDens import pixelAfeFeh
_NOTDONEYET= True
#lABELS
labels= {}
labels['fh']= r'$f_h$'
labels['vc']= r'$V_c\, (\mathrm{km\,s}^{-1})$'
labels['pout']= r'$P_{\mathrm{out}}$'
labels['fracfaint']= r'$\mathrm{Fraction\ on\ faint\ plates}$'
labels['nfaint']= r'$\mathrm{Number\ on\ faint\ plates}$'
labels['hz']= r'$h_z\ (\mathrm{pc})$'
labels['hr']= r'$h_R\ (\mathrm{kpc})$'
labels['rd']= r'$R_d\ (\mathrm{kpc})$'
labels['zh']= r'$z_h\ (\mathrm{pc})$'
labels['vcdvcro']= r'$V_{c,\mathrm{disk}}/V_c\,(R_0)$'
labels['vcdvc']= r'$V_{c,\mathrm{disk}}/V_c\,(2.2\,R_d)$'
labels['surfzdisk']= r'$\Sigma_{\mathrm{disk}}(R_0)\ (M_{\odot}\,\mathrm{pc}^{-2})$'
labels['massdisk']= r'$M_{\mathrm{disk}}\ (10^{10}\,M_{\odot})$'
labels['plhalo']= r'$\alpha_{h}$'
labels['rhodm']= r'$\rho_{\mathrm{DM}}\,(R_0,0)\ (M_{\odot}\,\mathrm{pc}^{-3})$'
labels['dlnvcdlnr']= r'$\frac{\mathrm{d}\ln V_c}{\mathrm{d}\ln R}\,(R_0)$'
labels['rhoo']= r'$\rho_{\mathrm{total}}\,(R_0,0)\ (M_{\odot}\,\mathrm{pc}^{-3})$'
labels['surfz800']= r'$\Sigma(R_0,|Z|\leq 0.8\,\mathrm{kpc})\ (M_{\odot}\,\mathrm{pc}^{-2})$'
labels['surfz']= r'$\Sigma(R_0,|Z|\leq 1.1\,\mathrm{kpc})\ (M_{\odot}\,\mathrm{pc}^{-2})$'
#RANGES
ranges= {}
ranges['fh']= [0.,1.]
ranges['vc']= [150.,270.]
ranges['pout']= [0.,0.5]
ranges['fracfaint']= [0.,0.5]
ranges['nfaint']= [0.,200.]
ranges['hz']= [150.,1000.]
ranges['hr']= [1.5,5.]
ranges['rd']= [1.9,3.5]
ranges['zh']= [100.,500.]
ranges['surfz']= [50.,100.]
ranges['surfzdisk']= [20.,90.]
ranges['rhodm']= [0.,0.014]
ranges['rhoo']= [0.,0.2]
ranges['plhalo']= [0.,3.]
ranges['massdisk']= [0.,10.]
ranges['dlnvcdlnr']= [-10.,2.]
ranges['vcdvc']= [0.,1.]
ranges['vcdvcro']= [0.,1.]
def plot1d(options,args):
    """Make a plot of a quantity's best-fit vs. FeH and aFe"""
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= 54
    if options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    savefile.close()
    #First calculate the derivative properties
    if not options.multi is None:
        derivProps= multi.parallel_map((lambda x: calcAllDerivProps(x,options,args)),
                                  range(npops),
                                  numcores=numpy.amin([options.multi,
                                                       npops,
                                                       multiprocessing.cpu_count()]))
    else:
        derivProps= []
        for ii in range(npops):
            derivProps.append(calcAllDerivProps(ii,options,args))
    #Load into plotthis
    plotthis= numpy.zeros(npops)+numpy.nan
    for ii in range(npops):
        if numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii],
                                         k=(options.sample.lower() == 'k')) /8.) > -0.5 \
                or (options.sample.lower() == 'g' and ii < 6) \
                or (options.sample.lower() == 'k' and ii < 7):
            continue
        plotthis[ii]= derivProps[ii][options.subtype.lower()]
    #Now plot
    bovy_plot.bovy_print()
    monoAbundanceMW.plotPixelFunc(fehs,afes,plotthis,
                                  zlabel=labels[options.subtype.lower()])
    bovy_plot.bovy_end_print(options.outfilename)
    return None        

def plotbestr(options,args):
    """Make a plot of a quantity's best-fit vs. FeH and aFe"""
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= 54
    if options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    savefile.close()
    #First calculate the derivative properties
    if not options.multi is None:
        derivProps= multi.parallel_map((lambda x: calcAllSurfErr(x,options,args)),
                                  range(npops),
                                  numcores=numpy.amin([options.multi,
                                                       npops,
                                                       multiprocessing.cpu_count()]))
    else:
        derivProps= []
        for ii in range(npops):
            derivProps.append(calcAllSurfErr(ii,options,args))
    #If a second argument is given, this gives a set of rs at which also to calculate the surface density
    if len(args) > 1:
        if os.path.exists(args[1]):
            surffile= open(args[1],'rb')
            altsurfrs= pickle.load(surffile)
            surffile.close()
            calcExtra= True
        else:
            raise IOError("extra savefilename with surface-densities has to exist when it is specified")
    else:
        calcExtra= False
    if not calcExtra: #Fiducial, for which we also calculate everything at the mean radius of each MAP
        #Load g orbits
        orbitsfile= 'gOrbitsNew.sav'
        savefile= open(orbitsfile,'rb')
        orbits= pickle.load(savefile)
        savefile.close()
        #Cut to S/N, logg, and EBV
        indx= (orbits.sna > 15.)*(orbits.logga > 4.2)*(orbits.ebv < 0.3)
        orbits= orbits[indx]
        #Load the orbits into the pixel structure
        pix= pixelAfeFeh(orbits,dfeh=0.1,dafe=0.05)
        #Now calculate meanr
        rmean= numpy.zeros(npops)
        for ii in range(npops):
            data= pix(fehs[ii],afes[ii])
            vals= data.densrmean*8.
            if False:#True:
                rmean[ii]= numpy.mean(vals)
            else:
                rmean[ii]= numpy.median(vals)
    #Load into plotthis
    plotthis= numpy.zeros(npops)+numpy.nan
    plotthis_y= numpy.zeros(npops)+numpy.nan
    plotthis_y_err= numpy.zeros(npops)+numpy.nan
    plotthiskz_y= numpy.zeros(npops)+numpy.nan
    plotthiskz_y_err= numpy.zeros(npops)+numpy.nan
    altplotthis= numpy.zeros(npops)+numpy.nan
    altplotthis_y= numpy.zeros(npops)+numpy.nan
    altplotthis_y_err= numpy.zeros(npops)+numpy.nan
    altplotthiskz_y= numpy.zeros(npops)+numpy.nan
    altplotthiskz_y_err= numpy.zeros(npops)+numpy.nan
    for ii in range(npops):
        if numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii],
                                         k=(options.sample.lower() == 'k')) /8.) > -0.5 \
                or (options.sample.lower() == 'g' and (ii < 0 or ii == 50)) \
                or (options.sample.lower() == 'k' and ii < 7):
            continue
        #Determine best-r
        #indx= numpy.argmin(derivProps[ii][:,2]/numpy.fabs(derivProps[ii][:,1]))
        indx= numpy.argmin(numpy.fabs(derivProps[ii][:,2]))
        if indx == 0: indx= int(numpy.floor(numpy.random.uniform()*10))
        plotthis[ii]= derivProps[ii][indx,0]
        plotthis_y[ii]= derivProps[ii][indx,1]
        plotthis_y_err[ii]= derivProps[ii][indx,3]
        plotthiskz_y[ii]= derivProps[ii][indx,4]
        plotthiskz_y_err[ii]= derivProps[ii][indx,5]
        if calcExtra:
            indx= numpy.argmin(numpy.fabs(derivProps[ii][:,0]-altsurfrs[ii]))
            altplotthis[ii]= derivProps[ii][indx,0]
            altplotthis_y[ii]= derivProps[ii][indx,1]
            altplotthis_y_err[ii]= derivProps[ii][indx,3]           
            altplotthiskz_y[ii]= derivProps[ii][indx,4]
            altplotthiskz_y_err[ii]= derivProps[ii][indx,5]           
        else:
            indx= numpy.argmin(numpy.fabs(derivProps[ii][:,0]-rmean[ii]))
            altplotthis[ii]= derivProps[ii][indx,0]
            altplotthis_y[ii]= derivProps[ii][indx,1]
            altplotthis_y_err[ii]= derivProps[ii][indx,3]           
            altplotthiskz_y[ii]= derivProps[ii][indx,4]
            altplotthiskz_y_err[ii]= derivProps[ii][indx,5]           
    #Now plot
    bovy_plot.bovy_print()
    monoAbundanceMW.plotPixelFunc(fehs,afes,plotthis,
                                  zlabel=r'$R_\Sigma\ (\mathrm{kpc})$')
    bovy_plot.bovy_end_print(options.outfilename)
    bovy_plot.bovy_print()
    print plotthis, plotthis_y
    bovy_plot.bovy_plot(plotthis,plotthis_y,'ko',
                        xlabel=r'$R\ (\mathrm{kpc})$',
                        ylabel=r'$\Sigma(R,|Z| \leq 1.1\,\mathrm{kpc})\ (M_\odot\,\mathrm{pc}^{-2})$',
                        xrange=[4.,10.],
                        yrange=[10,1050.],#,numpy.nanmin(plotthis_y)-10.,
#                                numpy.nanmax(plotthis_y)+10.],
                        semilogy=True)
    pyplot.errorbar(plotthis,
                    plotthis_y,
                    yerr=plotthis_y_err,
                    elinewidth=1.,capsize=3,zorder=0,
                    color='k',linestyle='none')  
    trs= numpy.linspace(4.3,9.,1001)
    pyplot.plot(trs,72.*numpy.exp(-(trs-8.)/3.),'k--')
    pyplot.plot(trs,72.*numpy.exp(-(trs-8.)/2.),'k-.')
    pyplot.plot(trs,72.*numpy.exp(-(trs-8.)/4.),'k:')
    #Fit exponential
    #indx= (plotthis < 8.)
    #plotthis= plotthis[indx]
    #plotthis_y= plotthis_y[indx]
    #plotthis_y_err= plotthis_y_err[indx]
    exp_params= optimize.fmin_powell(expcurve,
                                     numpy.log(numpy.array([72.,2.5])),
                                     args=(plotthis,plotthis_y,plotthis_y_err))
    pyplot.plot(trs,numpy.exp(exp_params[0]-(trs-8.)/numpy.exp(exp_params[1])),
                'k-',lw=2.)
    print numpy.exp(exp_params) 
    bovy_plot.bovy_end_print(options.outfilename.replace('.png','_rvssurf.png'))
    #Save
    if calcExtra:
        save_pickles(options.outfilename.replace('.png','_rvssurf.sav'),
                     plotthis,plotthis_y,plotthis_y_err,
                     plotthiskz_y,plotthiskz_y_err,
                     altplotthis,altplotthis_y,altplotthis_y_err,
                     altplotthiskz_y,altplotthiskz_y_err)
    else:
        save_pickles(options.outfilename.replace('.png','_rvssurf.sav'),
                     plotthis,plotthis_y,plotthis_y_err,
                     plotthiskz_y,plotthiskz_y_err,
                     altplotthis,altplotthis_y,altplotthis_y_err,
                     altplotthiskz_y,altplotthiskz_y_err)
    return None        
    
def expcurve(params,x,y,err):
    so= numpy.exp(params[0])
    rd= numpy.exp(params[1])
    q= 9.
    return numpy.nansum(q*(y-so*numpy.exp(-(x-8.)/rd))**2./(q*err**2.+(y-so*numpy.exp(-(x-8.)/rd))**2.))

def plotbestz(options,args):
    """Make a plot of a quantity's best-fit vs. FeH and aFe"""
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= 54
    if options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    savefile.close()
    #First calculate the derivative properties
    if not options.multi is None:
        derivProps= multi.parallel_map((lambda x: calcAllSurfErrZ(x,options,args)),
                                  range(npops),
                                  numcores=numpy.amin([options.multi,
                                                       npops,
                                                       multiprocessing.cpu_count()]))
    else:
        derivProps= []
        for ii in range(npops):
            derivProps.append(calcAllSurfErrZ(ii,options,args))
    #Load into plotthis
    plotthis= numpy.zeros(npops)+numpy.nan
    plotthis_y= numpy.zeros(npops)+numpy.nan
    plotthis_y_err= numpy.zeros(npops)+numpy.nan
    for ii in range(npops):
        if numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii],
                                         k=(options.sample.lower() == 'k')) /8.) > -0.5 \
                or (options.sample.lower() == 'g' and ii < 6) \
                or (options.sample.lower() == 'k' and ii < 7):
            continue
        #Determine best-r
        #indx= numpy.argmin(derivProps[ii][:,2]/numpy.fabs(derivProps[ii][:,1]))
        indx= numpy.argmin(numpy.fabs(derivProps[ii][:,2]))
        plotthis[ii]= derivProps[ii][indx,0]
        plotthis_y[ii]= derivProps[ii][indx,1]
        plotthis_y_err[ii]= derivProps[ii][indx,3]
    #Now plot
    bovy_plot.bovy_print()
    print plotthis, plotthis_y
    bovy_plot.bovy_plot(plotthis,plotthis_y,'ko',
                        xlabel=r'$Z\ (\mathrm{kpc})$',
                        ylabel=r'$\Sigma(R_0,|Z|)\ (M_\odot\,\mathrm{pc}^{-2})$',
                        xrange=[0.,5.],
                        yrange=[10.,1050.],#,numpy.nanmin(plotthis_y)-10.,
#                                numpy.nanmax(plotthis_y)+10.],
                        semilogy=True)
    pyplot.errorbar(plotthis,
                    plotthis_y,
                    yerr=plotthis_y_err,
                    elinewidth=1.,capsize=3,zorder=0,
                    color='k',linestyle='none')  
    #trs= numpy.linspace(4.3,9.,1001)
    #pyplot.plot(trs,72.*numpy.exp(-(trs-8.)/3.),'k--')
    #pyplot.plot(trs,72.*numpy.exp(-(trs-8.)/2.),'k-.')
    #pyplot.plot(trs,72.*numpy.exp(-(trs-8.)/4.),'k:')
    bovy_plot.bovy_end_print(options.outfilename.replace('.png','_zvssurf.png'))
    return None        
    
def plot2d(options,args):
    """Make a plot of a quantity's best-fit vs. FeH and aFe"""
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= 54
    if options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    savefile.close()
    #First calculate the derivative properties
    if not options.multi is None:
        derivProps= multi.parallel_map((lambda x: calcAllDerivProps(x,options,args)),
                                  range(npops),
                                  numcores=numpy.amin([options.multi,
                                                       npops,
                                                       multiprocessing.cpu_count()]))
    else:
        derivProps= []
        for ii in range(npops):
            derivProps.append(calcAllDerivProps(ii,options,args))
    xprop= options.subtype.split(',')[0]
    yprop= options.subtype.split(',')[1]
    if xprop == 'fracfaint' or yprop == 'fracfaint':
        #Read the data
        print "Reading the data ..."
        raw= read_rawdata(options)
        #Bin the data
        binned= pixelAfeFeh(raw,dfeh=0.1,dafe=0.05)
        for ii in range(npops):
            if numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii],
                                            k=(options.sample.lower() == 'k')) /8.) > -0.5 \
                                            or (options.sample.lower() == 'g' and (ii == 50 or ii == 57)) \
                                            or (options.sample.lower() == 'k' and ii < 7):
                                            continue
            data= binned(fehs[ii],afes[ii])
            indx= (data.dered_r > 17.8)
            derivProps[ii]['fracfaint']= numpy.sum(indx)/float(len(indx))
            derivProps[ii]['fracfaint_err']= 0.
    if xprop == 'nfaint' or yprop == 'nfaint':
        #Read the data
        print "Reading the data ..."
        raw= read_rawdata(options)
        #Bin the data
        binned= pixelAfeFeh(raw,dfeh=0.1,dafe=0.05)
        for ii in range(npops):
            if numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii],
                                            k=(options.sample.lower() == 'k')) /8.) > -0.5 \
                                            or (options.sample.lower() == 'g' and ii < 6) \
                                            or (options.sample.lower() == 'k' and ii < 7):
                                            continue
            data= binned(fehs[ii],afes[ii])
            indx= (data.dered_r > 17.8)
            derivProps[ii]['nfaint']= numpy.sum(indx)
            derivProps[ii]['nfaint_err']= 0.
    if xprop == 'hz' or yprop == 'hz':
        for ii in range(npops):
            if numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii],
                                            k=(options.sample.lower() == 'k')) /8.) > -0.5 \
                                            or (options.sample.lower() == 'g' and ii < 6) \
                                            or (options.sample.lower() == 'k' and ii < 7):
                                            continue
            hz, hzerr= monoAbundanceMW.hz(fehs[ii],afes[ii],
                                          k=(options.sample.lower() == 'k'),
                                          err=True)
            derivProps[ii]['hz']= hz
            derivProps[ii]['hz_err']= hzerr    
    if xprop == 'hr' or yprop == 'hr':
        for ii in range(npops):
            if numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii],
                                            k=(options.sample.lower() == 'k')) /8.) > -0.5 \
                                            or (options.sample.lower() == 'g' and ii < 6) \
                                            or (options.sample.lower() == 'k' and ii < 7):
                                            continue
            hr, hrerr= monoAbundanceMW.hr(fehs[ii],afes[ii],
                                          k=(options.sample.lower() == 'k'),
                                          err=True)
            derivProps[ii]['hr']= hr
            derivProps[ii]['hr_err']= hrerr    
    #Load into plotthis
    plotthis_x= numpy.zeros(npops)+numpy.nan
    plotthis_y= numpy.zeros(npops)+numpy.nan
    plotthis_x_err= numpy.zeros(npops)+numpy.nan
    plotthis_y_err= numpy.zeros(npops)+numpy.nan
    for ii in range(npops):
        if numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii],
                                         k=(options.sample.lower() == 'k')) /8.) > -0.5 \
                or (options.sample.lower() == 'g' and ii < 6) \
                or (options.sample.lower() == 'k' and ii < 7):
            continue
        plotthis_x[ii]= derivProps[ii][xprop]
        plotthis_y[ii]= derivProps[ii][yprop]
        plotthis_x_err[ii]= derivProps[ii][xprop+'_err']
        plotthis_y_err[ii]= derivProps[ii][yprop+'_err']
    #Now plot
    bovy_plot.bovy_print(fig_width=6.)
    bovy_plot.bovy_plot(plotthis_x,plotthis_y,
                        s=25.,c=afes,
                        cmap='jet',
                        xlabel=labels[xprop],ylabel=labels[yprop],
                        clabel=r'$[\alpha/\mathrm{Fe}]$',
                        xrange=ranges[xprop],yrange=ranges[yprop],
                        vmin=0.,vmax=0.5,
                        scatter=True,edgecolors='none',
                        colorbar=True)
    colormap = cm.jet
    for ii in range(npops):
        if numpy.isnan(plotthis_x[ii]): continue
        pyplot.errorbar(plotthis_x[ii],
                        plotthis_y[ii],
                        xerr=plotthis_x_err[ii],
                        yerr=plotthis_y_err[ii],
                        color=colormap(_squeeze(afes[ii],
                                                numpy.amax([numpy.amin(afes)]),
                                                            numpy.amin([numpy.amax(afes)]))),
                        elinewidth=1.,capsize=3,zorder=0)  
    bovy_plot.bovy_end_print(options.outfilename)
    return None        
    
def plotCombinedPDF(options,args):
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= 54
    if options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    savefile.close()
    #First calculate the derivative properties
    if not options.group is None:
        gafes, gfehs, legend= getMultiComparisonBins(options)
    else:
        legend= None
    if not options.multi is None:
        PDFs= multi.parallel_map((lambda x: calcAllPDFs(x,options,args)),
                                  range(npops),
                                  numcores=numpy.amin([options.multi,
                                                       npops,
                                                       multiprocessing.cpu_count()]))
    else:
        PDFs= []
        for ii in range(npops):
            PDFs.append(calcAllPDFs(ii,options,args))
    #Go through and combine
    combined_lnpdf= numpy.zeros((options.nrds,options.nfhs))
    for ii in range(npops):
        if not options.group is None:
            if numpy.amin((gfehs-fehs[ii])**2./0.1+(gafes-afes[ii])**2./0.0025) > 0.001:
                continue
        combined_lnpdf+= PDFs[ii]
    alogl= combined_lnpdf-numpy.nanmax(combined_lnpdf)
    #Now plot
    bovy_plot.bovy_print()
    bovy_plot.bovy_dens2d(numpy.exp(alogl).T,
                          origin='lower',cmap='gist_yarg',
                          interpolation='nearest',
                          xrange=[1.9,3.5],yrange=[-1./32.,1.+1./32.],
                          xlabel=r'$R_d\ (\mathrm{kpc})$',ylabel=r'$f_h$')
    if not legend is None:
        bovy_plot.bovy_text(legend,top_left=True,
                            size=14.)
    bovy_plot.bovy_end_print(options.outfilename)
    #Calculate and print derived properties
    derivProps= rawDerived(alogl,options,
                           vo=options.fixvc/_REFV0,zh=options.fixzh,
                           dlnvcdlnr=options.dlnvcdlnr)
    for key in derivProps.keys():
        if not '_err' in key:
            print key, derivProps[key], derivProps[key+'_err'], \
                derivProps[key]/derivProps[key+'_err'] 
    return None

def calcAllPDFs(ii,options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    savefile.close()
    if numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii],
                                    k=(options.sample.lower() == 'k')) /8.) > -0.5 \
                or (options.sample.lower() == 'g' and ii < 6) \
                or (options.sample.lower() == 'k' and ii < 7):
        return numpy.zeros((options.nrds,options.nfhs))
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
    for jj in range(marglogl.shape[0]):
        for kk in range(marglogl.shape[1]):
            marglogl[jj,kk]= misc.logsumexp(logl[jj,0,0,kk,:,:,:,0].flatten())
    if marglogl[-1,-1] < -10000000000000.:
        return numpy.zeros((options.nrds,options.nfhs))
    else:
        return marglogl

def calcAllDerivProps(ii,options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    savefile.close()
    if numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii],
                                    k=(options.sample.lower() == 'k')) /8.) > -0.5:
        return None
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
    derivProps= calcDerivProps(newname,vo=options.fixvc/_REFV0,zh=options.fixzh,
                               dlnvcdlnr=options.dlnvcdlnr)
    return derivProps

def calcAllSurfErr(ii,options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    savefile.close()
    if numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii],
                                    k=(options.sample.lower() == 'k')) /8.) > -0.5:
        return numpy.zeros((101,6))
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
    rs,mean_surfz, cov, std_surfz, mean_kz, std_kz= calcSurfRdCorr(newname,vo=options.fixvc/_REFV0,
                                  zh=options.fixzh,
                                  dlnvcdlnr=options.dlnvcdlnr)
    if rs is None: return numpy.zeros((101,6))
    out= numpy.zeros((len(rs),6))
    out[:,0]= rs
    out[:,1]= mean_surfz
    out[:,2]= cov
    out[:,3]= std_surfz
    out[:,4]= mean_kz
    out[:,5]= std_kz
    return out

def calcAllSurfErrZ(ii,options,args):
    #Go through all of the bins
    if options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    savefile.close()
    if numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii],
                                    k=(options.sample.lower() == 'k')) /8.) > -0.5:
        return numpy.zeros((101,4))
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
    #zs,mean_surfz, cov, std_surfz= calcSurfRdCorrZ(newname,vo=options.fixvc/_REFV0,
    zs,mean_surfz, std_surfz= calcSurfErrZ(newname,vo=options.fixvc/_REFV0,
                                              zh=options.fixzh,
                                  dlnvcdlnr=options.dlnvcdlnr)
    if zs is None: return numpy.zeros((101,4))
    out= numpy.zeros((len(zs),4))
    out[:,0]= zs
    out[:,1]= mean_surfz
    out[:,2]= std_surfz/mean_surfz
    out[:,3]= std_surfz
    return out

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    if options.type.lower() == '1d':
        plot1d(options,args)
    elif options.type.lower() == '2d':
        plot2d(options,args)
    elif options.type.lower() == 'bestr':
        plotbestr(options,args)
    elif options.type.lower() == 'bestz':
        plotbestz(options,args)
    elif options.type.lower() == 'combined':
        plotCombinedPDF(options,args)
