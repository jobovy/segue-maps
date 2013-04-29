import sys
import os, os.path
import cPickle as pickle
import numpy
from scipy import maxentropy, integrate
import multiprocessing
from matplotlib.patches import Ellipse
from matplotlib import pyplot, cm
from galpy.util import bovy_plot, multi
from galpy.df_src.quasiisothermaldf import quasiisothermaldf
from galpy import potential
import monoAbundanceMW
from segueSelect import _ERASESTR
from pixelFitDF import get_options, approxFitResult, _REFV0, _REFR0, \
    setup_potential, setup_aA, setup_dfgrid, nnsmooth
from calcDerivProps import rawDerived, calcDerivProps
from selectFigs import _squeeze
_NOTDONEYET= True
#lABELS
labels= {}
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
ranges['rd']= [1.9,3.5]
ranges['surfz']= [50.,100.]
ranges['surfzdisk']= [20.,90.]
ranges['rhodm']= [0.,0.014]
ranges['rhoo']= [0.,0.2]
ranges['plhalo']= [0.,2.]
ranges['massdisk']= [0.,10.]
ranges['vcdvc']= [0.,1.]
ranges['vcdvcro']= [0.,1.]
def plot1d(options,args):
    """Make a plot of a quantity's best-fit vs. FeH and aFe"""
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= 30
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
        if options.sample.lower() == 'g' \
                and (numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii]) /8.) > -0.5 \
                         or ii < 6):
            continue
        plotthis[ii]= derivProps[ii][options.subtype.lower()]
    #Now plot
    bovy_plot.bovy_print()
    monoAbundanceMW.plotPixelFunc(fehs,afes,plotthis,
                                  zlabel=labels[options.subtype.lower()])
    bovy_plot.bovy_end_print(options.outfilename)
    return None        
    
def plot2d(options,args):
    """Make a plot of a quantity's best-fit vs. FeH and aFe"""
    if options.sample.lower() == 'g':
        npops= 62
    elif options.sample.lower() == 'k':
        npops= 30
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
    plotthis_x= numpy.zeros(npops)+numpy.nan
    plotthis_y= numpy.zeros(npops)+numpy.nan
    plotthis_x_err= numpy.zeros(npops)+numpy.nan
    plotthis_y_err= numpy.zeros(npops)+numpy.nan
    xprop= options.subtype.split(',')[0]
    yprop= options.subtype.split(',')[1]
    for ii in range(npops):
        if options.sample.lower() == 'g' \
                and (numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii]) /8.) > -0.5 \
                         or ii < 6):
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
        npops= 30
    if options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    savefile.close()
    #First calculate the derivative properties
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
        combined_lnpdf+= PDFs[ii]
    alogl= combined_lnpdf-numpy.nanmax(combined_lnpdf)
    #Now plot
    bovy_plot.bovy_print()
    bovy_plot.bovy_dens2d(numpy.exp(alogl).T,
                          origin='lower',cmap='gist_yarg',
                          interpolation='nearest',
                          xrange=[1.9,3.5],yrange=[-1./32.,1.+1./32.],
                          xlabel=r'$R_d\ (\mathrm{kpc})$',ylabel=r'$f_h$')
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
    if options.sample.lower() == 'g' \
            and numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii]) /8.) > -0.5:
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
            marglogl[jj,kk]= maxentropy.logsumexp(logl[jj,0,0,kk,:,:,:,0].flatten())
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
    if options.sample.lower() == 'g' \
            and numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii]) /8.) > -0.5:
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

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    if options.type.lower() == '1d':
        plot1d(options,args)
    elif options.type.lower() == '2d':
        plot2d(options,args)
    elif options.type.lower() == 'combined':
        plotCombinedPDF(options,args)
