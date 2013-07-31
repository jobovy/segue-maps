import cPickle as pickle
import numpy
from scipy import maxentropy, integrate, interpolate
from galpy import potential
from galpy.util import bovy_plot, multi, save_pickles
from matplotlib import pyplot
from matplotlib.ticker import NullFormatter
from pixelFitDF import get_options, approxFitResult, _REFV0, _REFR0, \
    setup_potential, setup_aA, setup_dfgrid, nnsmooth, read_rawdata
from calcDerivProps import rawDerived, calcSurfErr, calcSurfRdCorr, \
    calcSurfRdCorrZ, calcSurfErrZ, calcDerivProps
from plotOverview import calcAllSurfErr
_NOTDONEYET= True
def illustrateBestR(options,args):
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
    #For bin 10, calculate the correlation between sigma and Rd
    bin= options.index
    derivProps= calcAllSurfErr(bin,options,args)
    #rs= numpy.linspace(4.5,9.,101)
    #derivProps= numpy.zeros((101,6))
    rs= derivProps[:,0]
    #also calculate the full surface density profile for each Rd's best fh
    if _NOTDONEYET:
        spl= options.restart.split('.')
    else:
        spl= args[0].split('.')
    newname= ''
    for jj in range(len(spl)-1):
        newname+= spl[jj]
        if not jj == len(spl)-2: newname+= '.'
    newname+= '_%i.' % bin
    newname+= spl[-1]
    options.potential= 'dpdiskplhalofixbulgeflatwgasalt'
    options.fitdvt= False
    savefile= open(newname,'rb')
    try:
        if not _NOTDONEYET:
            params= pickle.load(savefile)
            mlogl= pickle.load(savefile)
        logl= pickle.load(savefile)
    except:
        raise
    finally:
        savefile.close()
    if _NOTDONEYET:
        logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
    logl[numpy.isnan(logl)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
    marglogl= numpy.zeros((logl.shape[0],logl.shape[3]))
    sigs= numpy.zeros((logl.shape[0],len(rs)))
    rds= numpy.linspace(2.,3.4,8)
    fhs= numpy.linspace(0.,1.,16)
    hiresfhs= numpy.linspace(0.,1.,101)
    ro= 1.
    vo= options.fixvc/_REFV0
    for jj in range(logl.shape[0]):
        for kk in range(logl.shape[3]):
            marglogl[jj,kk]= maxentropy.logsumexp(logl[jj,0,0,kk,:,:,:,0].flatten())
        #interpolate
        tindx= marglogl[jj,:] > -1000000000.
        intp= interpolate.InterpolatedUnivariateSpline(fhs[tindx],marglogl[jj,tindx],
                                                       k=3)
        ml= intp(hiresfhs)
        indx= numpy.argmax(ml)
        indx2= numpy.argmax(marglogl[jj,:])
        #Setup potential to calculate stuff
        potparams= numpy.array([numpy.log(rds[jj]/8.),vo,numpy.log(options.fixzh/8000.),hiresfhs[indx],options.dlnvcdlnr])
        try:
            pot= setup_potential(potparams,options,0,returnrawpot=True)
        except RuntimeError:
            continue
        #Total surface density
        for ll in range(len(rs)):
            surfz= 2.*integrate.quad((lambda zz: potential.evaluateDensities(rs[ll]/_REFR0,zz,pot)),0.,options.height/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro
            sigs[jj,ll]= surfz
    #Now plot sigs
    bovy_plot.bovy_print(fig_height=7.,fig_width=5.)
    left, bottom, width, height= 0.1, 0.35, 0.8, 0.4
    axTop= pyplot.axes([left,bottom,width,height])
    fig= pyplot.gcf()
    fig.sca(axTop)
    ii= 0
    bovy_plot.bovy_plot(rs,sigs[ii,:],'k-',
                        xlabel=r'$R\ (\mathrm{kpc})$',
                        ylabel=r'$\Sigma(R,|Z| \leq 1.1\,\mathrm{kpc})\ (M_\odot\,\mathrm{pc}^{-2})$',
                        xrange=[4.,10.],
                        yrange=[10,1050.],
                        semilogy=True,overplot=True)
    for ii in range(1,logl.shape[0]):
        bovy_plot.bovy_plot(rs,sigs[ii,:],'k-',overplot=True)
    thisax= pyplot.gca()
    thisax.set_yscale('log')
    nullfmt   = NullFormatter()         # no labels
    thisax.xaxis.set_major_formatter(nullfmt)
    thisax.set_ylim(10.,1050.)
    thisax.set_xlim(4.,10.)
    pyplot.ylabel(r'$\Sigma_{1.1}(R)\ (M_\odot\,\mathrm{pc}^{-2})$')
    bovy_plot._add_ticks(yticks=False)
    bovy_plot.bovy_text(r'$[\mathrm{Fe/H}] = %.2f$' % (fehs[bin])
                        +'\n'
                        r'$[\alpha/\mathrm{Fe}] = %.3f$' % (afes[bin]),
                        size=16.,top_right=True)

    left, bottom, width, height= 0.1, 0.1, 0.8, 0.25
    ax2= pyplot.axes([left,bottom,width,height])
    fig= pyplot.gcf()
    fig.sca(ax2)
    bovy_plot.bovy_plot(rs,derivProps[:,2],'k-',lw=2.,
                        xrange=[4.,10.],
                        overplot=True)
    indx= numpy.argmin(numpy.fabs(derivProps[:,2]))
    bovy_plot.bovy_plot([4.,10.],[0.,0.],'-',color='0.5',overplot=True)
    bovy_plot.bovy_plot([rs[indx],rs[indx]],[derivProps[indx,2],1000.],
                        'k--',overplot=True)
    thisax= pyplot.gca()
    pyplot.ylabel(r'$\mathrm{Correlation\ between}$'+'\n'+r'$\!\!\!\!\!\!\!\!R_d\ \&\ \Sigma_{1.1}(R)$')
    pyplot.xlabel(r'$R\ (\mathrm{kpc})$')   
    pyplot.xlim(4.,10.)
    pyplot.ylim(-.5,.5)
    bovy_plot._add_ticks()
    pyplot.sca(axTop)
    bovy_plot.bovy_plot([rs[indx],rs[indx]],[0.01,sigs[3,indx]],
                        'k--',overplot=True)   
    bovy_plot.bovy_end_print(options.outfilename)
        

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    illustrateBestR(options,args)
