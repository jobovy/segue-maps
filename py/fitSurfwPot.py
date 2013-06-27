import os, os.path
import sys
import math
import numpy
import pickle
from scipy import optimize, integrate
from optparse import OptionParser
from galpy import potential
from galpy.util import save_pickles, bovy_plot, multi
import multiprocessing
from matplotlib import pyplot
import bovy_mcmc
from pixelFitDF import _REFR0, _REFV0, setup_potential, logprior_dlnvcdlnr
from fitDensz import cb
from calcDFResults import setup_options
import readTerminalData
from plotOverview import labels, ranges
from plotXDPotPDFs import _eval_gauss_grid
def fitSurfwPot(options,args):
    #First read the surface densities
    if not options.surffile is None and os.path.exists(options.surffile):
        surffile= open(options.surffile,'rb')
        surfrs= pickle.load(surffile)
        surfs= pickle.load(surffile)
        surferrs= pickle.load(surffile)
        kzs= pickle.load(surffile)
        kzerrs= pickle.load(surffile)
        surffile.close()
    else:
        raise IOError("-i has to be set")
    if True:
        surfrs[50]= numpy.nan
        surfrs[57]= numpy.nan
    indx= True - numpy.isnan(surfrs)
    surfrs= surfrs[indx]
    surfs= surfs[indx]
    surferrs= surferrs[indx]
    kzs= kzs[indx]
    kzerrs= kzerrs[indx]
    if options.surfaskz:
        surfs= kzs
        surferrs= kzerrs
    #Read the terminal velocity data if necessary
    if options.fitterminal:
        cl_glon, cl_vterm, cl_corr= readTerminalData.readClemens(dsinl=options.termdsinl)
        mc_glon, mc_vterm, mc_corr= readTerminalData.readMcClureGriffiths(dsinl=options.termdsinl)
        termdata= (cl_glon,cl_vterm/_REFV0,cl_corr,
                   mc_glon,mc_vterm/_REFV0,mc_corr)
    else:
        termdata= None
    #Read the APOGEE PDF if necessary
    apogeefile= '../potential-fits/apogee_XD.sav'
    if options.apogeeprior and os.path.exists(apogeefile):
        savefile= open(apogeefile,'rb')
        apogee_vcdxamp= pickle.load(savefile)
        apogee_vcdxmean= pickle.load(savefile)
        apogee_vcdxcovar= pickle.load(savefile)
        savefile.close()
        apogeeprior= (apogee_vcdxamp,apogee_vcdxmean,apogee_vcdxcovar)
    elif not os.path.exists(apogeefile):
        raise IOError("At least one input file has to exist ...")
    else:
        apogeeprior= None
    #Setup
    if options.mcsample:
        if not options.initfile is None and os.path.exists(options.initfile):
            initfile= open(options.initfile,'rb')
            init_params= pickle.load(initfile)
            initfile.close()
        else:
            raise IOError("--init has to be set when MCMC sampling")
    else:
        init_params= [numpy.log(2.5/_REFR0),1.,numpy.log(400./8000.),0.5,0.]
        if options.fitterminal: #Add Solar velocity parameters
            init_params.extend([0.,0.])
        init_params= numpy.array(init_params)
    #Fit/sample
    potoptions= setup_options(None)
    potoptions.potential= 'dpdiskplhalofixbulgeflatwgasalt'
    potoptions.fitdvt= False
    funcargs= (options,surfrs,surfs,surferrs,potoptions,
               numpy.log(1.5/8.),numpy.log(6./8.),
               numpy.log(100./8000.),numpy.log(500./8000.),
               termdata,apogeeprior)
    if not options.mcsample:
        #Optimize likelihood
        params= optimize.fmin_powell(like_func,init_params,
                                     args=funcargs,
                                     callback=cb)
        save_pickles(args[0],params)
        #Make a plot
        if True:
            pot= setup_potential(params,potoptions,0,returnrawpot=True)
            bovy_plot.bovy_print()
            bovy_plot.bovy_plot(surfrs,surfs,'ko',
                        xlabel=r'$R\ (\mathrm{kpc})$',
                                ylabel=r'$\Sigma(R,|Z| \leq 1.1\,\mathrm{kpc})\ (M_\odot\,\mathrm{pc}^{-2})$',
                                xrange=[4.,10.],
                                yrange=[10,1050.],#,numpy.nanmin(plotthis_y)-10.,
                                #                                numpy.nanmax(plotthis_y)+10.],
                                semilogy=True)
            pyplot.errorbar(surfrs,
                            surfs,
                            yerr=surferrs,
                            elinewidth=1.,capsize=3,zorder=0,
                            color='k',linestyle='none')  
            rs= numpy.linspace(4.5,9.,21)/8.
            msurfs= numpy.zeros_like(rs)
            ro= 1.
            vo= params[1]
            for ii in range(len(rs)):
                msurfs[ii]= 2.*integrate.quad((lambda zz: potential.evaluateDensities(rs[ii],zz,pot)),0.,1.1/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro            
            pyplot.plot(rs*8.,msurfs,'k-')
            bovy_plot.bovy_end_print(options.plotfile)
    else:
        isDomainFinite= [[False,False],
                         [True,False],
                         [False,False],
                         [True,True],
                         [False,False]]
        domain= [[0.,0.],
                 [0.,0.],
                 [0.,0.],
                 [0.,1.],
                 [0.,0.]]
        step= [0.2,0.2,0.4,0.2,5.]
        if options.fitterminal:
            isDomainFinite.extend([[False,False],
                                   [False,False]])
            domain.extend([[0.,0.],[0.,0.]])
            step.extend([0.1,0.1])
        thesesamples= bovy_mcmc.markovpy(init_params,
                                         step,
                                         pdf_func,
                                         funcargs,
                                         isDomainFinite=isDomainFinite,
                                         domain=domain,
                                         nsamples=options.nsamples)
        for kk in range(len(init_params)):
            xs= numpy.array([s[kk] for s in thesesamples])
            print numpy.mean(xs), numpy.std(xs)
        save_pickles(args[0],thesesamples)
    return None

def like_func(params,options,surfrs,surfs,surferrs,
              potoptions,
              rdmin,rdmax,
              zhmin,zhmax,
              termdata,apogeeprior):
    #Check ranges
    if params[1] < 0.: return numpy.finfo(numpy.dtype(numpy.float64)).max
    if params[3] < 0. or params[3] > 1.: return numpy.finfo(numpy.dtype(numpy.float64)).max
    if params[0] < rdmin or params[0] > rdmax: return numpy.finfo(numpy.dtype(numpy.float64)).max
    if params[2] < zhmin or params[2] > zhmax: return numpy.finfo(numpy.dtype(numpy.float64)).max   
    #Setup potential
    try:
        pot= setup_potential(params,potoptions,0,returnrawpot=True)
    except RuntimeError:
        return numpy.finfo(numpy.dtype(numpy.float64)).max
    #Calculate model surface density at surfrs
    vo= params[1]
    ro= 1.
    if not options.dontfitsurf:
        modelsurfs= numpy.zeros_like(surfs)
        for ii in range(len(surfrs)):
            if options.surfaskz:
                modelsurfs[ii]= -potential.evaluatezforces(surfrs[ii]/_REFR0,1.1/_REFR0,pot)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro/2./numpy.pi
            else:
                modelsurfs[ii]= 2.*integrate.quad((lambda zz: potential.evaluateDensities(surfrs[ii]/_REFR0,zz,pot)),0.,1.1/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro
        out= 0.5*numpy.sum((surfs-modelsurfs)**2./surferrs**2.)
    else:
        out= 0.
    #Add terminal velocities
    if options.fitterminal:
        vrsun= params[5]/vo
        vtsun= params[6]/vo
        cl_glon, cl_vterm, cl_corr, mc_glon, mc_vterm, mc_corr= termdata
        #Calculate terminal velocities at data glon
        cl_vterm_model= numpy.zeros_like(cl_vterm)
        for ii in range(len(cl_glon)):
            cl_vterm_model[ii]= potential.vterm(pot,cl_glon[ii])
        cl_vterm_model+= vrsun/vo*numpy.cos(cl_glon/180.*numpy.pi)\
            -vtsun/vo*numpy.sin(cl_glon/180.*numpy.pi)
        mc_vterm_model= numpy.zeros_like(mc_vterm)
        for ii in range(len(mc_glon)):
            mc_vterm_model[ii]= potential.vterm(pot,mc_glon[ii])
        mc_vterm_model+= vrsun/vo*numpy.cos(mc_glon/180.*numpy.pi)\
            -vtsun/vo*numpy.sin(mc_glon/180.*numpy.pi)
        cl_dvterm= (cl_vterm/vo-cl_vterm_model)/options.termsigma*_REFV0*vo
        mc_dvterm= (mc_vterm/vo-mc_vterm_model)/options.termsigma*_REFV0*vo
        out+= 0.5*numpy.sum(cl_dvterm*numpy.dot(cl_corr,cl_dvterm))
        out+= 0.5*numpy.sum(mc_dvterm*numpy.dot(mc_corr,mc_dvterm))
    #Add priors
    if options.bovy09voprior:
        out+= 0.5*(vo-236./_REFV0)**2./(11./_REFV0)**2.
    elif options.bovy12voprior:
        out+= 0.5*(vo-218./_REFV0)**2./(6./_REFV0)**2.
    if not options.nodlnvcdlnrprior and not options.apogeeprior: #don't double
        out-= logprior_dlnvcdlnr(params[4],options)
    #K dwarfs
    if options.lanprior:
        out+= 0.5*(2.*integrate.quad((lambda zz: potential.evaluateDensities(1.,zz,pot)),0.,1.0/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro-67.)**2./36.
        out+= 0.5*(2.*pot[0].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.*numpy.exp(params[2])*ro*_REFR0*1000.-42.)**2./36.
#        print 2.*integrate.quad((lambda zz: potential.evaluateDensities(1.,zz,pot)),0.,1.0/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro, \
#            2.*pot[0].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.*numpy.exp(params[2])*ro*_REFR0*1000.
    if options.apogeeprior:
        out-= _eval_gauss_grid([numpy.log(vo)],[params[4]],*apogeeprior)[0,0]
    return out

def pdf_func(params,*args):
    return -like_func(params,*args)

def calcDerived(options,args):
    """Calculate derived parameters for this best-fit or samples"""
    if os.path.exists(options.initfile):
        initfile= open(options.initfile,'rb')
        init_params= pickle.load(initfile)
        initfile.close()
    else:
        raise IOError("%s not found" % args[0])
    potoptions= setup_options(None)
    potoptions.potential= 'dpdiskplhalofixbulgeflatwgasalt'
    potoptions.fitdvt= False
    if options.mcsample:
        derived_params= multi.parallel_map((lambda x: calcDerivedSingle(init_params[x],options,potoptions)),
                                           range(len(init_params)),
                                           numcores=numpy.amin([len(init_params),
                                                                multiprocessing.cpu_count(),
                                                                options.multi]))
        for kk in range(len(derived_params[0])):
            xs= numpy.array([s[kk] for s in derived_params])
            print numpy.mean(xs), numpy.std(xs)
    else:
        derived_params= calcDerivedSingle(init_params,options,potoptions)
        print derived_params
    save_pickles(args[0],derived_params)
    return None

def calcDerivedSingle(params,options,potoptions):
    pot= setup_potential(params,potoptions,0,returnrawpot=True)
    ro= 1.
    vo= params[1]
    zh= numpy.exp(params[2])
    rd= numpy.exp(params[0])
    #First up, total surface density
    surfz= 2.*integrate.quad((lambda zz: potential.evaluateDensities(1.,zz,pot)),0.,1.1/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro
    #Disk density
    surfzdisk= 2.*pot[0].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.*zh*ro*_REFR0*1000.
    #halo density
    rhodm= pot[1].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.
    #total density
    rhoo= potential.evaluateDensities(1.,0.,pot)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.
    #mass of the disk
    rhod= pot[0].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.
    massdisk= rhod*2.*zh*numpy.exp(1./rd)*rd**2.*2.*numpy.pi*(ro*_REFR0)**3./10.
    #pl halo
    alpha= pot[1].alpha
    #vcdvc
    vcdvc= pot[0].vcirc(2.2*rd)/potential.vcirc(pot,2.2*rd)
    out= [surfz,surfzdisk,rhodm,rhoo,massdisk,alpha,vcdvc]
    return out
    
def plotStuff(options,args):
    if options.type == '2d':
        plot2dStuff(options,args)
    elif options.type == 'bestfitsurf':
        plotBestfitSurf(options,args)
    elif options.type == 'bestfitvterm':
        plotBestfitVterm(options,args)
    return None

def plotBestfitSurf(options,args):
    """Plot the best-fit surface density profile"""
    #First read the surface densities
    if not options.surffile is None and os.path.exists(options.surffile):
        surffile= open(options.surffile,'rb')
        surfrs= pickle.load(surffile)
        surfs= pickle.load(surffile)
        surferrs= pickle.load(surffile)
        surffile.close()
    else:
        raise IOError("-i has to be set")
    if True:
        surfrs[50]= numpy.nan
        surfrs[57]= numpy.nan
    indx= True - numpy.isnan(surfrs)
    surfrs= surfrs[indx]
    surfs= surfs[indx]
    surferrs= surferrs[indx]
    if os.path.exists(args[0]):
        initfile= open(args[0],'rb')
        init_params= pickle.load(initfile)
        initfile.close()
    else:
        raise IOError("%s not found" % args[0])
    #Setup potential
    potoptions= setup_options(None)
    potoptions.potential= 'dpdiskplhalofixbulgeflatwgasalt'
    potoptions.fitdvt= False
    pot= setup_potential(init_params,potoptions,0,returnrawpot=True)
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(surfrs,surfs,'ko',
                        xlabel=r'$R\ (\mathrm{kpc})$',
                        ylabel=r'$\Sigma(R,|Z| \leq 1.1\,\mathrm{kpc})\ (M_\odot\,\mathrm{pc}^{-2})$',
                        xrange=[4.,10.],
                        yrange=[10,1050.],
                        semilogy=True)
    pyplot.errorbar(surfrs,
                    surfs,
                    yerr=surferrs,
                    elinewidth=1.,capsize=3,zorder=0,
                    color='k',linestyle='none')  
    pyplot.errorbar([8.],[69.],yerr=[6.],marker='d',
                    elinewidth=1.,capsize=3,zorder=0,
                    color='k',linestyle='none')  
    rs= numpy.linspace(4.5,9.,21)/8.
    msurfs= numpy.zeros_like(rs)
    ro= 1.
    vo= init_params[1]
    for ii in range(len(rs)):
        msurfs[ii]= 2.*integrate.quad((lambda zz: potential.evaluateDensities(rs[ii],zz,pot)),0.,1.1/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro            
    pyplot.plot(rs*8.,msurfs,'k-')
    bovy_plot.bovy_end_print(options.plotfile)

def plotBestfitVterm(options,args):
    """Plot vterm + best-fit"""
    cl_glon, cl_vterm, cl_corr= readTerminalData.readClemens(dsinl=options.termdsinl)
    mc_glon, mc_vterm, mc_corr= readTerminalData.readMcClureGriffiths(dsinl=options.termdsinl)
    if os.path.exists(args[0]):
        initfile= open(args[0],'rb')
        init_params= pickle.load(initfile)
        initfile.close()
    else:
        raise IOError("%s not found" % args[0])
    #Setup potential
    potoptions= setup_options(None)
    potoptions.potential= 'dpdiskplhalofixbulgeflatwgasalt'
    potoptions.fitdvt= False
    pot= setup_potential(init_params,potoptions,0,returnrawpot=True)
    vo= init_params[1]
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(cl_glon,cl_vterm,'ko',
                        xlabel=r'$\mathrm{Galactic\ longitude\, (deg)}$',
                        ylabel=r'$\mathrm{Terminal\ velocity}\, (\mathrm{km\,s}^{-1})$',
                        xrange=[-100.,100.],
                        yrange=[-150.,150.])
    bovy_plot.bovy_plot(mc_glon-360.,mc_vterm,'ko',overplot=True)
    ls1= numpy.linspace(-90.,-20.,1001)
    ls2= numpy.linspace(20.,90.,1001)
    vt1= numpy.zeros_like(ls1)
    vt2= numpy.zeros_like(ls2)
    for ii in range(len(ls1)):
        vt1[ii]= potential.vterm(pot,ls1[ii])*_REFV0*vo
    for ii in range(len(ls2)):
        vt2[ii]= potential.vterm(pot,ls2[ii])*_REFV0*vo
    bovy_plot.bovy_plot(ls1,vt1,'k-',overplot=True)
    bovy_plot.bovy_plot(ls2,vt2,'k-',overplot=True)
    bovy_plot.bovy_end_print(options.plotfile)
    return None

def plot2dStuff(options,args):
    """Make MCMC plots"""
    #First load the chains
    savefile= open(args[0],'rb')
    thesesamples= pickle.load(savefile)
    savefile.close()
    if not options.derivedfile is None:
        if os.path.exists(options.derivedfile):
            derivedfile= open(options.derivedfile,'rb')
            derivedsamples= pickle.load(derivedfile)
            derivedfile.close()
        else:
            raise IOError("--derivedfile given but does not exist ...")
    samples= {}
    scaleDict= {}
    paramnames= ['rd','vc','zh','fh','dlnvcdlnr','usun','vsun']
    scale= [_REFR0,_REFV0,1000.*_REFR0,1.,1./30.*_REFV0/_REFR0,_REFV0,_REFV0]
    if len(thesesamples[0]) == 5:
        paramnames.pop()
        paramnames.pop()
        scale.pop()
        scale.pop()
    if not options.derivedfile is None:
        paramnames.extend(['surfz','surfzdisk','rhodm',
                           'rhoo','massdisk','plhalo','vcdvc'])
        scale.extend([1.,1.,1.,1.,1.,1.,1.])
    for kk in range(len(thesesamples[0])):
        xs= numpy.array([s[kk] for s in thesesamples])
        if paramnames[kk] == 'rd' or paramnames[kk] == 'zh':
            xs= numpy.exp(xs)
        samples[paramnames[kk]]= xs
        scaleDict[paramnames[kk]]= scale[kk]
    if not options.derivedfile is None:
        for ll in range(len(thesesamples[0]),
                        len(thesesamples[0])+len(derivedsamples[0])):
            kk= ll-len(thesesamples[0])
            xs= numpy.array([s[kk] for s in derivedsamples])
            samples[paramnames[ll]]= xs
            scaleDict[paramnames[ll]]= scale[ll]
    #samples['dlnvcdlnr']*= samples['vc']
    xprop= options.subtype.split(',')[0]
    yprop= options.subtype.split(',')[1]
    bovy_plot.bovy_print()
    bovy_plot.scatterplot(samples[xprop]*scaleDict[xprop],
                          samples[yprop]*scaleDict[yprop],
                          'k,',
                          xlabel=labels[xprop],
                          ylabel=labels[yprop],
                          xrange=ranges[xprop],
                          yrange=ranges[yprop],
                          bins=11,
                          contours=True,
                          onedhists=True,
                          cmap='gist_yarg')
    bovy_plot.bovy_end_print(options.plotfile)
    return None                                  

def get_options():
    usage = "usage: %prog [options] <savefile>\n\nsavefile= name of the file that the fits/samples will be saved to"
    parser = OptionParser(usage=usage)
    #Data options
    parser.add_option("-i",dest='surffile',default=None,
                      help="Name of the file that has the surface densities")
    parser.add_option("--dontfitsurf",action="store_true", 
                      dest="dontfitsurf",
                      default=False,
                      help="If set, don't fit the surface-density")
    parser.add_option("--fitterminal",action="store_true", 
                      dest="fitterminal",
                      default=False,
                      help="If set, fit the terminal velocities")
    parser.add_option("--termdsinl",dest='termdsinl',default=0.125,type='float',
                      help="Correlation length for terminal velocity residuals")
    parser.add_option("--termsigma",dest='termsigma',default=7.,type='float',
                      help="sigma for terminal velocity residuals")
    parser.add_option("--surfaskz",action="store_true", 
                      dest="surfaskz",
                      default=False,
                      help="If set, treat the surface-density measurements as measurements of kz")
    #Fit options
    parser.add_option("--init",dest='initfile',default=None,
                      help="Name of the file that has the best-fits")
    #Plot
    parser.add_option("-o",dest='plotfile',default=None,
                      help="Name of the file that has the plot")
    #Sampling options
    parser.add_option("--mcsample",action="store_true", dest="mcsample",
                      default=False,
                      help="If set, sample around the best fit, save in args[1]")
    parser.add_option("--nsamples",dest='nsamples',default=1000,type='int',
                      help="Number of MCMC samples to obtain")
    #seed
    parser.add_option("--seed",dest='seed',default=1,type='int',
                      help="seed for random number generator")
    #Priors
    parser.add_option("--bovy09voprior",action="store_true", 
                      dest="bovy09voprior",
                      default=False,
                      help="If set, apply the Bovy, Rix, & Hogg vo prior (225+/- 15)")
    parser.add_option("--bovy12voprior",action="store_true", 
                      dest="bovy12voprior",
                      default=False,
                      help="If set, apply the Bovy, et al. 2012 prior")
    parser.add_option("--nodlnvcdlnrprior",action="store_true",
                      dest="nodlnvcdlnrprior",
                      default=False,
                      help="If set, do not apply a prior on the logarithmic derivative of the rotation curve")
    parser.add_option("--lanprior",action="store_true", 
                      dest="lanprior",
                      default=False,
                      help="If set, apply priors from Lan's K dwarf analysis")
    parser.add_option("--apogeeprior",action="store_true", 
                      dest="apogeeprior",
                      default=False,
                      help="If set, apply priors on Vc and dlnvcdlnr from my APOGEE analysis")
    #calc
    parser.add_option("--calcderived",action="store_true", 
                      dest="calcderived",
                      default=False,
                      help="If set, calculate derived parameters")
    parser.add_option("-m","--multi",dest='multi',default=None,type='int',
                      help="number of cpus to use")
    parser.add_option("--derivedfile",dest='derivedfile',default=None,
                      help="Name of the file that has the derived parameters")
    #plot
    parser.add_option("--plot",action="store_true", 
                      dest="plot",
                      default=False,
                      help="If set, make a plot")
    parser.add_option("-t","--type",dest='type',default=None,
                      help="Type of thing to do")
    parser.add_option("--subtype",dest='subtype',default=None,
                      help="Sub-type of thing to do")
    #Gaussians
    parser.add_option("-g",dest='g', default=2,type='int',
                      help="Number of Gaussians to fit the samples with")    
    return parser

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    numpy.random.seed(options.seed)
    if options.plot:
        plotStuff(options,args)
    elif options.calcderived:
        calcDerived(options,args)
    else:
        fitSurfwPot(options,args)
