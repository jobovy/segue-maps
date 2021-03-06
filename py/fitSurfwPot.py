#export PYTHONPATH=~/Repos/extreme-deconvolution-ngerrors/py:$PYTHONPATH
import re
import os, os.path
import sys
import math
import numpy
import pickle
from scipy import optimize, integrate, special
from optparse import OptionParser
from galpy import potential
from galpy.util import save_pickles, bovy_plot, multi, bovy_conversion
from matplotlib import rc
import multiprocessing
from matplotlib import pyplot, cm
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
    elif options.apogeeprior and not os.path.exists(apogeefile):
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
    #potoptions.potential= 'dpdiskplhalofixbulgeflatwgasalt'
    potoptions.potential= 'dpdiskplhalofixcutbulgeflatwgasalt'
    potoptions.ro= options.ro
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
            ro= options.ro
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
    ro= options.ro
    if not options.dontfitsurf:
        modelsurfs= numpy.zeros_like(surfs)
        for ii in range(len(surfrs)):
            if options.surfaskz:
                modelsurfs[ii]= -potential.evaluatezforces((surfrs[ii]+ro*_REFR0-_REFR0)/_REFR0/ro,1.1/_REFR0/ro,pot)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro/2./numpy.pi
            else:
                modelsurfs[ii]= 2.*integrate.quad((lambda zz: potential.evaluateDensities(surfrs[ii]/_REFR0,zz,pot)),0.,1.1/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro
        out= 0.5*numpy.sum((surfs-modelsurfs)**2./surferrs**2.)
    else:
        out= 0.
    if options.simulatekz2:
        kz_sim= numpy.array([ 174.18150542,  159.58268322,  113.03887889,  191.62756714,
        189.77283767,  194.099383  ,  188.959876  ,  269.59353188,
         91.74695728,  141.75678914,  124.04418425,  120.69900535,
        194.54341779,  101.95363797,  132.9994155 ,  105.9104183 ,
        118.90328041,   95.50914944,   82.06312126,  111.13069026,
        111.71858284,   99.16022414,   95.231596  ,  107.70878227,
        111.94037148,  129.99798029,  108.31376433,  125.57669991,
        110.47896706,  190.39379472,   87.36444244,  116.0331996 ,
        136.62771727,   80.67013004,  111.22420434,   76.9248815 ,
        145.16041964,  114.41359543,   92.18998455,   73.76903729,
         89.39307008,  149.25123878,  157.64909484])
        kzerr_sim= numpy.array([ 40.07171688,  33.45236284,  26.07853875,  33.59104962,
        42.81852806,  30.73041237,  25.80878778,  33.55744612,
        17.80812298,  23.03100187,  14.29570739,  13.04545928,
        31.97373304,  17.44361627,  15.93027308,   8.67242324,
        10.61320147,  23.36652115,  15.84786188,  10.14177951,
        10.33841369,  15.37377585,  10.86032101,   8.4722236 ,
         9.05734992,  20.92978674,  10.24728671,  12.01645118,
         9.93090843,  20.42427064,  11.15484933,  10.91975199,
        18.15117833,   9.9792092 ,  11.38083172,   9.58490215,
        16.98585868,  16.23936587,   8.42164264,   6.4145673 ,
         9.6335548 ,  20.36969524,  35.01855949])
        modelsurfs2= numpy.zeros_like(surfs)
        for ii in range(len(surfrs)):
            if options.surfaskz:
                modelsurfs2[ii]= -potential.evaluatezforces((surfrs[ii]+ro*_REFR0-_REFR0)/_REFR0/ro,2.1/_REFR0/ro,pot)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro/2./numpy.pi
            else:
                raise NotImplementedError("not using the data as kz not implemented")
        out+= 0.5*numpy.sum((kz_sim-modelsurfs2)**2./kzerr_sim**2.)
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
        if options.surfaskz:
            out+= 0.5*(-potential.evaluatezforces(1.,1.1/_REFR0/ro,pot)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro/2./numpy.pi-67.)**2./36.
        else:
            out+= 0.5*(2.*integrate.quad((lambda zz: potential.evaluateDensities(1.,zz,pot)),0.,1.0/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro-67.)**2./36.
        out+= 0.5*(2.*pot[0].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.*numpy.exp(params[2])*ro*_REFR0*1000.-42.)**2./36.
#        print 2.*integrate.quad((lambda zz: potential.evaluateDensities(1.,zz,pot)),0.,1.0/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro, \
#            2.*pot[0].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.*numpy.exp(params[2])*ro*_REFR0*1000.
    if options.apogeeprior:
        out+= 0.5*(potential.vcirc(pot,1.)*vo*_REFV0-218.)**2./36.
        out+= 0.5*(potential.vcirc(pot,.5)*vo*_REFV0-218.)**2./9.
        out+= 0.5*(potential.vcirc(pot,1.5)*vo*_REFV0-218.)**2./100.
        #out-= _eval_gauss_grid([numpy.log(vo)],[params[4]],*apogeeprior)[0,0]
    if options.localdensprior:
        out+= 0.5*(potential.evaluateDensities(1.,0.,pot)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.-0.102)**2./0.01**2.
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
    #potoptions.potential= 'dpdiskplhalofixbulgeflatwgasalt'
    potoptions.potential= 'dpdiskplhalofixcutbulgeflatwgasalt'
    potoptions.fitdvt= False
    if options.mcsample:
        derived_params= multi.parallel_map((lambda x: calcDerivedSingle(init_params[x],options,potoptions)),
                                           range(len(init_params)),
                                           numcores=numpy.amin([len(init_params),
                                                                multiprocessing.cpu_count(),
                                                                options.multi]))
        for kk in range(len(derived_params[0])):
            xs= numpy.array([s[kk] for s in derived_params])
            print numpy.mean(xs[True-numpy.isnan(xs)]), \
                numpy.std(xs[True-numpy.isnan(xs)])
    else:
        derived_params= calcDerivedSingle(init_params,options,potoptions)
        print derived_params
    save_pickles(args[0],derived_params)
    return None

def calcDerivedSingle(params,options,potoptions):
    try:
        pot= setup_potential(params,potoptions,0,returnrawpot=True)
    except RuntimeError:
        return [numpy.nan for ii in range(9)]
    ro= options.ro
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
    massdisk= rhod*2.*zh*numpy.exp(1./rd*ro)*rd**2./ro**2.*2.*numpy.pi*(ro*_REFR0)**3./10.
    #pl halo
    alpha= pot[1].alpha
    #vcdvc
    vcdvc= pot[0].vcirc(2.2*rd/ro)/potential.vcirc(pot,2.2*rd/ro)
    #mass of the disk
    rhod= pot[3].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.
    massgasdisk= rhod*2.*130./8000.*numpy.exp(1./rd/2.*ro)*rd**2./ro**2.*4.*2.*numpy.pi*(ro*_REFR0)**3./10.*(1.+3./_REFR0)*numpy.exp(-3./_REFR0/2./rd*ro)#cut out central 3 kpc
    #Mass of the halo within 10 kpc
    masshalo= pot[1].mass(10./_REFR0/ro)\
        *bovy_conversion.mass_in_1010msol(_REFV0*vo,_REFR0*ro)
    out= [surfz,surfzdisk,rhodm,rhoo,massdisk,alpha,vcdvc,massdisk+massgasdisk,
          masshalo]
    return out
    
def calcRotcurves(options,args):
    """Calculate rotation curves for this best-fit or samples"""
    if os.path.exists(options.initfile):
        initfile= open(options.initfile,'rb')
        init_params= pickle.load(initfile)
        initfile.close()
    else:
        raise IOError("%s not found" % args[0])
    potoptions= setup_options(None)
    #potoptions.potential= 'dpdiskplhalofixbulgeflatwgasalt'
    potoptions.potential= 'dpdiskplhalofixcutbulgeflatwgasalt'
    potoptions.fitdvt= False
    if options.mcsample:
        rotcurves= multi.parallel_map((lambda x: calcRotcurvesSingle(init_params[x],options,potoptions)),
                                           range(len(init_params)),
                                           numcores=numpy.amin([len(init_params),
                                                                multiprocessing.cpu_count(),
                                                                options.multi]))
    else:
        rotcurves= calcRotcurvesSingle(init_params,options,potoptions)
    save_pickles(args[0],rotcurves)
    return None

def calcRotcurvesSingle(params,options,potoptions):
    pot= setup_potential(params,potoptions,0,returnrawpot=True)
    ro= options.ro
    vo= params[1]
    rs= numpy.linspace(0.0001,2.,1001)
    out= []
    #First full
    out.append(vo*potential.calcRotcurve(pot,rs))
    #Disk
    out.append(vo*potential.calcRotcurve(pot[0],rs))
    #Halo
    out.append(vo*potential.calcRotcurve(pot[1],rs))
    return out
    
def calcDensities(options,args):
    """Calculate (mid-plane) densities for this best-fit or samples"""
    if os.path.exists(options.initfile):
        initfile= open(options.initfile,'rb')
        init_params= pickle.load(initfile)
        initfile.close()
    else:
        raise IOError("%s not found" % args[0])
    potoptions= setup_options(None)
    #potoptions.potential= 'dpdiskplhalofixbulgeflatwgasalt'
    potoptions.potential= 'dpdiskplhalofixcutbulgeflatwgasalt'
    potoptions.fitdvt= False
    if options.mcsample:
        densities= multi.parallel_map((lambda x: calcDensitiesSingle(init_params[x],options,potoptions)),
                                           range(len(init_params)),
                                           numcores=numpy.amin([len(init_params),
                                                                multiprocessing.cpu_count(),
                                                                options.multi]))
    else:
        densities= calcDensitiesSingle(init_params,options,potoptions)
    save_pickles(args[0],densities)
    return None

def calcDensitiesSingle(params,options,potoptions):
    rs= numpy.linspace(0.0001,2.,1001)
    try:
        pot= setup_potential(params,potoptions,0,returnrawpot=True)
    except RuntimeError: #Fix bad halos
        out= []
        out.append(rs+numpy.nan)
        out.append(rs+numpy.nan)
        out.append(rs+numpy.nan)
        return out
    ro= options.ro
    vo= params[1]
    out= []
    #First full
    out.append(potential.evaluateDensities(rs,0.,pot)\
                   *bovy_conversion.dens_in_msolpc3(vo*_REFV0,ro*_REFR0))
    #Disk
    out.append(potential.evaluateDensities(rs,0.,pot[0])\
                   *bovy_conversion.dens_in_msolpc3(vo*_REFV0,ro*_REFR0))
    #Halo
    out.append(potential.evaluateDensities(rs,0.,pot[1])\
                   *bovy_conversion.dens_in_msolpc3(vo*_REFV0,ro*_REFR0))
    return out
    
def plotStuff(options,args):
    if options.type == '2d':
        plot2dStuff(options,args)
    elif options.type == 'bestfitsurf':
        plotBestfitSurf(options,args)
    elif options.type == 'bestfitkz':
        plotBestfitSurf(options,args)
    elif options.type == 'bestfitvterm':
        plotBestfitVterm(options,args)
    elif options.type == 'rotcurves':
        plotRotcurves(options,args)
    elif options.type == 'rotcurvessamples':
        plotRotcurvesSamples(options,args)
    elif options.type == 'densities':
        plotDensities(options,args)
    return None

def plotRotcurvesSamples(options,args):
    """Plot the rotation curves"""
    #Load the samples
    if os.path.exists(args[0]):
        savefile= open(args[0],'rb')
        rotcurves_samples= pickle.load(savefile)
        savefile.close()
    else:
        raise IOError("args[0] must be set to sample rotation curves ...")
    nsamples= len(rotcurves_samples)
    nrs= len(rotcurves_samples[0][0])
    rs= numpy.linspace(0.0001,2.,nrs)
    indx= numpy.random.permutation(nsamples)[0:10]
    #Sort them on the halo V_c(R_0)
    roindx= numpy.argmin(numpy.fabs(rs-1.))
    halovc= [rotcurves_samples[indx[ii]][2][roindx] for ii in range(len(indx))]
    indx= [x for (y,x) in sorted(zip(halovc,indx))]
    overplot= False
    bovy_plot.bovy_print()
    colormap = cm.hot
    colors= numpy.linspace(0.,9.,len(indx))/len(indx)
    for ii in range(len(indx)):
        bovy_plot.bovy_plot(rs*_REFR0,rotcurves_samples[indx[ii]][0]*_REFV0,
                            '-',lw=2.,color=colormap(colors[ii]),
                            xrange=[0.,10],
                            yrange=[0.,280.],
                            xlabel=r'$R\ (\mathrm{kpc})$',
                            ylabel=r'$V_c(R)\ (\mathrm{km\,s}^{-1})$',
                            overplot=overplot)
        overplot= True
        bovy_plot.bovy_plot(rs*_REFR0,rotcurves_samples[indx[ii]][1]*_REFV0,
                            '-',lw=2.,color=colormap(colors[ii]),
                            overplot=True)
        bovy_plot.bovy_plot(rs*_REFR0,rotcurves_samples[indx[ii]][2]*_REFV0,
                            '-',lw=2.,color=colormap(colors[ii]),
                            overplot=True)
    bovy_plot.bovy_text(8.5,175,r'$\mathrm{Disk}$',size=16.)
    bovy_plot.bovy_text(8.5,100.,r'$\mathrm{Halo}$',size=16.)
    if 'apogee' in options.plotfile:
        bovy_plot.bovy_text(r'$+\mathrm{APOGEE}\ V_c(R)$',
                            top_right=True,size=16.)
    bovy_plot.bovy_end_print(options.plotfile)

def plotRotcurves(options,args):
    """Plot the rotation curves"""
    #Load the best-fit rotcurves
    if os.path.exists(options.initfile):
        savefile= open(options.initfile,'rb')
        rotcurves_bf= pickle.load(savefile)
        savefile.close()
    else:
        raise IOError("initfile must be set to best-fit rotation curves ...")
    if True:
        #Now plot the uncertainty
        if os.path.exists(args[0]):
            savefile= open(args[0],'rb')
            rotcurves_samples= pickle.load(savefile)
            savefile.close()
        else:
            raise IOError("args[0] must be set to sample rotation curves ...")
        #Total
        nsamples= len(rotcurves_samples)
        nrs= len(rotcurves_samples[0][0])
        rs= numpy.linspace(0.0001,2.,nrs)
        allrcs= numpy.zeros((len(rs),nsamples))
        for ii in range(nsamples):
            allrcs[:,ii]= rotcurves_samples[ii][0]
        #Determine range and plot
        nsigs= 1
        rcsigs= numpy.zeros((len(rs),2*nsigs))
        for ii in range(nsigs):
            for jj in range(len(rs)):
                thisf= sorted(allrcs[jj,:])
                thiscut= 0.5*special.erfc((ii+1.)/math.sqrt(2.))
                #thiscut= 0.25#0.5*special.erfc((ii+1.)/math.sqrt(2.))
                rcsigs[jj,2*ii]= thisf[int(math.floor(thiscut*nsamples))]
                thiscut= 1.-thiscut
                rcsigs[jj,2*ii+1]= thisf[int(math.floor(thiscut*nsamples))]
        bovy_plot.bovy_print()
        bovy_plot.bovy_plot(rs*_REFR0,numpy.median(allrcs,axis=1)*_REFV0,
                            'k-',lw=2.,
                            xrange=[0.,10],
                            yrange=[0.,280.],
                            xlabel=r'$R\ (\mathrm{kpc})$',
                            ylabel=r'$V_c(R)\ (\mathrm{km\,s}^{-1})$',
                            zorder=10)
        colord, cc= (1.-0.75)/(nsigs+1.), 1
        nsigma= nsigs
        p= pyplot.fill_between(rs*_REFR0,_REFV0*rcsigs[:,0],_REFV0*rcsigs[:,1],
                               color='k',rasterized=True)
        p.set_facecolors("none")
        ax1= pyplot.gca()
        from matplotlib.patches import PathPatch
        for path in p.get_paths():
            p1 = PathPatch(path, fc="none",ec='0.5', hatch="x")
            ax1.add_patch(p1)
            p1.set_zorder(p.get_zorder()-0.1)
        #Disk
        allrcs= numpy.zeros((len(rs),nsamples))
        for ii in range(nsamples):
            allrcs[:,ii]= rotcurves_samples[ii][1]
        bovy_plot.bovy_plot(rs*_REFR0,numpy.median(allrcs,axis=1)*_REFV0,
                            'k-',lw=2.,
                            overplot=True)
        #Determine range and plot
        rcsigs= numpy.zeros((len(rs),2*nsigs))
        for ii in range(nsigs):
            for jj in range(len(rs)):
                thisf= sorted(allrcs[jj,:])
                thiscut= 0.5*special.erfc((ii+1.)/math.sqrt(2.))
                #thiscut= 0.25#0.5*special.erfc((ii+1.)/math.sqrt(2.))
                rcsigs[jj,2*ii]= thisf[int(math.floor(thiscut*nsamples))]
                thiscut= 1.-thiscut
                rcsigs[jj,2*ii+1]= thisf[int(math.floor(thiscut*nsamples))]
        colord, cc= (1.-0.75)/(nsigs+1.), 1
        nsigma= nsigs
        p= pyplot.fill_between(rs*_REFR0,_REFV0*rcsigs[:,0],_REFV0*rcsigs[:,1],
                            color='k',rasterized=True)
        p.set_facecolors("none")
        ax1= pyplot.gca()
        from matplotlib.patches import PathPatch
        for path in p.get_paths():
            p1 = PathPatch(path, fc="none",ec='0.5',
                           hatch="\\")
            ax1.add_patch(p1)
            p1.set_zorder(p.get_zorder()-0.1)
        #Halo
        allrcs= numpy.zeros((len(rs),nsamples))
        for ii in range(nsamples):
            allrcs[:,ii]= rotcurves_samples[ii][2]
        bovy_plot.bovy_plot(rs*_REFR0,numpy.median(allrcs,axis=1)*_REFV0,
                            'k-',lw=2.,
                            overplot=True)
        #Determine range and plot
        rcsigs= numpy.zeros((len(rs),2*nsigs))
        for ii in range(nsigs):
            for jj in range(len(rs)):
                thisf= sorted(allrcs[jj,:])
                thiscut= 0.5*special.erfc((ii+1.)/math.sqrt(2.))
                #thiscut= 0.25#0.5*special.erfc((ii+1.)/math.sqrt(2.))
                rcsigs[jj,2*ii]= thisf[int(math.floor(thiscut*nsamples))]
                thiscut= 1.-thiscut
                rcsigs[jj,2*ii+1]= thisf[int(math.floor(thiscut*nsamples))]
        colord, cc= (1.-0.75)/(nsigs+1.), 1
        nsigma= nsigs
        p= pyplot.fill_between(rs*_REFR0,_REFV0*rcsigs[:,0],_REFV0*rcsigs[:,1],
                               color='k',rasterized=True)
        p.set_facecolors("none")
        ax1= pyplot.gca()
        from matplotlib.patches import PathPatch
        for path in p.get_paths():
            p1 = PathPatch(path, fc="none",ec='0.5', hatch="/")
            ax1.add_patch(p1)
            p1.set_zorder(p.get_zorder()-0.1)
    bovy_plot.bovy_text(8.5,170,r'$\mathrm{Disk}$',size=16.)
    bovy_plot.bovy_text(8.5,80.,r'$\mathrm{Halo}$',size=16.)
    if 'apogee' in options.plotfile:
        bovy_plot.bovy_text(r'$+\mathrm{APOGEE}\ V_c(R)$',
                            top_right=True,size=16.)
    pyplot.savefig(options.plotfile,format=re.split(r'\.',options.plotfile)[-1],
                   dpi=100)
    return None

def plotDensities(options,args):
    """Plot the density curves"""
    #Load the best-fit densities
    if os.path.exists(options.initfile):
        savefile= open(options.initfile,'rb')
        densities_bf= pickle.load(savefile)
        savefile.close()
    else:
        raise IOError("initfile must be set to best-fit density curves ...")
    if True:
        #Now plot the uncertainty
        if os.path.exists(args[0]):
            savefile= open(args[0],'rb')
            densities_samples= pickle.load(savefile)
            savefile.close()
        else:
            raise IOError("args[0] must be set to sample density curves ...")
        #Total
        nsamples= len(densities_samples)
        nrs= len(densities_samples[0][0])
        rs= numpy.linspace(0.0001,2.,nrs)
        #Disk
        allrcs= numpy.zeros((len(rs),nsamples))
        for ii in range(nsamples):
            allrcs[:,ii]= densities_samples[ii][1]
        allrcs= allrcs[:,True-numpy.isnan(allrcs[0,:])]
        rindx= rs > 2./_REFR0
        #Determine range and plot
        nsigs= 1
        rcsigs= numpy.zeros((len(rs),2*nsigs))
        for ii in range(nsigs):
            for jj in range(len(rs)):
                thisf= sorted(allrcs[jj,:])
                thiscut= 0.5*special.erfc((ii+1.)/math.sqrt(2.))
                #thiscut= 0.25#0.5*special.erfc((ii+1.)/math.sqrt(2.))
                rcsigs[jj,2*ii]= thisf[int(math.floor(thiscut*nsamples))]
                thiscut= 1.-thiscut
                rcsigs[jj,2*ii+1]= thisf[int(math.floor(thiscut*nsamples))]
        bovy_plot.bovy_print()
        bovy_plot.bovy_plot(rs[rindx]*_REFR0,numpy.median(allrcs,axis=1)[rindx],
                            'k-',lw=2.,
                            xrange=[0.,10],
                            yrange=[0.001,10.],
                            xlabel=r'$R\ (\mathrm{kpc})$',
                            ylabel=r'$\rho(R,|Z|=0)\ (M_\odot\,\mathrm{pc}^{-3})$',
                            zorder=10,semilogy=True)
        colord, cc= (1.-0.75)/(nsigs+1.), 1
        nsigma= nsigs
        p= pyplot.fill_between(rs[rindx]*_REFR0,rcsigs[rindx,0],rcsigs[rindx,1],
                               color='k',rasterized=True)
        p.set_facecolors("none")
        ax1= pyplot.gca()
        from matplotlib.patches import PathPatch
        for path in p.get_paths():
            p1 = PathPatch(path, fc="none",ec='0.5', hatch="x")
            ax1.add_patch(p1)
            p1.set_zorder(p.get_zorder()-0.1)
        #Halo
        allrcs= numpy.zeros((len(rs),nsamples))
        for ii in range(nsamples):
            allrcs[:,ii]= densities_samples[ii][2]
        allrcs= allrcs[:,True-numpy.isnan(allrcs[0,:])]
        bovy_plot.bovy_plot(rs[rindx]*_REFR0,numpy.median(allrcs,axis=1)[rindx],
                            'k-',lw=2.,
                            overplot=True)
        #Determine range and plot
        rcsigs= numpy.zeros((len(rs),2*nsigs))
        for ii in range(nsigs):
            for jj in range(len(rs)):
                thisf= sorted(allrcs[jj,:])
                thiscut= 0.5*special.erfc((ii+1.)/math.sqrt(2.))
                #thiscut= 0.25#0.5*special.erfc((ii+1.)/math.sqrt(2.))
                rcsigs[jj,2*ii]= thisf[int(math.floor(thiscut*nsamples))]
                thiscut= 1.-thiscut
                rcsigs[jj,2*ii+1]= thisf[int(math.floor(thiscut*nsamples))]
        colord, cc= (1.-0.75)/(nsigs+1.), 1
        nsigma= nsigs
        p= pyplot.fill_between(rs[rindx]*_REFR0,rcsigs[rindx,0],rcsigs[rindx,1],
                            color='k',rasterized=True)
        p.set_facecolors("none")
        ax1= pyplot.gca()
        from matplotlib.patches import PathPatch
        for path in p.get_paths():
            p1 = PathPatch(path, fc="none",ec='0.5',
                           hatch="\\")
            ax1.add_patch(p1)
            p1.set_zorder(p.get_zorder()-0.1)
        pyplot.plot(rs*_REFR0,(rs)**-2.*numpy.median(allrcs,axis=1)[numpy.argmin(numpy.fabs(rs-1.))],
                    'k--',lw=2.)
        pyplot.plot(rs*_REFR0,(rs)**-1.*numpy.median(allrcs,axis=1)[numpy.argmin(numpy.fabs(rs-1.))],
                    'k-.',lw=2.)
        pyplot.plot(rs*_REFR0,(rs)**-0.*numpy.median(allrcs,axis=1)[numpy.argmin(numpy.fabs(rs-1.))],
                    'k:',lw=2.)
    #bovy_plot.bovy_text(8.5,170,r'$\mathrm{Disk}$',size=16.)
    #bovy_plot.bovy_text(8.5,80.,r'$\mathrm{Halo}$',size=16.)
    #if 'apogee' in options.plotfile:
    #    bovy_plot.bovy_text(r'$+\mathrm{APOGEE}\ V_c(R)$',
    #                        top_right=True,size=16.)
    pyplot.savefig(options.plotfile,format=re.split(r'\.',options.plotfile)[-1],
                   dpi=100)
    #Plot with just the models
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(rs*_REFR0,(rs)**-2.*numpy.median(allrcs,axis=1)[numpy.argmin(numpy.fabs(rs-1.))],
                        'k--',lw=2.,
                        xrange=[0.,10],
                        yrange=[0.001,10.],
                        xlabel=r'$R\ (\mathrm{kpc})$',
                        ylabel=r'$\rho(R,|Z|=0)\ (M_\odot\,\mathrm{pc}^{-3})$',
                        zorder=10,semilogy=True)
    pyplot.plot(rs*_REFR0,(rs)**-1.*numpy.median(allrcs,axis=1)[numpy.argmin(numpy.fabs(rs-1.))],
                'k-.',lw=2.)
    pyplot.plot(rs*_REFR0,(rs)**-0.*numpy.median(allrcs,axis=1)[numpy.argmin(numpy.fabs(rs-1.))],
                    'k:',lw=2.)
    pyplot.savefig('../tex-oort/PotDensities_justmodels.png',
                   format=re.split(r'\.',options.plotfile)[-1],
                   dpi=100)
    return None

def plotBestfitSurf(options,args):
    """Plot the best-fit surface density profile"""
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
    if os.path.exists(args[0]):
        initfile= open(args[0],'rb')
        init_params= pickle.load(initfile)
        initfile.close()
    else:
        raise IOError("%s not found" % args[0])
    #Setup potential
    potoptions= setup_options(None)
    #potoptions.potential= 'dpdiskplhalofixbulgeflatwgasalt'
    potoptions.potential= 'dpdiskplhalofixcutbulgeflatwgasalt'
    potoptions.fitdvt= False
    pot= setup_potential(init_params,potoptions,0,returnrawpot=True)
    if options.type == 'bestfitkz':
        surfs= kzs
        surferrs= kzerrs
        ylabel=r'$K_{Z}(R,|Z| = 1.1\,\mathrm{kpc})\ (2\pi G\,M_\odot\,\mathrm{pc}^{-2})$'
    else:
        ylabel=r'$\Sigma(R,|Z| \leq 1.1\,\mathrm{kpc})\ (M_\odot\,\mathrm{pc}^{-2})$'
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(surfrs,surfs,'ko',
                        xlabel=r'$R\ (\mathrm{kpc})$',
                        ylabel=ylabel,
                        xrange=[4.,10.],
                        yrange=[10,1050.],
                        semilogy=True)
    pyplot.errorbar(surfrs,
                    surfs,
                    yerr=surferrs,
                    elinewidth=1.,capsize=3,zorder=0,
                    color='k',linestyle='none')  
    pyplot.errorbar([8.],[69.],yerr=[6.],marker='d',ms=10.,
                    elinewidth=1.,capsize=3,zorder=10,
                    color='0.4',linestyle='none')
    rs= numpy.linspace(4.5,9.,21)/_REFR0
    msurfs= numpy.zeros_like(rs)
    ro= options.ro
    vo= init_params[1]
    for ii in range(len(rs)):
        if options.type == 'bestfitkz':
            msurfs[ii]= -potential.evaluatezforces(rs[ii],1.1/_REFR0,pot)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro/2./numpy.pi
        else:
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
    #potoptions.potential= 'dpdiskplhalofixbulgeflatwgasalt'
    potoptions.potential= 'dpdiskplhalofixcutbulgeflatwgasalt'
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
                      default=True,
                      help="If set, treat the surface-density measurements as measurements of kz")
    #Fit options
    parser.add_option("--init",dest='initfile',default=None,
                      help="Name of the file that has the best-fits")
    parser.add_option("--ro",dest='ro',default=1.,type='float',
                      help="Use a different value for R_0")
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
    parser.add_option("--localdensprior",action="store_true", 
                      dest="localdensprior",
                      default=False,
                      help="If set, apply prior on the local density from Holmberg & Flynn (2000)")
    parser.add_option("--apogeeprior",action="store_true", 
                      dest="apogeeprior",
                      default=False,
                      help="If set, apply priors on Vc and dlnvcdlnr from my APOGEE analysis")
    parser.add_option("--simulatekz2",action="store_true", 
                      dest="simulatekz2",
                      default=False,
                      help="If set, use a simulated data set of KZ at Z=2.1 kpc to see whether that measures alpha_DM")
    #calc
    parser.add_option("--calcderived",action="store_true", 
                      dest="calcderived",
                      default=False,
                      help="If set, calculate derived parameters")
    parser.add_option("-m","--multi",dest='multi',default=None,type='int',
                      help="number of cpus to use")
    parser.add_option("--derivedfile",dest='derivedfile',default=None,
                      help="Name of the file that has the derived parameters")
    parser.add_option("--calcrotcurves",action="store_true", 
                      dest="calcrotcurves",
                      default=False,
                      help="If set, calculate rotation curves")
    parser.add_option("--calcdensities",action="store_true", 
                      dest="calcdensities",
                      default=False,
                      help="If set, calculate (mid-plane) densities")
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
    elif options.calcrotcurves:
        calcRotcurves(options,args)
    elif options.calcdensities:
        calcDensities(options,args)
    else:
        fitSurfwPot(options,args)
