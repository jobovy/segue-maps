#for testing: python pixelFitDF.py --dfeh=0.5 --dafe=0.25 --mcall --mcout --singlefeh=-0.2 --singleafe=0.2 -p 1880 --minndata=1
#
#for running: python pixelFitDF.py --dfeh=0.25 --dafe=0.2 --novoprior --nmcv=100 --justpot --singles --aAmethod=staeckel -m 7 ../fakeDF/fakeDFFit_dfeh0.25_dafe0.2_q0.7_staeckel_justpot_singles.sav -f ../fakeDF/fakeDF_dfeh0.25_dafe0.2_q0.7_staeckel.fits
#
# ADDING NEW POTENTIAL MODEL:
#    edit: - logprior_pot
#          - get_potparams
#          - set_potparams
#          - get_vo
#          - get_npotparams
#          functions for setting up the domain
#
import os, os.path
import shutil
import sys
import copy
import tempfile
import time
import subprocess
import math
import numpy
from scipy import optimize, interpolate, special
from scipy.misc import logsumexp
from scipy.stats import nanmedian
#import cPickle as pickle
import pickle
from optparse import OptionParser
import multi
import multiprocessing
import pdb
from galpy.util import bovy_coords, save_pickles
from galpy import potential
from galpy.actionAngle import actionAngleAdiabatic
from galpy.actionAngle import actionAngleAdiabaticGrid
from galpy.actionAngle import actionAngleStaeckel
from galpy.actionAngle import actionAngleStaeckelGrid
from galpy.df_src.quasiisothermaldf import quasiisothermaldf
import bovy_mcmc
import markovpy as mpy
import emcee
try:
    from emcee.utils import MPIPool
except ImportError:
    pass
#    print "Warning: could not import MPIPool"
import acor
import monoAbundanceMW
from segueSelect import read_gdwarfs, read_kdwarfs, _GDWARFFILE, _KDWARFFILE, \
    segueSelect, _mr_gi, _gi_gr, _ERASESTR, _append_field_recarray, \
    ivezic_dist_gr
from fitDensz import cb, _ZSUN, DistSpline, _ivezic_dist, _NDS, _DblExpDensity, \
    _PowerDensity
from compareDataModel import _predict_rdist_plate, comparernumberPlate
from pixelFitDens import pixelAfeFeh
import AnDistance
_DEBUG= False
_REFR0= 8. #kpc
_REFV0= 220. #km/s
_VRSUN=-11.1 #km/s
_VTSUN= 245. #km/s
_PMSGRA= 30.24 #km/s/kpc
_VZSUN= 7.25 #km/s
_GMBULGE= 17208.0 #kpc (km/s)^2 = 4 x 10^9 Msolar
_ABULGE= 0.6
_GMBULGECUT= 43020.0 #kpc (km/s)^2 = 10^10 Msolar
_RCBULGE= 1.9
_PLBULGE= 1.8
_NGR= 11
_NFEH=11
_DEGTORAD= math.pi/180.
_SRHALO= 150. #km/s
_SPHIHALO= 100. #km/s
_SZHALO= 100. #km/s
_PRECALCVSAMPLES= False
_SURFSUBTRACTEXPON= True
_SURFNRS= 16
_SURFNZS= 16
_BFGS_CONSTRAINED= False
_BFGS= False
_CUSTOMSAMPLING= True
_MULTIWHOLEGRID= False
_MULTIDFGRID= True
_INITWESTIMATES= True
_SMOOTHDISPS= True
_SIMPLEOPTDF= True
_JUSTSIMPLEOPTDF= False
_NEWDFRANGES= False
_NEWESTDFRANGES= True
_NEWRDRANGE= True
_NEWSAVE= True
_INTEGRATEMARGINALIZE= True
_VARYHSZ= True
_RESUBMIT= False
_FIXOUT= True
#GL
_DEFAULTNGL=10
_DEFAULTNGL2=20
_glxdef, _glwdef= \
    numpy.polynomial.legendre.leggauss(_DEFAULTNGL)
_glxdef2, _glwdef2= \
    numpy.polynomial.legendre.leggauss(_DEFAULTNGL2)
_glxdef12, _glwdef12= \
    numpy.polynomial.legendre.leggauss(_DEFAULTNGL/2)
def pixelFitDF(options,args,pool=None):
    print "WARNING: IGNORING NUMPY FLOATING POINT WARNINGS ..."
    numpy.seterr(all='ignore')
    #Check whether the savefile already exists
    if os.path.exists(args[0]) and not options.fixbadvals:
        savefile= open(args[0],'rb')
        params= pickle.load(savefile)
        if options.mcsample:
            npops= pickle.load(savefile)
        savefile.close()
        if options.mcsample:
            print_samples_qa(params,options,npops)
        else:
            print params
        print "Savefile already exists, not re-fitting and overwriting ..."
        return None
    #Read the data
    print "Reading the data ..."
    raw= read_rawdata(options)
    #Setup error mc integration
    if not options.singles and options.singlefeh is None \
            and options.singlebin is None:
        print "Setting up error integration ..."
        raw, errstuff= setup_err_mc(raw,options)
    #Bin the data
    binned= pixelAfeFeh(raw,dfeh=options.dfeh,dafe=options.dafe)
    #Map the bins with ndata > minndata in 1D
    fehs, afes, ndatas= [], [], []
    for ii in range(len(binned.fehedges)-1):
        for jj in range(len(binned.afeedges)-1):
            data= binned(binned.feh(ii),binned.afe(jj))
            if len(data) < options.minndata:
                continue
            #print binned.feh(ii), binned.afe(jj), len(data)
            fehs.append(binned.feh(ii))
            afes.append(binned.afe(jj))
            ndatas.append(len(data))
    nabundancebins= len(fehs)
    fehs= numpy.array(fehs)
    afes= numpy.array(afes)
    ndatas= numpy.array(ndatas)
    if not options.singlebin is None:
        options.singlefeh= fehs[options.singlebin]
        options.singleafe= afes[options.singlebin]
    if not options.singlefeh is None:
        if options.loo:
            pass
        else:
            #Set up single feh
            indx= binned.callIndx(options.singlefeh,options.singleafe)
            if numpy.sum(indx) == 0:
                raise IOError("Bin corresponding to singlefeh and singleafe is empty ...")
            allraw= copy.copy(raw)
            raw= copy.copy(binned.data[indx])
            newerrstuff= []
            print "Setting up error integration ..."
            raw, errstuff= setup_err_mc(raw,options)
            #for ii in range(len(binned.data)):
            #    if indx[ii]: newerrstuff.append(errstuff[ii])
            #errstuff= newerrstuff
            print "Using %i data points ..." % (len(raw))
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
        if numpy.log(monoAbundanceMW.hr(fehs[0],afes[0],
                                            k=(options.sample.lower() == 'k'))/8.) > -0.5 and not options.conditionalr:
            #Don't run, because we cannot model these populations with our model
            return None
    else:
        allraw= raw
    #thissavefile= open('binmapping_k.sav','wb')
    #pickle.dump(fehs,thissavefile)
    #pickle.dump(afes,thissavefile)
    #pickle.dump(ndatas,thissavefile)
    #thissavefile.close()
    #return None
    if options.singles:
        run_abundance_singles(options,args,fehs,afes)
        return None
    if options.andistances:
        data= binned(fehs[0],afes[0])
        distfac= AnDistance.AnDistance(data.dered_g-data.dered_r,
                                       data.feh)
        if options.fixdm is None:
            options.fixdm= numpy.log10(distfac)*5.
        else:
            options.fixdm= options.fixdm+numpy.log10(distfac)*5.
        options.andistances= False
        #Start over
        pixelFitDF(options,args)
        return None
    #Setup everything for the selection function
    print "Setting up stuff for the normalization integral ..."
    normintstuff= setup_normintstuff(options,raw,binned,fehs,afes,allraw)
    if not options.init is None and os.path.exists(options.init):
        #Load initial parameters from file
        print "Loading parameters for file "+options.init
        savefile= open(options.init,'rb')
        params= pickle.load(savefile)
        savefile.close()
    else:
        if options.mcsample:
            print "WARNING: USING DEFAULT INITIALIZATION BECAUSE INITIALIZATION IS NOT SET ..."
        params= initialize(options,fehs,afes)
    if not options.mcsample and not options.grid and not options.gridall:
        if options.justdf:
            params= indiv_optimize_df(params,fehs,afes,binned,options,
                                      normintstuff,errstuff)
        elif options.justpot:
            params= indiv_optimize_potential(params,fehs,afes,binned,options,
                                             normintstuff,errstuff)
        else:
            #Optimize DF w/ fixed potential and potential w/ fixed DF
            for cc in range(options.ninit):
                print "Iteration %i  / %i ..." % (cc+1,options.ninit)
                print "Optimizing individual DFs with fixed potential ..."
                params= indiv_optimize_df(params,fehs,afes,binned,options,
                                          normintstuff,errstuff)
                print "Optimizing potential with individual DFs fixed ..."
                params= indiv_optimize_potential(params,fehs,afes,binned,
                                                 options,
                                                 normintstuff,errstuff)
                save_pickles(args[0],params)
            #Optimize full model
            optout= full_optimize(params,fehs,afes,binned,options,normintstuff,
                                  errstuff)
            params= optout[0]
            mloglikemax= optout[1]
        #Save
        save_pickles(args[0],params,mloglikemax)
    elif options.grid:
        gridOut= gridLike(fehs,afes,binned,options,normintstuff,
                 errstuff)
        #Find best-fit
        indx= numpy.unravel_index(numpy.argmax(gridOut[0]),gridOut[0].shape)
        params= list(gridOut[1][indx])
        if options.potential.lower() == 'dpdiskplhalofixbulgeflatwgasalt':
            params.append(gridOut[2][indx[0]])
            params.append(gridOut[3][indx[1]])
            params.append(gridOut[4][indx[2]])
            params.append(gridOut[5][indx[3]])
            params.append(options.dlnvcdlnr)
        elif options.potential.lower() == 'bt':
            params.append(gridOut[2][indx[0]])
        params= numpy.array(params)
        save_pickles(args[0],params,-gridOut[0][indx],*gridOut)
    elif options.gridall:
        gridOut= gridallLike(fehs,afes,binned,options,normintstuff,
                             errstuff)
        save_pickles(args[0],None) #Just so that there is a record of the fact that the fit is done
    else:
        #Sample
        if options.justdf:
            samples= indiv_sample_df(params,fehs,afes,binned,options,
                                       normintstuff,errstuff)
        elif options.justpot:
            samples= indiv_sample_potential(params,fehs,afes,binned,options,
                                            normintstuff,errstuff)
        else:
            #Setup everything necessary for sampling
            isDomainFinite, domain, step, create_method= setup_domain(options,len(fehs)) 
            if not options.restart is None and os.path.exists(options.restart):
                #Load previous state from file
                print "Loading state from file "+options.restart
                savefile= open(options.restart,'rb')
                params= pickle.load(savefile)
                mloglikemax= pickle.load(savefile)
                samples= pickle.load(savefile)
                dumm= pickle.load(savefile)
                pos= pickle.load(savefile)
                prob= pickle.load(savefile)
                state= pickle.load(savefile)
                savefile.close()
            else:
                pos= None
                prob= None
                state= None
            if _CUSTOMSAMPLING:
                if options.mpi:
                    nwalkers= 40
                else:
                    nwalkers= 14
                samples, lnps, pos,prob,state= custom_markovpy(options,len(fehs),params,
                                               step,
                                               loglike,
                                               (fehs,afes,binned,options,normintstuff,
                                                errstuff),
                                               nsamples=options.nsamples,
                                               nwalkers=nwalkers,
                                               sliceinit=False,
                                               skip=1,
                                               create_method=create_method,
                                               returnLnprob=True,
                                                               pos=pos,
                                                               prob=prob,
                                                               state=state,
                                                               use_emcee=options.mpi,
                                                               pool=pool)
            else:
                samples, lnps= bovy_mcmc.markovpy(params,
                                                  step,
                                                  loglike,
                                                  (fehs,afes,binned,options,normintstuff,
                                                   errstuff),
                                                  nsamples=options.nsamples,
                                                  nwalkers=len(params)+2,
                                                  sliceinit=False,
                                                  skip=1,
                                                  create_method=create_method,
                                                  returnLnprob=True)
                
                pos= None
                prob= None
                state= None
            indxMax= numpy.argmax(lnps)
            params= samples[indxMax]
            mloglikemax= -lnps[indxMax]
        #Save
        save_pickles(args[0],params,mloglikemax,samples,len(fehs),
                     pos,prob,state)
        print_samples_qa(samples,options,len(fehs))
    return None

##DATA
def read_rawdata(options):
    if not options.fixdm is None:
        distfac= 10.**(options.fixdm/5.)
    else:
        distfac= None
    if options.sample.lower() == 'g':
        if not options.fakedata is None:
            raw= read_gdwarfs(options.fakedata,logg=True,ebv=True,sn=options.snmin,nosolar=True,norcut=True,distfac=distfac)
        elif options.select.lower() == 'program':
            raw= read_gdwarfs(_GDWARFFILE,logg=True,ebv=True,sn=options.snmin,nosolar=True,distfac=distfac)
        else:
            raw= read_gdwarfs(logg=True,ebv=True,sn=options.snmin,nosolar=True,
                              distfac=distfac)
    elif options.sample.lower() == 'k':
        if not options.fakedata is None:
            raw= read_kdwarfs(options.fakedata,logg=True,ebv=True,sn=options.snmin,nosolar=True,norcut=True,distfac=distfac)
        elif options.select.lower() == 'program':
            raw= read_kdwarfs(_KDWARFFILE,logg=True,ebv=True,sn=options.snmin,nosolar=True,distfac=distfac)
        else:
            raw= read_kdwarfs(logg=True,ebv=True,sn=options.snmin,nosolar=True,
                              distfac=distfac)
    if not options.bmin is None:
        #Cut on |b|
        raw= raw[(numpy.fabs(raw.b) > options.bmin)]
    if not options.zmax is None:
        raw= raw[(numpy.fabs(raw.zc) <= options.zmax)]
    if not options.fehmin is None:
        raw= raw[(raw.feh >= options.fehmin)]
    if not options.fehmax is None:
        raw= raw[(raw.feh < options.fehmax)]
    if not options.afemin is None:
        raw= raw[(raw.afe >= options.afemin)]
    if not options.afemax is None:
        raw= raw[(raw.afe < options.afemax)]
    if not options.plate is None and not options.loo:
        raw= raw[(raw.plate == options.plate)]
    elif not options.plate is None:
        raw= raw[(raw.plate != options.plate)]
    if options.conditionalr and True:
        #Cut out bright stars on faint plates and vice versa
        plates= numpy.array(list(set(list(raw.plate))),dtype='int') #Only load plates that we use
        sf= segueSelect(plates=plates,type_faint='tanhrcut',
                        sample=options.sample,type_bright='tanhrcut',
                        sn=options.snmin,select=options.select,
                        indiv_brightlims=options.indiv_brightlims)
        indx= []
        nfaintbright, nbrightfaint= 0, 0
        for ii in range(len(raw.feh)):
            if sf.platebright[str(raw[ii].plate)] and raw[ii].dered_r >= 17.8:
                indx.append(False)
                nbrightfaint+= 1
            elif not sf.platebright[str(raw[ii].plate)] and raw[ii].dered_r < 17.8:
                indx.append(False)
                nfaintbright+= 1
            else:
                indx.append(True)
        print "nbrightfaint, nfaintbright", nbrightfaint, nfaintbright
        indx= numpy.array(indx,dtype='bool')
        raw= raw[indx]       
    return raw

##LOG LIKELIHOODS
def mloglike(*args,**kwargs):
    """minus log likelihood"""
    return -loglike(*args,**kwargs)

def loglike(params,fehs,afes,binned,options,normintstuff,errstuff,
            testgood=False):
    """log likelihood"""
    if numpy.any(numpy.isnan(params)):
        return -numpy.finfo(numpy.dtype(numpy.float64)).max
    #Priors
    for ii in range(len(fehs)):
        logoutfracprior= logprior_outfrac(get_outfrac(params,ii,options),
                                          options)
        if logoutfracprior == -numpy.finfo(numpy.dtype(numpy.float64)).max:
            return logoutfracprior
        logdfprior= logprior_dfparams(params,ii,options)
        if logdfprior == -numpy.finfo(numpy.dtype(numpy.float64)).max:
            return logdfprior
    logroprior= logprior_ro(get_ro(params,options),options)
    if logroprior == -numpy.finfo(numpy.dtype(numpy.float64)).max:
        return logroprior
    logpotprior= logprior_pot(params,options,len(fehs))
    if logpotprior == -numpy.finfo(numpy.dtype(numpy.float64)).max:
        return logpotprior
    #Set up potential and actionAngle
    if _DEBUG:
        print params
    try:
        pot= setup_potential(params,options,len(fehs),returnrawpot=testgood)
    except RuntimeError: #if this set of parameters gives a nonsense potential
        return -numpy.finfo(numpy.dtype(numpy.float64)).max
    if testgood: return 0.
    aA= setup_aA(pot,options)
    out= logdf(params,pot,aA,fehs,afes,binned,normintstuff,errstuff)
    returnThis= out+logroprior+logpotprior
    if numpy.isnan(returnThis):
        return -numpy.finfo(numpy.dtype(numpy.float64)).max
    else:
        return returnThis

def logdf(params,pot,aA,fehs,afes,binned,normintstuff,errstuff):
    logl= numpy.zeros(len(fehs))
    #Evaluate individual DFs
    args= (pot,aA,fehs,afes,binned,normintstuff,len(fehs),errstuff,options)
    if False:#not options.multi is None:
        logl= multi.parallel_map((lambda x: indiv_logdf(params,x,
                                                        *args)),
                                 range(len(fehs)),
                                 numcores=numpy.amin([len(fehs),
                                                      multiprocessing.cpu_count(),
                                                      options.multi]))
    else:
        for ii in range(len(fehs)):
            print ii
            logl[ii]= indiv_logdf(params,ii,*args)
    return numpy.sum(logl)

def indiv_logdf(params,indx,pot,aA,fehs,afes,binned,normintstuff,npops,
                errstuff,options):
    """Individual population log likelihood"""
    dfparams= get_dfparams(params,indx,options,log=False)
    vo= get_vo(params,options,npops)
    ro= get_ro(params,options)
    logoutfrac= numpy.log(get_outfrac(params,indx,options))
    loghalodens= numpy.log(ro*outDens(1.,0.,None))
    if options.dfmodel.lower() == 'qdf':
        #Normalize
        hr= dfparams[0]/ro
        sr= dfparams[1]/vo
        sz= dfparams[2]/vo
        hsr= dfparams[3]/ro
        hsz= dfparams[4]/ro
        #Setup
        qdf= quasiisothermaldf(hr,sr,sz,hsr,hsz,pot=pot,aA=aA,cutcounter=True)
    #Calculate surface(R=1.) for relative outlier normalization
    logoutfrac+= numpy.log(qdf.surfacemass_z(1.,ngl=options.ngl))
    #Get data ready
    R,vR,vT,z,vz= prepare_coordinates(params,indx,fehs,afes,binned,errstuff,
                                      options,len(fehs))
    if options.fitdvt or not options.fixdvt is None:
        dvt= get_dvt(params,options)
        vT+= dvt/vo
    ndata= R.shape[0]
    data_lndf= numpy.zeros((ndata,2*options.nmcerr))
    srhalo= _SRHALO/vo/_REFV0
    sphihalo= _SPHIHALO/vo/_REFV0
    szhalo= _SZHALO/vo/_REFV0
    data_lndf= numpy.empty((ndata,2*options.nmcerr))
    if False: #options.marginalizevt:
        data_lndf[:,0:options.nmcerr]= numpy.log(qdf.pvRvz(vR.flatten(),vz.flatten(),
                                                           R.flatten(),z.flatten())).reshape((ndata,options.nmcerr))
        data_lndf[:,options.nmcerr:2*options.nmcerr]= logoutfrac+loghalodens\
            -numpy.log(srhalo)-numpy.log(szhalo)\
            -0.5*(vR**2./srhalo**2.+vz**2./szhalo**2.)\
            -1.*numpy.log(2.*math.pi)
    else:
        data_lndf[:,0:options.nmcerr]= qdf(R.flatten(),vR.flatten(),vT.flatten(),
                                       z.flatten(),vz.flatten(),log=True).reshape((ndata,options.nmcerr))
        data_lndf[:,options.nmcerr:2*options.nmcerr]= logoutfrac+loghalodens\
            -numpy.log(srhalo)-numpy.log(sphihalo)-numpy.log(szhalo)\
            -0.5*(vR**2./srhalo**2.+vz**2./szhalo**2.+vT**2./sphihalo**2.)\
            -1.5*numpy.log(2.*math.pi)
    #Sum data and outlier df, for all MC samples
    data_lndf= mylogsumexp(data_lndf,axis=1)
    if False: #options.marginalizevt:
        data_lndf+= numpy.log(vo)
    #Normalize
    normalization= calc_normint(qdf,indx,normintstuff,params,npops,options,
                                logoutfrac)
    out= numpy.sum(data_lndf)\
        -ndata*(numpy.log(normalization)+numpy.log(options.nmcerr)) #latter so we can compare
    if _DEBUG:
        print fehs[indx], afes[indx], params, numpy.sum(data_lndf)-ndata*numpy.log(options.nmcerr),ndata*numpy.log(normalization), numpy.log(normalization), numpy.sum(data_lndf)-ndata*(numpy.log(normalization)+numpy.log(options.nmcerr))
    else:
        pass
#        print -out, params
    return out

def indiv_optimize_df_loglike(*args,**kwargs):
    return -indiv_optimize_df_mloglike(*args,**kwargs)
def indiv_optimize_df_mloglike(params,fehs,afes,binned,options,pot,aA,
                               indx,_bigparams,normintstuff,errstuff):
    """Minus log likelihood when optimizing the parameters of a single DF"""
    #_bigparams is a hack to propagate the parameters to the overall like
    theseparams= set_dfparams(params,_bigparams,indx,options)
    logoutfracprior= logprior_outfrac(get_outfrac(theseparams,indx,options),
                                      options)
    if logoutfracprior == -numpy.finfo(numpy.dtype(numpy.float64)).max:
        return -logoutfracprior
    logdfprior= logprior_dfparams(theseparams,indx,options)
    if logdfprior == -numpy.finfo(numpy.dtype(numpy.float64)).max:
        return -logdfprior
    ml= -indiv_logdf(theseparams,indx,pot,aA,fehs,afes,binned,normintstuff,
                     len(fehs),errstuff,options)
    print params, ml
    if numpy.isnan(ml):
        return numpy.finfo(numpy.dtype(numpy.float64)).max
    else:
        return ml

def indiv_optimize_pot_loglike(*args,**kwargs):
    return -indiv_optimize_pot_mloglike(*args,**kwargs)
def indiv_optimize_pot_mloglike(params,fehs,afes,binned,options,
                                _bigparams,normintstuff,errstuff):
    """Minus log likelihood when optimizing the parameters of a single DF"""
    #_bigparams is a hack to propagate the parameters to the overall like
    theseparams= set_potparams(params,_bigparams,options,len(fehs))
    ml= mloglike(theseparams,fehs,afes,binned,options,normintstuff,errstuff)
    print params, ml
    if numpy.isnan(ml):
        return numpy.finfo(numpy.dtype(numpy.float64)).max
    else:
        return ml

#Grid-based approach
def gridLike(fehs,afes,binned,options,normintstuff,errstuff):
    #Set up the grid
    if not options.fixro is None:
        ro= options.fixro
    else:
        ro= 1.
    if options.potential.lower() == 'dpdiskplhalofixbulgeflatwgasalt':
        vcs= numpy.array([200./_REFV0,220./_REFV0,240./_REFV0])
        zhs= numpy.array([300./1000./_REFR0,400./1000./_REFR0,500./1000./_REFR0])
        print "BOVY: ADJUST VC AND ZH"
        vcs= numpy.array([options.fixvc/_REFV0])
        zhs= numpy.array([options.fixzh/1000./_REFR0/ro])
        if _NEWRDRANGE:
            rds= numpy.linspace(1.8,3.2,options.nrds)/_REFR0/ro
        else:
            rds= numpy.linspace(1.5,4.5,options.nrds)/_REFR0/ro
        fhs= numpy.linspace(0.,1.,options.nfhs)
        #print "BOVY: ADJUST RDS AND FHS"
        #rds= numpy.array([3.])/_REFR0
        #fhs= numpy.array([0.5])
        rds= numpy.log(rds)
        zhs= numpy.log(zhs)
        if not options.restart is None and os.path.exists(options.restart):
                 #Load previous state from file
            print "Loading state from file "+options.restart
            savefile= open(options.restart,'rb')
            out= pickle.load(savefile)
            dfparams= pickle.load(savefile)
            ii= pickle.load(savefile)
            jj= pickle.load(savefile)
            kk= pickle.load(savefile)
            savefile.close()
        else:
            ii, jj, kk, ll= 0, 0, 0, 0 
            out= numpy.zeros((len(rds),len(vcs),len(zhs),len(fhs)))
            if options.fitdvt:
                ndfparams= 7
            else:
                ndfparams= 6
            dfparams= numpy.zeros((len(rds),len(vcs),len(zhs),len(fhs),ndfparams))
        while ii < len(rds):
            while jj < len(vcs):
                while kk <len(zhs):
                    print "Working on %i,%i,%i" % (ii,jj,kk)
                    if _MULTIWHOLEGRID and not options.multi is None:
                        multOut= multi.parallel_map((lambda x: loglike_optdf([rds[ii],vcs[jj],zhs[kk],fhs[x]],fehs,afes,binned,options,normintstuff,errstuff)),
                                                range(len(fhs)),
                                                numcores=numpy.amin([len(fhs),
                                                                     multiprocessing.cpu_count(),
                                                                     options.multi]))
                        for ll in range(len(fhs)):
                            optout= multOut[ll]
                            out[ii,jj,kk,ll]= optout[0]
                            dfparams[ii,jj,kk,ll,:]= optout[1:len(optout)+1]
                    else:
                        for ll in range(len(fhs)):
                            print "Working on %i,%i,%i,%i" % (ii,jj,kk,ll)
                            optout= loglike_optdf([rds[ii],vcs[jj],zhs[kk],fhs[ll]],fehs,afes,binned,options,normintstuff,errstuff)                   
                            out[ii,jj,kk,ll]= optout[0]
                            dfparams[ii,jj,kk,ll,:]= optout[1:len(optout)+1]
                    kk+= 1
                kk= 0
                jj+= 1
                if not options.restart is None:
                    save_pickles(options.restart,
                                 out,dfparams,ii,jj,kk)
            jj= 0
            ii+= 1          
        return (out,dfparams,rds,vcs,zhs,fhs)
    elif options.potential.lower() == 'bt':
        types= [0,1]
        out= numpy.zeros((len(types)))
        if options.fitdvt:
            ndfparams= 7
        else:
            ndfparams= 6
        dfparams= numpy.zeros((len(types),ndfparams))
        if _MULTIWHOLEGRID and not options.multi is None:
            multOut= multi.parallel_map((lambda x: loglike_optdf([types[x]],fehs,afes,binned,options,normintstuff,errstuff)),
                                                range(len(types)),
                                                numcores=numpy.amin([len(types),
                                                                     multiprocessing.cpu_count(),
                                                                     options.multi]))
            for ii in range(len(types)):
                optout= multOut[ii]
                out[ii]= optout[0]
                dfparams[ii,:]= optout[1:len(optout)+1]
        else:
            for ii in range(len(types)):
                optout= loglike_optdf([types[ii]],fehs,afes,binned,options,normintstuff,errstuff)                   
                out[ii]= optout[0]
                dfparams[ii,:]= optout[1:len(optout)+1]
        return (out,dfparams,types)         

def gridallLike(fehs,afes,binned,options,normintstuff,errstuff):
    #Pre-calculate outlier normalization
    #Set up the grid
    if options.potential.lower() == 'dpdiskplhalofixbulgeflatwgasalt':
        out_params= [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.]
        if options.conditionalr:
            normalization_out= calc_normint_fixedpot_conditionalr(None,0,
                                                                  normintstuff,
                                                                  out_params,
                                                                  1,
                                                                  options,
                                                                  0.,
                                                                  None,None,
                                                                  None,None,None,
                                                                  None,None,None,
                                                                  None,fqdf=0.)
        else:
            normalization_out= calc_normint_fixedpot(None,0,normintstuff,
                                                     out_params,
                                                     1,
                                                     options,
                                                     0.,
                                                     None,None,None,None,None,
                                                     None,None,None,None,fqdf=0.)
        dblexphr= monoAbundanceMW.hr(fehs[0],afes[0],k=(options.sample.lower() == 'k'))
        dblexphz= monoAbundanceMW.hz(fehs[0],afes[0],k=(options.sample.lower() == 'k'))
        if options.conditionalr:
            normalization_dout= calc_normint_fixedpot_conditionalr(None,0,
                                                                   normintstuff,
                                                                   out_params,
                                                                   1,
                                                                   options,
                                                                   0.,
                                                                   None,None,
                                                                   None,None,None,
                                                                   None,None,
                                                                   None,None,
                                                                   fqdf=0.,
                                                                   dblexp=True,
                                                                   dblexphr=dblexphr,
                                                                   dblexphz=dblexphz)
        else:
            normalization_dout= calc_normint_fixedpot(None,0,normintstuff,
                                                      out_params,
                                                      1,
                                                      options,
                                                      0.,
                                                      None,None,None,None,None,
                                                      None,None,None,None,fqdf=0.,
                                                      dblexp=True,
                                                      dblexphr=dblexphr,
                                                      dblexphz=dblexphz)
        if not options.fixro is None:
            ro= options.fixro
        else:
            ro= 1.
        vcs= numpy.array([200./_REFV0,220./_REFV0,240./_REFV0])
        zhs= numpy.array([300./1000./_REFR0,400./1000./_REFR0,500./1000./_REFR0])
        print "BOVY: ADJUST VC AND ZH"
        vcs= numpy.array([options.fixvc/_REFV0])
        zhs= numpy.array([options.fixzh/1000./_REFR0/ro])
        if _NEWRDRANGE:
            rds= numpy.linspace(2.0,3.4,options.nrds)/_REFR0/ro
            #rds= rds[3:]
            #print "BOVY: YOU ARE CHANGING THE RD RANGE FOR TESTING"
        else:
            rds= numpy.linspace(1.5,4.5,options.nrds)/_REFR0/ro
        fhs= numpy.linspace(0.,1.,options.nfhs)
        #rds= rds[3:]
        #fhs= fhs[8:]
        #print "BOVY: YOU ARE CHANGING RD/FH"
        #print "BOVY: ADJUST RDS AND FHS"
        #rds= numpy.array([3.])/_REFR0
        #fhs= numpy.array([0.5])
        rds= numpy.log(rds)
        zhs= numpy.log(zhs)
        if not options.restart is None and os.path.exists(options.restart):
                 #Load previous state from file
            print "Loading state from file "+options.restart
            savefile= open(options.restart,'rb')
            out= pickle.load(savefile)
            ii= pickle.load(savefile)
            jj= pickle.load(savefile)
            kk= pickle.load(savefile)
            try:
                ll= pickle.load(savefile)
            except EOFError:
                ll= 0
            savefile.close()
        else:
            ii, jj, kk, ll= 0, 0, 0, 0 
            if _NEWSAVE:
                out= numpy.zeros((len(rds),len(vcs),len(zhs),len(fhs),
                                  options.nhrs,options.nsrs,options.nszs,
                                  5)) #latter are nuisance params+1
            elif options.fitdvt:
                out= numpy.zeros((len(rds),len(vcs),len(zhs),len(fhs),
                                  options.nhrs,options.nsrs,options.nszs,
                                  options.ndvts,options.npouts,1,1))
            else:
                out= numpy.zeros((len(rds),len(vcs),len(zhs),len(fhs),
                                  options.nhrs,options.nsrs,options.nszs,
                                  options.npouts,1,1))
        if options.fixbadvals:
            #Check for bad values and flag for re-calculation
            logl= copy.copy(out)
            #logl[numpy.isnan(logl)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
            marglogl= numpy.zeros((logl.shape[0],logl.shape[3]))
            for mm in range(marglogl.shape[0]):
                for nn in range(marglogl.shape[1]):
                    marglogl[mm,nn]= logsumexp(logl[mm,0,0,nn,:,:,:,0].flatten())
            #Now find bad values
            badvals= numpy.zeros((logl.shape[0],logl.shape[3]),dtype='bool')+False
            badvals[numpy.isnan(marglogl)]= True
            smooth_marglogl, dev_marglogl= nnsmooth(marglogl)
            badvals[(numpy.fabs(marglogl-smooth_marglogl) > 5.*dev_marglogl)]= True
            #badvals[(marglogl == -numpy.finfo(numpy.dtype(numpy.float64)).max)]= False #These aren't bad
            #badvals[5,2]= True
            print "Found %i bad values" % (numpy.sum(badvals))
            #Reset to go through everything
            if os.path.exists(options.restart+'.fix'):
                 #Load previous state from file
                print "Loading state from file "+options.restart+".fix"
                savefile= open(options.restart+'.fix','rb')
                out= pickle.load(savefile)
                ii= pickle.load(savefile)
                jj= pickle.load(savefile)
                kk= pickle.load(savefile)
                try:
                    ll= pickle.load(savefile)
                except EOFError:
                    ll= 0
                savefile.close()
            else:
                ii, jj, kk, ll= 0, 0, 0, 0 
        global _MULTIWHOLEGRID
        while ii < len(rds):
            while jj < len(vcs):
                while kk <len(zhs):
                    print "Working on %i,%i,%i" % (ii,jj,kk)
                    if options.fixbadvals: _MULTIWHOLEGRID= False
                    if _MULTIWHOLEGRID and not options.multi is None:
                        multOut= multi.parallel_map((lambda x: loglike_gridall([rds[ii],vcs[jj],zhs[kk],fhs[x]],fehs,afes,binned,options,normintstuff,errstuff,normalization_out,normalization_dout)),
                                                range(len(fhs)-1,-1,-1),
                                                numcores=numpy.amin([len(fhs),
                                                                     multiprocessing.cpu_count(),
                                                                     options.multi]))
                        for ll in range(len(fhs)-1,-1,-1):
                            optout= multOut[ll]
                            if _NEWSAVE:
                                out[ii,jj,kk,ll,:,:,:,:]= optout
                            else:
                                out[ii,jj,kk,ll,:,:,:,:,:,:,:]= optout
                    else:
                        while ll <len(fhs):
                            if options.fixbadvals:
                                if not badvals[ii,ll]: 
                                    ll+= 1
                                    continue
                            print "Working on %i,%i,%i,%i" % (ii,jj,kk,ll)
                            start= time.time()
                            optout= loglike_gridall([rds[ii],vcs[jj],zhs[kk],fhs[ll]],fehs,afes,binned,options,normintstuff,errstuff,normalization_out,normalization_dout)                   
                            if _NEWSAVE:
                                out[ii,jj,kk,ll,:,:,:,:]= optout
                            else:
                                out[ii,jj,kk,ll,:,:,:,:,:,:,:]= optout
                            if (time.time()-start) > 10.:
                                if options.fixbadvals:
                                    save_pickles(options.restart+'.fix',
                                                 out,ii,jj,kk,ll)
                                elif not options.restart is None:
                                    save_pickles(options.restart,
                                                 out,ii,jj,kk,ll)
                            ll+= 1
                    ll= 0
                    kk+= 1
                    if options.fixbadvals:
                        save_pickles(options.restart+'.fix',
                                     out,ii,jj,kk,ll)
                    elif not options.restart is None:
                        save_pickles(options.restart,
                                     out,ii,jj,kk,ll)
                kk= 0
                jj+= 1
            jj= 0
            ii+= 1          
        return (out,rds,vcs,zhs,fhs)

def nnsmooth(marglogl):
    """Returns nearest-neighbors smoothed version, plus sigma"""
    #first make a bigger array
    tmarglogl= numpy.zeros((marglogl.shape[0]+4,marglogl.shape[1]+4,9))+numpy.nan
    tmarglogl[2:-2,2:-2,0]= copy.copy(marglogl)
    tmarglogl[:,:,1]= numpy.roll(tmarglogl[:,:,0],1,axis=0)
    tmarglogl[:,:,2]= numpy.roll(tmarglogl[:,:,0],-1,axis=0)
    tmarglogl[:,:,3]= numpy.roll(tmarglogl[:,:,0],1,axis=1)
    tmarglogl[:,:,4]= numpy.roll(tmarglogl[:,:,0],-1,axis=1)
    tmarglogl[:,:,5]= numpy.roll(tmarglogl[:,:,0],2,axis=0)
    tmarglogl[:,:,6]= numpy.roll(tmarglogl[:,:,0],-2,axis=0)
    tmarglogl[:,:,7]= numpy.roll(tmarglogl[:,:,0],2,axis=1)
    tmarglogl[:,:,8]= numpy.roll(tmarglogl[:,:,0],-2,axis=1)
    out= nanmedian(tmarglogl[:,:,0:9],axis=2)
    tmarglogl[:,:,1]-= out
    tmarglogl[:,:,2]-= out
    tmarglogl[:,:,3]-= out
    tmarglogl[:,:,4]-= out
    tmarglogl[:,:,5]-= out
    tmarglogl[:,:,6]-= out
    tmarglogl[:,:,7]-= out
    tmarglogl[:,:,8]-= out
    dev= nanmedian(numpy.fabs(tmarglogl[:,:,1:9]),axis=2)
    return (out[2:-2,2:-2],dev[2:-2,2:-2])
          
def loglike_optdf(params,fehs,afes,binned,options,normintstuff,errstuff):
    """log likelihood, ASSUMES A SINGLE BIN"""
    toptions= copy.copy(options)
    if _MULTIWHOLEGRID:
        toptions.multi= toptions.multi2 #Set multi to the second multi
    if toptions.fitdvt:
        out= numpy.empty(8)
        out[1:8]= numpy.zeros(7)+numpy.nan
    else:
        out= numpy.empty(7)
        out[1:7]= numpy.zeros(6)+numpy.nan
    if toptions.potential.lower() == 'dpdiskplhalofixbulgeflatwgasalt':
        potparams= numpy.array([params[0],params[1],params[2],params[3],toptions.dlnvcdlnr])
    elif toptions.potential.lower() == 'bt':
        potparams= numpy.array([])
        if params[0] == 0:
            toptions.potential = 'bti'
        elif params[0] == 1:
            toptions.potential = 'btii'
    tparams= initialize(toptions,fehs,afes)    
    tparams= set_potparams(potparams,tparams,toptions,len(fehs))
    if numpy.any(numpy.isnan(tparams)):
        out[0]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        return out
    logpotprior= logprior_pot(tparams,toptions,len(fehs))
    if logpotprior == -numpy.finfo(numpy.dtype(numpy.float64)).max:
        out[0]= logpotprior
        return out
    #Set up potential and actionAngle
    if _DEBUG:
        print tparams
    try:
        pot= setup_potential(tparams,toptions,len(fehs))
    except RuntimeError: #if this set of parameters gives a nonsense potential
        out[0]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        return out
    aA= setup_aA(pot,toptions)
    #Set-up the fiducial DF
    dfparams= get_dfparams(tparams,0,toptions,log=False)
    vo= get_vo(tparams,toptions,len(fehs))
    ro= get_ro(tparams,toptions)
    if _SIMPLEOPTDF:
        print "Optimizing simple estimate ..."
        dfparams= get_dfparams(tparams,0,toptions,log=True)[0:5]
        optout= optimize.fmin_powell(chi2_simpleoptdf,dfparams,
                                     args=(numpy.exp(copy.copy(dfparams)),
                                           pot,aA,toptions,ro,vo),
                                     xtol=10.**-3.,
                                     ftol=0.0005,
                                     full_output=True,
                                     maxiter=options.maxiter,
                                     maxfun=10000)
        print "SIMPLEOPT: DFPARAMS=", optout[0], params
        dfparams= list(numpy.exp(optout[0]))
        dfparams.append(0.05) #Add outlier fraction
        dfparams= numpy.array(dfparams)
    if toptions.dfmodel.lower() == 'qdf':
        #Normalize
        hr= dfparams[0]/ro
        sr= dfparams[1]/vo
        sz= dfparams[2]/vo
        hsr= dfparams[3]/ro
        hsz= dfparams[4]/ro
        #Setup
        qdf= quasiisothermaldf(hr,sr,sz,hsr,hsz,pot=pot,aA=aA,cutcounter=True)
    #Pre-calculate all actions
    nrs, nzs= _SURFNRS, _SURFNZS
    thisRmin, thisRmax= 4./_REFR0, 15./_REFR0
    thiszmin, thiszmax= 0., .8
    Rgrid= numpy.linspace(thisRmin,thisRmax,nrs)
    zgrid= numpy.linspace(thiszmin,thiszmax,nzs)
    surfgrid= numpy.empty((nrs,nzs))
    jrs= numpy.empty((nrs,nzs,toptions.ngl**3))
    lzs= numpy.empty((nrs,nzs,toptions.ngl**3))
    jzs= numpy.empty((nrs,nzs,toptions.ngl**3))
    rgs= numpy.empty((nrs,nzs,toptions.ngl**3))
    kappas= numpy.empty((nrs,nzs,toptions.ngl**3))
    nus= numpy.empty((nrs,nzs,toptions.ngl**3))
    Omegas= numpy.empty((nrs,nzs,toptions.ngl**3))
    normsrs= numpy.empty((nrs,nzs))
    normszs= numpy.empty((nrs,nzs))
    if not toptions.multi is None:
        multOut= multi.parallel_map((lambda x: setup_optdf_actions(Rgrid[x],
                                                                   zgrid,nzs,
                                                                   toptions,qdf)),
                                    range(nrs),
                                    numcores=numpy.amin([nrs,
                                                         multiprocessing.cpu_count(),
                                                         toptions.multi]))
        for ii in range(nrs):
            jrs[ii,:,:]= multOut[ii][0,:,:]
            lzs[ii,:,:]= multOut[ii][1,:,:]
            jzs[ii,:,:]= multOut[ii][2,:,:]
            rgs[ii,:,:]= multOut[ii][3,:,:]
            kappas[ii,:,:]= multOut[ii][4,:,:]
            nus[ii,:,:]= multOut[ii][5,:,:]
            Omegas[ii,:,:]= multOut[ii][6,:,:]
    else:
        for ii in range(nrs):
            for jj in range(nzs):
                surfgrid[ii,jj], tjr, tlz, tjz, trg, tkappa, tnu, tOmega= qdf.vmomentdensity(Rgrid[ii],zgrid[jj],
                                                                   0.,0.,0.,
                                                                   gl=True,
                                                                   ngl=toptions.ngl,
                                                                   _return_actions=True,_return_freqs=True)
                jrs[ii,jj,:]= tjr
                lzs[ii,jj,:]= tlz
                jzs[ii,jj,:]= tjz
                rgs[ii,jj,:]= trg
                kappas[ii,jj,:]= tkappa
                nus[ii,jj,:]= tnu
                Omegas[ii,jj,:]= tOmega
    for ii in range(nrs):
        normsrs[ii,:]= qdf._sr*numpy.exp((1.-Rgrid[ii])/qdf._hsr)
        normszs[ii,:]= qdf._sz*numpy.exp((1.-Rgrid[ii])/qdf._hsz)
    #Initial parameters
    dfparams= get_dfparams(tparams,0,toptions,log=True)
    if toptions.fitdvt:
        init_params= [(toptions.fixvc-235.)/_REFV0]
        init_params.extend(list(dfparams))
        bounds= [(-0.15,0.15)]
        bounds.extend([(numpy.amax([-1.9,numpy.log(qdf._hr*ro/2.)]),
                        numpy.amin([2.53,numpy.log(qdf._hr*ro*2.)])),
                       (numpy.amax([-3.1,numpy.log(qdf._sr*vo/2.)]),
                        numpy.amin([-0.4,numpy.log(qdf._sr*vo*2.)])),
                       (numpy.amax([-3.1,numpy.log(qdf._sz*vo/2.)]),
                        numpy.amin([-0.4,numpy.log(qdf._sz*vo*2.)])),
                       (-0.3,2.53),
                       (-0.3,2.53),(0.,.15)])
    else:
        init_params= list(dfparams)
        bounds= [(numpy.amax([-1.9,numpy.log(qdf._hr*ro/2.)]),
                  numpy.amin([2.53,numpy.log(qdf._hr*ro*2.)])),
                 (numpy.amax([-3.1,numpy.log(qdf._sr*vo/2.)]),
                  numpy.amin([-0.4,numpy.log(qdf._sr*vo*2.)])),
                 (numpy.amax([-3.1,numpy.log(qdf._sz*vo/2.)]),
                  numpy.amin([-0.4,numpy.log(qdf._sz*vo*2.)])),
                 (-0.3,2.53),
                 (-0.3,2.53),(0.,.15)]
    init_params= numpy.array(init_params)
    if _SIMPLEOPTDF and _JUSTSIMPLEOPTDF:
        optout= []
        expec_off= (toptions.fixvc-235.)/_REFV0
        if expec_off > 0.15: expec_off= 0.14
        elif expec_off < -0.15: expec_off= -0.14
        optout.append([expec_off])
        optout[0].extend(dfparams)
        optout.append(mloglike_optdf_2optimize(optout[0],
                                               tparams,
                                               pot,aA,fehs,afes,binned,normintstuff,
                                               len(fehs),errstuff,toptions,vo,ro,
                                               jrs,lzs,jzs,normsrs,normszs,
                                               qdf._sr*vo,qdf._sz*vo,qdf._hr*ro,
                                               rgs,kappas,nus,Omegas))
    elif _BFGS_CONSTRAINED:
        optout= optimize.fmin_l_bfgs_b(mloglike_optdf_2optimize,
                                       init_params,
                                       args=(tparams,
                                         pot,aA,fehs,afes,binned,normintstuff,
                                         len(fehs),errstuff,toptions,vo,ro,
                                         jrs,lzs,jzs,normsrs,normszs,
                                         qdf._sr*vo,qdf._sz*vo,qdf._hr*ro,
                                         rgs,kappas,nus,Omegas),
                                       approx_grad=True,
                                       bounds=bounds)
    elif _BFGS:
        if toptions.fitdvt:
            expec_off= (toptions.fixvc-235.)/_REFV0
            if expec_off > 0.15: expec_off= 0.14
            elif expec_off < -0.15: expec_off= -0.14
            init_params= [logit((expec_off+0.15)/0.3)]
        else:
            init_params= []
        for kk in range(len(dfparams)):
            init_params.append(logit((dfparams[kk]-bounds[kk+1][0])/(bounds[kk+1][1]-bounds[kk+1][0])))
        optout= optimize.fmin_bfgs(mloglike_optdf_2optimize,
                                   init_params,
                                   args=(tparams,
                                         pot,aA,fehs,afes,binned,normintstuff,
                                         len(fehs),errstuff,toptions,vo,ro,
                                         jrs,lzs,jzs,normsrs,normszs,
                                         qdf._sr*vo,qdf._sz*vo,qdf._hr*ro,
                                         rgs,kappas,nus,Omegas,bounds),
                                   full_output=True,
                                   maxiter=options.maxiter)
        #Post-process
        for kk in range(len(optout[0])):
            optout[0][kk]= ilogit(optout[0][kk])*(bounds[kk][1]-bounds[kk][0])+bounds[kk][0]        
    else:
        optout= optimize.fmin_powell(mloglike_optdf_2optimize,init_params,
                                     args=(tparams,
                                           pot,aA,fehs,afes,binned,normintstuff,
                                           len(fehs),errstuff,toptions,vo,ro,
                                           jrs,lzs,jzs,normsrs,normszs,
                                           qdf._sr*vo,qdf._sz*vo,qdf._hr*ro,
                                           rgs,kappas,nus,Omegas),
                                     callback=cb,
                                     xtol=10.**-3.,
                                     full_output=True,
                                     maxiter=options.maxiter,
                                     maxfun=1000)
    final_params= optout[0]
    mloglikemax= optout[1]
    print final_params, params, mloglikemax
    out= numpy.empty(len(final_params)+1)
    if numpy.isnan(mloglikemax):
        out[0]= -numpy.finfo(numpy.dtype(numpy.float64)).max
    else:
        out[0]= -mloglikemax
    out[1:len(final_params)+1]= final_params
    return out

def loglike_gridall(params,fehs,afes,binned,options,normintstuff,errstuff,
                    normalization_out,normalization_dout):
    """log likelihood, ASSUMES A SINGLE BIN"""
    toptions= copy.copy(options)
    if _MULTIWHOLEGRID:
        toptions.multi= toptions.multi2 #Set multi to the second multi
    if _NEWSAVE:
        out= numpy.zeros((options.nhrs,options.nsrs,options.nszs,
                          5))-numpy.finfo(numpy.dtype(numpy.float64)).max
    elif toptions.fitdvt:
        out= numpy.zeros((options.nhrs,options.nsrs,options.nszs,
                          options.ndvts,options.npouts,1,1))-numpy.finfo(numpy.dtype(numpy.float64)).max
    else:
        out= numpy.zeros((options.nhrs,options.nsrs,options.nszs,
                          options.npouts,1,1))-numpy.finfo(numpy.dtype(numpy.float64)).max
    if toptions.potential.lower() == 'dpdiskplhalofixbulgeflatwgasalt':
        if False:
            print "BOVY: YOU ARE FIXING THE POTENTIAL FOR DEBUGGING"
            params[0]= numpy.log(2.5/8.)
            params[3]= 0.466666666666
        potparams= numpy.array([params[0],params[1],params[2],params[3],toptions.dlnvcdlnr])
    elif toptions.potential.lower() == 'bt':
        potparams= numpy.array([])
        if params[0] == 0:
            toptions.potential = 'bti'
        elif params[0] == 1:
            toptions.potential = 'btii'
    tparams= initialize(toptions,fehs,afes)    
    tparams= set_potparams(potparams,tparams,toptions,len(fehs))
    if numpy.any(numpy.isnan(tparams)):
        if _NEWSAVE:
            out[:,:,:,:]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        else:
            out[:,:,:,:,:,:,:]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        return out
    #logpotprior= logprior_pot(tparams,toptions,len(fehs))
    #if logpotprior == -numpy.finfo(numpy.dtype(numpy.float64)).max:
    #    if _NEWSAVE:
    #        out[:,:,:,:]= logpotprior
    #    else:
    #        out[:,:,:,:,:,:,:]= logpotprior
    #    return out
    #Set up potential and actionAngle
    if _DEBUG:
        print tparams
    try:
        pot= setup_potential(tparams,toptions,len(fehs))
    except RuntimeError: #if this set of parameters gives a nonsense potential
        if _NEWSAVE:
            out[:,:,:,:]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        else:
            out[:,:,:,:,:,:,:]= -numpy.finfo(numpy.dtype(numpy.float64)).max
        return out
    aA= setup_aA(pot,toptions)
    #Set-up the fiducial DF
    dfparams= get_dfparams(tparams,0,toptions,log=False)
    vo= get_vo(tparams,toptions,len(fehs))
    ro= get_ro(tparams,toptions)
    if toptions.dfmodel.lower() == 'qdf':
        #Normalize
        hr= dfparams[0]/ro
        sr= dfparams[1]/vo
        sz= dfparams[2]/vo
        hsr= dfparams[3]/ro
        hsz= dfparams[4]/ro
        #Setup
        qdf= quasiisothermaldf(hr,sr,sz,hsr,hsz,pot=pot,aA=aA,cutcounter=True)
    #Pre-calculate all actions
    nrs, nzs= _SURFNRS, _SURFNZS
    thisRmin, thisRmax= 4./_REFR0, 15./_REFR0
    thiszmin, thiszmax= 0., .8
    Rgrid= numpy.linspace(thisRmin,thisRmax,nrs)
    zgrid= numpy.linspace(thiszmin,thiszmax,nzs)
    surfgrid= numpy.empty((nrs,nzs))
    jrs= numpy.empty((nrs,nzs,toptions.ngl**3))
    lzs= numpy.empty((nrs,nzs,toptions.ngl**3))
    jzs= numpy.empty((nrs,nzs,toptions.ngl**3))
    rgs= numpy.empty((nrs,nzs,toptions.ngl**3))
    kappas= numpy.empty((nrs,nzs,toptions.ngl**3))
    nus= numpy.empty((nrs,nzs,toptions.ngl**3))
    Omegas= numpy.empty((nrs,nzs,toptions.ngl**3))
    normsrs= numpy.empty((nrs,nzs))
    normszs= numpy.empty((nrs,nzs))
    if not toptions.multi is None:
        multOut= multi.parallel_map((lambda x: setup_optdf_actions(Rgrid[x],
                                                                   zgrid,nzs,
                                                                   toptions,qdf)),
                                    range(nrs),
                                    numcores=numpy.amin([nrs,
                                                         multiprocessing.cpu_count(),
                                                         toptions.multi]))
        for ii in range(nrs):
            jrs[ii,:,:]= multOut[ii][0,:,:]
            lzs[ii,:,:]= multOut[ii][1,:,:]
            jzs[ii,:,:]= multOut[ii][2,:,:]
            rgs[ii,:,:]= multOut[ii][3,:,:]
            kappas[ii,:,:]= multOut[ii][4,:,:]
            nus[ii,:,:]= multOut[ii][5,:,:]
            Omegas[ii,:,:]= multOut[ii][6,:,:]
    else:
        for ii in range(nrs):
            for jj in range(nzs):
                surfgrid[ii,jj], tjr, tlz, tjz, trg, tkappa, tnu, tOmega= qdf.vmomentdensity(Rgrid[ii],zgrid[jj],
                                                                   0.,0.,0.,
                                                                   gl=True,
                                                                   ngl=toptions.ngl,
                                                                   _return_actions=True,_return_freqs=True)
                jrs[ii,jj,:]= tjr
                lzs[ii,jj,:]= tlz
                jzs[ii,jj,:]= tjz
                rgs[ii,jj,:]= trg
                kappas[ii,jj,:]= tkappa
                nus[ii,jj,:]= tnu
                Omegas[ii,jj,:]= tOmega
    for ii in range(nrs):
        normsrs[ii,:]= qdf._sr*numpy.exp((1.-Rgrid[ii])/qdf._hsr)
        normszs[ii,:]= qdf._sz*numpy.exp((1.-Rgrid[ii])/qdf._hsz)
    #Now for the data
    if _NEWESTDFRANGES:
        dvts= [0.] #numpy.linspace(-0.35,0.05,options.ndvts) #could be centered on (0.8Vc-200.)/220.
    elif _NEWDFRANGES:
        dvts= numpy.linspace(-0.35,0.05,options.ndvts) #could be centered on (0.8Vc-200.)/220.
    else:
        dvts= numpy.linspace(-0.1,0.1,toptions.ndvts)  
    vRgl= None
    vRglw= None
    vTgl= None
    vTglw= None
    vzgl= None
    vzglw= None
    if _INTEGRATEMARGINALIZE and options.marginalizevrvt:
        #Get data ready
        R,vR,vT,z,vz, datasr, datasz, datarhorz= prepare_coordinates(tparams,0,fehs,afes,binned,errstuff,
                                          toptions,len(fehs))
        ndata= R.shape[0]
        datajrs= numpy.empty((ndata,toptions.ngl,options.ndvts,toptions.ngl**2))
        datalzs= numpy.empty((ndata,toptions.ngl,options.ndvts,toptions.ngl**2))
        datajzs= numpy.empty((ndata,toptions.ngl,options.ndvts,toptions.ngl**2))
        datargs= numpy.empty((ndata,toptions.ngl,options.ndvts,toptions.ngl**2))
        datakappas= numpy.empty((ndata,toptions.ngl,options.ndvts,toptions.ngl**2))
        datanus= numpy.empty((ndata,toptions.ngl,options.ndvts,toptions.ngl**2))
        dataOmegas= numpy.empty((ndata,toptions.ngl,options.ndvts,toptions.ngl**2))
        datanormsrs= qdf._sr*numpy.exp((1.-R)/qdf._hsr)
        #setup GL
        if toptions.ngl == _DEFAULTNGL:
            glx, glw= _glxdef, _glwdef
            glx12, glw12= _glxdef12, _glwdef12
        elif toptions.ngl == _DEFAULTNGL2:
            glx, glw= _glxdef2, _glwdef2
            glx12, glw12= _glxdef, _glwdef
        else:
            glx, glw= numpy.polynomial.legendre.leggauss(toptions.ngl)
            glx12, glw12= numpy.polynomial.legendre.leggauss(toptions.ngl/2)
        vzgl= (glx12+1.)
        vzgl= list(vzgl)
        vzgl.extend(-(glx12+1.))
        vzgl= numpy.array(vzgl)
        vzglw= glw12
        vzglw= list(vzglw)
        vzglw.extend(glw12)
        vzglw= numpy.array(vzglw)
        vzgl= numpy.tile(vzgl,(ndata,1))
        vzglw= numpy.tile(vzglw,(ndata,1))
        vzgl*= numpy.tile(4.*datasz/2.,(toptions.ngl,1)).T
        tdatasz= numpy.tile(datasz,(toptions.ngl,1)).T
        vzglw*= 1./numpy.sqrt(2.*numpy.pi)/tdatasz*numpy.exp(-vzgl**2./2./tdatasz**2.)
        vzgl+= numpy.tile(vz[:,0],(toptions.ngl,1)).T
        if not toptions.multi is None:
            multOut= multi.parallel_map((lambda x: setup_optdf_data_actions_gl(vzgl[:,x],R,z,toptions,qdf,datanormsrs)),
                                        range(options.ngl),
                                        numcores=numpy.amin([options.ngl,
                                                             multiprocessing.cpu_count(),
                                                             toptions.multi]))
            for ii in range(options.ngl):
                datajrs[:,ii,0,:]= multOut[ii][0,:,:]
                datalzs[:,ii,0,:]= multOut[ii][1,:,:]
                datajzs[:,ii,0,:]= multOut[ii][2,:,:]
                datargs[:,ii,0,:]= multOut[ii][3,:,:]
                datakappas[:,ii,0,:]= multOut[ii][4,:,:]
                datanus[:,ii,0,:]= multOut[ii][5,:,:]
                dataOmegas[:,ii,0,:]= multOut[ii][6,:,:]
        else:
            for x in range(options.ngl):
                multOut= setup_optdf_data_actions_gl(vzgl[:,x],R,z,toptions,qdf,datanormsrs)
                ii= x
                datajrs[:,ii,0,:]= multOut[0,:,:]
                datalzs[:,ii,0,:]= multOut[1,:,:]
                datajzs[:,ii,0,:]= multOut[2,:,:]
                datargs[:,ii,0,:]= multOut[3,:,:]
                datakappas[:,ii,0,:]= multOut[4,:,:]
                datanus[:,ii,0,:]= multOut[5,:,:]
                dataOmegas[:,ii,0,:]= multOut[6,:,:]
    elif _INTEGRATEMARGINALIZE and options.marginalizevt:
        #Get data ready
        R,vR,vT,z,vz, datasr, datasz, datarhorz= prepare_coordinates(tparams,
                                                                     0,fehs,
                                                                     afes,
                                                                     binned,
                                                                     errstuff,
                                                                     toptions,
                                                                     len(fehs))
        ndata= R.shape[0]
        datajrs= numpy.empty((ndata,options.ndvts,toptions.ngl**3))
        datalzs= numpy.empty((ndata,options.ndvts,toptions.ngl**3))
        datajzs= numpy.empty((ndata,options.ndvts,toptions.ngl**3))
        datargs= numpy.empty((ndata,options.ndvts,toptions.ngl**3))
        datakappas= numpy.empty((ndata,options.ndvts,toptions.ngl**3))
        datanus= numpy.empty((ndata,options.ndvts,toptions.ngl**3))
        dataOmegas= numpy.empty((ndata,options.ndvts,toptions.ngl**3))
        datanormsrs= qdf._sr*numpy.exp((1.-R)/qdf._hsr)
        #setup GL
        if toptions.ngl == _DEFAULTNGL:
            glx, glw= _glxdef, _glwdef
            glx12, glw12= _glxdef12, _glwdef12
        elif toptions.ngl == _DEFAULTNGL2:
            glx, glw= _glxdef2, _glwdef2
            glx12, glw12= _glxdef, _glwdef
        else:
            glx, glw= numpy.polynomial.legendre.leggauss(toptions.ngl)
            glx12, glw12= numpy.polynomial.legendre.leggauss(toptions.ngl/2)
        vTgl= 1.5/2.*(glx+1.)
        vzgl= (glx12+1.)
        vzgl= list(vzgl)
        vzgl.extend(-(glx12+1.))
        vzgl= numpy.array(vzgl)
        vzglw= glw12
        vzglw= list(vzglw)
        vzglw.extend(glw12)
        vzglw= numpy.array(vzglw)
        vRgl= (glx12+1.)
        vRgl= list(vRgl)
        vRgl.extend(-(glx12+1.))
        vRgl= numpy.array(vRgl)
        vRglw= glw12
        vRglw= list(vRglw)
        vRglw.extend(glw12)
        vRglw= numpy.array(vRglw)
        #tile
        #First tile
        vTgl= numpy.tile(vTgl,(options.ngl,options.ngl,1)).T
        vRgl= numpy.tile(numpy.reshape(vRgl,(1,options.ngl)).T,
                         (options.ngl,1,options.ngl))
        vzgl= numpy.tile(vzgl,(options.ngl,options.ngl,1))
        vTglw= numpy.tile(glw,(options.ngl,options.ngl,1)).T #also tile weights
        vRglw= numpy.tile(numpy.reshape(vRglw,(1,options.ngl)).T,
                          (options.ngl,1,options.ngl))
        vzglw= numpy.tile(vzglw,(options.ngl,options.ngl,1))
        #Second, data tile
        vTgl= numpy.tile(vTgl,(ndata,1,1,1))
        vTglw= numpy.tile(vTglw,(ndata,1,1,1))
        vRgl= numpy.tile(vRgl,(ndata,1,1,1))
        vRglw= numpy.tile(vRglw,(ndata,1,1,1))
        vzgl= numpy.tile(vzgl,(ndata,1,1,1))
        vzglw= numpy.tile(vzglw,(ndata,1,1,1))
        #Adjust
        vzgl*= numpy.tile(4.*datasz/2.,
                          (toptions.ngl,toptions.ngl,toptions.ngl,1)).T
        vRgl*= numpy.tile(4.*datasr/2.,
                          (toptions.ngl,toptions.ngl,toptions.ngl,1)).T
        datacovrz= datarhorz*datasr*datasz                   
        vRvzgl= numpy.empty((ndata,toptions.ngl**3,2))
        vRvzgl[:,:,0]= numpy.reshape(vRgl,(ndata,toptions.ngl**3))
        vRvzgl[:,:,1]= numpy.reshape(vzgl,(ndata,toptions.ngl**3))
        for ii in range(ndata):
            cov= numpy.array([[datasr[ii]**2.,datacovrz[ii]],
                              [datacovrz[ii],datasz[ii]**2.]])
            covinv= numpy.linalg.inv(cov)
            covdet= numpy.linalg.det(cov)
            vzglw[ii,:,:,:]*= numpy.reshape(1./2.*numpy.pi/numpy.sqrt(covdet)*numpy.exp(-0.5*numpy.sum(vRvzgl[ii,:,:]*numpy.dot(vRvzgl[ii,:,:],covinv),axis=1)),(toptions.ngl,toptions.ngl,toptions.ngl))
        vRgl+= numpy.tile(vR[:,0],(toptions.ngl,toptions.ngl,toptions.ngl,1)).T
        vzgl+= numpy.tile(vz[:,0],(toptions.ngl,toptions.ngl,toptions.ngl,1)).T
        #Now flatten
        vRgl= numpy.reshape(vRgl,(ndata,toptions.ngl**3))
        vTgl= numpy.reshape(vTgl,(ndata,toptions.ngl**3))
        vzgl= numpy.reshape(vzgl,(ndata,toptions.ngl**3))
        vRglw= numpy.reshape(vRglw,(ndata,toptions.ngl**3))
        vTglw= numpy.reshape(vTglw,(ndata,toptions.ngl**3))
        vzglw= numpy.reshape(vzglw,(ndata,toptions.ngl**3))
        if not toptions.multi is None:
            multOut= multi.parallel_map((lambda x: setup_optdf_data_actions_margvr_gl(vRgl[x,:],vTgl[x,:],vzgl[x,:],R[x,0],z[x,0],toptions,qdf)),
                                        range(ndata),
                                        numcores=numpy.amin([ndata,
                                                             multiprocessing.cpu_count(),
                                                             toptions.multi]))
            for ii in range(ndata):
                datajrs[ii,0,:]= multOut[ii][0,:]
                datalzs[ii,0,:]= multOut[ii][1,:]
                datajzs[ii,0,:]= multOut[ii][2,:]
                datargs[ii,0,:]= multOut[ii][3,:]
                datakappas[ii,0,:]= multOut[ii][4,:]
                datanus[ii,0,:]= multOut[ii][5,:]
                dataOmegas[ii,0,:]= multOut[ii][6,:]
        else:
            for x in range(options.ngl):
                multOut= setup_optdf_data_actions_gl(vRgl[x,:],vTgl[x,:],
                                                     vzgl[x,:],R[x,0],z[x,0],toptions,qdf)
                ii= x
                datajrs[ii,0,:]= multOut[0,:]
                datalzs[ii,0,:]= multOut[1,:]
                datajzs[ii,0,:]= multOut[2,:]
                datargs[ii,0,:]= multOut[3,:]
                datakappas[ii,0,:]= multOut[4,:]
                datanus[ii,0,:]= multOut[5,:]
                dataOmegas[ii,0,:]= multOut[6,:]
    elif False:
        #Get data ready
        R,vR,vT,z,vz= prepare_coordinates(tparams,0,fehs,afes,binned,errstuff,
                                          toptions,len(fehs))
        ndata= R.shape[0]
        datajrs= numpy.empty((ndata,options.nmcerr,options.ndvts,toptions.ngl**2))
        datalzs= numpy.empty((ndata,options.nmcerr,options.ndvts,toptions.ngl**2))
        datajzs= numpy.empty((ndata,options.nmcerr,options.ndvts,toptions.ngl**2))
        datargs= numpy.empty((ndata,options.nmcerr,options.ndvts,toptions.ngl**2))
        datakappas= numpy.empty((ndata,options.nmcerr,options.ndvts,toptions.ngl**2))
        datanus= numpy.empty((ndata,options.nmcerr,options.ndvts,toptions.ngl**2))
        dataOmegas= numpy.empty((ndata,options.nmcerr,options.ndvts,toptions.ngl**2))
        datanormsrs= qdf._sr*numpy.exp((1.-R)/qdf._hsr)
        if not toptions.multi is None:
            multOut= multi.parallel_map((lambda x: setup_optdf_data_actions_alt(R[:,x],z[:,x],vz[:,x],toptions,qdf,datanormsrs[:,x])),
                                        range(options.nmcerr),
                                        numcores=numpy.amin([options.nmcerr,
                                                             multiprocessing.cpu_count(),
                                                             toptions.multi]))
            for ii in range(options.nmcerr):
                datajrs[:,ii,0,:]= multOut[ii][0,:,:]
                datalzs[:,ii,0,:]= multOut[ii][1,:,:]
                datajzs[:,ii,0,:]= multOut[ii][2,:,:]
                datargs[:,ii,0,:]= multOut[ii][3,:,:]
                datakappas[:,ii,0,:]= multOut[ii][4,:,:]
                datanus[:,ii,0,:]= multOut[ii][5,:,:]
                dataOmegas[:,ii,0,:]= multOut[ii][6,:,:]
        else:
            for x in range(options.nmcerr):
                multOut= setup_optdf_data_actions_alt(R[:,x],z[:,x],vz[:,x],toptions,qdf,datanormsrs[:,x])
                ii= x
                datajrs[:,ii,0,:]= multOut[0,:,:]
                datalzs[:,ii,0,:]= multOut[1,:,:]
                datajzs[:,ii,0,:]= multOut[2,:,:]
                datargs[:,ii,0,:]= multOut[3,:,:]
                datakappas[:,ii,0,:]= multOut[4,:,:]
                datanus[:,ii,0,:]= multOut[5,:,:]
                dataOmegas[:,ii,0,:]= multOut[6,:,:]
    elif False:
        #Get data ready
        R,vR,vT,z,vz= prepare_coordinates(tparams,0,fehs,afes,binned,errstuff,
                                          toptions,len(fehs))
        ndata= R.shape[0]
        datajrs= numpy.empty((ndata,options.ndvts,options.nmcerr))
        datalzs= numpy.empty((ndata,options.ndvts,options.nmcerr))
        datajzs= numpy.empty((ndata,options.ndvts,options.nmcerr))
        datargs= numpy.empty((ndata,options.ndvts,options.nmcerr))
        datakappas= numpy.empty((ndata,options.ndvts,options.nmcerr))
        datanus= numpy.empty((ndata,options.ndvts,options.nmcerr))
        dataOmegas= numpy.empty((ndata,options.ndvts,options.nmcerr))
        datanormsrs= qdf._sr*numpy.exp((1.-R)/qdf._hsr)
        if not toptions.multi is None:
            multOut= multi.parallel_map((lambda x: setup_optdf_data_actions(R,
                                                                            vR,
                                                                            vT,
                                                                            z,
                                                                            vz,
                                                                            dvts[x],
                                                                            toptions,
                                                                            qdf,vo)),
                                        range(len(dvts)),
                                        numcores=numpy.amin([len(dvts),
                                                             multiprocessing.cpu_count(),
                                                             toptions.multi]))
            for ii in range(options.ndvts):
                datajrs[:,ii,:]= multOut[ii][0,:]
                datalzs[:,ii,:]= multOut[ii][1,:]
                datajzs[:,ii,:]= multOut[ii][2,:]
                datargs[:,ii,:]= multOut[ii][3,:]
                datakappas[:,ii,:]= multOut[ii][4,:]
                datanus[:,ii,:]= multOut[ii][5,:]
                dataOmegas[:,ii,:]= multOut[ii][6,:]
        else:
            for ii in range(options.ndvts):
                multOut= setup_optdf_data_actions(R,
                                                  vR,
                                                  vT,
                                                  z,
                                                  vz,
                                                  dvts[ii],
                                                  toptions,
                                                  qdf,vo)
                datajrs[:,ii,:]= multOut[0,:]
                datalzs[:,ii,:]= multOut[1,:]
                datajzs[:,ii,:]= multOut[2,:]
                datargs[:,ii,:]= multOut[3,:]
                datakappas[:,ii,:]= multOut[4,:]
                datanus[:,ii,:]= multOut[5,:]
                dataOmegas[:,ii,:]= multOut[6,:]
    #Go through the grid
    #IF YOU EDIT THIS, ALSO EDIT IT ABOVE
    if toptions.physicaldfparams:
        lnhr, lnsr, lnsz, rehr, resr, resz= approxFitResult(fehs[0],afes[0],
                                                            relerr=True,
                                                            sample=options.sample)
        #if rehr > 0.3: rehr= 0.3 #regularize
        if True: rehr= 0.3 #regularize
        if resr > 0.3: resr= 0.3
        if resz > 0.3: resz= 0.3
        hrs= numpy.linspace(lnhr-1.5*rehr,lnhr+1.5*rehr,options.nhrs)
        srs= numpy.linspace(lnsr-0.66*resz,lnsz+0.66*resz,options.nsrs)#USE ESZ
        szs= numpy.linspace(lnsz-0.66*resz,lnsz+0.66*resz,options.nszs)
    elif _NEWESTDFRANGES:
        hrs, srs, szs=  setup_dfgrid(fehs,afes,options)
    elif _NEWDFRANGES:
        lnhr, lnsr, lnsz, rehr, resr, resz= approxFitResult(fehs[0],afes[0],
                                                            relerr=True,
                                                            sample=options.sample)
        #if rehr > 0.3: rehr= 0.3 #regularize
        if True: rehr= 0.3 #regularize
        #if resr > 0.3: resr= 0.3
        #if resz > 0.3: resz= 0.3
        if True: resr= 0.3
        if True: resz= 0.3
        hrs= numpy.linspace(lnhr-6.*rehr,lnhr+6.*rehr,options.nhrs)
        srs= numpy.linspace(lnsr-0.66*resz,lnsr+0.66*resz,options.nsrs)#USE ESZ
        szs= numpy.linspace(lnsz-0.66*resz,lnsz+0.66*resz,options.nszs)
    else:
        hrs= numpy.log(numpy.linspace(1.5,5.,options.nhrs)/_REFR0)
        srs= numpy.log(numpy.linspace(25.,70.,options.nsrs)/_REFV0)
        szs= numpy.log(numpy.linspace(15.,60.,options.nszs)/_REFV0)
    #start= time.time()
    for ii in range(options.nszs):
        #print "Working on DF %i, dt= %f" % (ii,time.time()-start)
        #start= time.time()
        if _MULTIDFGRID and True:#options.multi > options.nhrs:
            multOut= multi.parallel_map((lambda x: mmloglike_gridall(tparams,
                                                            hrs[x % options.nhrs],srs[x / options.nhrs],szs[ii],
                                                            pot,aA,fehs,afes,binned,normintstuff,
                                                            len(fehs),errstuff,toptions,vo,ro,
                                                            jrs,lzs,jzs,normsrs,normszs,
                                                            qdf._sr*vo,qdf._sz*vo,qdf._hr*ro,
                                                            rgs,kappas,nus,Omegas,
                                                                        normalization_out,normalization_dout,
                                                                     datajrs,datalzs,datajzs,
                                                                     datargs,datakappas,datanus,dataOmegas,
                                                                     datanormsrs,vzgl,vzglw,vRgl,vRglw,vTgl,vTglw)),
                                            range(options.nhrs*options.nsrs),
                                            numcores=numpy.amin([options.nhrs*options.nsrs,
                                                                 multiprocessing.cpu_count(),
                                                                 options.multi]))
            for jj in range(options.nsrs*options.nhrs):
                if _NEWSAVE:
                    out[jj % options.nhrs,jj/options.nhrs,ii,:]= multOut[jj]
                else:
                    out[jj % options.nhrs,jj/options.nhrs,ii,:,:,:,:]= multOut[jj]
        else:
            for jj in range(options.nsrs):
                if _MULTIDFGRID:
                    multOut= multi.parallel_map((lambda x: mmloglike_gridall(tparams,
                                                            hrs[x],srs[jj],szs[ii],
                                                            pot,aA,fehs,afes,binned,normintstuff,
                                                            len(fehs),errstuff,toptions,vo,ro,
                                                            jrs,lzs,jzs,normsrs,normszs,
                                                            qdf._sr*vo,qdf._sz*vo,qdf._hr*ro,
                                                            rgs,kappas,nus,Omegas,
                                                                        normalization_out,normalization_dout,
                                                                             datajrs,datalzs,datajzs,
                                                                             datargs,datakappas,datanus,dataOmegas,
                                                                             datanormsrs,vzgl,vzglw,vRgl,vRglw,vTgl,vTglw)),
                                                range(options.nhrs),
                                            numcores=numpy.amin([options.nhrs,
                                                                 multiprocessing.cpu_count(),
                                                                 options.multi]))
                    for kk in range(options.nhrs):
                        if _NEWSAVE:
                            out[kk,jj,ii,:]= multOut[kk]
                        else:
                            out[kk,jj,ii,:,:,:,:]= multOut[kk]
                else:
                    for kk in range(options.nhrs):
                        if _NEWSAVE:
                            out[kk,jj,ii,:]= mmloglike_gridall(tparams,
                                                           hrs[kk],srs[jj],szs[ii],
                                                           pot,aA,fehs,afes,binned,normintstuff,
                                                           len(fehs),errstuff,toptions,vo,ro,
                                                           jrs,lzs,jzs,normsrs,normszs,
                                                           qdf._sr*vo,qdf._sz*vo,qdf._hr*ro,
                                                           rgs,kappas,nus,Omegas,
                                                           normalization_out,normalization_dout,
                                                           datajrs,datalzs,datajzs,
                                                           datargs,datakappas,datanus,dataOmegas,
                                                           datanormsrs,vzgl,vzglw,vRgl,vRglw,vTgl,vTglw)
                        else:
                            out[kk,jj,ii,:,:,:,:]= mmloglike_gridall(tparams,
                                                                 hrs[kk],srs[jj],szs[ii],
                                                                 pot,aA,fehs,afes,binned,normintstuff,
                                                                 len(fehs),errstuff,toptions,vo,ro,
                                                                 jrs,lzs,jzs,normsrs,normszs,
                                                                 qdf._sr*vo,qdf._sz*vo,qdf._hr*ro,
                                                                 rgs,kappas,nus,Omegas,
                                                                 normalization_out,normalization_dout,
                                                                 datajrs,datalzs,datajzs,
                                                                 datargs,datakappas,datanus,dataOmegas,
                                                                 datanormsrs,vzgl,vzglw,vRgl,vRglw,vTgl,vTglw)
    return out

def setup_dfgrid(fehs,afes,options):
    lnhr, lnsr, lnsz, rehr, resr, resz= approxFitResult(fehs[0],afes[0],
                                                        relerr=True,
                                                        sample=options.sample)
    resz= 0.3
    hrs= numpy.linspace(-1.85714286,0.9,options.nhrs)
    if _VARYHSZ:
        srs= numpy.linspace(numpy.log(0.5),numpy.log(2.),options.nsrs)#hsz now
    else:
        srs= numpy.linspace(lnsr-0.6*resz,lnsr+0.6*resz,options.nsrs)#USE ESZ
    #Adjustments to hsz
    if _VARYHSZ and options.sample.lower() == 'g':
        abindx= abToIndx(fehs[0],afes[0],sample=options.sample.lower())
        if abindx == 47 or abindx == 52:
            srs= numpy.linspace(numpy.log(2.),numpy.log(8.),options.nsrs)#hsz now
    szs= numpy.linspace(lnsz-0.6*resz,lnsz+0.6*resz,options.nszs)
    #Adustements based on inspection
    if options.sample.lower() == 'g':
        abindx= abToIndx(fehs[0],afes[0],sample=options.sample.lower())
        if abindx < 2:
            szs-= 0.3
        elif abindx < 4:
            szs-= 0.2
        elif abindx < 8:
            szs-= 0.15
        elif abindx < 10:
            szs-= 0.10
        elif abindx < 13:
            szs-= 0.125
        elif abindx > 15 and abindx < 19:
            szs-= 0.2
        elif abindx > 18 and abindx < 22:
            szs+= 0.2
        elif abindx == 25:
            szs-= 0.1
        elif abindx == 50:
            szs-= 0.75
        elif abindx == 57 or abindx == 58:
            szs-= 0.2
        elif abindx == 60:
            szs= numpy.linspace(lnsz-0.9*resz,lnsz+0.9*resz,options.nszs)
        elif abindx == 61:
            szs= numpy.linspace(lnsz-1.1*resz,lnsz+1.1*resz,options.nszs)
    #Further adjustements based on inspection of PDFs
    if options.andistances:
        if options.sample.lower() == 'g':
            abindx= abToIndx(fehs[0],afes[0],sample=options.sample.lower())
            if abindx == 0 or abindx == 1 or abindx == 2:
                szs-= 0.15
            elif abindx == 4 or abindx == 6:
                szs-= 0.1
            elif abindx == 7:
                szs-= 0.15
            elif abindx == 9 or abindx == 10:
                szs-= 0.08
            elif abindx == 11:
                szs-= 0.15
            elif abindx == 12:
                szs-= 0.1
            elif abindx == 14:
                szs-= 0.2
            elif abindx == 32:
                szs-= 0.1
            elif abindx == 39:
                szs-= 0.15
            elif abindx == 44:
                szs+= 0.15
            elif abindx == 50:
                szs+= 0.30
            elif abindx == 57:
                szs+= 0.8
            elif abindx == 58:
                szs+= 0.15
            elif abindx == 59:
                szs+= 0.15
    return (hrs,srs,szs)

def abToIndx(feh,afe,sample='g'):
    mapfehs= monoAbundanceMW.fehs(k=(sample.lower() == 'k'))
    mapafes= monoAbundanceMW.afes(k=(sample.lower() == 'k'))
    abindx= numpy.argmin((feh-mapfehs)**2./0.01 \
                             +(afe-mapafes)**2./0.0025)
    return abindx

def chi2_simpleoptdf(dfparams,goal_params,pot,aA,options,ro,vo):
    #Setup DF
    if options.dfmodel.lower() == 'qdf':
        #Normalize
        hr= numpy.exp(dfparams[0])/ro
        sr= numpy.exp(dfparams[1])/vo
        sz= numpy.exp(dfparams[2])/vo
        hsr= numpy.exp(dfparams[3])/ro
        hsz= numpy.exp(dfparams[4])/ro
        if options.dfmodel.lower() == 'qdf':
            if sr > 2.*goal_params[1] or sr < goal_params[1]/vo/2. \
                    or sz > 2.*goal_params[2]/vo or sz < goal_params[2]/vo/2. \
                    or hr > 2.*goal_params[0]/ro or hr < goal_params[0]/ro/2.:
                #Don't allow parameters too different from the initial parameters
                return numpy.finfo(numpy.dtype(numpy.float64)).max
        #Setup
        qdf= quasiisothermaldf(hr,sr,sz,hsr,hsz,pot=pot,aA=aA,cutcounter=True)
    #print numpy.exp(dfparams), goal_params
    #Estimate scale lengths and dispersions
    #this_hr= qdf.estimate_hr(1.,z=None,dR=0.33/ro,gl=True)
    this_hr= qdf.estimate_hr(1.,z=0.8/8.,dR=0.33/ro,gl=True)
    this_hsr= qdf.estimate_hsr(1.,z=0.8/8./ro,dR=0.33/ro,gl=True)
    this_hsz= qdf.estimate_hsz(1.,z=0.8/8./ro,dR=0.33/ro,gl=True)
    this_sr= numpy.sqrt(qdf.sigmaR2(1.,.8/8./ro,gl=True))
    this_sz= numpy.sqrt(qdf.sigmaz2(1.,.8/8./ro,gl=True))
    return 1.+(this_hr-goal_params[0]/ro)**2./goal_params[0]**2.*ro**2.\
        +(this_hsr-goal_params[3]/ro)**2./goal_params[3]**2.*ro**2.\
        +(this_hsz-goal_params[4]/ro)**2./goal_params[4]**2.*ro**2.\
        +(this_sr-goal_params[1]/vo)**2./goal_params[1]**2.*vo**2.\
        +(this_sz-goal_params[2]/vo)**2./goal_params[2]**2.*vo**2.

def logit(x):
    return numpy.log(x/(1.-x))

def ilogit(x):
    return numpy.exp(x)/(1.+numpy.exp(x))

def setup_optdf_data_actions_margvr_gl(vR,vT,vz,R,z,toptions,qdf):
    js= numpy.zeros((7,toptions.ngl**3))
    dumm, tjr, tlz, tjz, trg, tkappa, tnu, tOmega= qdf(R+numpy.zeros(toptions.ngl**3),
                                                       vR,
                                                       vT,
                                                       z+numpy.zeros(toptions.ngl**3),
                                                       vz,
                                                       _return_actions=True,
                                                       _return_freqs=True)
    js[0,:]= tjr
    js[1,:]= tlz
    js[2,:]= tjz
    js[3,:]= trg
    js[4,:]= tkappa
    js[5,:]= tnu
    js[6,:]= tOmega
    return js

def setup_optdf_data_actions_gl(vz,R,z,toptions,qdf,datanormsrs):
    js= numpy.zeros((7,R.shape[0],options.ngl*options.ngl))
    dumm, tjr, tlz, tjz, trg, tkappa, tnu, tOmega= qdf.pvz(vz,
                                                           R.flatten(),
                                                           z.flatten(),
                                                           gl=True,
                                                           ngl=options.ngl,
                                                           _return_actions=True,
                                                           _return_freqs=True,
                                                           _sigmaR1=datanormsrs.flatten())
    js[0,:,:]= numpy.reshape(tjr,(R.shape[0],options.ngl*options.ngl))
    js[1,:,:]= numpy.reshape(tlz,(R.shape[0],options.ngl*options.ngl))
    js[2,:,:]= numpy.reshape(tjz,(R.shape[0],options.ngl*options.ngl))
    js[3,:,:]= numpy.reshape(trg,(R.shape[0],options.ngl*options.ngl))
    js[4,:,:]= numpy.reshape(tkappa,(R.shape[0],options.ngl*options.ngl))
    js[5,:,:]= numpy.reshape(tnu,(R.shape[0],options.ngl*options.ngl))
    js[6,:,:]= numpy.reshape(tOmega,(R.shape[0],options.ngl*options.ngl))
    return js

def setup_optdf_data_actions_alt(R,z,vz,toptions,qdf,datanormsrs):
    js= numpy.zeros((7,R.shape[0],options.ngl*options.ngl))
    dumm, tjr, tlz, tjz, trg, tkappa, tnu, tOmega= qdf.pvz(vz.flatten(),
                                                           R.flatten(),
                                                           z.flatten(),
                                                           gl=True,
                                                           ngl=options.ngl,
                                                           _return_actions=True,
                                                           _return_freqs=True,
                                                           _sigmaR1=datanormsrs.flatten())
    js[0,:,:]= numpy.reshape(tjr,(R.shape[0],options.ngl*options.ngl))
    js[1,:,:]= numpy.reshape(tlz,(R.shape[0],options.ngl*options.ngl))
    js[2,:,:]= numpy.reshape(tjz,(R.shape[0],options.ngl*options.ngl))
    js[3,:,:]= numpy.reshape(trg,(R.shape[0],options.ngl*options.ngl))
    js[4,:,:]= numpy.reshape(tkappa,(R.shape[0],options.ngl*options.ngl))
    js[5,:,:]= numpy.reshape(tnu,(R.shape[0],options.ngl*options.ngl))
    js[6,:,:]= numpy.reshape(tOmega,(R.shape[0],options.ngl*options.ngl))
    return js

def setup_optdf_data_actions(R,vR,vT,z,vz,dvt,toptions,qdf,vo):
    js= numpy.zeros((7,R.shape[0]*options.nmcerr))
    tvT= vT+dvt/vo
    dumm, tjr, tlz, tjz, trg, tkappa, tnu, tOmega= qdf(R.flatten(),
                                                       vR.flatten(),
                                                       tvT.flatten(),
                                                       z.flatten(),vz.flatten(),
                                                       log=True,
                                                       _return_actions=True,
                                                       _return_freqs=True)
    js[0,:]= tjr
    js[1,:]= tlz
    js[2,:]= tjz
    js[3,:]= trg
    js[4,:]= tkappa
    js[5,:]= tnu
    js[6,:]= tOmega
    return js

def setup_optdf_actions(R,zgrid,nzs,options,qdf):
    js= numpy.zeros((7,nzs,options.ngl**3))
    for jj in range(nzs):
        dumm, tjr, tlz, tjz, trg, tkappa, tnu, tOmega= qdf.vmomentdensity(R,zgrid[jj],
                                                0.,0.,0.,
                                                gl=True,
                                                ngl=options.ngl,
                                                _return_actions=True,
                                                _return_freqs=True)
        js[0,jj,:]= tjr
        js[1,jj,:]= tlz
        js[2,jj,:]= tjz
        js[3,jj,:]= trg
        js[4,jj,:]= tkappa
        js[5,jj,:]= tnu
        js[6,jj,:]= tOmega
    return js

def mloglike_optdf_2optimize(params,fullparams,
                             pot,aA,fehs,afes,binned,normintstuff,
                             npops,errstuff,options,vo,ro,
                             jrs,lzs,jzs,normsrs,normszs,initsr,initsz,inithr,
                             rgs,kappas,nus,Omegas,bounds=None):
    """Actual minus loglikelihood to optimize, SINGLE POPULATION"""
    tparams= copy.copy(fullparams)
    startindx= 0
    if options.fitdvt: startindx+= 1
    if options.fitdm: startindx+= 1
    if options.fitro: startindx+= 1
    if options.fitvsun: startindx+= 3
    elif options.fitvsun: startindx+= 1
    ndfparams= get_ndfparams(options)
    startindx+= ndfparams*0
    if options.dfmodel.lower() == 'qdf':
        startindx= startindx+5
    if bounds is None:
        tparams[0:startindx+1]= params
    else: #use ilogit
        for kk in range(len(params)):
            tparams[kk]= ilogit(params[kk])*(bounds[kk][1]-bounds[kk][0])+bounds[kk][0]
    #Priors
    dvt= get_dvt(tparams,options)
    if not _BFGS_CONSTRAINED and not _BFGS:
        if dvt < -0.15 or dvt > 0.15: return numpy.finfo(numpy.dtype(numpy.float64)).max #don't allow crazy dvt
        for ii in range(1):
            logoutfracprior= logprior_outfrac(get_outfrac(tparams,ii,options),
                                              options)
            if logoutfracprior == -numpy.finfo(numpy.dtype(numpy.float64)).max:
                return -logoutfracprior
            logdfprior= logprior_dfparams(tparams,ii,options)
            if logdfprior == -numpy.finfo(numpy.dtype(numpy.float64)).max:
                return -logdfprior
    #logroprior= logprior_ro(get_ro(params,options),options)
    #if logroprior == -numpy.finfo(numpy.dtype(numpy.float64)).max:
    #    return -logroprior
    #Evaluate
    dfparams= get_dfparams(tparams,0,options,log=False)
    #More prior
    if not _BFGS and not _BFGS_CONSTRAINED:
        if options.dfmodel.lower() == 'qdf':
            if dfparams[1] > 2.*initsr or dfparams[1] < initsr/2. \
                    or dfparams[2] > 2.*initsz or dfparams[2] < initsz/2. \
                    or dfparams[0] > 2.*inithr or dfparams[0] < inithr/2.:
                #Don't allow parameters too different from the initial parameters
                return numpy.finfo(numpy.dtype(numpy.float64)).max
        outfrac= get_outfrac(tparams,0,options)
        out= logprior_outfrac(outfrac,options)
    else:
        outfrac= get_outfrac(tparams,0,options)
        out= 0.
    logoutfrac= numpy.log(outfrac)
    loghalodens= numpy.log(ro*outDens(1.,0.,None))
    if options.dfmodel.lower() == 'qdf':
        #Normalize
        hr= dfparams[0]/ro
        sr= dfparams[1]/vo
        sz= dfparams[2]/vo
        hsr= dfparams[3]/ro
        hsz= dfparams[4]/ro
        #Setup
        qdf= quasiisothermaldf(hr,sr,sz,hsr,hsz,pot=pot,aA=aA,cutcounter=True)
    #Calculate surface(R=1.) for relative outlier normalization
    #print "BOVY: SURFACEMASS_Z NEEDS TO BE SPED UP"
    #logoutfrac+= numpy.log(qdf.surfacemass_z(1.,ngl=options.ngl))
    logoutfrac+= numpy.log(qdf.surfacemass_z(1.,ngl=options.ngl))
    #Get data ready
    R,vR,vT,z,vz= prepare_coordinates(tparams,0,fehs,afes,binned,errstuff,
                                      options,npops)
    if options.fitdvt or not options.fixdvt is None:
        dvt= get_dvt(tparams,options)
        vT+= dvt/vo
    ndata= R.shape[0]
    data_lndf= numpy.zeros((ndata,2*options.nmcerr))
    srhalo= _SRHALO/vo/_REFV0
    sphihalo= _SPHIHALO/vo/_REFV0
    szhalo= _SZHALO/vo/_REFV0
    data_lndf= numpy.empty((ndata,2*options.nmcerr))
    if False: #options.marginalizevt:
        data_lndf[:,0:options.nmcerr]= numpy.log(qdf.pvRvz(vR.flatten(),vz.flatten(),
                                                           R.flatten(),z.flatten())).reshape((ndata,options.nmcerr))
        data_lndf[:,options.nmcerr:2*options.nmcerr]= logoutfrac+loghalodens\
            -numpy.log(srhalo)-numpy.log(szhalo)\
            -0.5*(vR**2./srhalo**2.+vz**2./szhalo**2.)\
            -1.*numpy.log(2.*math.pi)
    else:
        data_lndf[:,0:options.nmcerr]= qdf(R.flatten(),vR.flatten(),vT.flatten(),
                                       z.flatten(),vz.flatten(),log=True).reshape((ndata,options.nmcerr))
        data_lndf[:,options.nmcerr:2*options.nmcerr]= logoutfrac+loghalodens\
            -numpy.log(srhalo)-numpy.log(sphihalo)-numpy.log(szhalo)\
            -0.5*(vR**2./srhalo**2.+vz**2./szhalo**2.+vT**2./sphihalo**2.)\
            -1.5*numpy.log(2.*math.pi)
    #Sum data and outlier df, for all MC samples
    data_lndf= mylogsumexp(data_lndf,axis=1)
    if False: #options.marginalizevt:
        data_lndf+= numpy.log(vo)
    #Normalize
    normalization= calc_normint_fixedpot(qdf,0,normintstuff,tparams,npops,options,
                                         logoutfrac,jrs,lzs,jzs,normsrs,normszs,rgs,kappas,nus,Omegas)
    out+= numpy.sum(data_lndf)\
        -ndata*(numpy.log(normalization)+numpy.log(options.nmcerr)) #latter so we can compare
    if _DEBUG:
        print fehs[0], afes[0], tparams, numpy.sum(data_lndf)-ndata*numpy.log(options.nmcerr),ndata*numpy.log(normalization), numpy.log(normalization), numpy.sum(data_lndf)-ndata*(numpy.log(normalization)+numpy.log(options.nmcerr))
    else:
        pass
#        print -out, tparams
    return -out

def mmloglike_gridall(fullparams,hr,sr,sz,
                      pot,aA,fehs,afes,binned,normintstuff,
                      npops,errstuff,options,vo,ro,
                      jrs,lzs,jzs,normsrs,normszs,initsr,initsz,inithr,
                      rgs,kappas,nus,Omegas,normalization_out,
                      normalization_dout,
                      datajrs,datalzs,datajzs,
                      datargs,datakappas,datanus,dataOmegas,
                      datanormsrs,vzgl,vzglw,vRgl,vRglw,vTgl,vTglw):
    """Actual minus loglikelihood to optimize, SINGLE POPULATION"""
    toptions= copy.copy(options)
    if _MULTIDFGRID:
        toptions.multi= toptions.multi2 #Set multi to the second multi
    tparams= copy.copy(fullparams)
    startindx= 0
    if toptions.fitdvt: startindx+= 1
    if toptions.fitdm: startindx+= 1
    if toptions.fitro: startindx+= 1
    if toptions.fitvsun: startindx+= 3
    elif toptions.fitvsun: startindx+= 1
    tparams[startindx]= hr
    if _VARYHSZ:
        tparams[startindx+4]= sr #this is actually hsz, sr is fixed
    else:
        tparams[startindx+1]= sr
    tparams[startindx+2]= sz
    #Setup out
    if _NEWSAVE:
        out= numpy.zeros(5)
        tmp_out= numpy.zeros((toptions.ndvts,toptions.npouts))
        tmp_out2= numpy.zeros((toptions.ndvts,toptions.npouts))
    else:
        out= numpy.zeros((toptions.ndvts,toptions.npouts,1,1))
    if options.conditionalr: #Don't vary hr
        dblexphr= monoAbundanceMW.hr(fehs[0],afes[0],k=(options.sample.lower() == 'k'))
        #Closest hr
        lnhr= numpy.log(dblexphr/_REFR0)
        hrsGrid= numpy.linspace(-1.85714286,0.9,options.nhrs)
        clindx= numpy.argmin(numpy.fabs(lnhr-hrsGrid))
        if hr != hrsGrid[clindx]:
            out[0]= -numpy.finfo(numpy.dtype(numpy.float64)).max
            return out
    #Setup everything for fast calculations
    loghalodens= numpy.log(ro*outDens(1.,0.,None))
    dfparams= get_dfparams(tparams,0,toptions,log=False)
    if toptions.physicaldfparams:
        dfparams= get_dfparams(tparams,0,toptions,log=True)
        dfparams= dfparams[0:5]
        optout= optimize.fmin_powell(chi2_simpleoptdf,dfparams,
                                     args=(numpy.exp(copy.copy(dfparams)),
                                           pot,aA,toptions,ro,vo),
                                     xtol=10.**-3.,
                                     ftol=0.0005,
                                     full_output=True,
                                     maxiter=options.maxiter,
                                     maxfun=10000,disp=False)
        dfparams= list(numpy.exp(optout[0]))
        dfparams= numpy.array(dfparams)
    if toptions.dfmodel.lower() == 'qdf':
        #Normalize
        hr= dfparams[0]/ro
        sr= dfparams[1]/vo
        sz= dfparams[2]/vo
        hsr= dfparams[3]/ro
        hsz= dfparams[4]/ro
        #Setup
        #pdb.set_trace()
        qdf= quasiisothermaldf(hr,sr,sz,hsr,hsz,pot=pot,aA=aA,cutcounter=True)
    #Calculate surface(R=1.) for relative outlier normalization
    #logoutfrac= numpy.log(qdf.surfacemass_z(1.,ngl=toptions.ngl))
    #Calculate normalizations
    if options.conditionalr:
        normalization_qdf= calc_normint_fixedpot_conditionalr(qdf,0,normintstuff,tparams,npops,
                                                 toptions,
                                                 -numpy.finfo(numpy.dtype(numpy.float64)).max,
                                                 jrs,lzs,jzs,normsrs,normszs,
                                                 rgs,kappas,nus,Omegas)
    else:
        normalization_qdf= calc_normint_fixedpot(qdf,0,normintstuff,tparams,npops,
                                                 toptions,
                                                 -numpy.finfo(numpy.dtype(numpy.float64)).max,
                                                 jrs,lzs,jzs,normsrs,normszs,
                                                 rgs,kappas,nus,Omegas)
    #tnormalization_out= numpy.exp(logoutfrac)*normalization_out*vo**3.
    tnormalization_out= normalization_out*vo**3.
#    tnormalization_dout= normalization_dout*vo**3.
    if not _FIXOUT:
        if options.conditionalr:
            raise NotImplementedError
        logoutfrac= numpy.log(normalization_qdf/tnormalization_out)
        tnormalization_out= normalization_qdf
    else:
        logoutfrac= 0.
    #Get data ready
    R,vR,vT,z,vz, datasr, datasz, datarhorz= prepare_coordinates(tparams,0,
                                                                 fehs,afes,
                                                                 binned,
                                                                 errstuff,
                                                                 toptions,
                                                                 npops)
    ndata= R.shape[0]
    if _INTEGRATEMARGINALIZE:
        R= R[:,0]
        vR= vR[:,0]
        vT= vT[:,0]
        z= z[:,0]
        vz= vz[:,0]
    #ndata= datajrs.shape[0]
    data_lndf= numpy.zeros((ndata,2*toptions.nmcerr))
    if options.marginalizevt:
        mdata_df= numpy.zeros((ndata,toptions.ngl**3))
    else:
        mdata_df= numpy.zeros((ndata,toptions.ngl))
    srhalo= _SRHALO/vo/_REFV0
    sphihalo= _SPHIHALO/vo/_REFV0
    szhalo= _SZHALO/vo/_REFV0*numpy.exp(-(R-1.)/6.*_REFR0*ro)
    #Evaluate outliers
#    dblexphr= monoAbundanceMW.hr(fehs[0],afes[0],k=(options.sample.lower() == 'k'))
    dblexphz= monoAbundanceMW.hz(fehs[0],afes[0],k=(options.sample.lower() == 'k'))
    if dblexphz > 500.: szhalo*= 2.
    if options.marginalizevrvt:
        if _INTEGRATEMARGINALIZE:
            szhalo= numpy.sqrt(datasz**2.+szhalo**2.)
            fac= 0.
        else:
            fac= 2.
        data_lndf[:,toptions.nmcerr:2*toptions.nmcerr]= numpy.reshape(logoutfrac+loghalodens\
            -numpy.log(szhalo)-fac*numpy.log(300./_REFV0/vo)\
            -0.5*(vz**2./szhalo**2.)+2.*numpy.log(vo)\
            -0.5*numpy.log(2.*math.pi),(ndata,toptions.nmcerr))
#        data_lndf[:,1*toptions.nmcerr:2*toptions.nmcerr]= numpy.reshape(numpy.log(_DblExpDensity(R*_REFR0*ro,z*_REFR0*ro,[numpy.log(dblexphz/1000.),-numpy.log(dblexphr)]))\
#            -numpy.log(szhalo)-fac*numpy.log(300./_REFV0/vo)\
#            -0.5*(vz**2./szhalo**2.)+2.*numpy.log(vo)\
#            -0.5*numpy.log(2.*math.pi),(ndata,toptions.nmcerr))
    elif options.marginalizevt:
        if _INTEGRATEMARGINALIZE:
            szhalo= numpy.sqrt(datasz**2.+szhalo**2.)
            srhalo= numpy.sqrt(datasr**2.+srhalo**2.)
            fac= 0.
        else:
            fac= 1.
        data_lndf[:,toptions.nmcerr:2*toptions.nmcerr]= numpy.reshape(logoutfrac+loghalodens\
            -numpy.log(srhalo)-numpy.log(szhalo)-fac*numpy.log(300./_REFV0/vo)\
            -0.5*(vz**2./szhalo**2.+vR**2./srhalo**2.)\
            -1.*numpy.log(2.*math.pi),(ndata,toptions.nmcerr))
    else:
        data_lndf[:,toptions.nmcerr:2*toptions.nmcerr]= logoutfrac+loghalodens\
            -numpy.log(srhalo)-numpy.log(sphihalo)-numpy.log(szhalo)\
            -0.5*(vR**2./srhalo**2.+vz**2./szhalo**2.+vT**2./sphihalo**2.)\
            -1.5*numpy.log(2.*math.pi)
    if _FIXOUT:
        data_lndf[:,toptions.nmcerr:2*toptions.nmcerr]-= numpy.log(tnormalization_out)
#        data_lndf[:,2*toptions.nmcerr:3*toptions.nmcerr]-= numpy.log(tnormalization_dout)
    #Run through the grid
    if _NEWESTDFRANGES:
        dvts= numpy.zeros(1)#numpy.linspace(-0.35,0.05,options.ndvts) #could be centered on (0.8Vc-200.)/220.
        pouts= numpy.linspace(10.**-5.,.5,options.npouts)
#        pouts2= numpy.linspace(10.**-5.,.45,options.npouts)
        #pouts= pouts/(1.-pouts)
    elif _NEWDFRANGES:
        dvts= numpy.linspace(-0.35,0.05,options.ndvts) #could be centered on (0.8Vc-200.)/220.
        pouts= numpy.linspace(10.**-5.,.5,options.npouts)
    else:
        dvts= numpy.linspace(-0.1,0.1,toptions.ndvts)
        pouts= numpy.linspace(10.**-5.,.3,toptions.npouts)
    for ii in range(toptions.ndvts):
        #dvt= dvts[ii]
        #tvT= vT+dvt/vo
        #data_lndf[:,0:toptions.nmcerr]= qdf(R.flatten(),vR.flatten(),
        #                                   tvT.flatten(),
        #                                   z.flatten(),
        #                                   vz.flatten(),log=True).reshape((ndata,toptions.nmcerr))
        #data_lndf[:,0:toptions.nmcerr]= qdf((datajrs[:,ii],
        #                                     datalzs[:,ii],
        #                                     datajzs[:,ii]),
        #                                    rg=datargs[:,ii],
        #                                    kappa=datakappas[:,ii],
        #                                    nu=datanus[:,ii],
        #                                    Omega=dataOmegas[:,ii],
        #                                    log=True).reshape((ndata,toptions.nmcerr))
        if _INTEGRATEMARGINALIZE:
            if options.marginalizevrvt:
                for ll in range(options.ngl):
                    mdata_df[:,ll]= qdf.pvz(vzgl[:,ll],
                                            R.flatten(),
                                            z.flatten(),
                                            gl=True,
                                            ngl=options.ngl,
                                            _jr=datajrs[:,ll,ii,:].flatten(),
                                            _lz=datalzs[:,ll,ii,:].flatten(),
                                            _jz=datajzs[:,ll,ii,:].flatten(),
                                            _rg=datargs[:,ll,ii,:].flatten(),
                                            _kappa=datakappas[:,ll,ii,:].flatten(),
                                            _nu=datanus[:,ll,ii,:].flatten(),
                                            _Omega=dataOmegas[:,ll,ii,:].flatten(),
                                            _sigmaR1=datanormsrs.flatten())
                data_lndf[:,0]= numpy.log(numpy.sum(mdata_df*vzglw,axis=1)*datasz*2.)+2.*numpy.log(vo)
            elif options.marginalizevt:
                for ll in range(ndata):
                    mdata_df[ll,:]= qdf((datajrs[ll,ii,:],
                                         datalzs[ll,ii,:],
                                         datajzs[ll,ii,:]),
                                        rg=datargs[ll,ii,:],
                                        kappa=datakappas[ll,ii,:],
                                        nu=datanus[ll,ii,:],
                                        Omega=dataOmegas[ll,ii,:],
                                        log=False,
                                        _return_actions=False,
                                        _return_freqs=False)
                data_lndf[:,0]= numpy.log(numpy.sum(mdata_df*vzglw*vRglw*vTglw,
                                                    axis=1)*datasz*datasr*3.)+numpy.log(vo)
        else:
            data_lndf[:,0:toptions.nmcerr]= numpy.log(qdf.pvz(vz.flatten(),
                                                              R.flatten(),
                                                              z.flatten(),
                                                              gl=True,
                                                              ngl=options.ngl,
                                                _jr=datajrs[:,:,ii,:].flatten(),
                                                _lz=datalzs[:,:,ii,:].flatten(),
                                                _jz=datajzs[:,:,ii,:].flatten(),
                                                _rg=datargs[:,:,ii,:].flatten(),
                                                _kappa=datakappas[:,:,ii,:].flatten(),
                                                _nu=datanus[:,:,ii,:].flatten(),
                                                _Omega=dataOmegas[:,:,ii,:].flatten(),
                                                _sigmaR1=datanormsrs.flatten())).reshape((ndata,toptions.nmcerr))+2.*numpy.log(vo)-2.*numpy.log(300./_REFV0/vo)
        if _FIXOUT:
            data_lndf[:,0:toptions.nmcerr]-= numpy.log(normalization_qdf)
        for jj in range(toptions.npouts):
            #Sum data and outlier df, for all MC samples
            data_lndf[:,toptions.nmcerr:2*toptions.nmcerr]+= numpy.log(pouts[jj])
#            for kk in range(toptions.npouts):
                #Sum data and outlier df, for all MC samples
#                data_lndf[:,2*toptions.nmcerr:3*toptions.nmcerr]+= numpy.log(pouts2[kk])
            if _FIXOUT:
                data_lndf[:,0:toptions.nmcerr]+= numpy.log(1.-pouts[jj])#-pouts2[kk])
            sumdata_lndf= mylogsumexp(data_lndf,axis=1)
            if _FIXOUT:
                tmp_out[ii,jj]= numpy.sum(sumdata_lndf)
#                    poutlier= mylogsumexp(data_lndf[:,toptions.nmcerr:2*toptions.nmcerr],axis=1)-sumdata_lndf
#                    tmp_out2[ii,jj]= numpy.sum(numpy.exp(poutlier))/ndata
            else:
                tmp_out[ii,jj]= numpy.sum(sumdata_lndf)\
                    -ndata*(numpy.log(normalization_qdf+pouts[jj]*tnormalization_out)+numpy.log(toptions.nmcerr)) #latter so we can compare
            #tmp_out1[ii,jj,kk,ll]= numpy.sum(sumdata_lndf)
            #tmp_out2[ii,jj,kk,ll]= -ndata*(numpy.log(normalization_qdf+pouts[jj]*tnormalization_out+pouts[kk]*tnormalization_dout)+numpy.log(toptions.nmcerr)) #latter so we can compare
            if _FIXOUT:
                data_lndf[:,0:toptions.nmcerr]-= numpy.log(1.-pouts[jj])#-pouts2[kk])
#            data_lndf[:,2*toptions.nmcerr:3*toptions.nmcerr]-= numpy.log(pouts2[kk])
            data_lndf[:,toptions.nmcerr:2*toptions.nmcerr]-= numpy.log(pouts[jj])
 #Reset
    out[0]= logsumexp(tmp_out)
    indx= numpy.unravel_index(numpy.argmax(tmp_out),tmp_out.shape)
    if not options.conditionalr:
        out[1]= normalization_qdf*pouts[indx[1]]/tnormalization_out
    out[2]= pouts[indx[1]]
#    if _FIXOUT:
#        out[3]= tmp_out2[indx[0],indx[1]]
    if False:
        #Calculate outlier probability for best-fit
        ii= 0
        for ll in range(options.ngl):
            mdata_df[:,ll]= qdf.pvz(vzgl[:,ll],
                                    R.flatten(),
                                    z.flatten(),
                                    gl=True,
                                    ngl=options.ngl,
                                    _jr=datajrs[:,ll,ii,:].flatten(),
                                    _lz=datalzs[:,ll,ii,:].flatten(),
                                    _jz=datajzs[:,ll,ii,:].flatten(),
                                    _rg=datargs[:,ll,ii,:].flatten(),
                                    _kappa=datakappas[:,ll,ii,:].flatten(),
                                    _nu=datanus[:,ll,ii,:].flatten(),
                                    _Omega=dataOmegas[:,ll,ii,:].flatten(),
                                    _sigmaR1=datanormsrs.flatten())
        data_lndf[:,0]= numpy.log(numpy.sum(mdata_df*vzglw,axis=1)*datasz*2.)+2.*numpy.log(vo)
        if _FIXOUT:
            data_lndf[:,0:toptions.nmcerr]-= numpy.log(normalization_qdf)
        data_lndf[:,toptions.nmcerr:2*toptions.nmcerr]+= numpy.log(pouts[indx[1]])
        data_lndf[:,0:toptions.nmcerr]+= numpy.log(1.-pouts[jj])
        sumdata_lndf= mylogsumexp(data_lndf,axis=1)
        if _FIXOUT:
            tmp_out[ii,jj]= numpy.sum(sumdata_lndf)
            poutlier= mylogsumexp(data_lndf[:,toptions.nmcerr:2*toptions.nmcerr],axis=1)-sumdata_lndf
        pdb.set_trace()
    return out

##PRIORS
def logprior_ro(ro,options):
    """Prior on ro"""
    if not options.fitro: return 0.
    if options.noroprior: return 0.
    return -(ro-_REFR0/_REFR0)**2./(0.5/_REFR0)**2. #assume sig ro = 0.5 kpc

def logprior_outfrac(outfrac,options):
    """Prior on the outlier fraction"""
    if outfrac <= 0. or outfrac >= 1.:
        return -numpy.finfo(numpy.dtype(numpy.float64)).max
    return numpy.log(1./(1.+numpy.exp((outfrac-0.2)/0.02)))

def logprior_dfparams(p,ii,options):
    """Prior on the DF"""
    #get params
    theseparams= get_dfparams(p,ii,options,log=True)
    if options.fitdm:
        dm= get_dm(p,options)
        if dm < -0.4 or dm > 0.4:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max   
    if options.dfmodel.lower() == 'qdf':
        if theseparams[0] < -1.9 or theseparams[0] > 2.53:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        if theseparams[1] < -3.1 or theseparams[1] > -0.4:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        if theseparams[2] < -3.1 or theseparams[2] > -0.4:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        if theseparams[3] < -0.3 or theseparams[3] > 2.53:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        if theseparams[4] < -0.3 or theseparams[4] > 2.53:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        if theseparams[1] < theseparams[2]-0.7: #don't let sr < sz/??
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        return 0.

def logprior_pot(params,options,npops):
    """Prior on the potential"""
    vo= get_vo(params,options,npops)
    if vo < 100./_REFV0 or vo > 350./_REFV0: return -numpy.finfo(numpy.dtype(numpy.float64)).max #don't allow crazy vo
    dvt= get_dvt(params,options)
    if dvt < -0.5 or dvt > 0.5: return -numpy.finfo(numpy.dtype(numpy.float64)).max #don't allow crazy dvt
    out= 0.
    if options.novoprior: pass
    else:
        if options.bovy09voprior:
            out-= 0.5*(vo-236./_REFV0)**2./(11./_REFV0)**2.
        elif options.bovy12voprior:
            out-= 0.5*(vo-218./_REFV0)**2./(6./_REFV0)**2.
        else:
            out-= 0.5*(vo-225./_REFV0)**2./(15./_REFV0)**2.
    potparams= get_potparams(params,options,npops)
    if options.potential.lower() == 'flatlog' or options.potential.lower() == 'flatlogdisk':
        q= potparams[0]
        if (not options.noqprior and q <= 0.53) or (options.noqprior and q <= 0.): #minimal flattening for positive density at R > 5 kpc, |Z| < 4 kpc, ALSO CHANGE IN SETUP_DOMAIN
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
    elif options.potential.lower() == 'mwpotentialsimplefit' \
            or options.potential.lower() == 'mwpotentialfixhalo':
        if potparams[0] < -2.1 or potparams[0] > -0.3:#2.53:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        if potparams[2-(1-(options.fixvo is None))] < -5.1 or potparams[2-(1-(options.fixvo is None))] > 1.4:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        if potparams[3-(1-(options.fixvo is None))] < 0. or potparams[3-(1-(options.fixvo is None))] > 1.:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        if options.potential.lower() == 'mwpotentialfixhalo' \
                and (potparams[4] < 0.9 or potparams[4] > 1.):
#                         or (potparams[3]+potparams[4] > 1.) \
#                         or (potparams[3]+potparams[4] < .9)):
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
    elif options.potential.lower() == 'mwpotentialfixhaloflat':
        if potparams[0] < -2.1 or potparams[0] > -0.3:#2.53:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        if potparams[2-(1-(options.fixvo is None))] < -5.1 or potparams[2-(1-(options.fixvo is None))] > 1.4:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        if potparams[4] < 0.0 or potparams[4] > 0.1:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        return logprior_dlnvcdlnr(potparams[3],options)
    elif options.potential.lower() == 'mpdiskplhalofixbulgeflat' \
            or options.potential.lower() == 'dpdiskplhalofixbulgeflat' \
            or options.potential.lower() == 'dpdiskplhalofixbulgeflatwgas':
        if potparams[0] < -2.1 or potparams[0] > -0.3:#2.53:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        if potparams[2-(1-(options.fixvo is None))] < -5.1 or potparams[2-(1-(options.fixvo is None))] > 1.4:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        if potparams[4] < 0.0 or potparams[4] > 3.:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        return logprior_dlnvcdlnr(potparams[3],options)
    elif options.potential.lower() == 'dpdiskplhalofixbulgeflatwgasalt':
        if potparams[0] < -2.1 or potparams[0] > -0.3:#2.53:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        if potparams[2-(1-(options.fixvo is None))] < -5.1 or potparams[2-(1-(options.fixvo is None))] > 1.4:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        if potparams[3] < 0.0 or potparams[3] > 1.:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        return logprior_dlnvcdlnr(potparams[4],options)
    elif options.potential.lower() == 'mpdiskflplhalofixplfixbulgeflat':
        if potparams[0] < -2.1 or potparams[0] > -0.3:#2.53:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        if potparams[2-(1-(options.fixvo is None))] < -5.1 or potparams[2-(1-(options.fixvo is None))] > 1.4:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        if potparams[4] < 0.4 or potparams[4] > 1.15: #rough prior from Evans 94
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        return logprior_dlnvcdlnr(potparams[3],options)
    elif options.potential.lower() == 'dpdiskflplhalofixbulgeflatwgas':
        if potparams[0] < -2.1 or potparams[0] > -0.3:#2.53:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        if potparams[2-(1-(options.fixvo is None))] < -5.1 or potparams[2-(1-(options.fixvo is None))] > 1.4:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        if potparams[4] < 0.4 or potparams[4] > 1.15: #rough prior from Evans 94
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        if potparams[5] < -2.0 or potparams[5] > 1.0:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        return logprior_dlnvcdlnr(potparams[3],options)
    elif options.potential.lower() == 'bti' \
            or options.potential.lower() == 'btii':
        return 0.
    return out

def logprior_dlnvcdlnr(dlnvcdlnr,options):
    if options.nodlnvcdlnrprior: return 0.
    if True:
        sb= 0.04
        if dlnvcdlnr/30. > sb or dlnvcdlnr/30. < -0.5:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        return numpy.log((sb-dlnvcdlnr/30.)/sb)-(sb-dlnvcdlnr/30.)/sb

def approxFitResult(feh,afe,relerr=False,sample='g'):
    """Return the result from the paper I and III fits, smoothed and transformed to DF input parameters
    returns (lnhR,lnSr,lnSz)
    also returns hRerr/hR,sRerr/sR,sZerr/sz if relerr=True (these are not transformed, as the transformation shouldn't change these too much"""
    #Get smoothed monoAbundance results
    hr= monoAbundanceMW.hr(feh,afe,k=(sample.lower() == 'k'))/_REFR0 #No smoothing for this
    sr= monoAbundanceMW.sigmar(feh,afe,smooth=True,k=(sample.lower() == 'k'))/_REFV0
    sz= monoAbundanceMW.sigmaz(feh,afe,smooth=True,k=(sample.lower() == 'k'))/_REFV0
    if relerr:
        rehr= monoAbundanceMW.hr(feh,afe,err=True,
                                 k=(sample.lower() == 'k'))[1]/_REFR0/hr #No smoothing for this
        resr= monoAbundanceMW.sigmar(feh,afe,smooth=False,err=True,
                                     k=(sample.lower() == 'k'))[1]/_REFV0/sr
        resz= monoAbundanceMW.sigmaz(feh,afe,smooth=False,err=True,
                                     k=(sample.lower() == 'k'))[1]/_REFV0/sz
    #Special case the two most metal-poor G dwarf bins
    if feh < -1.1:
        sr= monoAbundanceMW.sigmar(-1.05,afe,smooth=True,
                                    k=(sample.lower() == 'k'))/_REFV0
    #Load the results from qdfProperties that relates input and output parameters, first the scale length
    savefile= open('hrhrhr.sav','rb')
    hrhrhr= pickle.load(savefile)
    qdfhrs= pickle.load(savefile)
    qdfsrs= pickle.load(savefile)
    savefile.close()
    indx= numpy.argmin((sr*_REFV0-qdfsrs)**2.) #Closest radial dispersion, roughly
    hrSpline= interpolate.UnivariateSpline(numpy.log(hrhrhr[:,indx]*qdfhrs[:,indx]/_REFR0),
                                   numpy.log(qdfhrs[:,indx]/_REFR0),
                                           k=3)
#    try:
    lnhrin= hrSpline(numpy.log(hr))
    if lnhrin < -1.6: lnhrin= -1.6
    #Now find the input sigmas: sigmaz
    savefile= open('szszsz.sav','rb')
    szszsz= pickle.load(savefile)
    qdfszs= pickle.load(savefile)
    qdfhrs= pickle.load(savefile)
    savefile.close()
    indx= numpy.argmin((numpy.exp(lnhrin)*_REFR0-qdfhrs)**2.) #Closest radial scale length
    szSpline= interpolate.UnivariateSpline(numpy.log(szszsz[:,indx]*qdfszs[:,indx]/_REFV0),
                                   numpy.log(qdfszs[:,indx]/_REFV0),
                                   k=3)
#    try:
    lnszin= szSpline(numpy.log(sz))
    #sigmar:
    savefile= open('srsrsr.sav','rb')
    srsrsr= pickle.load(savefile)
    qdfsrs= pickle.load(savefile)
    qdfhrs= pickle.load(savefile)
    savefile.close()
    indx= numpy.argmin((numpy.exp(lnhrin)*_REFR0-qdfhrs)**2.) #Closest radial scale length
    srSpline= interpolate.UnivariateSpline(numpy.log(srsrsr[:,indx]*qdfsrs[:,indx]/_REFV0),
                                   numpy.log(qdfsrs[:,indx]/_REFV0),
                                   k=3)
#    try:
    lnsrin= srSpline(numpy.log(sr))
    if relerr:
        return (lnhrin,lnsrin,lnszin,rehr,resr,resz)
    else:
        return (lnhrin,lnsrin,lnszin)
    
##SETUP AND CALCULATE THE NORMALIZATION INTEGRAL
def calc_normint(qdf,indx,normintstuff,params,npops,options,logoutfrac,
                     fqdf=1.):
    """Calculate the normalization integral"""
    if options.mcall or options.mcwdf: #evaluation is the same for these
        return calc_normint_mcall(qdf,indx,normintstuff,params,npops,options,
                                  logoutfrac)
    else:
        return calc_normint_mcv(qdf,indx,normintstuff,params,npops,options,
                                logoutfrac,fqdf=fqdf)

def calc_normint_mcall(qdf,indx,normintstuff,params,npops,options,logoutfrac):
    """calculate the normalization integral by monte carlo integrating over everything"""
    thisnormintstuff= normintstuff[indx]
    mock= unpack_normintstuff(thisnormintstuff,options)
    out= 0.
    ro= get_ro(params,options)
    vo= get_vo(params,options,npops)
    #logoutfrac= numpy.log(get_outfrac(params,indx,options))
    loghalodens= numpy.log(ro/12.)
    srhalo= _SRHALO/vo/_REFV0
    sphihalo= _SPHIHALO/vo/_REFV0
    szhalo= _SZHALO/vo/_REFV0
    #Calculate (R,z)s
    l= numpy.array([m[3] for m in mock])
    b= numpy .array([m[4] for m in mock])
    d= numpy.array([m[6] for m in mock])
    XYZ= bovy_coords.lbd_to_XYZ(l,b,d,
                                degree=True)
    R= ((ro*_REFR0-XYZ[:,0])**2.+XYZ[:,1]**2.)**(0.5)/ro/_REFR0
    XYZ[:,2]+= _ZSUN
    z= XYZ[:,2]/ro/_REFR0
    vlos= numpy.array([m[8] for m in mock])
    pmll= numpy.array([m[9] for m in mock])
    pmbb= numpy.array([m[10] for m in mock])
    vxvyvz= bovy_coords.vrpmllpmbb_to_vxvyvz(vlos,pmll,pmbb,l,b,d,degree=False)
    vsun = numpy.array(list(get_vsun(params,options)))*_REFV0
    vR,vT,vz= bovy_coords.vxvyvz_to_galcencyl(vxvyvz[:,0],
                                              vxvyvz[:,1],
                                              vxvyvz[:,2],
                                              ro*_REFR0-XYZ[:,0],
                                              XYZ[:,1],
                                              XYZ[:,2],
                                              vsun=vsun,
                                              galcen=False)
    vR/= _REFV0*vo
    vT/= _REFV0*vo
    vz/= _REFV0*vo
    """
    vR= numpy.array([m[8] for m in mock])/vo/_REFV0
    vT= numpy.array([m[9] for m in mock])/vo/_REFV0
    vz= numpy.array([m[10] for m in mock])/vo/_REFV0
    """
    thislogdf= numpy.zeros((len(R),2))
    thislogfiddf= numpy.array([m[11] for m in mock])
    thislogdf[:,0]= qdf(R,vR,vT,z,vz,log=True)
    thislogdf[:,1]= logoutfrac+loghalodens\
            -numpy.log(srhalo)-numpy.log(sphihalo)-numpy.log(szhalo)\
            -0.5*(vR**2./srhalo**2.+vz**2./szhalo**2.+vT**2./sphihalo**2.)
    #Sum data and outlier df
    thislogdf= mylogsumexp(thislogdf,axis=1)
    out= numpy.exp(-thislogfiddf+thislogdf)
    #print numpy.log(numpy.mean(out)), numpy.std(out)/numpy.mean(out)/numpy.sqrt(options.nmc)
    return numpy.mean(out)

def calc_normint_mcv(qdf,indx,normintstuff,params,npops,options,logoutfrac,
                     fqdf=1.):
    """calculate the normalization integral by monte carlo integrating over v, but grid integrating over everything else"""
    thisnormintstuff= normintstuff[indx]
    if _PRECALCVSAMPLES:
        sf, plates,platel,plateb,platebright,platefaint,grmin,grmax,rmin,rmax,fehmin,fehmax,feh,colordist,fehdist,gr,rhogr,rhofeh,mr,dmin,dmax,ds, surfscale, hr, hz, colorfehfac,normR, normZ,surfnrs, surfnzs, surfRgrid, surfzgrid, surfvrs, surfvts, surfvzs= unpack_normintstuff(thisnormintstuff,options)
    else:
        sf, plates,platel,plateb,platebright,platefaint,grmin,grmax,rmin,rmax,fehmin,fehmax,feh,colordist,fehdist,gr,rhogr,rhofeh,mr,dmin,dmax,ds, surfscale, hr, hz, colorfehfac, normR, normZ= unpack_normintstuff(thisnormintstuff,options)
    out= 0.
    ro= get_ro(params,options)
#    outfrac= get_outfrac(params,indx,options)
    outfrac= numpy.exp(logoutfrac)
    halodens= ro*outDens(1.,0.,None)
    globalInterp= True
    start= time.time()
    if globalInterp and not fqdf == 0.:
        nrs, nzs= _SURFNRS, _SURFNZS
        thisRmin, thisRmax= 4./_REFR0, 15./_REFR0
        thiszmin, thiszmax= 0., .8
        Rgrid= numpy.linspace(thisRmin,thisRmax,nrs)
        zgrid= numpy.linspace(thiszmin,thiszmax,nzs)
        surfgrid= numpy.empty((nrs,nzs))
        if _PRECALCVSAMPLES:
            nrs, nzs= surfnrs, surfnzs
            Rgrid= surfRgrid
            zgrid= surfzgrid
        for ii in range(nrs):
            for jj in range(nzs):
                if _PRECALCVSAMPLES:
                    surfgrid[ii,jj]= qdf.density(surfRgrid[ii],
                                                 surfzgrid[jj],
                                                 _vrs=surfvrs[jj+ii*surfnzs],
                                                 _vts=surfvts[jj+ii*surfnzs],
                                                 _vzs=surfvzs[jj+ii*surfnzs],
                                                 nmc=options.nmcv,
                                                 _rawgausssamples=True)
                else:
                    surfgrid[ii,jj]= qdf.density(Rgrid[ii],zgrid[jj],
                                                 ngl=options.ngl,
                                                 nmc=options.nmcv)
        if _SURFSUBTRACTEXPON:
            Rs= numpy.tile(Rgrid,(nzs,1)).T
            Zs= numpy.tile(zgrid,(nrs,1))
            ehr= qdf.estimate_hr(1.,z=0.125)
#            ehz= qdf.estimate_hz(1.,zmin=0.5,zmax=0.7)#Get large z behavior right
            ehz= qdf.estimate_hz(1.,z=0.125)
            surfInterp= interpolate.RectBivariateSpline(Rgrid,zgrid,
                                                        numpy.log(surfgrid)
                                                        +Rs/ehr+numpy.fabs(Zs)/ehz,
                                                        kx=3,ky=3,
                                                        s=0.)
            #s=10.*float(nzs*nrs))
        else:
            surfInterp= interpolate.RectBivariateSpline(Rgrid,zgrid,
                                                        numpy.log(surfgrid),
                                                        kx=3,ky=3,
                                                        s=0.)
#                                                        s=10.*float(nzs*nrs))
    if options.mcvalt:
        #Alternative manner that uses well-tested compareDataModel code
        if _SURFSUBTRACTEXPON and not fqdf == 0.:
            compare_func= lambda x,y,du: numpy.exp(surfInterp.ev(x/ro/_REFR0,numpy.fabs(y)/ro/_REFR0)-x/ro/_REFR0/ehr-numpy.fabs(y)/ehz/ro/_REFR0)+outfrac*halodens
        elif not fqdf == 0.:
            compare_func= lambda x,y,du: numpy.exp(surfInterp.ev(x/ro/_REFR0,numpy.fabs(y)/ro/_REFR0))+outfrac*halodens
        else:
            compare_func= lambda x,y,du: outfrac*halodens
        distfac= 10.**(get_dm(params,options)/5.)
        mid= time.time()
        n= comparernumberPlate(compare_func,
                               None,sf,
                               colordist,fehdist,None,
                               'all',zmax=options.zmax,
                               rmin=rmin,rmax=rmax,
                               grmin=grmin,grmax=grmax,
                               fehmin=fehmin,
                               fehmax=fehmax,
                               feh=feh,
                               noplot=True,nodata=True,distfac=distfac,
                               R0=_REFR0*ro,
                               colorfehfac=colorfehfac,normR=normR,normZ=normZ,
                               numcores=options.multi)
        vo= get_vo(params,options,npops)
        end= time.time()
        #print "Times: %f, %f, %f" % ((mid-start)/(end-start),
        #                             (end-mid)/(end-start),
        #                             end-start)
        return numpy.sum(n)*vo**3.
    for ii in range(len(plates)):
        #if _DEBUG: print plates[ii], sf(plates[ii])
        if options.sample.lower() == 'k' and options.indiv_brightlims:
            #For now, to be sure
            thisrmin= rmin
            thisrmax= rmax
        elif sf.platebright[str(plates[ii])] and not sf.type_bright.lower() == 'sharprcut':
            thisrmin= rmin
            thisrmax= 17.8
        elif sf.platebright[str(plates[ii])] and sf.type_bright.lower() == 'sharprcut':
            thisrmin= rmin
            thisrmax= numpy.amin([sf.rcuts[str(plates[ii])],17.8])
        elif not sf.type_faint.lower() == 'sharprcut':
            thisrmin= 17.8
            thisrmax= rmax
        elif sf.type_faint.lower() == 'sharprcut':
            thisrmin= 17.8
            thisrmax= numpy.amin([sf.rcuts[str(plates[ii])],rmax])
        #Compute integral by binning everything in distance
        thisout= numpy.zeros(_NDS)
        #BOVY: THIS CAN BE PUT IN SETUP?
        for kk in range(_NGR):
            for jj in range(_NFEH):
                #What rs do these ds correspond to
                rs= 5.*numpy.log10(ds)+10.+mr[kk,jj]
                thisout+= sf(plates[ii],r=rs)*rhogr[kk]*rhofeh[jj]
                #We know the following are zero
                thisout[(rs < thisrmin)]= 0.
                thisout[(rs > thisrmax)]= 0.
        #Calculate (R,z)s
        XYZ= bovy_coords.lbd_to_XYZ(numpy.array([platel[ii] for dd in range(len(ds))]),
                                    numpy.array([plateb[ii] for dd in range(len(ds))]),
                                    ds,degree=True)
        R= ((ro*_REFR0-XYZ[:,0])**2.+XYZ[:,1]**2.)**(0.5)/ro/_REFR0
        XYZ[:,2]+= _ZSUN
        z= XYZ[:,2]/ro/_REFR0
        if not options.zmin is None and not options.zmax is None:
            indx= (numpy.fabs(XYZ[:,2]) < options.zmin)
            thisout[indx]= 0.
            indx= (numpy.fabs(XYZ[:,2]) >= options.zmax)
            thisout[indx]= 0.
        if not options.rmin is None and not options.rmax is None:
            indx= (R < options.rmin)
            thisout[indx]= 0.
            indx= (R >= options.rmax)
            thisout[indx]= 0.
        if not globalInterp:
            #Calculate the surfacemass on a rough grid, interpolate, and integrate
            ndsgrid= numpy.amax([int(round((dmax-dmin)/surfscale[ii]*options.nscale)),7]) #at least 7
#        print surfscale[ii], ndsgrid
            dsgrid= numpy.linspace(dmin,dmax,ndsgrid)
            XYZgrid= bovy_coords.lbd_to_XYZ(numpy.array([platel[ii] for dd in range(ndsgrid)]),
                                            numpy.array([plateb[ii] for dd in range(ndsgrid)]),
                                            dsgrid,degree=True)
            Rgrid= ((ro*_REFR0-XYZ[:,0])**2.+XYZ[:,1]**2.)**(0.5)/ro/_REFR0
            XYZgrid[:,2]+= _ZSUN
            zgrid= XYZgrid[:,2]/ro/_REFR0       
            surfgrid= numpy.zeros(ndsgrid)
            for kk in range(ndsgrid):
                surfgrid[kk]= qdf.density(Rgrid[kk],zgrid[kk],nmc=options.nmcv)
#            print kk, dsgrid[kk], Rgrid[kk], zgrid[kk], surfgrid[kk]
        #Interpolate
#        surfinterpolate= interpolate.InterpolatedUnivariateSpline(dsgrid/ro/_REFR0,
            surfinterpolate= interpolate.UnivariateSpline(dsgrid/ro/_REFR0,
                                                          numpy.log(surfgrid),
                                                          k=3)
            thisout*= ds**2.*(numpy.exp(surfinterpolate(ds/ro/_REFR0))\
                                  +outfrac*halodens)#*(2.*math.pi)**-1.5)
        else:
            if _SURFSUBTRACTEXPON:
                thisout*= ds**2.*(numpy.exp(surfInterp.ev(R,numpy.fabs(z))
                                            -R/ehr-numpy.fabs(z)/ehz)
                                  +outfrac*halodens)#*(2.*math.pi)**-1.5)
            else:
                thisout*= ds**2.*(numpy.exp(surfInterp.ev(R,numpy.fabs(z)))
                                  +outfrac*halodens)#*(2.*math.pi)**-1.5)
            #To be sure we are not bothered by extrapolation
            thisout[(R < thisRmin)]= 0.
            thisout[(R > thisRmax)]= 0.
            thisout[(numpy.fabs(z) > thiszmax)]= 0.
#        print ii, len(plates)
        out+= numpy.sum(thisout)
    vo= get_vo(params,options,npops)
    return out*vo**3.

def calc_normint_fixedpot(qdf,indx,normintstuff,params,npops,options,
                          logoutfrac,jrs,lzs,jzs,normsrs,normszs,
                          rgs,kappas,nus,Omegas,fqdf=1.,
                          dblexp=False,dblexphr=None,dblexphz=None):
    """Calculate the normalization integral"""
    if options.mcall or options.mcwdf: #evaluation is the same for these
        raise NotImplementedError("mcall and mcwdf not implemented for fixed potential")
    else:
        return calc_normint_mcv_fixedpot(qdf,indx,normintstuff,params,npops,
                                         options,
                                         logoutfrac,
                                         jrs,lzs,jzs,normsrs,normszs,
                                         rgs,kappas,nus,Omegas,fqdf,
                                         dblexp,dblexphr,dblexphz)

def calc_normint_mcv_fixedpot(qdf,indx,normintstuff,params,npops,options,
                              logoutfrac,jrs,lzs,jzs,normsrs,normszs,
                              rgs,kappas,nus,Omegas,fqdf,dblexp,
                              dblexphr,dblexphz):
    """calculate the normalization integral by monte carlo integrating over v, but grid integrating over everything else"""
    thisnormintstuff= normintstuff[indx]
    if _PRECALCVSAMPLES:
        sf, plates,platel,plateb,platebright,platefaint,grmin,grmax,rmin,rmax,fehmin,fehmax,feh,colordist,fehdist,gr,rhogr,rhofeh,mr,dmin,dmax,ds, surfscale, hr, hz, colorfehfac,normR, normZ,surfnrs, surfnzs, surfRgrid, surfzgrid, surfvrs, surfvts, surfvzs= unpack_normintstuff(thisnormintstuff,options)
    else:
        sf, plates,platel,plateb,platebright,platefaint,grmin,grmax,rmin,rmax,fehmin,fehmax,feh,colordist,fehdist,gr,rhogr,rhofeh,mr,dmin,dmax,ds, surfscale, hr, hz, colorfehfac, normR, normZ= unpack_normintstuff(thisnormintstuff,options)
    out= 0.
    ro= get_ro(params,options)
#    outfrac= get_outfrac(params,indx,options)
    outfrac= numpy.exp(logoutfrac)
    halodens= ro*outDens(1.,0.,None)
    globalInterp= True
    #start= time.time()
    if globalInterp and not fqdf == 0.:
        nrs, nzs= _SURFNRS, _SURFNZS
        thisRmin, thisRmax= 4./_REFR0, 15./_REFR0
        thiszmin, thiszmax= 0., .8
        Rgrid= numpy.linspace(thisRmin,thisRmax,nrs)
        zgrid= numpy.linspace(thiszmin,thiszmax,nzs)
        surfgrid= numpy.empty((nrs,nzs))
        if not options.multi is None:
            multOut= multi.parallel_map((lambda x: _calc_surfgrid_actions(Rgrid[x],
                                                                       zgrid,nzs,
                                                                       options,qdf,jrs[x,:,:],lzs[x,:,:],jzs[x,:,:],normsrs[x,:],normszs[x,:],
                                                                          rgs[x,:,:],kappas[x,:,:],nus[x,:,:],Omegas[x,:,:])),
                                        range(nrs),
                                        numcores=numpy.amin([nrs,
                                                             multiprocessing.cpu_count(),
                                                             options.multi]))
            for ii in range(nrs):
                surfgrid[ii,:]= multOut[ii]
        else:
            for ii in range(nrs):
                for jj in range(nzs):
                    surfgrid[ii,jj]= qdf.density(Rgrid[ii],zgrid[jj],
                                                 ngl=options.ngl,
                                                 nmc=options.nmcv,
                                                 _jr=jrs[ii,jj,:],
                                                 _lz=lzs[ii,jj,:],
                                                 _jz=jzs[ii,jj,:],
                                                 _rg=rgs[ii,jj,:],
                                                 _kappa=kappas[ii,jj,:],
                                                 _nu=nus[ii,jj,:],
                                                 _Omega=Omegas[ii,jj,:],
                                                 _sigmaR1=normsrs[ii,jj],
                                                 _sigmaz1=normszs[ii,jj])
        if _SURFSUBTRACTEXPON:
            Rs= numpy.tile(Rgrid,(nzs,1)).T
            Zs= numpy.tile(zgrid,(nrs,1))
            ehr= qdf.estimate_hr(1.,z=0.125)
#            ehz= qdf.estimate_hz(1.,zmin=0.5,zmax=0.7)#Get large z behavior right
            ehz= qdf.estimate_hz(1.,z=0.125)
            surfInterp= interpolate.RectBivariateSpline(Rgrid,zgrid,
                                                        numpy.log(surfgrid)
                                                        +Rs/ehr+numpy.fabs(Zs)/ehz,
                                                        kx=3,ky=3,
                                                        s=0.)
            #s=10.*float(nzs*nrs))
        else:
            surfInterp= interpolate.RectBivariateSpline(Rgrid,zgrid,
                                                        numpy.log(surfgrid),
                                                        kx=3,ky=3,
                                                        s=0.)
#                                                        s=10.*float(nzs*nrs))
    if options.mcvalt:
        #Alternative manner that uses well-tested compareDataModel code
        if _SURFSUBTRACTEXPON and not fqdf == 0.:
            compare_func= lambda x,y,du: numpy.exp(surfInterp.ev(x/ro/_REFR0,numpy.fabs(y)/ro/_REFR0)-x/ro/_REFR0/ehr-numpy.fabs(y)/ehz/ro/_REFR0)+outfrac*halodens
        elif not fqdf == 0.:
            compare_func= lambda x,y,du: numpy.exp(surfInterp.ev(x/ro/_REFR0,numpy.fabs(y)/ro/_REFR0))+outfrac*halodens
        elif dblexp: #just do the outliers
            compare_func= lambda x,y,du: outfrac*_DblExpDensity(x,y,[numpy.log(dblexphz/1000.),-numpy.log(dblexphr)])
        else: #just do the outliers
            compare_func= lambda x,y,du: outfrac*halodens
        distfac= 10.**(get_dm(params,options)/5.)
        #mid= time.time()
        n= comparernumberPlate(compare_func,
                               None,sf,
                               colordist,fehdist,None,
                               'all',zmax=options.zmax,
                               rmin=rmin,rmax=rmax,
                               grmin=grmin,grmax=grmax,
                               fehmin=fehmin,
                               fehmax=fehmax,
                               feh=feh,
                               noplot=True,nodata=True,distfac=distfac,
                               R0=_REFR0*ro,
                               colorfehfac=colorfehfac,normR=normR,normZ=normZ,
                               numcores=options.multi)
        vo= get_vo(params,options,npops)
        #end= time.time()
        #print "Times: %f, %f, %f" % ((mid-start)/(end-start),
        #                             (end-mid)/(end-start),
        #                             end-start)
        return numpy.nansum(n)*vo**3.

def calc_normint_fixedpot_conditionalr(qdf,indx,normintstuff,params,npops,
                                       options,
                                       logoutfrac,jrs,lzs,jzs,normsrs,normszs,
                                       rgs,kappas,nus,Omegas,fqdf=1.,
                                       dblexp=False,dblexphr=None,dblexphz=None):
    """Calculate the normalization integral"""
    if options.mcall or options.mcwdf: #evaluation is the same for these
        raise NotImplementedError("mcall and mcwdf not implemented for fixed potential")
    else:
        return calc_normint_mcv_fixedpot_conditionalr(qdf,indx,normintstuff,
                                                      params,npops,
                                                      options,
                                                      logoutfrac,
                                                      jrs,lzs,jzs,
                                                      normsrs,normszs,
                                                      rgs,kappas,nus,Omegas,fqdf,
                                                      dblexp,dblexphr,dblexphz)

def calc_normint_mcv_fixedpot_conditionalr(qdf,indx,normintstuff,params,npops,
                                           options,
                                           logoutfrac,jrs,lzs,jzs,
                                           normsrs,normszs,
                                           rgs,kappas,nus,Omegas,fqdf,dblexp,
                                           dblexphr,dblexphz):
    """calculate the normalization integral by monte carlo integrating over v, but grid integrating over everything else"""
    thisnormintstuff= normintstuff[indx]
    if _PRECALCVSAMPLES:
        raise NotImplementedError
        sf, plates,platel,plateb,platebright,platefaint,grmin,grmax,rmin,rmax,fehmin,fehmax,feh,colordist,fehdist,gr,rhogr,rhofeh,mr,dmin,dmax,ds, surfscale, hr, hz, colorfehfac,normR, normZ,surfnrs, surfnzs, surfRgrid, surfzgrid, surfvrs, surfvts, surfvzs= unpack_normintstuff(thisnormintstuff,options)
    else:
        sf, plates,platel,plateb,platebright,platefaint,grmin,grmax,rmin,rmax,fehmin,fehmax,feh,colordist,fehdist,gr,rhogr,rhofeh,mr,dmin,dmax,ds, surfscale, hr, hz, colorfehfac, normR, normZ, conditional_d, conditional_R, conditional_z, conditional_colorfehfac, conditional_rderivd= unpack_normintstuff(thisnormintstuff,options)
    out= 0.
    ro= get_ro(params,options)
#    outfrac= get_outfrac(params,indx,options)
    outfrac= numpy.exp(logoutfrac)
    halodens= ro*outDens(1.,0.,None)
    globalInterp= True
    #start= time.time()
    if globalInterp and not fqdf == 0.:
        nrs, nzs= _SURFNRS, _SURFNZS
        thisRmin, thisRmax= 4./_REFR0, 15./_REFR0
        thiszmin, thiszmax= 0., .8
        Rgrid= numpy.linspace(thisRmin,thisRmax,nrs)
        zgrid= numpy.linspace(thiszmin,thiszmax,nzs)
        surfgrid= numpy.empty((nrs,nzs))
        if not options.multi is None:
            multOut= multi.parallel_map((lambda x: _calc_surfgrid_actions(Rgrid[x],
                                                                       zgrid,nzs,
                                                                       options,qdf,jrs[x,:,:],lzs[x,:,:],jzs[x,:,:],normsrs[x,:],normszs[x,:],
                                                                          rgs[x,:,:],kappas[x,:,:],nus[x,:,:],Omegas[x,:,:])),
                                        range(nrs),
                                        numcores=numpy.amin([nrs,
                                                             multiprocessing.cpu_count(),
                                                             options.multi]))
            for ii in range(nrs):
                surfgrid[ii,:]= multOut[ii]
        else:
            for ii in range(nrs):
                for jj in range(nzs):
                    surfgrid[ii,jj]= qdf.density(Rgrid[ii],zgrid[jj],
                                                 ngl=options.ngl,
                                                 nmc=options.nmcv,
                                                 _jr=jrs[ii,jj,:],
                                                 _lz=lzs[ii,jj,:],
                                                 _jz=jzs[ii,jj,:],
                                                 _rg=rgs[ii,jj,:],
                                                 _kappa=kappas[ii,jj,:],
                                                 _nu=nus[ii,jj,:],
                                                 _Omega=Omegas[ii,jj,:],
                                                 _sigmaR1=normsrs[ii,jj],
                                                 _sigmaz1=normszs[ii,jj])
        if _SURFSUBTRACTEXPON:
            Rs= numpy.tile(Rgrid,(nzs,1)).T
            Zs= numpy.tile(zgrid,(nrs,1))
            ehr= qdf.estimate_hr(1.,z=0.125)
#            ehz= qdf.estimate_hz(1.,zmin=0.5,zmax=0.7)#Get large z behavior right
            ehz= qdf.estimate_hz(1.,z=0.125)
            surfInterp= interpolate.RectBivariateSpline(Rgrid,zgrid,
                                                        numpy.log(surfgrid)
                                                        +Rs/ehr+numpy.fabs(Zs)/ehz,
                                                        kx=3,ky=3,
                                                        s=0.)
            #s=10.*float(nzs*nrs))
        else:
            surfInterp= interpolate.RectBivariateSpline(Rgrid,zgrid,
                                                        numpy.log(surfgrid),
                                                        kx=3,ky=3,
                                                        s=0.)
#                                                        s=10.*float(nzs*nrs))
    if options.mcvalt:
        #Alternative manner that uses well-tested compareDataModel code
        if _SURFSUBTRACTEXPON and not fqdf == 0.:
            compare_func= lambda x,y,du: numpy.exp(surfInterp.ev(x/ro/_REFR0,numpy.fabs(y)/ro/_REFR0)-x/ro/_REFR0/ehr-numpy.fabs(y)/ehz/ro/_REFR0)+outfrac*halodens
        elif not fqdf == 0.:
            compare_func= lambda x,y,du: numpy.exp(surfInterp.ev(x/ro/_REFR0,numpy.fabs(y)/ro/_REFR0))+outfrac*halodens
        elif dblexp: #just do the outliers
            compare_func= lambda x,y,du: outfrac*_DblExpDensity(x,y,[numpy.log(dblexphz/1000.),-numpy.log(dblexphr)])
        else: #just do the outliers
            compare_func= lambda x,y,du: outfrac*halodens
        ndata= conditional_d.shape[0]
        nplates= conditional_d.shape[1]
        out= numpy.zeros((ndata,nplates))
        for kk in range(nplates):
            out[:,kk]= conditional_d[:,kk]**2.*compare_func(conditional_R,
                                                            conditional_z[:,kk],
                                                            None)*conditional_colorfehfac[:,kk]/conditional_rderivd[:,kk]
        vo= get_vo(params,options,npops)
        return numpy.reshape(numpy.nansum(out,axis=1)*vo**3.,(ndata,1))

def _calc_surfgrid_actions(R,zgrid,nzs,options,qdf,
                           jrs,lzs,jzs,normsrs,normszs,
                           rgs,kappas,nus,Omegas):
    out= numpy.zeros(nzs)
    for jj in range(nzs):
        out[jj]= qdf.density(R,zgrid[jj],
                             ngl=options.ngl,
                             nmc=options.nmcv,
                             _jr=jrs[jj,:],
                             _lz=lzs[jj,:],
                             _jz=jzs[jj,:],
                             _rg=rgs[jj,:],
                             _kappa=kappas[jj,:],
                             _nu=nus[jj,:],
                             _Omega=Omegas[jj,:],
                             _sigmaR1=normsrs[jj],
                             _sigmaz1=normszs[jj])
    return out

def setup_normintstuff(options,raw,binned,fehs,afes,allraw):
    """Gather everything necessary for calculating the normalization integral"""
    if not options.savenorm is None and os.path.exists(options.savenorm):
        print "Reading normintstuff from file ..."
        savefile= open(options.savenorm,'rb')
        out= pickle.load(savefile)
        savefile.close()
        return out
    #Load selection function
    plates= numpy.array(list(set(list(allraw.plate))),dtype='int') #Only load plates that we use
    print "Using %i plates, %i stars ..." %(len(plates),len(raw))
    sf= segueSelect(plates=plates,type_faint='tanhrcut',
                    sample=options.sample,type_bright='tanhrcut',
                    sn=options.snmin,select=options.select,
                    indiv_brightlims=options.indiv_brightlims)
    plates= sf.plates #To make sure that these are matched up
    platelb= bovy_coords.radec_to_lb(sf.platestr.ra,sf.platestr.dec,
                                     degree=True)
    indx= [not 'faint' in name for name in sf.platestr.programname]
    platebright= numpy.array(indx,dtype='bool')
    indx= ['faint' in name for name in sf.platestr.programname]
    platefaint= numpy.array(indx,dtype='bool')
    if options.sample.lower() == 'g':
        grmin, grmax= 0.48, 0.55
        rmin,rmax= 14.5, 20.2
    if options.sample.lower() == 'k':
        grmin, grmax= 0.55, 0.75
        rmin,rmax= 14.5, 19.
    colorrange=[grmin,grmax]
    out= []
    mapfehs= monoAbundanceMW.fehs(k=(options.sample.lower() == 'k'))
    mapafes= monoAbundanceMW.afes(k=(options.sample.lower() == 'k'))
    if not options.multi is None:
        #Generate list of temporary files
        tmpfiles= []
        for ii in range(len(fehs)): tmpfiles.append(tempfile.mkstemp())
        try:
            dummy= multi.parallel_map((lambda x: indiv_setup_normintstuff(x,
                                                                          options,raw,binned,fehs,afes,
                                                                          plates,sf,platelb,
                                                                          platebright,platefaint,grmin,grmax,rmin,rmax,colorrange,mapfehs,mapafes,
                                                                          True,
                                                                          tmpfiles)),
                                      
                                      range(len(fehs)),
                                      numcores=numpy.amin([len(fehs),
                                                           multiprocessing.cpu_count(),
                                                           options.multi]))
            #Now read all of the temporary files
            for ii in range(len(fehs)):
                tmpfile= open(tmpfiles[ii][1],'rb')
                thisnormintstuff= pickle.load(tmpfile)
                tmpfile.close()
                out.append(thisnormintstuff)
        finally:
            for ii in range(len(fehs)):
                os.remove(tmpfiles[ii][1])
    else:
        for ii in range(len(fehs)):
            thisnormintstuff= indiv_setup_normintstuff(ii,
                                                       options,raw,binned,
                                                       fehs,afes,
                                                       plates,sf,platelb,
                                                       platebright,platefaint,
                                                       grmin,grmax,rmin,rmax,
                                                       colorrange,
                                                       mapfehs,mapafes,
                                                       False,None)
            out.append(thisnormintstuff)
    if not options.savenorm is None:
        savefile= open(options.savenorm,'wb')
        pickle.dump(out,savefile)
        savefile.close()
    return out

def indiv_setup_normintstuff(ii,options,raw,binned,fehs,afes,plates,sf,platelb,
                             platebright,platefaint,grmin,grmax,rmin,rmax,
                             colorrange,mapfehs,mapafes,savetopickle,tmpfiles):
    data= binned(fehs[ii],afes[ii])
    thisnormintstuff= normintstuffClass() #Empty object to act as container
    #Fit this data, set up feh and color
    feh= fehs[ii]
    fehrange= [feh-options.dfeh/2.,feh+options.dfeh/2.]
    #FeH
    fehdist= DistSpline(*numpy.histogram(data.feh,bins=5,
                                         range=fehrange),
                         xrange=fehrange,dontcuttorange=False)
    #Color
    colordist= DistSpline(*numpy.histogram(data.dered_g\
                                               -data.dered_r,
                                           bins=9,range=colorrange),
                           xrange=colorrange)
    if options.mcall:
        #Find nearest mono-abundance bin that has a measurement
        abindx= numpy.argmin((fehs[ii]-mapfehs)**2./0.01 \
                                 +(afes[ii]-mapafes)**2./0.0025)
        thishr= monoAbundanceMW.hr(mapfehs[abindx],mapafes[abindx],
                                   k=(options.sample.lower() == 'k'))
        thishz= monoAbundanceMW.hz(mapfehs[abindx],mapafes[abindx],
                                   k=(options.sample.lower() == 'k'))/1000.
        #Calculate the r-distribution for each plate
        nrs= 1001
        ngr, nfeh= 11, 11 #BOVY: INCREASE?
        tgrs= numpy.linspace(grmin,grmax,ngr)
        tfehs= numpy.linspace(fehrange[0],fehrange[1],nfeh)
        #Calcuate FeH and gr distriutions
        fehdists= numpy.zeros(nfeh)
        for jj in range(nfeh): fehdists[jj]= fehdist(tfehs[jj])
        fehdists= numpy.cumsum(fehdists)
        fehdists/= fehdists[-1]
        colordists= numpy.zeros(ngr)
        for jj in range(ngr): colordists[jj]= colordist(tgrs[jj])
        colordists= numpy.cumsum(colordists)
        colordists/= colordists[-1]
        rs= numpy.linspace(rmin,rmax,nrs)
        rdists= numpy.zeros((len(sf.plates),nrs,ngr,nfeh))
        if options.mcout:
            fidoutfrac= 0.025
            rdistsout= numpy.zeros((len(sf.plates),nrs,ngr,nfeh))
        for jj in range(len(sf.plates)):
            p= sf.plates[jj]
            sys.stdout.write('\r'+"Working on plate %i (%i/%i)" % (p,jj+1,len(sf.plates)))
            sys.stdout.flush()
            rdists[jj,:,:,:]= _predict_rdist_plate(rs,
                                                   lambda x,y,z: fidDens(x,y,thishr,thishz,z),
                                                   None,rmin,rmax,
                                                   platelb[jj,0],platelb[jj,1],
                                                   grmin,grmax,
                                                   fehrange[0],fehrange[1],feh,
                                                   colordist,
                                                   fehdist,sf,sf.plates[jj],
                                                   dontmarginalizecolorfeh=True,
                                                   ngr=ngr,nfeh=nfeh)
            if options.mcout:
                rdistsout[jj,:,:,:]= _predict_rdist_plate(rs,
                                                          lambda x,y,z: outDens(x,y,z),
                                                          None,rmin,rmax,
                                                          platelb[jj,0],platelb[jj,1],
                                                          grmin,grmax,
                                                          fehrange[0],fehrange[1],feh,
                                                          colordist,
                                                          fehdist,sf,sf.plates[jj],
                                                          dontmarginalizecolorfeh=True,
                                                          ngr=ngr,nfeh=nfeh)
        sys.stdout.write('\r'+_ERASESTR+'\r')
        sys.stdout.flush()
        numbers= numpy.sum(rdists,axis=3)
        numbers= numpy.sum(numbers,axis=2)
        numbers= numpy.sum(numbers,axis=1)
        numbers= numpy.cumsum(numbers)
        if options.mcout:
            totfid= numbers[-1]
        numbers/= numbers[-1]
        rdists= numpy.cumsum(rdists,axis=1)
        for ll in range(len(sf.plates)):
            for jj in range(ngr):
                for kk in range(nfeh):
                    rdists[ll,:,jj,kk]/= rdists[ll,-1,jj,kk]
        if options.mcout:
            numbersout= numpy.sum(rdistsout,axis=3)
            numbersout= numpy.sum(numbersout,axis=2)
            numbersout= numpy.sum(numbersout,axis=1)
            numbersout= numpy.cumsum(numbersout)
            totout= fidoutfrac*numbersout[-1]
            totnumbers= totfid+totout
            totfid/= totnumbers
            totout/= totnumbers
            #print totfid, totout
            numbersout/= numbersout[-1]
            rdistsout= numpy.cumsum(rdistsout,axis=1)
            for ll in range(len(sf.plates)):
                for jj in range(ngr):
                    for kk in range(nfeh):
                        rdistsout[ll,:,jj,kk]/= rdistsout[ll,-1,jj,kk]
        #Now sample until we're done
        thisout= []
        while len(thisout) < options.nmc:
            #First sample a plate
            ran= numpy.random.uniform()
            kk= 0
            while numbers[kk] < ran: kk+= 1
            #Also sample a Feh and a color
            ran= numpy.random.uniform()
            ff= 0
            while fehdists[ff] < ran: ff+= 1
            ran= numpy.random.uniform()
            cc= 0
            while colordists[cc] < ran: cc+= 1
            #plate==kk, feh=ff,color=cc; now sample from the rdist of this plate
            ran= numpy.random.uniform()
            jj= 0
            if options.mcout and numpy.random.uniform() < totout: #outlier
                while rdistsout[kk,jj,cc,ff] < ran: jj+= 1
                thisoutlier= True
            else:
                while rdists[kk,jj,cc,ff] < ran: jj+= 1
                thisoutlier= False
            #r=jj
            thisout.append([rs[jj],tgrs[cc],tfehs[ff],platelb[kk,0],platelb[kk,1],
                            sf.plates[kk],
                            _ivezic_dist(tgrs[cc],rs[jj],tfehs[ff]),
                            thisoutlier])
        #Add mock velocities
        #First calculate all R
        d= numpy.array([o[6] for o in thisout])
        l= numpy.array([o[3] for o in thisout])
        b= numpy.array([o[4] for o in thisout])
        XYZ= bovy_coords.lbd_to_XYZ(l,b,d,degree=True)
        R= ((_REFR0-XYZ[:,0])**2.+XYZ[:,1]**2.)**(0.5)
        XYZ[:,2]+= _ZSUN
        z= XYZ[:,2]
        phi= numpy.arcsin(XYZ[:,1]/R)
        phi[(_REFR0-XYZ[:,0] < 0.)]= numpy.pi-phi[(_REFR0-XYZ[:,0] < 0.)]
        for jj in range(options.nmc):
            sigz= monoAbundanceMW.sigmaz(mapfehs[abindx],
                                         mapafes[abindx],
                                         r=R[jj],
                                         k=(options.sample.lower() == 'k'))
            sigr= 2.*sigz #BOVY: FOR NOW
            sigphi= sigr/numpy.sqrt(2.) #BOVY: FOR NOW
            #Estimate asymmetric drift
            va= sigr**2./2./_REFV0\
                *(-.5+R[jj]*(1./thishr+2./7.))
            if options.mcout and thisout[jj][7]:
                #Sample from halo gaussian
                vz= numpy.random.normal()*_SZHALO
                vr= numpy.random.normal()*_SRHALO
                vphi= numpy.random.normal()*_SPHIHALO
            else:
                #Sample from disk gaussian
                vz= numpy.random.normal()*sigz
                vr= numpy.random.normal()*sigr
                vphi= numpy.random.normal()*sigphi+_REFV0-va
            #Calculate observables
            vx,vy,vz= bovy_coords.galcencyl_to_vxvyvz(vr,vphi,vz,phi[jj],
                                                      vsun=[_VRSUN,_VTSUN,
                                                            _VZSUN])
            vrpmllpmbb= bovy_coords.vxvyvz_to_vrpmllpmbb(vx,vy,vz,l[jj],b[jj],
                                                         d[jj],
                                                         XYZ=False,degree=True)
            #Append to out
            if options.mcout:
                fidlogeval= numpy.log(fidDens(R[jj],z[jj],thishr,thishz,
                                              None))\
                                              -numpy.log(sigr)\
                                              -numpy.log(sigphi)\
                                              -numpy.log(sigz)\
                                              -0.5*(vr**2./sigr**2.+vz**2./sigz**2.+(vphi-_REFV0+va)**2./sigphi**2.)
                outlogeval= numpy.log(fidoutfrac)\
                    +numpy.log(outDens(R[jj],z[jj],None))\
                    -numpy.log(_SRHALO)\
                    -numpy.log(_SPHIHALO)\
                    -numpy.log(_SZHALO)\
                    -0.5*(vr**2./_SRHALO**2.+vz**2./_SZHALO**2.+vphi**2./_SPHIHALO**2.)
#                thisout[jj].extend([vr,vphi,vz,#next is evaluation of f at mock
                thisout[jj].extend([vrpmllpmbb[0],vrpmllpmbb[1],vrpmllpmbb[2],#next is evaluation of f at mock
                                    logsumexp([fidlogeval,outlogeval])])
            else:
#                thisout[jj].extend([vr,vphi,vz,#next is evaluation of f at mock
                thisout[jj].extend([vrpmllpmbb[0],vrpmllpmbb[1],vrpmllpmbb[2],#next is evaluation of f at mock
                                    numpy.log(fidDens(R[jj],z[jj],thishr,thishz,None))\
                                        -numpy.log(sigr)-numpy.log(sigphi)-numpy.log(sigz)-0.5*(vr**2./sigr**2.+vz**2./sigz**2.+(vphi-_REFV0+va)**2./sigphi**2.)])
        #Load into thisnormintstuff
        thisnormintstuff.mock= thisout
        if savetopickle:
            #Save to temporary pickle
            tmpfile= open(tmpfiles[ii][1],'wb')
            pickle.dump(thisnormintstuff,tmpfile)
            tmpfile.close()
        else:
            return thisnormintstuff
    elif options.mcwdf:
        from fakeDFData import fakeDFData #Needs to be here to avoid recursion
        #Setup default qdf, pot, and aA
        normintparams= initialize(options,fehs,afes)
        normintpot= setup_potential(normintparams,options,len(fehs))
        normintaA= setup_aA(normintpot,options)
        normintdfparams= get_dfparams(normintparams,ii,options,log=False)
        normintvo= get_vo(normintparams,options,len(fehs))
        normintro= get_ro(normintparams,options)
        if options.dfmodel.lower() == 'qdf':
            #Normalize
            norminthr= normintdfparams[0]/normintro
            normintsr= normintdfparams[1]/normintvo
            normintsz= normintdfparams[2]/normintvo
            norminthsr= normintdfparams[3]/normintro
            norminthsz= normintdfparams[4]/normintro
            #Setup
            normintqdf= quasiisothermaldf(norminthr,normintsr,
                                          normintsz,norminthsr,
                                          norminthsz,
                                          pot=normintpot,aA=normintaA,
                                          cutcounter=True)
        
        thisnormintstuff.mock= fakeDFData(None,normintqdf,
                                          ii,normintparams,fehs,afes,
                                          options,rmin,rmax,platelb,
                                          grmin,grmax,
                                          fehrange,colordist,
                                          fehdist,feh,sf,
                                          mapfehs,mapafes,
                                          ndata=options.nmc,returnlist=True)
        if savetopickle:
            #Save to temporary pickle
            tmpfile= open(tmpfiles[ii][1],'wb')
            pickle.dump(thisnormintstuff,tmpfile)
            tmpfile.close()
        else:
            return thisnormintstuff
    else:
        #Integration grid when binning
        grs= numpy.linspace(grmin,grmax,_NGR)
        fehsgrid= numpy.linspace(fehrange[0],fehrange[1],_NFEH)
        rhogr= numpy.array([colordist(gr) for gr in grs])
        rhofeh= numpy.array([fehdist(tfeh) for tfeh in fehsgrid])
        mr= numpy.zeros((_NGR,_NFEH))
        for kk in range(_NGR):
            for ll in range(_NFEH):
                mr[kk,ll]= _mr_gi(_gi_gr(grs[kk]),fehsgrid[ll])
        #determine dmin and dmax
        allbright, allfaint= True, True
        #dmin and dmax for this rmin, rmax
        if not options.fixro is None:
            ro= options.fixro
        else:
            ro= 1.
        colorfehfac= numpy.zeros((len(sf.plates),_NDS))
        normR= numpy.zeros((len(sf.plates),_NDS))
        normZ= numpy.zeros((len(sf.plates),_NDS))
        #Zmin and Zmax for this rmin, rmax
        bs= []
        allbright, allfaint= False, False
        for p in sf.plates:
           #l and b?
            pindx= (sf.plates == p)
            plateb= platelb[pindx,1][0]
            bs.append(plateb)
            if 'faint' in sf.platestr[pindx].programname[0]:
                allbright= False
            else:
                allfaint= False
        bs= numpy.array(bs)
        bmin, bmax= numpy.amin(numpy.fabs(bs)),numpy.amax(numpy.fabs(bs))
        if allbright:
            thisrmin, thisrmax= rmin, 17.8
        elif allfaint:
            thisrmin, thisrmax= 17.8, rmax
        else:
            thisrmin, thisrmax= rmin, rmax
        _THISNGR, _THISNFEH= 51, 51
        tgrs= numpy.zeros((_THISNGR,_THISNFEH))
        tfehs= numpy.zeros((_THISNGR,_THISNFEH))
        for kk in range(_THISNGR):
            tfehs[kk,:]= numpy.linspace(fehrange[0],fehrange[1],_THISNFEH)
        for kk in range(_THISNFEH):
            tgrs[:,kk]= numpy.linspace(grmin,grmax,_THISNGR)
        if not options.fixdm is None:
            distfac= 10.**(options.fixdm/5.)
        else:
            distfac= 1.
        dmin= numpy.amin(_ivezic_dist(tgrs,thisrmin,tfehs))*distfac
        dmax= numpy.amax(_ivezic_dist(tgrs,thisrmax,tfehs))*distfac
        if options.zmax is None:
            zmax= dmax*numpy.sin(bmax*_DEGTORAD)
        zmin= dmin*numpy.sin(bmin*_DEGTORAD)
        zmin-= 2.*_ZSUN #Just to be sure we have the South covered
        zs= numpy.linspace(zmin,zmax,_NDS)
        _THISNGR, _THISNFEH= 51, 51
        thisgrs= numpy.zeros((_THISNGR,_THISNFEH))
        thisfehs= numpy.zeros((_THISNGR,_THISNFEH))
        for kk in range(_THISNGR):
            thisfehs[kk,:]= numpy.linspace(fehrange[0],fehrange[1],_THISNFEH)
        for kk in range(_THISNFEH):
            thisgrs[:,kk]= numpy.linspace(grmin,grmax,_THISNGR)
        dmin= numpy.amin(_ivezic_dist(thisgrs,thisrmin,thisfehs))*distfac
        dmax= numpy.amax(_ivezic_dist(thisgrs,thisrmax,thisfehs))*distfac
        ds= numpy.linspace(dmin,dmax,_NDS)     
        grs= numpy.linspace(grmin,grmax,_NGR)
        tfehs= fehsgrid
        for pindx in range(len(plates)):
            tmpout= numpy.zeros(len(zs))
            l= platelb[pindx,0]
            b= platelb[pindx,1]
            if b > 0.:
                tds= (zs-_ZSUN)/numpy.fabs(numpy.sin(b*_DEGTORAD))
            else:
                tds= (zs+_ZSUN)/numpy.fabs(numpy.sin(b*_DEGTORAD))
            norm= 0.
            logds= 5.*numpy.log10(tds/distfac)+10.
            for kk in range(len(tfehs)):
                for jj in range(len(grs)):
                    #What rs do these zs correspond to
                    gi= _gi_gr(grs[jj])
                    mr= _mr_gi(gi,tfehs[kk])
                    rs= logds+mr
                    select= numpy.array(sf(sf.plates[pindx],r=rs))
                    tmpout+= colordist(grs[jj])*fehdist(tfehs[kk])\
                        *select
                    norm+= colordist(grs[jj])*fehdist(tfehs[kk])
            colorfehfac[pindx,:]= tmpout/norm
            #Calculate (R,z)s
            XYZ= bovy_coords.lbd_to_XYZ(numpy.array([l for kk in range(len(tds))]),
                                        numpy.array([b for kk in range(len(tds))]),
                                        tds,degree=True)
            normR[pindx,:]= ((_REFR0*ro-XYZ[:,0])**2.+XYZ[:,1]**2.)**(0.5)
            normZ[pindx,:]= XYZ[:,2]+_ZSUN
        #Determine scale over which the surface-mass density changes for this plate
        #Find nearest mono-abundance bin that has a measurement
        abindx= numpy.argmin((fehs[ii]-mapfehs)**2./0.01 \
                                 +(afes[ii]-mapafes)**2./0.0025)
        thishr= monoAbundanceMW.hr(mapfehs[abindx],mapafes[abindx],
                                   k=(options.sample.lower() == 'k'))
        thishz= monoAbundanceMW.hz(mapfehs[abindx],mapafes[abindx],
                                   k=(options.sample.lower() == 'k'))/1000.
        surfscale= []
        surfds= numpy.linspace(dmin,dmax,11)
        for kk in range(len(plates)):
            if options.dfmodel.lower() == 'qdf':
                XYZ= bovy_coords.lbd_to_XYZ(numpy.array([platelb[kk,0] for dd in range(len(surfds))]),
                                            numpy.array([platelb[kk,1] for dd in range(len(surfds))]),
                                            surfds,degree=True)
                R= ((_REFR0-XYZ[:,0])**2.+XYZ[:,1]**2.)**(0.5)
                XYZ[:,2]+= _ZSUN
                z= XYZ[:,2]
                drdd= -(_REFR0-XYZ[:,0])/R*numpy.cos(platelb[kk,0]*_DEGTORAD)*numpy.cos(platelb[kk,1]*_DEGTORAD)\
                        +XYZ[:,1]/R*numpy.cos(platelb[kk,1]*_DEGTORAD)*numpy.sin(platelb[kk,0]*_DEGTORAD)
                dzdd= numpy.fabs(numpy.sin(platelb[kk,1]*_DEGTORAD))
                dlnnudd= -1./thishr*drdd-1./thishz*dzdd
                surfscale.append(numpy.amin(numpy.fabs(1./dlnnudd)))
            #print thishr, thishz, platelb[kk,0], platelb[kk,1], surfscale[-1]
        #Load into thisnormintstuff
        thisnormintstuff.sf= sf
        thisnormintstuff.plates= plates
        thisnormintstuff.platel= platelb[:,0]
        thisnormintstuff.plateb= platelb[:,1]
        thisnormintstuff.platebright= platebright
        thisnormintstuff.platefaint= platefaint
        thisnormintstuff.grmin= grmin
        thisnormintstuff.grmax= grmax
        thisnormintstuff.rmin= rmin
        thisnormintstuff.rmax= rmax
        thisnormintstuff.fehmin= fehrange[0]
        thisnormintstuff.fehmax= fehrange[1]
        thisnormintstuff.feh= feh
        thisnormintstuff.colordist= colordist
        thisnormintstuff.fehdist= fehdist
        thisnormintstuff.grs= grs
        thisnormintstuff.rhogr= rhogr
        thisnormintstuff.rhofeh= rhofeh        
        thisnormintstuff.mr= mr
        thisnormintstuff.dmin= dmin
        thisnormintstuff.dmax= dmax
        thisnormintstuff.ds= ds
        thisnormintstuff.surfscale= surfscale
        thisnormintstuff.hr= thishr
        thisnormintstuff.hz= thishz
        thisnormintstuff.colorfehfac= colorfehfac
        thisnormintstuff.normR= normR
        thisnormintstuff.normZ= normZ
        if options.conditionalr:
            #Pre-calculate for each datapoint and plate what the distance on that plate is for a given R etc.
            #First calculate R
            R= numpy.sqrt((_REFR0-data.xc)**2.+data.yc**2.)
            ndata= len(R)
            conditional_d= numpy.zeros((ndata,len(plates)))+numpy.nan
            conditional_z= numpy.zeros((ndata,len(plates)))
            conditional_colorfehfac= numpy.zeros((ndata,len(plates)))
            conditional_rderivd= numpy.zeros((ndata,len(plates)))
            for kk in range(len(plates)):
                #Calculate d, first calculate the Galactocentric azimuth, then 
                #calculate d
                thisl= platelb[kk,0]*numpy.ones(ndata)
                thisb= platelb[kk,1]*numpy.ones(ndata)
                #If this is the plate a star is on, use the star's l
                indx= (data.plate == plates[kk])
                thisl[indx]= data[indx].l
                thisb[indx]= data[indx].b
                #Outer disk plates do not have R < R_0
                oindx= (thisl >= 90.)*(thisl <= 270.)*(R < _REFR0)
                #Outer disk plate
                conditional_d[oindx,kk]= 0.
                conditional_z[oindx,kk]= 0.
                conditional_rderivd[oindx,kk]= 1.
                conditional_colorfehfac[oindx,kk]= 0.
                #Inner disk plates do not have R > R_0 or R < Rmin
                minR= _REFR0*numpy.fabs(numpy.sin(thisl*_DEGTORAD))
                oindx= (True-(thisl > 90.)*(thisl < 270.))*(R > _REFR0)
                conditional_d[oindx,kk]= 0.
                conditional_z[oindx,kk]= 0.
                conditional_rderivd[oindx,kk]= 1.
                conditional_colorfehfac[oindx,kk]= 0.
                oindx= (True-(thisl > 90.)*(thisl < 270.))*(R < minR)
                conditional_d[oindx,kk]= 0.
                conditional_z[oindx,kk]= 0.
                conditional_rderivd[oindx,kk]= 1.
                conditional_colorfehfac[oindx,kk]= 0.
                indx= numpy.isnan(conditional_d[:,kk])
                sinphil= _REFR0/R[indx]*numpy.sin(thisl[indx]*_DEGTORAD)
                phil= numpy.arcsin(sinphil)
                phi= numpy.pi-phil-thisl[indx]*_DEGTORAD
                d= R[indx]*numpy.sin(phi)/numpy.sin(thisl[indx]*_DEGTORAD)
                #For safety, also copy in the distances from the data if this is the data's plate
                plateindx= (data.plate == plates[kk])
                conditional_d[indx,kk]= d
                conditional_d[plateindx,kk]= data[plateindx].dist
                indx= ( conditional_d[:,kk] <= dmax ) \
                    * ( conditional_d[:,kk] >= dmin )
                badindx= True-indx
                conditional_z[badindx,kk]= 0.
                conditional_rderivd[badindx,kk]= 1.
                conditional_colorfehfac[badindx,kk]= 0.
                conditional_rderivd[indx,kk]= numpy.fabs(1./R[indx]*(conditional_d[indx,kk] * numpy.cos(thisb[indx]*_DEGTORAD)**2.-_REFR0*numpy.cos(thisl[indx]*_DEGTORAD)*numpy.cos(thisb[indx]*_DEGTORAD)))
                conditional_z[indx,kk]= conditional_d[indx,kk]*numpy.sin(thisb[indx]*_DEGTORAD)-_ZSUN
                theselogds= 5.*numpy.log10(conditional_d[indx,kk]/distfac)+10.
                tmpout= numpy.zeros(len(theselogds))
                norm= 0.
                for ll in range(len(tfehs)):
                    for jj in range(len(grs)):
                        #What rs do these zs correspond to
                        gi= _gi_gr(grs[jj])
                        mr= _mr_gi(gi,tfehs[ll])
                        rs= theselogds+mr
                        select= numpy.array(sf(plates[kk],r=rs))
                        tmpout+= colordist(grs[jj])*fehdist(tfehs[ll])\
                            *select
                        norm+= colordist(grs[jj])*fehdist(tfehs[ll])
                conditional_colorfehfac[indx,kk]= tmpout/norm
            thisnormintstuff.conditional_d= conditional_d
            thisnormintstuff.conditional_R= R
            thisnormintstuff.conditional_z= conditional_z
            thisnormintstuff.conditional_colorfehfac= conditional_colorfehfac
            thisnormintstuff.conditional_rderivd= conditional_rderivd
        if _PRECALCVSAMPLES:
            normintparams= initialize(options,fehs,afes)
            normintpot= setup_potential(normintparams,options,len(fehs))
            normintaA= setup_aA(normintpot,options)
            normintdfparams= get_dfparams(normintparams,ii,options,log=False)
            normintvo= get_vo(normintparams,options,len(fehs))
            normintro= get_ro(normintparams,options)
            if options.dfmodel.lower() == 'qdf':
               #Normalize
                norminthr= normintdfparams[0]/normintro
                normintsr= normintdfparams[1]/normintvo
                normintsz= normintdfparams[2]/normintvo
                norminthsr= normintdfparams[3]/normintro
                norminthsz= normintdfparams[4]/normintro
            #Setup
            normintqdf= quasiisothermaldf(norminthr,normintsr,
                                          normintsz,norminthsr,
                                          norminthsz,
                                          pot=normintpot,aA=normintaA,
                                          cutcounter=True)
            nrs, nzs= _SURFNRS, _SURFNZS
            thisrmin, thisrmax= 4./_REFR0, 15./_REFR0
            thiszmin, thiszmax= 0., .8
            Rgrid= numpy.linspace(thisrmin,thisrmax,nrs)
            zgrid= numpy.linspace(thiszmin,thiszmax,nzs)
            vrs= []
            vts= []
            vzs= []
            for jj in range(nrs):
                for kk in range(nzs):
                    dummy, thisvrs, thisvts, thisvzs= normintqdf.vmomentdensity(Rgrid[jj],zgrid[kk],0.,0.,0.,
                                                                             nmc=options.nmcv,
                                                                             _returnmc=True,_rawgausssamples=True)
                    vrs.append(thisvrs)
                    vts.append(thisvts)
                    vzs.append(thisvzs)
            thisnormintstuff.surfnrs= nrs
            thisnormintstuff.surfnzs= nzs
            thisnormintstuff.surfRgrid= Rgrid
            thisnormintstuff.surfzgrid= zgrid
            thisnormintstuff.surfvrs= vrs
            thisnormintstuff.surfvts= vts
            thisnormintstuff.surfvzs= vzs
        if savetopickle:
            #Save to temporary pickle
            tmpfile= open(tmpfiles[ii][1],'wb')
            pickle.dump(thisnormintstuff,tmpfile)
            tmpfile.close()
        else:
            return thisnormintstuff

def unpack_normintstuff(normintstuff,options):
    if options.mcall or options.mcwdf:
        return normintstuff.mock
    else:
        if _PRECALCVSAMPLES:
            return (normintstuff.sf,
                    normintstuff.plates,
                    normintstuff.platel,
                    normintstuff.plateb,
                    normintstuff.platebright,
                    normintstuff.platefaint,
                    normintstuff.grmin,
                    normintstuff.grmax,
                    normintstuff.rmin,
                    normintstuff.rmax,
                    normintstuff.fehmin,
                    normintstuff.fehmax,
                    normintstuff.feh,
                    normintstuff.colordist,
                    normintstuff.fehdist,
                    normintstuff.grs,
                    normintstuff.rhogr,
                    normintstuff.rhofeh,
                    normintstuff.mr,
                    normintstuff.dmin,
                    normintstuff.dmax,
                    normintstuff.ds,
                    normintstuff.surfscale,
                    normintstuff.hr,
                    normintstuff.hz,
                    normintstuff.colorfehfac,
                    normintstuff.normR,
                    normintstuff.normZ,
                    normintstuff.surfnrs,
                    normintstuff.surfnzs,
                    normintstuff.surfRgrid,
                    normintstuff.surfzgrid,
                    normintstuff.surfvrs,
                    normintstuff.surfvts,
                    normintstuff.surfvzs)
        elif options.conditionalr:
            return (normintstuff.sf,
                    normintstuff.plates,
                    normintstuff.platel,
                    normintstuff.plateb,
                    normintstuff.platebright,
                    normintstuff.platefaint,
                    normintstuff.grmin,
                    normintstuff.grmax,
                    normintstuff.rmin,
                    normintstuff.rmax,
                    normintstuff.fehmin,
                    normintstuff.fehmax,
                    normintstuff.feh,
                    normintstuff.colordist,
                    normintstuff.fehdist,
                    normintstuff.grs,
                    normintstuff.rhogr,
                    normintstuff.rhofeh,
                    normintstuff.mr,
                    normintstuff.dmin,
                    normintstuff.dmax,
                    normintstuff.ds,
                    normintstuff.surfscale,
                    normintstuff.hr,
                    normintstuff.hz,
                    normintstuff.colorfehfac,
                    normintstuff.normR,
                    normintstuff.normZ,
                    normintstuff.conditional_d,
                    normintstuff.conditional_R,
                    normintstuff.conditional_z,
                    normintstuff.conditional_colorfehfac,
                    normintstuff.conditional_rderivd)
        else:
            return (normintstuff.sf,
                    normintstuff.plates,
                    normintstuff.platel,
                    normintstuff.plateb,
                    normintstuff.platebright,
                    normintstuff.platefaint,
                    normintstuff.grmin,
                    normintstuff.grmax,
                    normintstuff.rmin,
                    normintstuff.rmax,
                    normintstuff.fehmin,
                    normintstuff.fehmax,
                    normintstuff.feh,
                    normintstuff.colordist,
                    normintstuff.fehdist,
                    normintstuff.grs,
                    normintstuff.rhogr,
                    normintstuff.rhofeh,
                    normintstuff.mr,
                    normintstuff.dmin,
                    normintstuff.dmax,
                    normintstuff.ds,
                    normintstuff.surfscale,
                    normintstuff.hr,
                    normintstuff.hz,
                    normintstuff.colorfehfac,
                    normintstuff.normR,
                    normintstuff.normZ)
        
class normintstuffClass:
    """Empty class to hold normalization integral necessities"""
    pass


##RUNNING SINGLE BINS IN A SINGLE CALL
def run_abundance_singles(options,args,fehs,afes):
    options.singles= False #First turn this off!
    savename= args[0]
    initname= options.init
    normname= options.savenorm
    restartname= options.restart
    if options.cluster:
        options.cluster= False
        start= 0
        stop= len(fehs)
        if options.batch1:
            stop= 31
        elif options.batch2:
            start= 31
        for ii in range(start,stop):
            run_abundance_singles_single_onCluster(options,args,fehs,afes,ii,
                                                   savename,initname,normname,
                                                   restartname)
    elif not options.multi is None:
        dummy= multi.parallel_map((lambda x: run_abundance_singles_single(options,args,fehs,afes,x,
                                                                          savename,initname,normname)),
                                  range(len(fehs)),
                                  numcores=numpy.amin([len(fehs),
                                                       multiprocessing.cpu_count(),
                                                       options.multi]))
    else:
        for ii in range(len(fehs)):
            run_abundance_singles_single(options,args,fehs,afes,ii,
                                         savename,initname,normname)
    return None

def unparse_cmd(options,args):
    out= ""
    for arg in args:
        out+= " "+arg
    parser= get_options()
    for opt in parser._long_opt.keys():
        if opt == '--help': continue
        dest= parser._long_opt[opt].dest
        opttype= parser._long_opt[opt].type
        action= parser._long_opt[opt].action
        if action.lower() == 'store_true' and options.__dict__[dest]:
            out+= " "+opt
        elif action.lower() == 'store_false' and not options.__dict__[dest]:
            out+= " "+opt
        elif action.lower() == 'store':
            if options.__dict__[dest] is None:
                continue
            elif opttype == 'int':
                out+= " "+opt+"=%i" % options.__dict__[dest]
            elif opttype == 'float':
                out+= " "+opt+"=%f" % options.__dict__[dest]
            elif opttype == 'string':
                out+= " "+opt+"=" + options.__dict__[dest]
    return out

def run_abundance_singles_single_onCluster(options,args,fehs,afes,ii,savename,
                                           initname,
                                           normname,
                                           restartname):
    #Prepare args and options
    spl= savename.split('.')
    newname= ''
    for jj in range(len(spl)-1):
        newname+= spl[jj]
        if not jj == len(spl)-2: newname+= '.'
    newname+= '_%i.' % ii
    newname+= spl[-1]
    args[0]= newname
    if options.gridall and os.path.exists(newname): #Fit is already done
        return None
    if numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii],
                                    k=(options.sample.lower() == 'k'))/8.) > -0.5:
        #Don't run, because we cannot model these populations with our model
        return None
    if options.mcsample and not initname is None:
        #Do the same for init
        spl= initname.split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        options.init= newname
    if not normname is None:
        #Do the same for init
        spl= normname.split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        options.savenorm= newname
    if not options.restart is None:
        #Do the same for restart
        spl= restartname.split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        options.restart= newname
    options.singlefeh= fehs[ii]
    options.singleafe= afes[ii]   
    #Now run
    cmd= unparse_cmd(options,args)
    print cmd
    cmd= "mpirun -x PYTHONPATH /home/bovy/local/bin/python pixelFitDF.py"+cmd
    #Create file that will submit the job
    cmdfilename='../cmds/'+os.path.basename(args[0])+'.sh'
    if options.grid or options.gridall:
        shutil.copyfile('submit_template_grid.txt',cmdfilename)
    else:
        shutil.copyfile('submit_template.txt',cmdfilename)
    cmdfile= open(cmdfilename,'a')
    cmdfile.write(cmd)
    if _RESUBMIT:
        cmdfile.write('\n')
        cmdfile.write('if [ -f %s" ] \n' % (args[0]))
        cmdfile.write('then\n')
        cmdfile.write('\t echo "All done"\n')
        cmdfile.write('else\n')
        cmdfile.write('\t qresub $JOB_ID\n')
        cmdfile.write('fi')
        cmdfile.write
    cmdfile.close()
    #Now submit
    if options.grid or options.gridall:
        #hold_jid created using qstat | grep bovy | awk '{print "-hold_jid "$1}'
        subprocess.call(["qsub","-w","n","-l","exclusive=true",
                         "-N",options.clustername+'_%i' % ii,
                         "-hold_jid",options.clusterholdname+'_%i' % ii,
                         "-l","h_rt=24:00:00",cmdfilename])
    else:
        subprocess.call(["qsub","-w","n","-l","exclusive=true",
                         "-l","h_rt=36:00:00",cmdfilename])
    return None    

def run_abundance_singles_single(options,args,fehs,afes,ii,savename,initname,
                                 normname):
    #Prepare args and options
    spl= savename.split('.')
    newname= ''
    for jj in range(len(spl)-1):
        newname+= spl[jj]
        if not jj == len(spl)-2: newname+= '.'
    newname+= '_%i.' % ii
    newname+= spl[-1]
    args[0]= newname
    if options.mcsample and not initname is None:
        #Do the same for init
        spl= initname.split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        options.init= newname
    if not normname is None:
        #Do the same for init
        spl= normname.split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        options.savenorm= newname
    options.singlefeh= fehs[ii]
    options.singleafe= afes[ii]
    options.multi=1
    #Now run
    pixelFitDF(options,args)

##COORDINATE TRANSFORMATIONS AND RO/VO NORMALIZATION
def prepare_coordinates(params,indx,fehs,afes,binned,errstuff,options,
                        npops):
    vo= get_vo(params,options,npops)
    ro= get_ro(params,options)
    vsun= get_vsun(params,options)
    data= copy.copy(binned(fehs[indx],afes[indx]))
    """
    #Create XYZ and R, vxvyvz, cov_vxvyvz
    R= ((1.-data.xc/_REFR0/ro)**2.+(data.yc/_REFR0/ro)**2.)**0.5
    #Confine to R-range?
    if not options.rmin is None and not options.rmax is None:
        dataindx= (R >= options.rmin/_REFR0/ro)*\
            (R < options.rmax/_REFR0/ro)
        data= data[dataindx]
        R= R[dataindx]
    #Confine to z-range?
    if not options.zmin is None and not options.zmax is None:
        dataindx= ((data.zc+_ZSUN) >= options.zmin)*\
            ((data.zc+_ZSUN) < options.zmax)
        data= data[dataindx]
        R= R[dataindx]
    XYZ= numpy.zeros((len(data),3))
    XYZ[:,0]= data.xc/_REFR0/ro
    XYZ[:,1]= data.yc/_REFR0/ro
    XYZ[:,2]= (data.zc+_ZSUN)/_REFR0/ro
    z= XYZ[:,2]
    """
    if not options.fitro and not options.fitvsun and not options.fitvtsun \
            and not options.fitdm:
        callindx= binned.callIndx(fehs[indx],afes[indx])
        R= numpy.array([e.R for e in errstuff])[callindx]/ro/_REFR0
        z= numpy.array([e.z for e in errstuff])[callindx]/ro/_REFR0
        vR= numpy.array([e.vR for e in errstuff])[callindx]/vo/_REFV0
        vT= numpy.array([e.vT for e in errstuff])[callindx]/vo/_REFV0
        vz= numpy.array([e.vz for e in errstuff])[callindx]/vo/_REFV0
        return (R,vR,vT,z,vz,
                binned.data[callindx].vRc_err/vo/_REFV0,
                binned.data[callindx].vzc_err/vo/_REFV0,
                binned.data[callindx].vRvzc_rho)
    elif (options.fitvsun or options.fitvtsun) and not options.fitro \
            and not options.fitdm:
        callindx= binned.callIndx(fehs[indx],afes[indx])
        R= numpy.array([e.R for e in errstuff])[callindx]/ro/_REFR0
        z= numpy.array([e.z for e in errstuff])[callindx]/ro/_REFR0
        vx= numpy.array([e.vx for e in errstuff])[callindx]/vo/_REFV0
        vy= numpy.array([e.vy for e in errstuff])[callindx]/vo/_REFV0
        vz= numpy.array([e.vz for e in errstuff])[callindx]/vo/_REFV0
        cosphi= numpy.array([e.cosphi for e in errstuff])[callindx]/vo/_REFV0
        sinphi= numpy.array([e.sinphi for e in errstuff])[callindx]/vo/_REFV0
        #Apply solar motion
        vx-= vsun[0]/vo
        vy+= vsun[1]/vo
        vz+= vsun[2]/vo
        vR= -vx*cosphi+vy*sinphi
        vT= vx*sinphi+vy*cosphi
        return (R,vR,vT,z,vz)       
    elif not (options.fitvsun or options.fitvtsun) and options.fitro \
            and not options.fitdm:
        callindx= binned.callIndx(fehs[indx],afes[indx])
        x= numpy.array([e.x for e in errstuff])[callindx]/ro/_REFR0
        y= numpy.array([e.y for e in errstuff])[callindx]/ro/_REFR0
        z= numpy.array([e.z for e in errstuff])[callindx]/ro/_REFR0
        vx= numpy.array([e.vx for e in errstuff])[callindx]/vo/_REFV0
        vy= numpy.array([e.vy for e in errstuff])[callindx]/vo/_REFV0
        vz= numpy.array([e.vz for e in errstuff])[callindx]/vo/_REFV0
        R= ((1.-x)**2.+y**2.)**0.5
        cosphi= (1.-x)/R
        sinphi= y/R
        vR= -vx*cosphi+vy*sinphi
        vT= vx*sinphi+vy*cosphi
        return (R,vR,vT,z,vz)       
    elif options.fitdm:
        callindx= binned.callIndx(fehs[indx],afes[indx])
        distfac= 10.**(get_dm(params,options)/5.)
        x= numpy.array([e.x for e in errstuff])[callindx]*distfac
        y= numpy.array([e.y for e in errstuff])[callindx]*distfac
        z= numpy.array([e.z for e in errstuff])[callindx]*distfac
        vr= numpy.array([e.vr for e in errstuff])[callindx]
        pmll= numpy.array([e.pmll for e in errstuff])[callindx]
        pmbb= numpy.array([e.pmbb for e in errstuff])[callindx]
        vxvyvz= bovy_coords.vrpmllpmbb_to_vxvyvz(vr.flatten(),
                                                 pmll.flatten(),
                                                 pmbb.flatten(),
                                                 x.flatten(),
                                                 y.flatten(),
                                                 z.flatten(),
                                                 XYZ=True,
                                                 degree=False)/vo/_REFV0
        #Apply solar motion
        vxvyvz[:,0]-= vsun[0]/vo
        vxvyvz[:,1]+= vsun[1]/vo
        vxvyvz[:,2]+= vsun[2]/vo
        R= ((ro*_REFR0-x)**2.+y**2.)**0.5/ro/_REFR0
        cosphi= (1.-x/ro/_REFR0)/R
        sinphi= y/R/ro/_REFR0
        vR= -vxvyvz[:,0]*cosphi.flatten()+vxvyvz[:,1]*sinphi.flatten()
        vT= vxvyvz[:,0]*sinphi.flatten()+vxvyvz[:,1]*cosphi.flatten()
        vz= numpy.reshape(vxvyvz[:,2],x.shape)
        vR= numpy.reshape(vR,x.shape)
        vT= numpy.reshape(vT,x.shape)
        return (R,vR,vT,(z+_ZSUN)/ro/_REFR0,vz)
    XYZ= numpy.zeros((len(data),3,options.nmcerr))
    vxvyvz= numpy.zeros((len(data),3,options.nmcerr))
    for ii in range(len(data)):
        for jj in range(options.nmcerr):
            XYZ[ii,0,jj]= data[ii].xdraws[jj][0]/_REFR0/ro
            XYZ[ii,1,jj]= data[ii].xdraws[jj][1]/_REFR0/ro
            XYZ[ii,2,jj]= (data[ii].xdraws[jj][2]+_ZSUN)/_REFR0/ro
            vxvyvz[ii,0,jj]= (data[ii].vdraws[jj][0]/_REFV0-vsun[0])/vo #minus OK
            vxvyvz[ii,1,jj]= (data[ii].vdraws[jj][1]/_REFV0+vsun[1])/vo
            vxvyvz[ii,2,jj]= (data[ii].vdraws[jj][2]/_REFV0+vsun[2])/vo
    R= ((1.-XYZ[:,0,:])**2.+(XYZ[:,1,:])**2.)**0.5
    z= XYZ[:,2]
    #Rotate to Galactocentric frame
    cosphi= (1.-XYZ[:,0,:])/R
    sinphi= XYZ[:,1,:]/R
    #cosphi= numpy.tile(cosphi,(options.nmcerr,1)).T
    #sinphi= numpy.tile(sinphi,(options.nmcerr,1)).T
    vR= -vxvyvz[:,0,:]*cosphi+vxvyvz[:,1,:]*sinphi
    vT= vxvyvz[:,0,:]*sinphi+vxvyvz[:,1,:]*cosphi
    #R= numpy.tile(R,(options.nmcerr,1)).T
    #z= numpy.tile(z,(options.nmcerr,1)).T
    return (R,vR,vT,z,vxvyvz[:,2,:])

##SETUP THE POTENTIAL IN EACH STEP
def setup_aA(pot,options):
    """Function for setting up the actionAngle object"""
    if options.multi is None:
        numcores= 1
    else:
        numcores= options.multi
    if options.aAmethod.lower() == 'adiabaticgrid':
        return actionAngleAdiabaticGrid(pot=pot,nR=options.aAnR,
                                        nEz=options.aAnEz,nEr=options.aAnEr,
                                        nLz=options.aAnLz,
                                        zmax=options.aAzmax,
                                        Rmax=options.aARmax,
                                        numcores=numcores,
                                        c=True)
    elif options.aAmethod.lower() == 'adiabatic':
        return actionAngleAdiabatic(pot=pot,gamma=1.,c=True)
    elif options.aAmethod.lower() == 'staeckel':
        return actionAngleStaeckel(pot=pot,delta=options.staeckeldelta,c=True)
    elif options.aAmethod.lower() == 'staeckelgrid':
        return actionAngleStaeckelGrid(pot=pot,delta=options.staeckeldelta,
                                       c=True,Rmax=options.staeckelRmax,
                                       nLz=options.staeckelnLz,
                                       nE=options.staeckelnE,
                                       npsi=options.staeckelnpsi)
    
def setup_potential(params,options,npops,
                    interpDens=False,interpdvcircdr=False,
                    returnrawpot=False):
    """Function for setting up the potential"""
    potparams= get_potparams(params,options,npops)
    if options.potential.lower() == 'flatlog':
        return potential.LogarithmicHaloPotential(normalize=1.,q=potparams[0])
    elif options.potential.lower() == 'mwpotential':
        return potential.MWPotential #Just used for fake data
    elif options.potential.lower() == 'mwpotentialsimplefit':
        ro= get_ro(params,options)
        ampd= 0.95*potparams[3-(1-(options.fixvo is None))]
        amph= 0.95*(1.-potparams[3-(1-(options.fixvo is None))])
        return [potential.MiyamotoNagaiPotential(a=numpy.exp(potparams[0])/ro,
                                                 b=numpy.exp(potparams[2-(1-(options.fixvo is None))])/ro,
                                                 normalize=ampd),
                potential.NFWPotential(a=4.5,normalize=amph),
                potential.HernquistPotential(a=0.6/8,normalize=0.05)]
    elif options.potential.lower() == 'mwpotentialfixhalo':
        ro= get_ro(params,options)
        ampdh= potparams[4]
        ampd= potparams[3]*ampdh
        amph= (1.-potparams[3])*ampdh
        ampb= 1.-ampd-amph
        return [potential.MiyamotoNagaiPotential(a=numpy.exp(potparams[0])/ro,
                                                 b=numpy.exp(potparams[2])/ro,
                                                 normalize=ampd),
                potential.NFWPotential(a=4.5,normalize=amph),
                potential.HernquistPotential(a=0.6/8,normalize=ampb)]
    elif options.potential.lower() == 'mwpotentialfixhaloflat':
        ro= get_ro(params,options)
        dlnvcdlnr= potparams[3]/30.
        ampb= potparams[4] 
        #normalize to 1 for calculation of ampd and amph
        dp= potential.MiyamotoNagaiPotential(a=numpy.exp(potparams[0])/ro,
                                             b=numpy.exp(potparams[2])/ro,
                                             normalize=1.)
        hp= potential.NFWPotential(a=4.5,normalize=1.)
        bp= potential.HernquistPotential(a=0.6/8,normalize=ampb)
        ampd, amph= fdfh_from_dlnvcdlnr(dlnvcdlnr,dp,bp,hp)
        if ampd <= 0. or amph <= 0.:
            raise RuntimeError
        dp= potential.MiyamotoNagaiPotential(a=numpy.exp(potparams[0])/ro,
                                             b=numpy.exp(potparams[2])/ro,
                                             normalize=ampd)
        hp= potential.NFWPotential(a=4.5,normalize=amph)
        return [dp,hp,bp]
    elif options.potential.lower() == 'mpdiskplhalofixbulgeflat':
        ro= get_ro(params,options)
        vo= get_vo(params,options,npops)
        dlnvcdlnr= potparams[3]/30.
        ampb= _GMBULGE/_ABULGE*(_REFR0*ro/_ABULGE)/(1.+(_REFR0*ro/_ABULGE))**2./_REFV0**2./vo**2.
        bp= potential.HernquistPotential(a=_ABULGE/_REFR0/ro,normalize=ampb)
        #normalize to 1 for calculation of ampd and amph
        dp= potential.MiyamotoNagaiPotential(a=numpy.exp(potparams[0])/ro,
                                             b=numpy.exp(potparams[2])/ro,
                                             normalize=1.)
        hp= potential.PowerSphericalPotential(alpha=potparams[4],normalize=1.)
        ampd, amph= fdfh_from_dlnvcdlnr(dlnvcdlnr,dp,bp,hp)
        if ampd <= 0. or amph <= 0.:
            raise RuntimeError
        dp= potential.MiyamotoNagaiPotential(a=numpy.exp(potparams[0])/ro,
                                             b=numpy.exp(potparams[2])/ro,
                                             normalize=ampd)
        hp= potential.PowerSphericalPotential(alpha=potparams[4],
                                              normalize=amph)
        return [dp,hp,bp]
    elif options.potential.lower() == 'dpdiskplhalofixbulgeflat':
        ro= get_ro(params,options)
        vo= get_vo(params,options,npops)
        dlnvcdlnr= potparams[3]/30.
        ampb= _GMBULGE/_ABULGE*(_REFR0*ro/_ABULGE)/(1.+(_REFR0*ro/_ABULGE))**2./_REFV0**2./vo**2.
        bp= potential.HernquistPotential(a=_ABULGE/_REFR0/ro,normalize=ampb)
        #normalize to 1 for calculation of ampd and amph
        dp= potential.DoubleExponentialDiskPotential(hr=numpy.exp(potparams[0])/ro,
                                                     hz=numpy.exp(potparams[2])/ro,
                                                     normalize=1.)
        hp= potential.PowerSphericalPotential(alpha=potparams[4],normalize=1.)
        ampd, amph= fdfh_from_dlnvcdlnr(dlnvcdlnr,dp,[bp],hp)
        if ampd <= 0. or amph <= 0.:
            raise RuntimeError
        dp= potential.DoubleExponentialDiskPotential(hr=numpy.exp(potparams[0])/ro,
                                                     hz=numpy.exp(potparams[2])/ro,
                                             normalize=ampd)
        hp= potential.PowerSphericalPotential(alpha=potparams[4],
                                              normalize=amph)
        #Use an interpolated version for speed
        if returnrawpot:
            return [dp,hp,bp]
        else:
            return potential.interpRZPotential(RZPot=[dp,hp,bp],rgrid=(numpy.log(0.01),numpy.log(20.),101),zgrid=(0.,1.,101),logR=True,interpepifreq=True,interpverticalfreq=True,interpvcirc=True,use_c=True,enable_c=True,interpPot=True,interpDens=interpDens,interpdvcircdr=interpdvcircdr)
    elif options.potential.lower() == 'dpdiskplhalofixbulgeflatwgas':
        ro= get_ro(params,options)
        vo= get_vo(params,options,npops)
        dlnvcdlnr= potparams[3]/30.
        ampb= _GMBULGE/_ABULGE*(_REFR0*ro/_ABULGE)/(1.+(_REFR0*ro/_ABULGE))**2./_REFV0**2./vo**2.
        bp= potential.HernquistPotential(a=_ABULGE/_REFR0/ro,normalize=ampb)
        #Also add 13 Msol/pc^2 with a scale height of 130 pc, and a scale length of ?
        gp= potential.DoubleExponentialDiskPotential(hr=2.*numpy.exp(potparams[0])/ro,
                                                     hz=0.130/ro/_REFR0,
                                                     normalize=1.)
        gassurfdens= 2.*gp.dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.*gp._hz*ro*_REFR0*1000.
        gp= potential.DoubleExponentialDiskPotential(hr=2.*numpy.exp(potparams[0])/ro,
                                                     hz=0.130/ro/_REFR0,
                                                     normalize=13./gassurfdens)
        #normalize to 1 for calculation of ampd and amph
        dp= potential.DoubleExponentialDiskPotential(hr=numpy.exp(potparams[0])/ro,
                                                     hz=numpy.exp(potparams[2])/ro,
                                                     normalize=1.)
        hp= potential.PowerSphericalPotential(alpha=potparams[4],normalize=1.)
        ampd, amph= fdfh_from_dlnvcdlnr(dlnvcdlnr,dp,[bp,gp],hp)
        if ampd <= 0. or amph <= 0.:
            raise RuntimeError
        dp= potential.DoubleExponentialDiskPotential(hr=numpy.exp(potparams[0])/ro,
                                                     hz=numpy.exp(potparams[2])/ro,
                                             normalize=ampd)
        hp= potential.PowerSphericalPotential(alpha=potparams[4],
                                              normalize=amph)
        #print ampb, 13./gassurfdens, ampd, amph, dp(1.,0.), hp(1.,0.)
        #Use an interpolated version for speed
        if returnrawpot:
            return [dp,hp,bp,gp]
        else:
            return potential.interpRZPotential(RZPot=[dp,hp,bp,gp],rgrid=(numpy.log(0.01),numpy.log(20.),101),zgrid=(0.,1.,101),logR=True,interpepifreq=True,interpverticalfreq=True,interpvcirc=True,use_c=True,enable_c=True,interpPot=True,interpDens=interpDens,interpdvcircdr=interpdvcircdr)
    elif options.potential.lower() == 'dpdiskplhalofixbulgeflatwgasalt':
        ro= get_ro(params,options)
        vo= get_vo(params,options,npops)
        dlnvcdlnr= potparams[4]/30.
        ampb= _GMBULGE/_ABULGE*(_REFR0*ro/_ABULGE)/(1.+(_REFR0*ro/_ABULGE))**2./_REFV0**2./vo**2.
        bp= potential.HernquistPotential(a=_ABULGE/_REFR0/ro,normalize=ampb)
        #Also add 13 Msol/pc^2 with a scale height of 130 pc, and a scale length of ?
        gp= potential.DoubleExponentialDiskPotential(hr=2.*numpy.exp(potparams[0])/ro,
                                                     hz=0.130/ro/_REFR0,
                                                     normalize=1.)
        gassurfdens= 2.*gp.dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.*gp._hz*ro*_REFR0*1000.
        gp= potential.DoubleExponentialDiskPotential(hr=2.*numpy.exp(potparams[0])/ro,
                                                     hz=0.130/ro/_REFR0,
                                                     normalize=13./gassurfdens)
        fdfh= 1.-13./gassurfdens-ampb
        fd= (1.-potparams[3])*fdfh
        fh= fdfh-fd
        dp= potential.DoubleExponentialDiskPotential(hr=numpy.exp(potparams[0])/ro,
                                                     hz=numpy.exp(potparams[2])/ro,
                                                     normalize=fd)
        plhalo= plhalo_from_dlnvcdlnr(dlnvcdlnr,dp,[bp,gp],options,fh)
        if plhalo < 0. or plhalo > 3:
            raise RuntimeError("plhalo=%f" % plhalo)
        hp= potential.PowerSphericalPotential(alpha=plhalo,
                                              normalize=fh)
        #print ampb, 13./gassurfdens, ampd, amph, dp(1.,0.), hp(1.,0.)
        #Use an interpolated version for speed
        if returnrawpot:
            return [dp,hp,bp,gp]
        else:
            return potential.interpRZPotential(RZPot=[dp,hp,bp,gp],rgrid=(numpy.log(0.01),numpy.log(20.),101),zgrid=(0.,1.,101),logR=True,interpepifreq=True,interpverticalfreq=True,interpvcirc=True,use_c=True,enable_c=True,interpPot=True,interpDens=interpDens,interpdvcircdr=interpdvcircdr)
    elif options.potential.lower() == 'dpdiskplhalofixcutbulgeflatwgasalt':
        ro= get_ro(params,options)
        vo= get_vo(params,options,npops)
        dlnvcdlnr= potparams[4]/30.
        ampb= _GMBULGECUT/2./numpy.pi/_RCBULGE**(3.-_PLBULGE/2.)/special.gamma(3.-_PLBULGE)*(_REFR0*ro)**(2.-_PLBULGE)/_REFV0**2./vo**2.
        #ampb= _GMBULGECUT/_REFR0/ro/_REFV0**2./vo**2. #Good enough approximation; this actually gives 10^10, but we used the above in the paper, which gives ~6x10^9
        bp= potential.PowerSphericalPotentialwCutoff(alpha=_PLBULGE,rc=_RCBULGE/_REFR0/ro,normalize=ampb)
        #Also add 13 Msol/pc^2 with a scale height of 130 pc, and a scale length of ?
        gp= potential.DoubleExponentialDiskPotential(hr=2.*numpy.exp(potparams[0])/ro,
                                                     hz=0.130/ro/_REFR0,
                                                     normalize=1.)
        gassurfdens= 2.*gp.dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.*gp._hz*ro*_REFR0*1000.
        gp= potential.DoubleExponentialDiskPotential(hr=2.*numpy.exp(potparams[0])/ro,
                                                     hz=0.130/ro/_REFR0,
                                                     normalize=13./gassurfdens)
        fdfh= 1.-13./gassurfdens-ampb
        fd= (1.-potparams[3])*fdfh
        fh= fdfh-fd
        dp= potential.DoubleExponentialDiskPotential(hr=numpy.exp(potparams[0])/ro,
                                                     hz=numpy.exp(potparams[2])/ro,
                                                     normalize=fd)
        plhalo= plhalo_from_dlnvcdlnr(dlnvcdlnr,dp,[bp,gp],options,fh)
        if plhalo < 0. or plhalo > 3:
            raise RuntimeError("plhalo=%f" % plhalo)
        hp= potential.PowerSphericalPotential(alpha=plhalo,
                                              normalize=fh)
        #print ampb, 13./gassurfdens, ampd, amph, dp(1.,0.), hp(1.,0.)
        #Use an interpolated version for speed
        if returnrawpot:
            return [dp,hp,bp,gp]
        else:
            return potential.interpRZPotential(RZPot=[dp,hp,bp,gp],rgrid=(numpy.log(0.01),numpy.log(20.),101),zgrid=(0.,1.,101),logR=True,interpepifreq=True,interpverticalfreq=True,interpvcirc=True,use_c=True,enable_c=True,interpPot=True,interpDens=interpDens,interpdvcircdr=interpdvcircdr)
    elif options.potential.lower() == 'dpdiskflplhalofixbulgeflatwgas':
        ro= get_ro(params,options)
        vo= get_vo(params,options,npops)
        dlnvcdlnr= potparams[3]/30.
        ampb= _GMBULGE/_ABULGE*(_REFR0*ro/_ABULGE)/(1.+(_REFR0*ro/_ABULGE))**2./_REFV0**2./vo**2.
        bp= potential.HernquistPotential(a=_ABULGE/_REFR0/ro,normalize=ampb)
        #Also add 13 Msol/pc^2 with a scale height of 130 pc, and a scale length of ?
        gp= potential.DoubleExponentialDiskPotential(hr=2.*numpy.exp(potparams[0])/ro,
                                                     hz=0.130/ro/_REFR0,
                                                     normalize=1.)
        gassurfdens= 2.*gp.dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.*gp._hz*ro*_REFR0*1000.
        gp= potential.DoubleExponentialDiskPotential(hr=2.*numpy.exp(potparams[0])/ro,
                                                     hz=0.130/ro/_REFR0,
                                                     normalize=13./gassurfdens)
        #normalize to 1 for calculation of ampd and amph
        dp= potential.DoubleExponentialDiskPotential(hr=numpy.exp(potparams[0])/ro,
                                                     hz=numpy.exp(potparams[2])/ro,
                                                     normalize=1.)
        hp= potential.FlattenedPowerPotential(alpha=potparams[5],
                                              q=potparams[4],
                                              normalize=1.)
        ampd, amph= fdfh_from_dlnvcdlnr(dlnvcdlnr,dp,[bp,gp],hp)
        if ampd <= 0. or amph <= 0.:
            raise RuntimeError
        dp= potential.DoubleExponentialDiskPotential(hr=numpy.exp(potparams[0])/ro,
                                                     hz=numpy.exp(potparams[2])/ro,
                                             normalize=ampd)
        hp= potential.FlattenedPowerPotential(alpha=potparams[5],
                                              q=potparams[4],
                                              normalize=amph)
        #print ampb, 13./gassurfdens, ampd, amph, dp(1.,0.), hp(1.,0.)
        #Use an interpolated version for speed
        if returnrawpot:
            return [dp,hp,bp,gp]
        else:
            return potential.interpRZPotential(RZPot=[dp,hp,bp,gp],rgrid=(numpy.log(0.01),numpy.log(20.),101),zgrid=(0.,1.,101),logR=True,interpepifreq=True,interpverticalfreq=True,interpvcirc=True,use_c=True,enable_c=True,interpPot=True,interpDens=interpDens,interpdvcircdr=interpdvcircdr)
    elif options.potential.lower() == 'dpdiskplhalodarkdiskfixbulgeflatwgasalt':
        ro= get_ro(params,options)
        vo= get_vo(params,options,npops)
        dlnvcdlnr= potparams[4]/30.
        ampb= _GMBULGE/_ABULGE*(_REFR0*ro/_ABULGE)/(1.+(_REFR0*ro/_ABULGE))**2./_REFV0**2./vo**2.
        bp= potential.HernquistPotential(a=_ABULGE/_REFR0/ro,normalize=ampb)
        #Also add 13 Msol/pc^2 with a scale height of 130 pc, and a scale length of ?
        gp= potential.DoubleExponentialDiskPotential(hr=2.*numpy.exp(potparams[0])/ro,
                                                     hz=0.130/ro/_REFR0,
                                                     normalize=1.)
        gassurfdens= 2.*gp.dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.*gp._hz*ro*_REFR0*1000.
        gp= potential.DoubleExponentialDiskPotential(hr=2.*numpy.exp(potparams[0])/ro,
                                                     hz=0.130/ro/_REFR0,
                                                     normalize=13./gassurfdens)
        fdfh= 1.-13./gassurfdens-ampb
        fd= (1.-potparams[3])*fdfh
        fh= fdfh-fd
        dp= potential.DoubleExponentialDiskPotential(hr=numpy.exp(potparams[0])/ro,
                                                     hz=numpy.exp(potparams[2])/ro,
                                                     normalize=fd)
        darkdp= potential.DoubleExponentialDiskPotential(hr=options.darkdiskrd/_REFR0/ro,
                                                         hz=1.5/_REFR0/ro,
                                                         normalize=1.)
        darkdrhoo= potential.evaluateDensities(1.,0.,darkdp)      
        gamma= potparams[5]/4./numpy.pi
        plhalo, ampdarkd, amph= plhaloamp_from_dlnvcdlnr_darkd(dlnvcdlnr,darkdp,[dp,bp,gp],options,fh,darkdrhoo,gamma)
        if ampdarkd < 0. and ampdarkd > -10.**-10.:
            ampdarkd= 0. #tweak for when ampdarkd gets slightly below zero
        if ampdarkd < 0. or amph < 0.:
            raise RuntimeError("Amplitude of one of the halo components is negaitve")
        #print ampb, 13./gassurfdens, ampd, amph, dp(1.,0.), hp(1.,0.)
        #Use an interpolated version for speed
        darkdp= potential.DoubleExponentialDiskPotential(hr=options.darkdiskrd/_REFR0/ro,
                                                         hz=1.5/_REFR0/ro,
                                                         normalize=ampdarkd)
        hp= potential.PowerSphericalPotential(alpha=plhalo,
                                              normalize=amph)
        if returnrawpot:
            return [dp,hp,bp,gp,darkdp]
        else:
            return potential.interpRZPotential(RZPot=[dp,hp,bp,gp,darkdp],rgrid=(numpy.log(0.01),numpy.log(20.),101),zgrid=(0.,1.,101),logR=True,interpepifreq=True,interpverticalfreq=True,interpvcirc=True,use_c=True,enable_c=True,interpPot=True,interpDens=interpDens,interpdvcircdr=interpdvcircdr)
    elif options.potential.lower() == 'mpdiskflplhalofixplfixbulgeflat':
        ro= get_ro(params,options)
        vo= get_vo(params,options,npops)
        dlnvcdlnr= potparams[3]/30.
        ampb= _GMBULGE/_ABULGE*(_REFR0*ro/_ABULGE)/(1.+(_REFR0*ro/_ABULGE))**2./_REFV0**2./vo**2.
        bp= potential.HernquistPotential(a=_ABULGE/_REFR0/ro,normalize=ampb)
        #normalize to 1 for calculation of ampd and amph
        dp= potential.MiyamotoNagaiPotential(a=numpy.exp(potparams[0])/ro,
                                             b=numpy.exp(potparams[2])/ro,
                                             normalize=1.)
        hp= potential.FlattenedPowerPotential(alpha=-0.8,q=potparams[4],
                                              normalize=1.)
        ampd, amph= fdfh_from_dlnvcdlnr(dlnvcdlnr,dp,bp,hp)
        if ampd <= 0. or amph <= 0.:
            raise RuntimeError
        dp= potential.MiyamotoNagaiPotential(a=numpy.exp(potparams[0])/ro,
                                             b=numpy.exp(potparams[2])/ro,
                                             normalize=ampd)
        hp= potential.FlattenedPowerPotential(alpha=-0.8,q=potparams[4],
                                              normalize=amph)
        return [dp,hp,bp]
    elif options.potential.lower() == 'flatlogdisk':
        return [potential.LogarithmicHaloPotential(normalize=.5,q=potparams[0]),
                potential.MiyamotoNagaiPotential(normalize=.5,a=0.5,b=0.1)]
    elif options.potential.lower() == 'bti': #model I from Binney & Tremaine (ish)
        ro= get_ro(params,options)
        vo= get_vo(params,options,npops)
        #Bulge
        ampb= _GMBULGE/_ABULGE*(_REFR0*ro/_ABULGE)/(1.+(_REFR0*ro/_ABULGE))**2./_REFV0**2./vo**2.
        bp= potential.HernquistPotential(a=_ABULGE/_REFR0/ro,normalize=ampb)
        #Halo
        fh= 0.35/0.95*(1.-ampb)
        hp= potential.PowerSphericalPotential(alpha=1.,
                                              normalize=fh)
        #Disks
        rd= 2./8.
        gp= potential.DoubleExponentialDiskPotential(hr=2.*rd/ro,
                                                     hz=0.080/ro/_REFR0,
                                                     normalize=1.)
        dp= potential.DoubleExponentialDiskPotential(hr=rd/ro,
                                                     hz=0.350/_REFR0/ro,
                                                     normalize=1.)
        gassurfdens= 2.*gp.dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.*gp._hz*ro*_REFR0*1000.
        stellarsurfdens= 2.*dp.dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.*dp._hz*ro*_REFR0*1000.
        fgr= 1./3.*stellarsurfdens/gassurfdens
        fdr= 1./(1.+fgr)
        fgr= fgr/(1.+fgr)
        fd= 0.6/0.95*(1.-ampb)*fdr
        fg= 0.6/0.95*(1.-ampb)*fgr
        gp= potential.DoubleExponentialDiskPotential(hr=2.*rd/ro,
                                                     hz=0.080/ro/_REFR0,
                                                     normalize=fg)
        dp= potential.DoubleExponentialDiskPotential(hr=rd/ro,
                                                     hz=0.350/_REFR0/ro,
                                                     normalize=fd)
        #Use an interpolated version for speed
        if returnrawpot:
            return [dp,hp,bp,gp]
        else:
            return potential.interpRZPotential(RZPot=[dp,hp,bp,gp],rgrid=(numpy.log(0.01),numpy.log(20.),101),zgrid=(0.,1.,101),logR=True,interpepifreq=True,interpverticalfreq=True,interpvcirc=True,use_c=True,enable_c=True,interpPot=True,interpDens=interpDens,interpdvcircdr=interpdvcircdr)

    elif options.potential.lower() == 'btii': #model II from Binney & Tremaine (ish)
        ro= get_ro(params,options)
        vo= get_vo(params,options,npops)
        #Bulge
        ampb= _GMBULGE/_ABULGE*(_REFR0*ro/_ABULGE)/(1.+(_REFR0*ro/_ABULGE))**2./_REFV0**2./vo**2.
        bp= potential.HernquistPotential(a=_ABULGE/_REFR0/ro,normalize=ampb)
        #Halo
        fh= 0.63/0.96*(1.-ampb)
        hp= potential.PowerSphericalPotential(alpha=1.9,
                                              normalize=fh)
        #Disks
        rd= 3.2/8.
        gp= potential.DoubleExponentialDiskPotential(hr=2.*rd/ro,
                                                     hz=0.080/ro/_REFR0,
                                                     normalize=1.)
        dp= potential.DoubleExponentialDiskPotential(hr=rd/ro,
                                                     hz=0.350/_REFR0/ro,
                                                     normalize=1.)
        gassurfdens= 2.*gp.dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.*gp._hz*ro*_REFR0*1000.
        stellarsurfdens= 2.*dp.dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.*dp._hz*ro*_REFR0*1000.
        fgr= 1./3.*stellarsurfdens/gassurfdens
        fdr= 1./(1.+fgr)
        fgr= fgr/(1.+fgr)
        fd= 0.33/0.96*(1.-ampb)*fdr
        fg= 0.33/0.96*(1.-ampb)*fgr
        gp= potential.DoubleExponentialDiskPotential(hr=2.*rd/ro,
                                                     hz=0.080/ro/_REFR0,
                                                     normalize=fg)
        dp= potential.DoubleExponentialDiskPotential(hr=rd/ro,
                                                     hz=0.350/_REFR0/ro,
                                                     normalize=fd)
        #Use an interpolated version for speed
        if returnrawpot:
            return [dp,hp,bp,gp]
        else:
            return potential.interpRZPotential(RZPot=[dp,hp,bp,gp],rgrid=(numpy.log(0.01),numpy.log(20.),101),zgrid=(0.,1.,101),logR=True,interpepifreq=True,interpverticalfreq=True,interpvcirc=True,use_c=True,enable_c=True,interpPot=True,interpDens=interpDens,interpdvcircdr=interpdvcircdr)

def fdfh_from_dlnvcdlnr(dlnvcdlnr,diskpot,bulgepot,halopot):
    """Calculate the halo amplitude corresponding to this rotation curve derivative"""
    #Make sure that diskpot and diskhalo are normalized!!
    #First calculate the derivatives dvc^2/dR of disk, halo, and bulge
    dvcdr_disk= -potential.evaluateRforces(1.,0.,diskpot)+potential.evaluateR2derivs(1.,0.,diskpot)
    dvcdr_halo= -potential.evaluateRforces(1.,0.,halopot)+potential.evaluateR2derivs(1.,0.,halopot)
    dvcdr_bulge= -potential.evaluateRforces(1.,0.,bulgepot)+potential.evaluateR2derivs(1.,0.,bulgepot)
    #calculate fd,fh
    oneminusvcb2= 1.-potential.vcirc(bulgepot,1.)**2.
    return ((oneminusvcb2*dvcdr_halo-(2.*dlnvcdlnr-dvcdr_bulge))/(dvcdr_halo-dvcdr_disk),
            (2.*dlnvcdlnr-dvcdr_bulge-dvcdr_disk*oneminusvcb2)/(dvcdr_halo-dvcdr_disk))

def plhalo_from_dlnvcdlnr(dlnvcdlnr,diskpot,bulgepot,options,fh):
    """Calculate the halo's shape corresponding to this rotation curve derivative"""
    #First calculate the derivatives dvc^2/dR of disk and bulge
    dvcdr_disk= -potential.evaluateRforces(1.,0.,diskpot)+potential.evaluateR2derivs(1.,0.,diskpot)
    dvcdr_bulge= -potential.evaluateRforces(1.,0.,bulgepot)+potential.evaluateR2derivs(1.,0.,bulgepot)
    if 'plhalo' in options.potential.lower():
        return 2.-(2.*dlnvcdlnr-dvcdr_disk-dvcdr_bulge)/fh

def plhaloamp_from_dlnvcdlnr_darkd(dlnvcdlnr,darkdpot,
                                   restpot,options,fh,darkdrhoo,gamma):
    """Calculate the halo's shape corresponding to this rotation curve derivative in the case of a dark disk"""
    #First calculate the derivatives dvc^2/dR of the rest and of the dark disk
    dvcdr_rest= -potential.evaluateRforces(1.,0.,restpot)\
        +potential.evaluateR2derivs(1.,0.,restpot)
    dvcdr_darkd= -potential.evaluateRforces(1.,0.,darkdpot)\
        +potential.evaluateR2derivs(1.,0.,darkdpot)
    try:
        alpha= optimize.brentq(_plhaloamp_eq,0.,3.,args=(darkdrhoo,dvcdr_darkd,dvcdr_rest,dlnvcdlnr,fh,gamma))
    except ValueError:
        raise RuntimeError('plhalo < 0 or plhalo > 3')
    else:
        delta= fh*darkdrhoo/(gamma*(3.-alpha)+darkdrhoo)
        eps= fh-delta
        return (alpha,eps,delta)

def _plhaloamp_eq(alpha,darkdrhoo,dvcdr_darkd,dvcdr_rest,dlnvcdlnr,fh,gamma):
    delta= fh*darkdrhoo/(gamma*(3.-alpha)+darkdrhoo)
    eps= fh-delta
    return dlnvcdlnr-eps*dvcdr_darkd-delta*(2.-alpha)-dvcdr_rest

##FULL OPTIMIZER
def full_optimize(params,fehs,afes,binned,options,normintstuff,errstuff):
    """Function for optimizing the full set of parameters"""
    if _BFGS:
        return optimize.fmin_bfgs(mloglike,params,
                                  args=(fehs,afes,binned,options,normintstuff,
                                        errstuff),
                                  callback=cb,
                                  full_output=True)
    else:
        typf= mloglike(params,fehs,afes,binned,options,normintstuff,errstuff)
        ftol= .1/numpy.fabs(typf)#should be plenty, assuming we're not too far off to start
        return optimize.fmin_powell(mloglike,params,
                                    args=(fehs,afes,binned,options,normintstuff,
                                          errstuff),
                                    callback=cb,
                                    ftol=ftol,xtol=10.**-3.,
                                    full_output=True,
                                    maxfun=5000)#Cut off long fits

##INDIVIDUAL OPTIMIZATIONS
def indiv_optimize_df(params,fehs,afes,binned,options,normintstuff,errstuff):
    """Function for optimizing individual DFs with potential fixed"""
    #Set up potential and actionAngle
    pot= setup_potential(params,options,len(fehs))
    aA= setup_aA(pot,options)
    if not options.multi is None:
        #Generate list of temporary files
        tmpfiles= []
        for ii in range(len(fehs)): tmpfiles.append(tempfile.mkstemp())
        try:
            logl= multi.parallel_map((lambda x: indiv_optimize_df_single(params,x,
                                                                         fehs,afes,binned,options,aA,pot,normintstuff,errstuff,tmpfiles)),
                                     range(len(fehs)),
                                     numcores=numpy.amin([len(fehs),
                                                          multiprocessing.cpu_count(),
                                                          options.multi]))
            #Now read all of the temporary files
            for ii in range(len(fehs)):
                tmpfile= open(tmpfiles[ii][1],'rb')
                new_dfparams= pickle.load(tmpfile)
                params= set_dfparams(new_dfparams,params,ii,options)
                tmpfile.close()
        finally:
            for ii in range(len(fehs)):
                os.remove(tmpfiles[ii][1])
    else:
        for ii in range(len(fehs)):
            print ii
            init_dfparams= list(get_dfparams(params,ii,options,log=True))
            typf= indiv_optimize_df_mloglike(init_dfparams,
                                             fehs,afes,binned,
                                             options,pot,aA,
                                             ii,copy.copy(params),
                                             normintstuff,errstuff)
            ftol= .1/numpy.fabs(typf)#should be plenty, assuming we're not too far off to start
            new_dfparams= optimize.fmin_powell(indiv_optimize_df_mloglike,
                                               init_dfparams,
                                           args=(fehs,afes,binned,
                                                 options,pot,aA,
                                                 ii,copy.copy(params),
                                                 normintstuff,errstuff),
                                               callback=cb,
                                               ftol=ftol,xtol=10.**-3.)
            params= set_dfparams(new_dfparams,params,ii,options)
    return params

def indiv_optimize_df_single(params,ii,fehs,afes,binned,options,aA,pot,normintstuff,errstuff,tmpfiles):
    """Function to optimize the DF params for a single population when holding the potential fixed and using multi-processing"""
    print ii
    init_dfparams= list(get_dfparams(params,ii,options,log=True))
    typf= indiv_optimize_df_mloglike(init_dfparams,
                                     fehs,afes,binned,
                                     options,pot,aA,
                                     ii,copy.copy(params),
                                     normintstuff,errstuff)
    ftol= .1/numpy.fabs(typf)#should be plenty, assuming we're not too far off to start
    new_dfparams= optimize.fmin_powell(indiv_optimize_df_mloglike,
                                       init_dfparams,
                                       args=(fehs,afes,binned,
                                             options,pot,aA,
                                             ii,copy.copy(params),
                                             normintstuff,errstuff),
                                       callback=cb,
                                       ftol=ftol,xtol=10.**-3.)
    #Now save to temporary pickle
    tmpfile= open(tmpfiles[ii][1],'wb')
    pickle.dump(new_dfparams,tmpfile)
    tmpfile.close()
    return None

def indiv_optimize_potential(params,fehs,afes,binned,options,normintstuff,
                             errstuff):
    """Function for optimizing the potential w/ individual DFs fixed"""
    init_potparams= numpy.array(get_potparams(params,options,len(fehs)))
    #Get a typical value to estimate necessary ftol for delta chi <~ 1
    typf= indiv_optimize_pot_mloglike(init_potparams,
                                      fehs,afes,binned,options,
                                      copy.copy(params),
                                      normintstuff,errstuff)
    ftol= .1/numpy.fabs(typf)#should be plenty, assuming we're not too far off to start
    new_potparams= optimize.fmin_powell(indiv_optimize_pot_mloglike,
                                        init_potparams,
                                        args=(fehs,afes,binned,options,
                                              copy.copy(params),
                                              normintstuff,errstuff),
                                        callback=cb,
                                        xtol=10.**-3.,
                                        ftol=ftol)
    params= set_potparams(new_potparams,params,options,len(fehs))
    return params

def indiv_sample_potential(params,fehs,afes,binned,options,normintstuff,
                           errstuff):
    """Function for sampling the potential w/ individual DFs fixed"""
    init_potparams= numpy.array(get_potparams(params,options,len(fehs)))
    #Setup everything necessary for sampling
    isDomainFinite, domain= setup_domain_indiv_potential(options,len(fehs))
    samples= bovy_mcmc.markovpy(init_potparams,
                                0.01,
                                indiv_optimize_pot_loglike,
                                (fehs,afes,binned,options,
                                 copy.copy(params),
                                 normintstuff,errstuff),
                                nsamples=options.nsamples,
                                nwalkers=4*len(init_potparams))
    #For now, don't merge pot samples with fixed DF
    return samples

def custom_markovpy(options,npops,initial_theta,step,lnpdf,pdf_params,
                    isDomainFinite=[False,False],domain=[0.,0.],
                    nsamples=1,nwalkers=None,threads=None,
                    sliceinit=False,skip=0,create_method='step_out',
                    returnLnprob=False,
                    pos=None,prob=None,state=None,
                    use_emcee=False,pool=None):
    try:
        ndim = len(initial_theta)
    except TypeError:
        ndim= 1
        if not sliceinit:
            initial_theta= numpy.array([initial_theta])
            isDomainFinite= [isDomainFinite]
            domain= [domain]
            step= [step]
    if not isinstance(isDomainFinite,numpy.ndarray):
        isDomainFinite= numpy.array(isDomainFinite)
    if not isinstance(domain,numpy.ndarray):
        domain= numpy.array(domain)
    if isinstance(step,list): step= numpy.array(step)
    if isinstance(step,(int,float)) or len(step) == 1:
        step= numpy.ones(ndim)*step
    if len(isDomainFinite.shape) == 1 and ndim > 1 and not sliceinit:
        dFinite= []
        for ii in range(ndim):
            dFinite.append(isDomainFinite)
        isDomainFinite= dFinite
    if len(domain.shape) == 1 and ndim > 1 and not sliceinit:
        dDomain= []
        for ii in range(ndim):
            dDomain.append(domain)
        domain= dDomain
    if ndim == 1: lambdafunc= lambda x: lnpdf(x[0],*pdf_params)
    else: lambdafunc= lambda x: lnpdf(x,*pdf_params)
    #Set-up walkers
    if nwalkers is None:
        if use_emcee:
            nwalkers= numpy.amax([6,2*ndim+2])
        else:
            nwalkers = numpy.amax([5,2*ndim])
    if threads is None:
        threads= 1
    nmarkovsamples= int(numpy.ceil(float(nsamples)/nwalkers))
    print nwalkers
    if pos is None:
        #Set up initial position
        initial_position= []
        lnprobs= []
        for ww in range(nwalkers):
            tlnp= -numpy.finfo(numpy.dtype(numpy.float64)).max
            while tlnp == -numpy.finfo(numpy.dtype(numpy.float64)).max:
                thisparams= []
                for pp in range(ndim):
                    prop= initial_theta[pp]+numpy.random.normal()*step[pp]
                    if (isDomainFinite[pp][0] and prop < domain[pp][0]):
                        prop= domain[pp][0]
                    elif (isDomainFinite[pp][1] and prop > domain[pp][1]):
                        prop= domain[pp][1]
                    thisparams.append(prop)
                ##CUSTOM##INTIALIZATION##OF##RD##AND##FH
                if options.starthigh:
                    newrd= numpy.log((numpy.random.beta(1.5,1.)*1.5+2.)/_REFR0)
                    #newfh= numpy.random.beta(1.5,1.)
                else:
                    newrd= numpy.log((numpy.random.beta(1.,1.5)*1.5+2.)/_REFR0)
                    #newfh= numpy.random.beta(1.,1.5)
                trd= numpy.exp(newrd)*_REFR0
                texp= numpy.fabs((newrd-2.75))/0.75*1.+1.
                if trd > 2.75:
                    newfh= numpy.random.beta(texp,1.)
                else:
                    newfh= numpy.random.beta(1.,texp)
                tpotparams= numpy.array(get_potparams(thisparams,options,npops))
                if options.potential.lower() == 'dpdiskplhalofixbulgeflatwgasalt':
                    tpotparams[0]= newrd
                    tpotparams[3]= newfh
                else:
                    raise NotImplementedError("custom markovpy not implemented for this potential")
                thisparams= set_potparams(tpotparams,thisparams,
                                          options,npops)
                print thisparams
                tlnp= lnpdf(thisparams,*pdf_params,testgood=True)
            lnprobs.append(tlnp)
            initial_position.append(numpy.array(thisparams))
        lnprobs= None
        if not lnprobs is None: lnprobs= numpy.array(lnprobs)
    #Set up sampler
    if use_emcee:
        sampler = emcee.EnsembleSampler(nwalkers,ndim,
                                        lambdafunc,
                                        threads=threads,
                                        pool=pool)
        #Sample
        if pos is None:
            pos, prob, state= sampler.run_mcmc(initial_position,nmarkovsamples,
                                               rstate0=numpy.random.mtrand.RandomState().get_state(),
                                               
                                               lnprob0=lnprobs)
        else:
            pos, prob, state= sampler.run_mcmc(pos,
                                               nmarkovsamples,
                                               rstate0=state,
                                               lnprob0=prob)
        #Get chain
        chain= sampler.chain
    else:
        sampler = mpy.EnsembleSampler(nwalkers,ndim,
                                      lambdafunc,
                                      threads=threads)
        #Sample
        if pos is None:
            pos, prob, state= sampler.run_mcmc(initial_position,
                                               numpy.random.mtrand.RandomState().get_state(),
                                               nmarkovsamples,
                                               lnprobinit=lnprobs)
        else:
            pos, prob, state= sampler.run_mcmc(pos,state,
                                               nmarkovsamples,
                                               lnprobinit=prob)
        #Get chain
        chain= sampler.get_chain()
    if returnLnprob:
        if use_emcee:
            lnp= sampler.lnprobability()
        else:
            lnp= sampler.get_lnprobability()
        lnps= []
    samples= []
    for ss in range(nmarkovsamples):
        for ww in range(nwalkers):
            thisparams= []
            for pp in range(ndim):
                if use_emcee:
                    thisparams.append(chain[ww,ss,pp])
                else:
                    thisparams.append(chain[ww,pp,ss])
            samples.append(numpy.array(thisparams))
            if returnLnprob:
                lnps.append(lnp[ww,ss])
    if len(samples) > nsamples:
        if returnLnprob:
            lnps= lnps[-nsamples:len(samples)]
        samples= samples[-nsamples:len(samples)]
    if returnLnprob:
        return (samples,lnps,pos,prob,state)
    else:
        return (samples,pos,prob,state)

##SETUP ERROR INTEGRATION
def setup_err_mc(data,options,indx=None):
    #First sample distances, then sample velocities 
    #Calculate r error
    if indx is None:
        ndata= len(data)
        indx= numpy.zeros(ndata,dtype='bool')+True
    else:
        ndata= numpy.sum(indx)
    if options.nmcerr > 1:
        rerr= ivezic_dist_gr(data.dered_g,
                             data.dered_r,
                             data.feh,
                             dg=data.g_err,
                             dr=data.r_err,
                             dfeh=data.feh_err,
                             return_error=True,
                             _returndmr=True)        
        rsamples= numpy.tile(data[indx].dered_r,(options.nmcerr,1)).T\
            +numpy.random.normal(size=(ndata,options.nmcerr))\
            *numpy.tile(rerr[indx],(options.nmcerr,1)).T
        dsamples= ivezic_dist_gr(numpy.tile(data[indx].dered_g-data[indx].dered_r,(options.nmcerr,1)).T+rsamples,
                                 rsamples,
                                 numpy.tile(data[indx].feh,(options.nmcerr,1)).T)[0]
        #if not options.fixdm is None:
        #    distfac= 10.**(options.fixdm/5.)
        #    dsamples*= distfac
        #dsamples[(dsamples > 15.)]= 15. #to make sure samples don't go nuts
        #Transform to XYZ
        lb= bovy_coords.radec_to_lb(data[indx].ra,data[indx].dec,degree=True)
        XYZ= bovy_coords.lbd_to_XYZ(numpy.tile(lb[:,0],(options.nmcerr,1)).T,
                                    numpy.tile(lb[:,1],(options.nmcerr,1)).T,
                                    dsamples,degree=True)
    else: #if nmcerr=1, just use data point
        dsamples= ivezic_dist_gr(data.dered_g,data.dered_r,
                                 data.feh)[0]
        #if not options.fixdm is None:
        #    distfac= 10.**(options.fixdm/5.)
        #    dsamples*= distfac
        #Transform to XYZ
        lb= bovy_coords.radec_to_lb(data.ra,data.dec,degree=True)
        XYZ= bovy_coords.lbd_to_XYZ(lb[:,0],
                                    lb[:,1],
                                    dsamples,degree=True).reshape((1,len(data),3))
    #draw velocity
    vdraws= numpy.zeros(len(data),dtype=list)
    xdraws= numpy.zeros(len(data),dtype=list)
    outvxvyvz= numpy.zeros((len(data),3,options.nmcerr))
    for ii in range(len(data)):
        if not indx[ii]: continue
#        xdraws[ii]= []
#        vdraws[ii]= []
        #First cholesky the covariance
        #L= linalg.cholesky(cov_vxvyvz[ii,:,:],lower=True)
        #Draw vr and proper motions
        if options.nmcerr > 1:
            vrsamples= data[ii].vr\
                +numpy.random.normal(size=(options.nmcerr))*data[ii].vr_err
            pmrasamples= data[ii].pmra\
                +numpy.random.normal(size=(options.nmcerr))*data[ii].pmra_err
            pmdecsamples= data[ii].pmdec\
                +numpy.random.normal(size=(options.nmcerr))*data[ii].pmdec_err
            pmllpmbb= bovy_coords.pmrapmdec_to_pmllpmbb(pmrasamples,
                                                        pmdecsamples,
                                                        numpy.ones(options.nmcerr)*data[ii].ra,
                                                        numpy.ones(options.nmcerr)*data[ii].dec,degree=True)
            l,b= bovy_coords.radec_to_lb(data[ii].ra,data[ii].dec,degree=True)
            if options.fitdm:
                outvxvyvz[ii,0,:]= vrsamples
                outvxvyvz[ii,1,:]= pmllpmbb[:,0]
                outvxvyvz[ii,2,:]= pmllpmbb[:,1]
            else:
                vxvyvz= bovy_coords.vrpmllpmbb_to_vxvyvz(vrsamples,
                                                         pmllpmbb[:,0],
                                                         pmllpmbb[:,1],
                                                         numpy.ones(options.nmcerr)*l,
                                                         numpy.ones(options.nmcerr)*b,
                                                         dsamples[ii,:],
                                                         degree=True)
                outvxvyvz[ii,:,:]= vxvyvz.T
        else:
            vrsamples= data[ii].vr
            pmrasamples= data[ii].pmra
            pmdecsamples= data[ii].pmdec
            pmll,pmbb= bovy_coords.pmrapmdec_to_pmllpmbb(pmrasamples,
                                                        pmdecsamples,
                                                        data[ii].ra,
                                                        data[ii].dec,degree=True)
            l,b= bovy_coords.radec_to_lb(data[ii].ra,data[ii].dec,degree=True)
            if options.fitdm:
                outvxvyvz[ii,0,0]= vrsamples
                outvxvyvz[ii,1,0]= pmll
                outvxvyvz[ii,2,0]= pmbb
            else:
                vxvyvz= bovy_coords.vrpmllpmbb_to_vxvyvz(vrsamples,
                                                         pmll,
                                                         pmbb,
                                                         l,
                                                         b,
                                                         dsamples[ii],
                                                         degree=True)
                outvxvyvz[ii,0,0]= vxvyvz[0]
                outvxvyvz[ii,1,0]= vxvyvz[1]
                outvxvyvz[ii,2,0]= vxvyvz[2]
        if (not options.fitro and not options.fitvsun and not options.fitvtsun \
                and not options.fitdm) \
                or ((options.fitvsun or options.fitvtsun) and not options.fitro \
                        and not options.fitdm) \
                        or (not (options.fitvsun or options.fitvtsun) and options.fitro \
                                and not options.fitdm) \
                                or options.fitdm:
            pass
        else:
            #Load into vdraws
            if options.nmcerr == 1:
                xdraws[ii]= [XYZ[0,ii,:]]
                vdraws[ii]= [vxvyvz]
            else:
                vdraws[ii]= [vxvyvz[jj,:] for jj in range(options.nmcerr)]
                xdraws[ii]= [XYZ[jj,ii,:] for jj in range(options.nmcerr)]
        """
        for jj in range(options.nmcerr):
            #vn= numpy.random.normal(size=(3))
            #if options.nmcerr == 1:
            #    vdraws[ii].append(vxvyvz[ii,:]) #if 1, just use observation
            #else:
            #    vdraws[ii].append(numpy.dot(L,vn)+vxvyvz[ii,:])
            if options.nmcerr == 1:
                xdraws[ii].append(XYZ[ii,:])
                vdraws[ii].append(vxvyvz)
            else:
                vdraws[ii].append(vxvyvz[jj,:])
                         xdraws[ii].append(XYZ[jj,ii,:])
         """
    if not options.fitro and not options.fitvsun and not options.fitvtsun \
            and not options.fitdm:
        if not options.fixro is None:
            ro= options.fixro
        else:
            ro= 1.
        vsun = [_VRSUN,_VTSUN,_VZSUN]
        #Do coordinate transformations
        outvxvyvz[:,0,:]-= vsun[0]
        outvxvyvz[:,1,:]+= vsun[1]
        outvxvyvz[:,2,:]+= vsun[2]
        R= ((_REFR0*ro-XYZ[:,:,0])**2.+(XYZ[:,:,1])**2.)**0.5 
        z= XYZ[:,:,2]+_ZSUN
        #Rotate to Galactocentric frame
        cosphi= (_REFR0*ro-XYZ[:,:,0])/R
        sinphi= XYZ[:,:,1]/R
        vR= -outvxvyvz[:,0,:]*cosphi.T+outvxvyvz[:,1,:]*sinphi.T
        vT= outvxvyvz[:,0,:]*sinphi.T+outvxvyvz[:,1,:]*cosphi.T
        if options.marginalizevrvt:
            vR= numpy.random.uniform(size=vR.shape)*300.-150.
            vT= numpy.random.uniform(size=vT.shape)*300.
        elif options.marginalizevt:
            vT= numpy.random.uniform(size=vT.shape)*300.
        #Load into structure
        errstuff= []
        for ii in range(len(data)):
            thiserrstuff= errstuffClass()
            thiserrstuff.R= R[:,ii]
            thiserrstuff.vR= vR[ii,:]
            thiserrstuff.vT= vT[ii,:]
            thiserrstuff.z= z[:,ii]
            thiserrstuff.vz= outvxvyvz[ii,2,:]
            errstuff.append(thiserrstuff)
    elif (options.fitvsun or options.fitvtsun) and not options.fitro \
            and not options.fitdm:
        if not options.fixro is None:
            ro= options.fixro
        else:
            ro= 1.
        R= ((_REFR0*ro-XYZ[:,:,0])**2.+(XYZ[:,:,1])**2.)**0.5 
        z= XYZ[:,:,2]+_ZSUN
        #Rotate to Galactocentric frame
        cosphi= (_REFR0*ro-XYZ[:,:,0])/R
        sinphi= XYZ[:,:,1]/R
        #Load into structure
        errstuff= []
        for ii in range(len(data)):
            thiserrstuff= errstuffClass()
            thiserrstuff.R= R[:,ii]
            thiserrstuff.vx= outvxvyvz[ii,0,:]
            thiserrstuff.vy= outvxvyvz[ii,1,:]
            thiserrstuff.z= z[:,ii]
            thiserrstuff.vz= outvxvyvz[ii,2,:]
            thiserrstuff.cosphi= cosphi[:,ii]
            thiserrstuff.sinphi= sinphi[:,ii]
            errstuff.append(thiserrstuff)
    elif not (options.fitvsun or options.fitvtsun) and options.fitro \
            and not options.fitdm:
        raise NotImplementedError("fitro needs work")
        vsun = [_VRSUN,_PMSGRA*_REFR0,_VZSUN]
        #Do coordinate transformations
        outvxvyvz[:,0,:]-= vsun[0]
        outvxvyvz[:,1,:]+= vsun[1]
        outvxvyvz[:,2,:]+= vsun[2]
        z= XYZ[:,:,2]+_ZSUN
        #Load into structure
        errstuff= []
        for ii in range(len(data)):
            thiserrstuff= errstuffClass()
            thiserrstuff.x= XYZ[:,ii,0]
            thiserrstuff.y= XYZ[:,ii,1]
            thiserrstuff.vx= outvxvyvz[ii,0,:]
            thiserrstuff.vy= outvxvyvz[ii,1,:]
            thiserrstuff.z= z[:,ii]
            thiserrstuff.vz= outvxvyvz[ii,2,:]
            errstuff.append(thiserrstuff)
    elif options.fitdm:
        #Load into structure
        errstuff= []
        for ii in range(len(data)):
            thiserrstuff= errstuffClass()
            thiserrstuff.x= XYZ[:,ii,0]
            thiserrstuff.y= XYZ[:,ii,1]
            thiserrstuff.z= XYZ[:,ii,2]
            thiserrstuff.vr= outvxvyvz[ii,0,:]
            thiserrstuff.pmll= outvxvyvz[ii,1,:]
            thiserrstuff.pmbb= outvxvyvz[ii,2,:]
            errstuff.append(thiserrstuff)      
    else:
        data= _append_field_recarray(data,'vdraws',vdraws)
        data= _append_field_recarray(data,'xdraws',xdraws)
        #Load into structure
        errstuff= []
        for ii in range(len(data)):
            thiserrstuff= errstuffClass()
            thiserrstuff.R= 0.
    return (data,errstuff)

class errstuffClass:
    """empty class for error stuff"""
    pass

##INITIALIZATION
def initialize(options,fehs,afes):
    """Function to initialize the fit; uses fehs and afes to initialize using MAPS"""
    p= []
    if options.fitdvt:
        p.append((options.fixvc-235.)/_REFV0)
    if options.fitdm:
        p.append(0.)
    if options.fitro:
        p.append(1.)
    if options.fitvsun:
        p.extend([0.,1.1,0.])
    elif options.fitvtsun:
        p.append(1.1)
    if not options.fixro is None:
        ro= options.fixro
    else:
        ro= 1.
    mapfehs= monoAbundanceMW.fehs(k=(options.sample.lower() == 'k'))
    mapafes= monoAbundanceMW.afes(k=(options.sample.lower() == 'k'))
    indx= (mapfehs != -0.45)*(mapafes != 0.075) #Pop this one, bc sz is crazy
    mapfehs= mapfehs[indx]
    mapafes= mapafes[indx]
    for ii in range(len(fehs)):
        if options.dfmodel.lower() == 'qdf':
            #Find nearest mono-abundance bin that has a measurement
            abindx= numpy.argmin((fehs[ii]-mapfehs)**2./0.01 \
                                     +(afes[ii]-mapafes)**2./0.0025)
            feh, afe= mapfehs[abindx], mapafes[abindx]
            if _INITWESTIMATES:
                lnhr, lnsr, lnsz= approxFitResult(feh,afe,sample=options.sample)
                thishr= numpy.exp(lnhr)*8.
                thissr= numpy.exp(lnsr)*220.
                thissz= numpy.exp(lnsz)*220.
            elif _SMOOTHDISPS:
                #Smooth sz
                up= monoAbundanceMW.sigmaz(feh+0.1,afe,
                                           k=(options.sample.lower() == 'k'))
                down= monoAbundanceMW.sigmaz(feh-0.1,afe,
                                             k=(options.sample.lower() == 'k'))
                left= monoAbundanceMW.sigmaz(feh,afe-0.05,
                                             k=(options.sample.lower() == 'k'))
                right= monoAbundanceMW.sigmaz(feh,afe+0.05,
                                              k=(options.sample.lower() == 'k'))
                here= monoAbundanceMW.sigmaz(feh,afe,
                                             k=(options.sample.lower() == 'k'))
                upright= monoAbundanceMW.sigmaz(feh+0.1,afe+0.05,
                                                k=(options.sample.lower() == 'k'))
                upleft= monoAbundanceMW.sigmaz(feh+0.1,afe-0.05,
                                               k=(options.sample.lower() == 'k'))
                downright= monoAbundanceMW.sigmaz(feh-0.1,afe+0.05,
                                                  k=(options.sample.lower() == 'k'))
                downleft= monoAbundanceMW.sigmaz(feh-0.1,afe-0.05,
                                                 k=(options.sample.lower() == 'k'))
                allsz= numpy.array([here,up,down,left,right,upright,upleft,downright,downleft])
                indx= True-numpy.isnan(allsz)
                thissz= numpy.mean(allsz[True-numpy.isnan(allsz)])
                #Smooth sr
                feh, afe= mapfehs[abindx], mapafes[abindx]
                up= monoAbundanceMW.sigmar(feh+0.1,afe,
                                           k=(options.sample.lower() == 'k'))
                down= monoAbundanceMW.sigmar(feh-0.1,afe,
                                             k=(options.sample.lower() == 'k'))
                left= monoAbundanceMW.sigmar(feh,afe-0.05,
                                             k=(options.sample.lower() == 'k'))
                right= monoAbundanceMW.sigmar(feh,afe+0.05,
                                              k=(options.sample.lower() == 'k'))
                here= monoAbundanceMW.sigmar(feh,afe,
                                             k=(options.sample.lower() == 'k'))
                upright= monoAbundanceMW.sigmar(feh+0.1,afe+0.05,
                                                k=(options.sample.lower() == 'k'))
                upleft= monoAbundanceMW.sigmar(feh+0.1,afe-0.05,
                                               k=(options.sample.lower() == 'k'))
                downright= monoAbundanceMW.sigmar(feh-0.1,afe+0.05,
                                                  k=(options.sample.lower() == 'k'))
                downleft= monoAbundanceMW.sigmar(feh-0.1,afe-0.05,
                                                 k=(options.sample.lower() == 'k'))
                allsr= numpy.array([here,up,down,left,right,upright,upleft,downright,downleft])
                indx= True-numpy.isnan(allsr)
                thissr= numpy.mean(allsr[True-numpy.isnan(allsr)])
                thishr= 0.9*monoAbundanceMW.hr(mapfehs[abindx],mapafes[abindx],
                                               k=(options.sample.lower() == 'k'))
            else:
                thissz= monoAbundanceMW.sigmaz(feh,afe,
                                               k=(options.sample.lower() == 'k'))
                thissr= monoAbundanceMW.sigmar(feh,afe,
                                               k=(options.sample.lower() == 'k'))
                thishr= 0.9*monoAbundanceMW.hr(mapfehs[abindx],mapafes[abindx],
                                               k=(options.sample.lower() == 'k'))
            #Put everthing together
            p.extend([numpy.log(thishr/_REFR0),
                      numpy.log(thissr/_REFV0), #sigmaR
                      numpy.log(thissz/_REFV0), #sigmaZ
                      numpy.log(8./_REFR0),numpy.log(7./_REFR0)]) #hsigR, hsigZ
            #Outlier fraction
            p.append(0.05)
    if options.potential.lower() == 'flatlog' or options.potential.lower() == 'flatlogdisk':
        p.extend([.7,1.])
    elif options.potential.lower() == 'mwpotentialsimplefit':
        if options.fixvo is None:
            p.extend([-1.,1.,-3.,0.5])
        else:
            p.extend([-1.,-3.,0.5])
    elif options.potential.lower() == 'mwpotentialfixhalo':
        p.extend([-1.,1.,-3.,0.5,0.95])
    elif options.potential.lower() == 'mwpotentialfixhaloflat':
        p.extend([-1.,1.,-3.,0.,0.05])
    elif options.potential.lower() == 'mpdiskplhalofixbulgeflat' \
            or options.potential.lower() == 'dpdiskplhalofixbulgeflat' \
            or options.potential.lower() == 'dpdiskplhalofixbulgeflatwgas':
        p.extend([-1.39,1.,-3.,0.,1.15])
        #p.extend([-1.,1.,-3.,0.,1.35])
        #p.extend([-.69,1.07,-3.,0.,2.2])
    elif options.potential.lower() == 'dpdiskplhalodarkdiskfixbulgeflatwgas':
        p.extend([-1.39,1.,-3.,0.,1.15,0.])
    elif options.potential.lower() == 'dpdiskplhalofixbulgeflatwgasalt':
        #p.extend([-1.39,1.,-3.,0.3,-0.05])
        #p.extend([-1.,1.,-3.,0.5,0.])
        if options.starthigh:
            p.extend([-.69,1.07,-3.,0.7,0.02])
        else:
            p.extend([-1.39,1.,-3.,0.3,-0.05])
    elif options.potential.lower() == 'dpdiskflplhalofixbulgeflatwgas':
        p.extend([-1.,1.,-3.,0.,1.,-0.8])
    elif options.potential.lower() == 'mpdiskflplhalofixplfixbulgeflat':
        p.extend([-1.,1.,-3.,0.,1.])
    return numpy.array(p)

##SETUP DOMAIN FOR MARKOVPY
def setup_domain(options,npops):
    """Setup isDomainFinite, domain for markovpy"""
    isDomainFinite= []
    domain= []
    step= []
    create_method= []
    if options.fitdvt:
        isDomainFinite.append([True,True])
        domain.append([-0.5,0.5])
        step.append(0.2)
        create_method.append('whole')
    if options.fitdm:
        isDomainFinite.append([True,True])
        domain.append([-0.5,0.5])
        step.append(0.2)
        create_method.append('whole')
    if options.fitro:
        isDomainFinite.append([True,True])
        domain.append([5./_REFR0,11./_REFR0])
        step.append(0.2)
        create_method.append('whole')
    if options.fitvsun:
        isDomainFinite.append([False,False])
        domain.append([0.,0.])
        step.append(0.2)
        create_method.append('step_out')
        isDomainFinite.append([False,False])
        domain.append([0.,0.])
        step.append(0.2)
        create_method.append('step_out')
        isDomainFinite.append([False,False])
        domain.append([0.,0.])
        step.append(0.2)
        create_method.append('step_out')
    elif options.fitvsun:
        isDomainFinite.append([False,False])
        domain.append([0.,0.])
        step.append(0.2)
        create_method.append('step_out')
    for ii in range(npops):
        if options.dfmodel.lower() == 'qdf':
            domain.append([-2.77,2.53])
            domain.append([-3.1,-0.4])
            domain.append([-3.1,-0.4])
            domain.append([-2.77,2.53])
            domain.append([-2.77,2.53])
            domain.append([0.,1.])
            isDomainFinite.append([True,True])
            isDomainFinite.append([True,True])
            isDomainFinite.append([True,True])
            isDomainFinite.append([True,True])
            isDomainFinite.append([True,True])
            isDomainFinite.append([True,True])
            step.append(0.2)
            step.append(0.2)
            step.append(0.2)
            step.append(0.2)
            step.append(0.2)
            step.append(0.2)
            create_method.append('whole')
            create_method.append('whole')
            create_method.append('whole')
            create_method.append('whole')
            create_method.append('whole')
            create_method.append('whole')
    if options.potential.lower() == 'flatlog' or options.potential.lower() == 'flatlogdisk':
        isDomainFinite.append([True,False])
        if not options.noqprior:
            domain.append([0.53,0.])
        else:
            domain.append([0.0,0.])
        step.append(0.2)
        create_method.append('step_out')
        isDomainFinite.append([True,True])
        domain.append([100./_REFV0,350./_REFV0])
        step.append(0.2)
        create_method.append('whole')
    elif options.potential.lower() == 'mpdiskplhalofixbulgeflat' \
            or options.potential.lower() == 'dpdiskplhalofixbulgeflat' \
            or options.potential.lower() == 'dpdiskplhalofixbulgeflatwgas':
        domain.append([-2.1,-0.3])
        isDomainFinite.append([True,True])
        step.append(0.2)
        create_method.append('whole')
        domain.append([100./_REFV0,350./_REFV0])
        isDomainFinite.append([True,True])
        step.append(0.2)
        create_method.append('whole')
        domain.append([-5.1,1.4])
        isDomainFinite.append([True,True])
        step.append(0.2)
        create_method.append('whole')
        domain.append([-0.5*30.,0.04*30.])
        isDomainFinite.append([True,True])
        step.append(0.2)
        create_method.append('whole')
        domain.append([0.,3.])
        isDomainFinite.append([True,True])
        step.append(0.2)
        create_method.append('whole')
    elif options.potential.lower() == 'dpdiskplhalofixbulgeflatwgasalt':
        domain.append([-2.1,-0.3])
        isDomainFinite.append([True,True])
        step.append(0.5)
        create_method.append('whole')
        domain.append([100./_REFV0,350./_REFV0])
        isDomainFinite.append([True,True])
        step.append(0.1)
        create_method.append('whole')
        domain.append([-5.1,1.4])
        isDomainFinite.append([True,True])
        step.append(0.2)
        create_method.append('whole')
        domain.append([0.,1.])
        isDomainFinite.append([True,True])
        step.append(0.2)
        create_method.append('whole')
        domain.append([-0.5*30.,0.04*30.])
        isDomainFinite.append([True,True])
        step.append(0.2)
        create_method.append('whole')
    elif options.potential.lower() == 'mwpotentialsimplefit' \
            or options.potential.lower() == 'mwpotentialfixhalo' \
            or options.potential.lower() == 'mwpotentialfixhaloflat' \
            or options.potential.lower() == 'dpdiskflplhalofixbulgeflatwgas' \
            or options.potential.lower() == 'mpdiskflplhalofixplfixbulgeflat':
        raise NotImplementedError("setup domain for sampling of mwpotentialsimplefit not setup")
    return (isDomainFinite,domain,step,create_method)

def setup_domain_indiv_df(options,npops):
    """Setup isDomainFinite, domain for markovpy"""
    raise NotImplementedError("setup_domain_indiv_df needs to be edited for new priors")
    isDomainFinite= []
    domain= []
    if options.dfmodel.lower() == 'qdf':
        ndfparams= get_ndfparams(options)
        for jj in range(ndfparams-1):
            isDomainFinite.append([False,False])
            domain.append([0.,0.])
        #Outlier fraction
        isDomainFinite.append([True,True])
        domain.append([0.,1.])
    return (isDomainFinite,domain)

def setup_domain_indiv_potential(options,npops):
    """Setup isDomainFinite, domain for markovpy for sampling the potential"""
    raise NotImplementedError("setup_domain_indiv_potential needs to be edited for new priors")
    isDomainFinite= []
    domain= []
    if options.potential.lower() == 'flatlog' or options.potential.lower() == 'flatlogdisk':
        isDomainFinite.append([True,False])
        if not options.noqprior:
            domain.append([0.53,0.])
        else:
            domain.append([0.0,0.])
        isDomainFinite.append([True,True])
        domain.append([100./_REFV0,350./_REFV0])
    elif options.potential.lower() == 'mpdiskplhalofixbulgeflat' \
            or options.potential.lower() == 'dpdiskplhalofixbulgeflat' \
            or options.potential.lower() == 'dpdiskplhalofixbulgeflatwgas':
        domain.append([-2.1,-0.3])
        isDomainFinite.append([True,True])
        domain.append([100./_REFV0,350./_REFV0])
        isDomainFinite.append([True,True])
        domain.append([-5.1,1.4])
        isDomainFinite.append([True,True])
        domain.append([-0.5,0.04])
        isDomainFinite.append([True,True])
        domain.append([0.,3.])
        isDomainFinite.append([True,True])
    elif options.potential.lower() == 'mwpotentialsimplefit' \
            or options.potential.lower() == 'mwpotentialfixhalo' \
            or options.potential.lower() == 'mwpotentialfixhaloflat' \
            or options.potential.lower() == 'dpdiskflplhalofixbulgeflatwgas' \
            or options.potential.lower() == 'mpdiskflplhalofixplfixbulgeflat':
        raise NotImplementedError("setup domain for sampling of mwpotentialsimplefit not setup")
    return (isDomainFinite,domain)

##GET AND SET THE PARAMETERS
def get_potparams(p,options,npops):
    """Function that returns the set of potential parameters for these options"""
    startindx= 0
    if options.fitdvt: startindx+= 1
    if options.fitdm: startindx+= 1
    if options.fitro: startindx+= 1
    if options.fitvsun: startindx+= 3
    elif options.fitvsun: startindx+= 1
    ndfparams= get_ndfparams(options)
    startindx+= ndfparams*npops
    if options.potential.lower() == 'flatlog' or options.potential.lower() == 'flatlogdisk':
        return (p[startindx],p[startindx+1]) #q, vo
    elif options.potential.lower() == 'mwpotential':
        return (1.) #vo
    elif options.potential.lower() == 'mwpotentialsimplefit':
        if not options.fixvo is None:
            return (p[startindx],p[startindx+1],p[startindx+2]) # hr, hz,ampd
        else:
            return (p[startindx],p[startindx+1],p[startindx+2],p[startindx+3]) # hr, vo, hz,ampd
    elif options.potential.lower() == 'mwpotentialfixhalo' \
            or options.potential.lower() == 'mwpotentialfixhaloflat':
        return (p[startindx],p[startindx+1],p[startindx+2],p[startindx+3],p[startindx+4]) # hr, vo, hz,ampd+h, ampd/d+h
    elif options.potential.lower() == 'mpdiskplhalofixbulgeflat' \
            or options.potential.lower() == 'dpdiskplhalofixbulgeflat' \
            or options.potential.lower() == 'dpdiskplhalofixbulgeflatwgas' \
            or options.potential.lower() == 'dpdiskplhalofixbulgeflatwgasalt' \
            or options.potential.lower() == 'dpdiskplhalofixcutbulgeflatwgasalt':
        return (p[startindx],p[startindx+1],p[startindx+2],p[startindx+3],p[startindx+4]) # hr, vo, hz,dlnvcdlnr,power-law halo
    elif options.potential.lower() == 'dpdiskplhalodarkdiskfixbulgeflatwgasalt':
        return (p[startindx],p[startindx+1],p[startindx+2],p[startindx+3],p[startindx+4],p[startindx+5]) # hr, vo, hz,dlnvcdlnr,power-law halo, relative dark disk density
    elif options.potential.lower() == 'dpdiskflplhalofixbulgeflatwgas':
        return (p[startindx],p[startindx+1],p[startindx+2],p[startindx+3],p[startindx+4],p[startindx+5]) # hr, vo, hz,dlnvcdlnr,flattening, power-law halo
    elif options.potential.lower() == 'mpdiskflplhalofixplfixbulgeflat':
        return (p[startindx],p[startindx+1],p[startindx+2],p[startindx+3],p[startindx+4]) # hr, vo, hz,dlnvcdlnr,halo flattening

def get_vo(p,options,npops):
    """Function that returns the vo parameter for these options"""
    if not options.fixvo is None: return options.fixvo
    startindx= 0
    if options.fitdvt: startindx+= 1
    if options.fitdm: startindx+= 1
    if options.fitro: startindx+= 1
    if options.fitvsun: startindx+= 3
    elif options.fitvsun: startindx+= 1
    ndfparams= get_ndfparams(options)
    startindx+= ndfparams*npops
    if options.potential.lower() == 'flatlog' or options.potential.lower() == 'flatlogdisk':
        return p[startindx+1]
    elif options.potential.lower() == 'mwpotential':
        return 1.
    elif options.potential.lower() == 'mwpotentialsimplefit':
        return p[startindx+1]
    elif options.potential.lower() == 'mwpotentialfixhalo' \
            or options.potential.lower() == 'mwpotentialfixhaloflat' \
            or options.potential.lower() == 'mpdiskplhalofixbulgeflat' \
            or options.potential.lower() == 'dpdiskplhalofixbulgeflat' \
            or options.potential.lower() == 'dpdiskflplhalofixbulgeflatwgas' \
            or options.potential.lower() == 'dpdiskplhalofixbulgeflatwgasalt' \
            or options.potential.lower() == 'dpdiskplhalofixcutbulgeflatwgasalt' \
            or options.potential.lower() == 'dpdiskplhalodarkdiskfixbulgeflatwgasalt' \
            or options.potential.lower() == 'dpdiskplhalofixbulgeflatwgas' \
            or options.potential.lower() == 'mpdiskflplhalofixplfixbulgeflat':
        return p[startindx+1]
    elif options.potential.lower() == 'bti' \
            or options.potential.lower() == 'btii':
        return 1.

def get_outfrac(p,indx,options):
    """Function that returns the outlier fraction for these options"""
    startindx= 0
    if options.fitdvt: startindx+= 1
    if options.fitdm: startindx+= 1
    if options.fitro: startindx+= 1
    if options.fitvsun: startindx+= 3
    elif options.fitvsun: startindx+= 1
    ndfparams= get_ndfparams(options)
    startindx+= ndfparams*indx
    if options.dfmodel.lower() == 'qdf':
        return p[startindx+5]

def set_potparams(p,params,options,npops):
    """Function that sets the set of potential parameters for these options"""
    startindx= 0
    if options.fitdvt: startindx+= 1
    if options.fitdm: startindx+= 1
    if options.fitro: startindx+= 1
    if options.fitvsun: startindx+= 3
    elif options.fitvsun: startindx+= 1
    ndfparams= get_ndfparams(options)
    startindx+= ndfparams*npops
    if options.potential.lower() == 'flatlog' or options.potential.lower() == 'flatlogdisk':
        params[startindx]= p[0]
        params[startindx+1]= p[1]
    elif options.potential.lower() == 'mwpotentialsimplefit':
        params[startindx]= p[0]
        params[startindx+1]= p[1]
        params[startindx+2]= p[2]
        if options.fixvo is None:
            params[startindx+3]= p[3]
    elif options.potential.lower() == 'mwpotentialfixhalo' \
            or options.potential.lower() == 'mwpotentialfixhaloflat' \
            or options.potential.lower() == 'mpdiskplhalofixbulgeflat' \
            or options.potential.lower() == 'dpdiskplhalofixbulgeflat' \
            or options.potential.lower() == 'dpdiskplhalofixbulgeflatwgas' \
            or options.potential.lower() == 'dpdiskplhalofixbulgeflatwgasalt' \
            or options.potential.lower() == 'dpdiskplhalofixcutbulgeflatwgasalt' \
            or options.potential.lower() == 'mpdiskflplhalofixplfixbulgeflat':
        params[startindx]= p[0]
        params[startindx+1]= p[1]
        params[startindx+2]= p[2]
        params[startindx+3]= p[3]
        params[startindx+4]= p[4]
    elif options.potential.lower() == 'dpdiskplhalodarkdiskfixbulgeflatwgasalt':
        params[startindx]= p[0]
        params[startindx+1]= p[1]
        params[startindx+2]= p[2]
        params[startindx+3]= p[3]
        params[startindx+4]= p[4]
        params[startindx+5]= p[5]
    elif options.potential.lower() == 'dpdiskflplhalofixbulgeflatwgas':
        params[startindx]= p[0]
        params[startindx+1]= p[1]
        params[startindx+2]= p[2]
        params[startindx+3]= p[3]
        params[startindx+4]= p[4]
        params[startindx+5]= p[5]
    elif options.potential.lower() == 'bti' \
            or options.potential.lower() == 'btii':
        pass
    return params

def get_dfparams(p,indx,options,log=False):
    """Function that returns the set of DF parameters for population indx for these options,
    Returns them as a set such that they can be given to the initialization"""
    startindx= 0
    if options.fitdvt: startindx+= 1
    if options.fitdm: startindx+= 1
    if options.fitro: startindx+= 1
    if options.fitvsun: startindx+= 3
    elif options.fitvsun: startindx+= 1
    ndfparams= get_ndfparams(options)
    startindx+= ndfparams*indx
    if options.dfmodel.lower() == 'qdf':
        if log:
            return (p[startindx],
                    p[startindx+1],
                    p[startindx+2],
                    p[startindx+3],
                    p[startindx+4],
                    p[startindx+5])
        else:
            return (numpy.exp(p[startindx]),
                    numpy.exp(p[startindx+1]),
                    numpy.exp(p[startindx+2]),
                    numpy.exp(p[startindx+3]),
                    numpy.exp(p[startindx+4]),
                    p[startindx+5]) #outlier fraction never gets exponentiated
        
def set_dfparams(p,params,indx,options):
    """Function that sets the set of DF parameters for population indx for these options"""
    startindx= 0
    if options.fitdvt: startindx+= 1
    if options.fitdm: startindx+= 1
    if options.fitro: startindx+= 1
    if options.fitvsun: startindx+= 3
    elif options.fitvsun: startindx+= 1
    ndfparams= get_ndfparams(options)
    startindx+= ndfparams*indx
    if options.dfmodel.lower() == 'qdf':
        for ii in range(ndfparams):
            params[startindx+ii]= p[ii]
    return params

def get_ndfparams(options):
    """Function that returns the number of DF parameters for a single population"""
    if options.dfmodel.lower() == 'qdf':
        return 6 #5 + outlier fraction

def get_npotparams(options):
    """Function that returns the number of potential parameters"""
    if options.potential.lower() == 'flatlog' or options.potential.lower() == 'flatlogdisk':
        return 2
    elif options.potential.lower() == 'mwpotentialsimplefit':
        if not options.fixvo is None:
            return 3
        else:
            return 4
    elif options.potential.lower() == 'mwpotentialfixhalo' \
            or options.potential.lower() == 'mwpotentialfixhaloflat' \
            or options.potential.lower() == 'mpdiskplhalofixbulgeflat' \
            or options.potential.lower() == 'dpdiskplhalofixbulgeflat' \
            or options.potential.lower() == 'dpdiskplhalofixbulgeflatwgas' \
            or options.potential.lower() == 'dpdiskplhalofixbulgeflatwgasalt' \
            or options.potential.lower() == 'dpdiskplhalofixcutbulgeflatwgasalt' \
            or options.potential.lower() == 'mpdiskflplhalofixplfixbulgeflat':
        return 5
    elif options.potential.lower() == 'dpdiskflplhalofixbulgeflatwgas' \
            or options.potential.lower() == 'dpdiskplhalodarkdiskfixbulgeflatwgas':
        return 6
    elif options.potential.lower() == 'bti' \
            or options.potential.lower() == 'btii':
        return 0

def get_ro(p,options):
    """Function that returns R0 for these options"""
    if options.fitro:
        return p[options.fitdm+options.fitdvt]
    elif not options.fixro is None:
        return options.fixro
    else:
        return 1.

def get_dm(p,options):
    """Function that returns the change in distance modulus these options"""
    if options.fitdm:
        return p[options.fitdvt]
    elif not options.fixdm is None:
        return options.fixdm
    else:
        return 0.

def get_dvt(p,options):
    """Function that returns the change in distance modulus these options"""
    if options.fitdvt:
        return p[0]
    elif not options.fixdvt is None:
        return options.fixdvt
    else:
        return 0.

def get_vsun(p,options):
    """Function to return motion of the Sun in the Galactocentric reference frame"""
    ro= get_ro(p,options)
    startindx= 0
    if options.fitdvt: startindx+= 1
    if options.fitdm: startindx+= 1
    if options.fitro: startindx+= 1
    if options.fitvsun:
        return (p[startindx],p[startindx+1],p[startindx+2])
    elif options.fitvtsun:
        return p[startindx]
    else:
        return (_VRSUN/_REFV0,_PMSGRA*ro*_REFR0/_REFV0,_VZSUN/_REFV0)

##FIDUCIAL DENSITIES FOR MC NORMALIZATION INTEGRATION
def fidDens(R,z,hr,hz,dummy):
    """Fiducial exponential density for normalization integral"""
    return 1./hz*numpy.exp(-(R-_REFR0)/hr-numpy.fabs(z)/hz)

def outDens(R,z,dummy):
    """Fiducial outlier density for normalization integral (constant)"""
    return 1./12.

def interpDens(R,z,surfInterp):
    """Function to give density using the interpolated representation"""
    return numpy.exp(surfInterp.ev(R/_REFR0,numpy.fabs(z)/_REFR0))

def interpDenswoutlier(R,z,params):
    """Function to give density using the interpolated representation"""
    return numpy.exp(params[0].ev(R/_REFR0,numpy.fabs(z)/_REFR0))+params[1]

##SAMPLES QA
def print_samples_qa(samples,options,npops):
    print "Mean, standard devs, acor tau, acor mean, acor s ..."
    #potparams
    if options.justpot:
        if options.potential.lower() == 'flatlog' or options.potential.lower() == 'flatlogdisk':
            nparams= 2
        for kk in range(nparams):
            xs= numpy.array([s[kk] for s in samples])
            #Auto-correlation time
            tau, m, s= acor.acor(xs)
            print numpy.mean(xs), numpy.std(xs), tau, m, s
    else:
        for kk in range(len(get_potparams(samples[0],options,npops))):
            xs= numpy.array([get_potparams(s,options,npops)[kk] for s in samples])
        #Auto-correlation time
            tau, m, s= acor.acor(xs)
            print numpy.mean(xs), numpy.std(xs), tau, m, s

##UTILITY
def mylogsumexp(arr,axis=0):
    """Faster logsumexp?"""
    minarr= numpy.amax(arr,axis=axis)
    if axis == 1:
        minarr= numpy.reshape(minarr,(arr.shape[0],1))
    if axis == 0:
        minminarr= numpy.tile(minarr,(arr.shape[0],1))
    elif axis == 1:
        minminarr= numpy.tile(minarr,(1,arr.shape[1]))
    elif axis == None:
        minminarr= numpy.tile(minarr,arr.shape)
    else:
        raise NotImplementedError("'mylogsumexp' not implemented for axis > 2")
    if axis == 1:
        minarr= numpy.reshape(minarr,(arr.shape[0]))
    return minarr+numpy.log(numpy.sum(numpy.exp(arr-minminarr),axis=axis))

def get_options():
    usage = "usage: %prog [options] <savefile>\n\nsavefile= name of the file that the fits will be saved to"
    parser = OptionParser(usage=usage)
    #Data options
    parser.add_option("--sample",dest='sample',default='g',
                      help="Use 'G' or 'K' dwarf sample")
    parser.add_option("--select",dest='select',default='all',
                      help="Select 'all' or 'program' stars")
    parser.add_option("--dfeh",dest='dfeh',default=0.1,type='float',
                      help="FeH bin size")   
    parser.add_option("--dafe",dest='dafe',default=0.05,type='float',
                      help="[a/Fe] bin size")   
    parser.add_option("--singles",action="store_true", dest="singles",
                      default=False,
                      help="If set, perform each bins independently, save as savefile_%i.sav' etc.")
    parser.add_option("--singlebin",dest='singlebin',default=None,type='int',
                      help="Bin when considering a single FeH")
    parser.add_option("--singlefeh",dest='singlefeh',default=None,type='float',
                      help="FeH when considering a single FeH (can be for loo)")   
    parser.add_option("--singleafe",dest='singleafe',default=None,type='float',
                      help="[a/Fe] when considering a single afe (can be for loo)")   
    parser.add_option("--minndata",dest='minndata',default=100,type='int',
                      help="Minimum number of objects in a bin to perform a fit")   
    parser.add_option("-f","--fakedata",dest='fakedata',default=None,
                      help="Name of the fake data filename")
    parser.add_option("--loo",action="store_true", dest="loo",
                      default=False,
                      help="If set, leave out the bin corresponding to singlefeh and singleafe, in leave-one-out fashion")
    parser.add_option("-p","--plate",dest='plate',default=None,type='int',
                      help="Single plate to use")
    parser.add_option("--bmin",dest='bmin',type='float',
                      default=None,
                      help="Minimum Galactic latitude")
    parser.add_option("--zmin",dest='zmin',type='float',
                      default=None,
                      help="Minimum height")
    parser.add_option("--zmax",dest='zmax',type='float',
                      default=None,
                      help="Maximum height")
    parser.add_option("--rmin",dest='rmin',type='float',
                      default=None,
                      help="Minimum radius")
    parser.add_option("--rmax",dest='rmax',type='float',
                      default=None,
                      help="Maximum radius")
    parser.add_option("--snmin",dest='snmin',type='float',
                      default=15.,
                      help="Minimum S/N")
    parser.add_option("--fehmin",dest='fehmin',type='float',
                      default=None,
                      help="Minimum [Fe/H]")
    parser.add_option("--fehmax",dest='fehmax',type='float',
                      default=None,
                      help="Maximum [Fe/H]")
    parser.add_option("--afemin",dest='afemin',type='float',
                      default=None,
                      help="Minimum [a/Fe]")
    parser.add_option("--afemax",dest='afemax',type='float',
                      default=None,
                      help="Maximum [a/Fe]")
    parser.add_option("--indiv_brightlims",action="store_true", 
                      dest="indiv_brightlims",
                      default=False,
                      help="indiv_brightlims keyword for segueSelect")
    #Potential model
    parser.add_option("--potential",dest='potential',default='flatlog',
                      help="Potential model to fit")
    #DF model
    parser.add_option("--dfmodel",dest='dfmodel',default='qdf',#Quasi-isothermal
                      help="DF model to fit")
    #Action-angle options
    parser.add_option("--aAmethod",dest='aAmethod',default='adiabaticgrid',
                      help="action angle method to use")
    parser.add_option("--aAnR",dest='aAnR',default=16,type='int',
                      help="Number of radii for Ez grid in aA")
    parser.add_option("--aAnEz",dest='aAnEz',default=16,type='int',
                      help="Number of Ez grid points in aA")
    parser.add_option("--aAnEr",dest='aAnEr',default=31,type='int',
                      help="Number of Er grid points in aA")
    parser.add_option("--aAnLz",dest='aAnLz',default=31,type='int',
                      help="Number of Lz grid points in aA")
    parser.add_option("--aAzmax",dest='aAzmax',default=1.,type='float',
                      help="zmax in aA")
    parser.add_option("--aARmax",dest='aARmax',default=5.,type='float',
                      help="Rmax in aA")
    parser.add_option("--staeckeldelta",dest='staeckeldelta',
                      default=.45,type='float',
                      help="Focal length for Staeckel approximation")
    parser.add_option("--staeckelRmax",dest='staeckelRmax',
                      default=5.,type='float',
                      help="Rmax in Staeckel AA")
    parser.add_option("--staeckelnLz",dest='staeckelnLz',default=60,type='int',
                      help="Number of Lz grid points in Staeckel")
    parser.add_option("--staeckelnE",dest='staeckelnE',default=50,type='int',
                      help="Number of E grid points in Staeckel")
    parser.add_option("--staeckelnpsi",dest='staeckelnpsi',
                      default=50,type='int',
                      help="Number of psi grid points in Staeckel")
    #Fit options
    parser.add_option("--fitro",action="store_true", dest="fitro",
                      default=False,
                      help="If set, fit for R_0")
    parser.add_option("--fitvsun",action="store_true", dest="fitvsun",
                      default=False,
                      help="If set, fit for v_sun")
    parser.add_option("--fitvtsun",action="store_true", dest="fitvtsun",
                      default=False,
                      help="If set, fit for v_{t,sun}")
    parser.add_option("--ninit",dest='ninit',default=1,type='int',
                      help="Number of initial optimizations to perform (indiv DF + potential w/ fixed DF")
    parser.add_option("--fitdm",action="store_true", dest="fitdm",
                      default=False,
                      help="If set, fit for a distance modulus offset")
    parser.add_option("--fixdm",dest="fixdm",type='float',
                      default=None,
                      help="If set, fix a distance modulus offset")
    parser.add_option("--andistances",action="store_true", dest="andistances",
                      default=False,
                      help="Use An-ish distances")
    parser.add_option("--fitdvt",action="store_true", dest="fitdvt",
                      default=False,
                      help="If set, fit for a vT offset")
    parser.add_option("--fixdvt",dest="fixdvt",type='float',
                      default=None,
                      help="If set, fix a vT offset")
    parser.add_option("--starthigh",action="store_true", dest="starthigh",
                      default=False,
                      help="Start with a high value for RD")
    #Errors
    parser.add_option("--nmcerr",dest='nmcerr',default=30,type='int',
                      help="Number of MC samples to use for Monte Carlo integration over error distribution")
    #Normalization integral
    parser.add_option("--nmcv",dest='nmcv',default=1000,type='int',
                      help="Number of MC samples to use for velocity integration")
    parser.add_option("--ngl",dest='ngl',default=10,type='int',
                      help="Order of Gauss-Legendre quadrature to use for velocity integration")
    parser.add_option("--nscale",dest='nscale',default=1.,type='float',
                      help="Number of 'scales' to calculate the surface scale over when integrating the density (can be float)")
    parser.add_option("--nmc",dest='nmc',default=1000,type='int',
                      help="Number of MC samples to use for Monte Carlo normalization integration")
    parser.add_option("--mcall",action="store_true", dest="mcall",
                      default=False,
                      help="If set, calculate the normalization integral by first calculating the normalization of the exponential density given the best-fit and then calculating the difference with Monte Carlo integration")
    parser.add_option("--mcwdf",action="store_true", dest="mcwdf",
                      default=False,
                      help="If set, calculate the normalization integral by first calculating the normalization of a fiducial DF given the best-fit MAP and then calculating the difference with Monte Carlo integration (based on an actual qdf)")
    parser.add_option("--nomcvalt",action="store_false", dest="mcvalt",
                      default=True,
                      help="If set, calculate the normalization integral by first mcv, but in an alternative implementation")
    parser.add_option("--mcout",action="store_true", dest="mcout",
                      default=False,
                      help="If set, add an outlier model to the mock data used for the normalization integral")
    parser.add_option("--savenorm",dest='savenorm',default=None,
                      help="If set, save normintstuff to this file")
    #priors
    parser.add_option("--noroprior",action="store_true", dest="noroprior",
                      default=False,
                      help="If set, do not apply an Ro prior")
    parser.add_option("--novoprior",action="store_true", dest="novoprior",
                      default=False,
                      help="If set, do not apply a vo prior (default: Bovy et al. 2012)")
    parser.add_option("--fixvo",dest="fixvo",type='float',
                      default=None,
                      help="If set, fix vo=V_c/220 to this value and do not fit for it")
    parser.add_option("--fixro",dest="fixro",type='float',
                      default=None,
                      help="If set, fix ro=R_0/8 kpc to this value")
    parser.add_option("--noqprior",action="store_true", dest="noqprior",
                      default=False,
                      help="If set, do not apply a q prior (default: q > 0.53)")
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
    #Sample?
    parser.add_option("--mcsample",action="store_true", dest="mcsample",
                      default=False,
                      help="If set, sample around the best fit, save in args[1]")
    parser.add_option("--nsamples",dest='nsamples',default=10000,type='int',
                      help="Number of MCMC samples to obtain")
    parser.add_option("--init",dest='init',default=None,
                      help="Initial parameters file")
    parser.add_option("--restart",dest='restart',default=None,
                      help="File that contains previous MCMC samples' output state for restarting the chain")
    parser.add_option("-m","--multi",dest='multi',default=None,type='int',
                      help="number of cpus to use")
    parser.add_option("--multi2",dest='multi2',default=None,type='int',
                      help="number of cpus to use (second one)")
    parser.add_option("--cluster",action="store_true", dest="cluster",
                      default=False,
                      help="If set, fit each bin on a separate node on the cluster (for --singles)")
    parser.add_option("--mpi",action="store_true", dest="mpi",
                      default=False,
                      help="If set, fit a single bin on the cluster in parallel using emcee")
    parser.add_option("--clustername",dest="clustername",
                      default='K',
                      help="Name of the job on the cluster")
    parser.add_option("--clusterholdname",dest="clusterholdname",
                      default='XXX',
                      help="Name of the job to wait until finish on the cluster")
    parser.add_option("--batch1",action="store_true", dest="batch1",
                      default=False,
                      help="If set, submit the first 31 G dwarf bins")
    parser.add_option("--batch2",action="store_true", dest="batch2",
                      default=False,
                      help="If set, submit the last 31 G dwarf bins")
    #Grid-based w/ DF optimization
    parser.add_option("--grid",action="store_true", dest="grid",
                      default=False,
                      help="If set, evaluate the likelihood on a grid in the potential parameters")
    parser.add_option("--nrds",dest='nrds',default=11,type='int',
                      help="Number of scale lengths to use in grid-based search")
    parser.add_option("--nfhs",dest='nfhs',default=11,type='int',
                      help="Number of halo contributions to use in grid-based search")
    parser.add_option("--fixzh",dest='fixzh',default=400.,type='float',
                      help="zh when it is fixed")
    parser.add_option("--fixvc",dest='fixvc',default=220.,type='float',
                      help="vc when it is fixed")
    parser.add_option("--dlnvcdlnr",dest='dlnvcdlnr',default=0.,type='float',
                      help="dlnvcdlnr when it is fixed")
    parser.add_option("--darkdiskrd",dest='darkdiskrd',default=9.,type='float',
                      help="Scale length of the dark disk")
    parser.add_option("--maxiter",dest='maxiter',default=8,type='int',
                      help="Maximum number of iterations in DF optimization")
    #Total grid-based
    parser.add_option("--gridall",action="store_true", dest="gridall",
                      default=False,
                      help="If set, evaluate the likelihood on a grid in the potential parameters *and* the DF parameters")
    parser.add_option("--nhrs",dest='nhrs',default=8,type='int',
                      help="Number of scale lengths to use in grid-based search")
    parser.add_option("--nsrs",dest='nsrs',default=8,type='int',
                      help="Number of radial dispersions to use in grid-based search")
    parser.add_option("--nszs",dest='nszs',default=16,type='int',
                      help="Number of vertical dispersions to use in grid-based search")
    if _NEWESTDFRANGES:
        parser.add_option("--ndvts",dest='ndvts',default=1,type='int',
                          help="Number of dvts to use in grid-based search")
        parser.add_option("--npouts",dest='npouts',default=25,type='int',
                          help="Number of pouts to use in grid-based search")
        parser.add_option("--npouts2",dest='npouts2',default=1,type='int',
                          help="Number of pouts to use in grid-based search")
    elif _NEWDFRANGES:
        parser.add_option("--ndvts",dest='ndvts',default=3,type='int',
                          help="Number of dvts to use in grid-based search")
        parser.add_option("--npouts",dest='npouts',default=25,type='int',
                          help="Number of pouts to use in grid-based search")
    else:
        parser.add_option("--ndvts",dest='ndvts',default=5,type='int',
                          help="Number of dvts to use in grid-based search")
        parser.add_option("--npouts",dest='npouts',default=31,type='int',
                          help="Number of pouts to use in grid-based search")
    parser.add_option("--nszouts",dest='nszouts',default=8,type='int',
                      help="Number of szouts to use in grid-based search")
    parser.add_option("--physicaldfparams",action="store_true",
                      dest="physicaldfparams",
                      default=False,
                      help="Use physical DF parameters, i.e., specify the actual dispersion and first find the DF params which give you the desired dispersions (and scale lengths")
    #Type of fit
    parser.add_option("--justdf",action="store_true", dest="justdf",
                      default=False,
                      help="If set, just fit the DF assuming fixed potential (CURRENTLY ONLY FOR FIT, NOT FOR SAMPLE")
    parser.add_option("--justpot",action="store_true", dest="justpot",
                      default=False,
                      help="If set, just fit the potential assuming fixed DF (CURRENTLY ONLY FOR FIT, NOT FOR SAMPLE")
    parser.add_option("--marginalizevt",action="store_true",
                      dest="marginalizevt",
                      default=False,
                      help="If set, don't use vT data")
    parser.add_option("--marginalizevrvt",action="store_true",
                      dest="marginalizevrvt",
                      default=False,
                      help="If set, don't use vR or vT data")
    parser.add_option("--fixbadvals",action="store_true",
                      dest="fixbadvals",
                      default=False,
                      help="If set, check a previous solution for bad values and re-calculate")
    parser.add_option("--conditionalr",action="store_true",
                      dest="conditionalr",
                      default=False,
                      help="If set, use the conditional probability, conditioned on R")
    #seed
    parser.add_option("--seed",dest='seed',default=2,type='int',
                      help="seed for random number generator")
    #Other options (not necessarily used in this file
    parser.add_option("-t","--type",dest='type',default=None,
                      help="Type of thing to do")
    parser.add_option("--subtype",dest='subtype',default=None,
                      help="Sub-type of thing to do")
    parser.add_option("--group",dest='group',default=None,
                      help="Group to consider (in plotDensComparisonDFMulti and others(?)")
    parser.add_option("-q","--flatten",dest='flatten',default=None,
                      type='float',
                      help="Shortcut to set fake flattening")
    parser.add_option("-o","--outfilename",dest='outfilename',default=None,
                      help="Name for an output file")
    parser.add_option("--ext",dest='ext',default='png',
                      help="Extension for output file")
    parser.add_option("--tighten",action="store_true", dest="tighten",
                      default=False,
                      help="If set, tighten axes")
    parser.add_option("--all",action="store_true", dest="all",
                      default=False,
                      help="Just make the 'all' figure")
    parser.add_option("--relative",action="store_true", dest="relative",
                      default=False,
                      help="Plot quantities relative to some reference value")
    parser.add_option("--noerrs",action="store_true", dest="noerrs",
                      default=False,
                      help="Don't add uncertainties")
    parser.add_option("--usemedianpotential",action="store_true",
                      dest="usemedianpotential",
                      default=False,
                      help="Use the median potential of all the bins")
    parser.add_option("--nv",dest='nv',default=201,type='int',
                      help="Number of vs for v pdf")
    parser.add_option("--height",dest='height',type='float',
                      default=1.1,
                      help="A 'height' (e.g., to compute the surface density at")
    parser.add_option("--index",dest='index',type='int',
                      default=0,
                      help="An index")
    parser.add_option("--allgroups",action="store_true", dest="allgroups",
                      default=False,
                      help="Make plots for all groups")
    parser.add_option("--justvel",action="store_true", dest="justvel",
                      default=False,
                      help="Make plots just for the velocities")
    parser.add_option("--restrictdvt",action="store_true", dest="restrictdvt",
                      default=False,
                      help="Restrict the range of dvt when displaying plot results")
    parser.add_option("--restrictdf",action="store_true", dest="restrictdf",
                      default=False,
                      help="Restrict the range of DF parameters")
    parser.add_option("--conditional",action="store_true", dest="conditional",
                      default=False,
                      help="Plot a conditional PDF")
    return parser
  
if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    numpy.random.seed(options.seed)
    if options.mpi:
        pool= MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    else:
        pool= None
    pixelFitDF(options,args,pool)

