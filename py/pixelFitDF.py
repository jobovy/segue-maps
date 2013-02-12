#for testing: python pixelFitDF.py --dfeh=0.5 --dafe=0.25 --mcall --mcout --singlefeh=-0.2 --singleafe=0.2 -p 1880 --minndata=1
#
#for running: python pixelFitDF.py --dfeh=0.25 --dafe=0.2 --novoprior --nmcv=100 --justpot --singles --aAmethod=staeckel -m 7 ../fakeDF/fakeDFFit_dfeh0.25_dafe0.2_q0.7_staeckel_justpot_singles.sav -f ../fakeDF/fakeDF_dfeh0.25_dafe0.2_q0.7_staeckel.fits
#
# TO DO:
#   - add sigmar to monoAbundanceMW
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
from scipy import optimize, interpolate, linalg
from scipy.maxentropy import logsumexp
import cPickle as pickle
from optparse import OptionParser
import multi
import multiprocessing
from galpy.util import bovy_coords, bovy_plot, save_pickles
from galpy import potential
from galpy.actionAngle import actionAngleAdiabatic
from galpy.actionAngle import actionAngleAdiabaticGrid
from galpy.actionAngle import actionAngleStaeckel
from galpy.actionAngle import actionAngleStaeckelGrid
from galpy.df_src.quasiisothermaldf import quasiisothermaldf
import bovy_mcmc
import acor
from galpy.util import save_pickles
import monoAbundanceMW
from segueSelect import read_gdwarfs, read_kdwarfs, _GDWARFFILE, _KDWARFFILE, \
    segueSelect, _mr_gi, _gi_gr, _ERASESTR, _append_field_recarray, \
    ivezic_dist_gr
from fitDensz import cb, _ZSUN, DistSpline, _ivezic_dist, _NDS
from compareDataModel import _predict_rdist_plate, comparernumberPlate
from pixelFitDens import pixelAfeFeh
_DEBUG= True
_REFR0= 8. #kpc
_REFV0= 220. #km/s
_VRSUN=-11.1 #km/s
_VTSUN= 245. #km/s
_PMSGRA= 30.24 #km/s/kpc
_VZSUN= 7.25 #km/s
_GMBULGE= 17208.0 #kpc (km/s)^2 = 1 x 10^10 Msolar
_ABULGE= 0.6
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
_BFGS= True
def pixelFitDF(options,args):
    print "WARNING: IGNORING NUMPY FLOATING POINT WARNINGS ..."
    numpy.seterr(all='ignore')
    #Check whether the savefile already exists
    if os.path.exists(args[0]):
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
    if not options.singles:
        print "Setting up error integration ..."
        raw, errstuff= setup_err_mc(raw,options)
    #Bin the data
    binned= pixelAfeFeh(raw,dfeh=options.dfeh,dafe=options.dafe)
    #Map the bins with ndata > minndata in 1D
    fehs, afes= [], []
    for ii in range(len(binned.fehedges)-1):
        for jj in range(len(binned.afeedges)-1):
            data= binned(binned.feh(ii),binned.afe(jj))
            if len(data) < options.minndata:
                continue
            #print binned.feh(ii), binned.afe(jj), len(data)
            fehs.append(binned.feh(ii))
            afes.append(binned.afe(jj))
    nabundancebins= len(fehs)
    fehs= numpy.array(fehs)
    afes= numpy.array(afes)
    if not options.singlefeh is None:
        if options.loo:
            pass
        else:
            #Set up single feh
            indx= binned.callIndx(options.singlefeh,options.singleafe)
            if numpy.sum(indx) == 0:
                raise IOError("Bin corresponding to singlefeh and singleafe is empty ...")
            raw= copy.copy(binned.data[indx])
            newerrstuff= []
            for ii in range(len(binned.data)):
                if indx[ii]: newerrstuff.append(errstuff[ii])
            errstuff= newerrstuff
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
    if options.singles:
        run_abundance_singles(options,args,fehs,afes)
        return None
    #Setup everything for the selection function
    print "Setting up stuff for the normalization integral ..."
    normintstuff= setup_normintstuff(options,raw,binned,fehs,afes)
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
    if not options.mcsample:
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
            indxMax= numpy.argmax(lnps)
            params= samples[indxMax]
            mloglikemax= -lnps[indxMax]
        #Save
        save_pickles(args[0],params,mloglikemax,samples,len(fehs))
        print_samples_qa(samples,options,len(fehs))
    return None

##DATA
def read_rawdata(options):
    if options.sample.lower() == 'g':
        if not options.fakedata is None:
            raw= read_gdwarfs(options.fakedata,logg=True,ebv=True,sn=options.snmin,nosolar=True,norcut=True)
        elif options.select.lower() == 'program':
            raw= read_gdwarfs(_GDWARFFILE,logg=True,ebv=True,sn=options.snmin,nosolar=True)
        else:
            raw= read_gdwarfs(logg=True,ebv=True,sn=options.snmin,nosolar=True)
    elif options.sample.lower() == 'k':
        if not options.fakedata is None:
            raw= read_kdwarfs(options.fakedata,logg=True,ebv=True,sn=options.snmin,nosolar=True,norcut=True)
        elif options.select.lower() == 'program':
            raw= read_kdwarfs(_KDWARFFILE,logg=True,ebv=True,sn=options.snmin,nosolar=True)
        else:
            raw= read_kdwarfs(logg=True,ebv=True,sn=options.snmin,nosolar=True)
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
    return raw

##LOG LIKELIHOODS
def mloglike(*args,**kwargs):
    """minus log likelihood"""
    return -loglike(*args,**kwargs)

def loglike(params,fehs,afes,binned,options,normintstuff,errstuff):
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
        pot= setup_potential(params,options,len(fehs))
    except RuntimeError: #if this set of parameters gives a nonsense potential
        return -numpy.finfo(numpy.dtype(numpy.float64)).max
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
        if theseparams[0] < -2.77 or theseparams[0] > 2.53:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        if theseparams[1] < -3.1 or theseparams[1] > -0.4:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        if theseparams[2] < -3.1 or theseparams[2] > -0.4:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        if theseparams[3] < -2.77 or theseparams[3] > 2.53:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        if theseparams[4] < -2.77 or theseparams[4] > 2.53:
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
    return out

def logprior_dlnvcdlnr(dlnvcdlnr,options):
    if options.nodlnvcdlnrprior: return 0.
    if True:
        sb= 0.04
        if dlnvcdlnr/30. > sb or dlnvcdlnr/30. < -0.5:
            return -numpy.finfo(numpy.dtype(numpy.float64)).max
        return numpy.log((sb-dlnvcdlnr/30.)/sb)-(sb-dlnvcdlnr/30.)/sb

##SETUP AND CALCULATE THE NORMALIZATION INTEGRAL
def calc_normint(qdf,indx,normintstuff,params,npops,options,logoutfrac):
    """Calculate the normalization integral"""
    if options.mcall or options.mcwdf: #evaluation is the same for these
        return calc_normint_mcall(qdf,indx,normintstuff,params,npops,options,
                                  logoutfrac)
    else:
        return calc_normint_mcv(qdf,indx,normintstuff,params,npops,options,
                                logoutfrac)

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

def calc_normint_mcv(qdf,indx,normintstuff,params,npops,options,logoutfrac):
    """calculate the normalization integral by monte carlo integrating over v, but grid integrating over everything else"""
    thisnormintstuff= normintstuff[indx]
    if _PRECALCVSAMPLES:
        sf, plates,platel,plateb,platebright,platefaint,grmin,grmax,rmin,rmax,fehmin,fehmax,feh,colordist,fehdist,gr,rhogr,rhofeh,mr,dmin,dmax,ds, surfscale, hr, hz, surfnrs, surfnzs, surfRgrid, surfzgrid, surfvrs, surfvts, surfvzs= unpack_normintstuff(thisnormintstuff,options)
    else:
        sf, plates,platel,plateb,platebright,platefaint,grmin,grmax,rmin,rmax,fehmin,fehmax,feh,colordist,fehdist,gr,rhogr,rhofeh,mr,dmin,dmax,ds, surfscale, hr, hz= unpack_normintstuff(thisnormintstuff,options)
    out= 0.
    ro= get_ro(params,options)
#    outfrac= get_outfrac(params,indx,options)
    outfrac= numpy.exp(logoutfrac)
    halodens= ro*outDens(1.,0.,None)
    globalInterp= True
    if globalInterp:
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
        if _SURFSUBTRACTEXPON:
            compare_func= lambda x,y,du: numpy.exp(surfInterp.ev(x/ro/_REFR0,numpy.fabs(y)/ro/_REFR0)-x/ro/_REFR0/ehr-numpy.fabs(y)/ehz/ro/_REFR0)+outfrac*halodens
        else:
            compare_func= lambda x,y,du: numpy.exp(surfInterp.ev(x/ro/_REFR0,numpy.fabs(y)/ro/_REFR0))+outfrac*halodens
        distfac= 10.**(get_dm(params,options)/5.)
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
                               numcores=options.multi)
        vo= get_vo(params,options,npops)
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

def setup_normintstuff(options,raw,binned,fehs,afes):
    """Gather everything necessary for calculating the normalization integral"""
    if not options.savenorm is None and os.path.exists(options.savenorm):
        print "Reading normintstuff from file ..."
        savefile= open(options.savenorm,'rb')
        out= pickle.load(savefile)
        savefile.close()
        return out
    #Load selection function
    plates= numpy.array(list(set(list(raw.plate))),dtype='int') #Only load plates that we use
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
    mapfehs= monoAbundanceMW.fehs()
    mapafes= monoAbundanceMW.afes()
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
        thishr= monoAbundanceMW.hr(mapfehs[abindx],mapafes[abindx])
        thishz= monoAbundanceMW.hz(mapfehs[abindx],mapafes[abindx])/1000.
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
                                         r=R[jj])
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
        rhofeh= numpy.array([fehdist(feh) for feh in fehsgrid])
        mr= numpy.zeros((_NGR,_NFEH))
        for kk in range(_NGR):
            for ll in range(_NFEH):
                mr[kk,ll]= _mr_gi(_gi_gr(grs[kk]),fehsgrid[ll])
        #determine dmin and dmax
        allbright, allfaint= True, True
        #dmin and dmax for this rmin, rmax
        for p in sf.plates:
            #l and b?
            pindx= (sf.plates == p)
            plateb= platelb[pindx,1][0]
            if 'faint' in sf.platestr[pindx].programname[0]:
                allbright= False
            else:
                allfaint= False
        if not (options.sample.lower() == 'k' \
                    and options.indiv_brightlims):
            if allbright:
                thisrmin, thisrmax= rmin, 17.8
            elif allfaint:
                thisrmin, thisrmax= 17.8, rmax
            else:
                thisrmin, thisrmax= rmin, rmax
        else:
            thisrmin, thisrmax= rmin, rmax
        _THISNGR, _THISNFEH= 51, 51
        thisgrs= numpy.zeros((_THISNGR,_THISNFEH))
        thisfehs= numpy.zeros((_THISNGR,_THISNFEH))
        for kk in range(_THISNGR):
            thisfehs[kk,:]= numpy.linspace(fehrange[0],fehrange[1],_THISNFEH)
        for kk in range(_THISNFEH):
            thisgrs[:,kk]= numpy.linspace(grmin,grmax,_THISNGR)
        dmin= numpy.amin(_ivezic_dist(thisgrs,thisrmin,thisfehs))
        dmax= numpy.amax(_ivezic_dist(thisgrs,thisrmax,thisfehs))
        ds= numpy.linspace(dmin,dmax,_NDS)
        #Determine scale over which the surface-mass density changes for this plate
        #Find nearest mono-abundance bin that has a measurement
        abindx= numpy.argmin((fehs[ii]-mapfehs)**2./0.01 \
                                 +(afes[ii]-mapafes)**2./0.0025)
        thishr= monoAbundanceMW.hr(mapfehs[abindx],mapafes[abindx])
        thishz= monoAbundanceMW.hz(mapfehs[abindx],mapafes[abindx])/1000.
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
                    normintstuff.surfnrs,
                    normintstuff.surfnzs,
                    normintstuff.surfRgrid,
                    normintstuff.surfzgrid,
                    normintstuff.surfvrs,
                    normintstuff.surfvts,
                    normintstuff.surfvzs)
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
                    normintstuff.hz)
        
class normintstuffClass:
    """Empty class to hold normalization integral necessities"""
    pass


##RUNNING SINGLE BINS IN A SINGLE CALL
def run_abundance_singles(options,args,fehs,afes):
    options.singles= False #First turn this off!
    savename= args[0]
    initname= options.init
    normname= options.savenorm
    if options.cluster:
        options.cluster= False
        for ii in range(len(fehs)):
            run_abundance_singles_single_onCluster(options,args,fehs,afes,ii,
                                                   savename,initname,normname)
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
    #Now run
    cmd= unparse_cmd(options,args)
    print cmd
    cmd= "mpirun -x PYTHONPATH /home/bovy/local/bin/python pixelFitDF.py"+cmd
    #Create file that will submit the job
    cmdfilename='../cmds/'+os.path.basename(args[0])+'.sh'
    shutil.copyfile('submit_template.txt',cmdfilename)
    cmdfile= open(cmdfilename,'a')
    cmdfile.write(cmd)
    cmdfile.close()
    #Now submit
    subprocess.call(["qsub","-w","n","-l","exclusive=true","-l","h_rt=36:00:00",cmdfilename])
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
        return (R,vR,vT,z,vz)
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
        fdfh= 1.-13./gassurfdens-ampb
        fd= (1.-potparams[3])*fdfh
        fh= fdfh- fd
        dp= potential.DoubleExponentialDiskPotential(hr=numpy.exp(potparams[0])/ro,
                                                     hz=numpy.exp(potparams[2])/ro,
                                                     normalize=fd)
        plhalo= plhalo_from_dlnvcdlnr(potparams[4],dp,[bp,gp],options)
        if plhalo < 0. or plhalo > 3:
            raise RuntimeError
        hp= potential.PowerSphericalPotential(alpha=plhalo,
                                              normalize=fh)
kk        #print ampb, 13./gassurfdens, ampd, amph, dp(1.,0.), hp(1.,0.)
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

def plhalo_from_dlnvcdlnr(dlnvcdlnr,diskpot,bulgepot,options):
    """Calculate the halo's shape corresponding to this rotation curve derivative"""
    #First calculate the derivatives dvc^2/dR of disk and bulge
    dvcdr_disk= -potential.evaluateRforces(1.,0.,diskpot)+potential.evaluateR2derivs(1.,0.,diskpot)
    dvcdr_bulge= -potential.evaluateRforces(1.,0.,bulgepot)+potential.evaluateR2derivs(1.,0.,bulgepot)
    if 'plhalo' in options.potential.lower():
        return 2.-(2.*dlnvcdlnr-dvcdr_disk-dvcdr_bulge)

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

##SETUP ERROR INTEGRATION
def setup_err_mc(data,options):
    #First sample distances, then sample velocities 
    #Calculate r error
    if options.nmcerr > 1:
        rerr= ivezic_dist_gr(data.dered_g,
                             data.dered_r,
                             data.feh,
                             dg=data.g_err,
                             dr=data.r_err,
                             dfeh=data.feh_err,
                             return_error=True,
                             _returndmr=True)        
        rsamples= numpy.tile(data.dered_r,(options.nmcerr,1)).T\
            +numpy.random.normal(size=(len(data),options.nmcerr))\
            *numpy.tile(rerr,(options.nmcerr,1)).T
        dsamples= ivezic_dist_gr(numpy.tile(data.dered_g-data.dered_r,(options.nmcerr,1)).T+rsamples,
                                 rsamples,
                                 numpy.tile(data.feh,(options.nmcerr,1)).T)[0]
        if not options.fixdm is None:
            distfac= 10.**(options.fixdm/5.)
            dsamples*= distfac
        #dsamples[(dsamples > 15.)]= 15. #to make sure samples don't go nuts
        #Transform to XYZ
        lb= bovy_coords.radec_to_lb(data.ra,data.dec,degree=True)
        XYZ= bovy_coords.lbd_to_XYZ(numpy.tile(lb[:,0],(options.nmcerr,1)).T,
                                    numpy.tile(lb[:,1],(options.nmcerr,1)).T,
                                    dsamples,degree=True)
    else: #if nmcerr=1, just use data point
        dsamples= ivezic_dist_gr(data.dered_g,data.dered_r,
                                 data.feh)[0]
        if not options.fixdm is None:
            distfac= 10.**(options.fixdm/5.)
            dsamples*= distfac
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
            if options.marginalizevt:
                #'sabotage' pm_ll to remove information in the plane
                pmllpmbb[:,0]= 2.*(numpy.random.uniform(size=options.nmcerr)-1.)*100.
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
        p.append(0.)
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
    mapfehs= monoAbundanceMW.fehs()
    mapafes= monoAbundanceMW.afes()
    indx= (mapfehs != -0.45)*(mapafes != 0.075) #Pop this one, bc sz is crazy
    mapfehs= mapfehs[indx]
    mapafes= mapafes[indx]
    for ii in range(len(fehs)):
        if options.dfmodel.lower() == 'qdf':
            #Find nearest mono-abundance bin that has a measurement
            abindx= numpy.argmin((fehs[ii]-mapfehs)**2./0.01 \
                                     +(afes[ii]-mapafes)**2./0.0025)
            p.extend([numpy.log(monoAbundanceMW.hr(mapfehs[abindx],mapafes[abindx])/_REFR0), #hR
numpy.log(2.*monoAbundanceMW.sigmaz(mapfehs[abindx],mapafes[abindx])/_REFV0), #sigmaR
                      numpy.log(monoAbundanceMW.sigmaz(mapfehs[abindx],mapafes[abindx])/_REFV0), #sigmaZ
                      numpy.log(7./_REFR0),numpy.log(7./_REFR0)]) #hsigR, hsigZ
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
    elif options.potential.lower() == 'dpdiskplhalofixbulgeflatwgasalt':
        p.extend([-1.39,1.,-3.,0.3,-0.05])
        #p.extend([-1.,1.,-3.,0.5,0.])
        #p.extend([-.69,1.07,-3.,0.7,0.02])
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
        domain.append([-0.5,0.04])
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
        step.append(0.5)
        create_method.append('whole')
        domain.append([-5.1,1.4])
        isDomainFinite.append([True,True])
        step.append(0.5)
        create_method.append('whole')
        domain.append([0.,1.])
        isDomainFinite.append([True,True])
        step.append(0.5)
        create_method.append('whole')
        domain.append([-0.5,0.04])
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
            or options.potential.lower() == 'dpdiskplhalofixbulgeflatwgasalt':
        return (p[startindx],p[startindx+1],p[startindx+2],p[startindx+3],p[startindx+4]) # hr, vo, hz,dlnvcdlnr,power-law halo
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
            or options.potential.lower() == 'dpdiskplhalofixbulgeflatwgas' \
            or options.potential.lower() == 'mpdiskflplhalofixplfixbulgeflat':
        return p[startindx+1]

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
            or options.potential.lower() == 'mpdiskflplhalofixplfixbulgeflat':
        params[startindx]= p[0]
        params[startindx+1]= p[1]
        params[startindx+2]= p[2]
        params[startindx+3]= p[3]
        params[startindx+4]= p[4]
    elif options.potential.lower() == 'dpdiskflplhalofixbulgeflatwgas':
        params[startindx]= p[0]
        params[startindx+1]= p[1]
        params[startindx+2]= p[2]
        params[startindx+3]= p[3]
        params[startindx+4]= p[4]
        params[startindx+5]= p[5]
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
            or options.potential.lower() == 'mpdiskflplhalofixplfixbulgeflat':
        return 5
    elif options.potential.lower() == 'dpdiskflplhalofixbulgeflatwgas':
        return 6

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
    parser.add_option("--fitdvt",action="store_true", dest="fitdvt",
                      default=False,
                      help="If set, fit for a vT offset")
    parser.add_option("--fixdvt",dest="fixdvt",type='float',
                      default=None,
                      help="If set, fix a vT offset")
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
    parser.add_option("-m","--multi",dest='multi',default=None,type='int',
                      help="number of cpus to use")
    parser.add_option("--cluster",action="store_true", dest="cluster",
                      default=False,
                      help="If set, fit each bin on a separate node on the cluster (for --singles)")
    #Type of fit
    parser.add_option("--justdf",action="store_true", dest="justdf",
                      default=False,
                      help="If set, just fit the DF assuming fixed potential (CURRENTLY ONLY FOR FIT, NOT FOR SAMPLE")
    parser.add_option("--justpot",action="store_true", dest="justpot",
                      default=False,
                      help="If set, just fit the potential assuming fixed DF (CURRENTLY ONLY FOR FIT, NOT FOR SAMPLE")
    parser.add_option("--marginalizevt",action="store_true", dest="marginalizevt",
                      default=False,
                      help="If set, don't use vT data")
    #seed
    parser.add_option("--seed",dest='seed',default=1,type='int',
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
    return parser
  
if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    numpy.random.seed(options.seed)
    pixelFitDF(options,args)

