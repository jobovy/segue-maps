import os, os.path
import sys
import copy
import tempfile
import math
import numpy
from scipy import optimize
#from scipy.maxentropy import logsumexp
import cPickle as pickle
from optparse import OptionParser
import multi
import multiprocessing
from galpy.util import bovy_coords, bovy_plot, save_pickles
from galpy import potential
from galpy.actionAngle_src.actionAngleAdiabaticGrid import  actionAngleAdiabaticGrid
from galpy.df_src.quasiisothermaldf import quasiisothermaldf
import bovy_mcmc
import monoAbundanceMW
from segueSelect import read_gdwarfs, read_kdwarfs, _GDWARFFILE, _KDWARFFILE, \
    segueSelect
from fitDensz import cb
from fitSigz import _ZSUN
from pixelFitDens import pixelAfeFeh
_REFR0= 8. #kpc
_REFV0= 220. #km/s
def pixelFitDynamics(options,args):
    #Read the data
    print "Reading the data ..."
    if options.sample.lower() == 'g':
        if options.select.lower() == 'program':
            raw= read_gdwarfs(_GDWARFFILE,logg=True,ebv=True,sn=options.snmin,nosolar=True)
        else:
            raw= read_gdwarfs(logg=True,ebv=True,sn=options.snmin,nosolar=True)
    elif options.sample.lower() == 'k':
        if options.select.lower() == 'program':
            raw= read_kdwarfs(_KDWARFFILE,logg=True,ebv=True,sn=options.snmin,nosolar=True)
        else:
            raw= read_kdwarfs(logg=True,ebv=True,sn=options.snmin,nosolar=True)
    if not options.bmin is None:
        #Cut on |b|
        raw= raw[(numpy.fabs(raw.b) > options.bmin)]
    #Bin the data
    binned= pixelAfeFeh(raw,dfeh=options.dfeh,dafe=options.dafe)
    #Map the bins with ndata > minndata in 1D
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
    #Load selection function
    plates= numpy.array(list(set(list(raw.plate))),dtype='int') #Only load plates that we use
    print "Using %i plates, %i stars ..." %(len(plates),len(raw))
    sf= segueSelect(plates=plates,type_faint='tanhrcut',
                    sample=options.sample,type_bright='tanhrcut',
                    sn=options.snmin,select=options.select,
                    indiv_brightlims=options.indiv_brightlims)
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
    #First initialization
    params= initialize(options,fehs,afes)
    #Optimize DF w/ fixed potential and potential w/ fixed DF
    for cc in range(options.ninit):
        print "Iteration %i  / %i ..." % (cc+1,options.ninit)
        print "Optimizing individual DFs with fixed potential ...."
        params= indiv_optimize_df(params,fehs,afes,binned,options)
        print "Optimizing potential with individual DFs fixed ..."
        params= indiv_optimize_potential(params,fehs,afes,binned,options)
    #Optimize full model
    params= full_optimize(params,fehs,afes,binned,options)
    #Save
    print "BOVY: SAVE"
    #Sample?
    return None

def mloglike(*args,**kwargs):
    """minus log likelihood"""
    return -loglike(*args,**kwargs)

def indiv_optimize_df_mloglike(params,fehs,afes,binned,options,pot,aA,
                               indx,_bigparams):
    """Minus log likelihood when optimizing the parameters of a single DF"""
    #_bigparams is a hack to propagate the parameters to the overall like
    theseparams= set_dfparams(params,_bigparams,indx,options,log=False)
    ml= -indiv_logdf(theseparams,indx,pot,aA,fehs,afes,binned)
    print params, ml
    return ml

def indiv_optimize_pot_mloglike(params,fehs,afes,binned,options,
                                _bigparams):
    """Minus log likelihood when optimizing the parameters of a single DF"""
    #_bigparams is a hack to propagate the parameters to the overall like
    theseparams= set_potparams(params,_bigparams,options,len(fehs))
    ml= mloglike(theseparams,fehs,afes,binned,options)
    print params, ml
    return ml#oglike(theseparams,fehs,afes,binned,options)

def loglike(params,fehs,afes,binned,options):
    """log likelihood"""
    #Set up potential and actionAngle
    pot= setup_potential(params,options,len(fehs))
    aA= setup_aA(pot,options)
    return logdf(params,pot,aA,fehs,afes,binned)

def logdf(params,pot,aA,fehs,afes,binned):
    logl= numpy.zeros(len(fehs))
    #Evaluate individual DFs
    args= (pot,aA,fehs,afes,binned)
    if not options.multi is None:
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

def indiv_logdf(params,indx,pot,aA,fehs,afes,binned):
    """Individual population log likelihood"""
    dfparams= get_dfparams(params,indx,options)
    qdf= quasiisothermaldf(*dfparams,pot=pot,aA=aA)
    #Get data ready
    R,vR,vT,z,vz,covv= prepare_coordinates(params,indx,fehs,afes,binned)
    data_lndf= numpy.zeros(len(R))
    for ii in range(len(R)):
        #print R[ii], vR[ii], vT[ii], z[ii], vz[ii]
        data_lndf[ii]= qdf(R[ii],vR[ii],vT[ii],z[ii],vz[ii],log=True)
        if data_lndf[ii] == -numpy.finfo(numpy.dtype(numpy.float64)).max:
            print "Warning; data likelihood is -inf"
            data_lndf[ii]= 0.
    #Normalize
    return numpy.sum(data_lndf)

def prepare_coordinates(params,indx,fehs,afes,binned):
    vc= get_potparams(params,options,len(fehs))[0] #Always zero-th
    ro= get_ro(params,options)
    vsun= get_vsun(params,options)
    #Create XYZ and R, vxvyvz, cov_vxvyvz
    data= copy.copy(binned(fehs[indx],afes[indx]))
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
    vxvyvz= numpy.zeros((len(data),3))
    vxvyvz[:,0]= data.vxc/_REFV0/vc-vsun[0]
    vxvyvz[:,1]= data.vyc/_REFV0/vc+vsun[1]
    vxvyvz[:,2]= data.vzc/_REFV0/vc+vsun[2]
    cov_vxvyvz= numpy.zeros((len(data),3,3))
    cov_vxvyvz[:,0,0]= data.vxc_err**2./_REFV0/_REFV0/vc/vc
    cov_vxvyvz[:,1,1]= data.vyc_err**2./_REFV0/_REFV0/vc/vc
    cov_vxvyvz[:,2,2]= data.vzc_err**2./_REFV0/_REFV0/vc/vc
    cov_vxvyvz[:,0,1]= data.vxvyc_rho*data.vxc_err*data.vyc_err/_REFV0/_REFV0/vc/vc
    cov_vxvyvz[:,0,2]= data.vxvzc_rho*data.vxc_err*data.vzc_err/_REFV0/_REFV0/vc/vc
    cov_vxvyvz[:,1,2]= data.vyvzc_rho*data.vyc_err*data.vzc_err/_REFV0/_REFV0/vc/vc
    #Rotate to Galactocentric frame
    cosphi= (1.-XYZ[:,0])/R
    sinphi= XYZ[:,1]/R
    vR= -vxvyvz[:,0]*cosphi+vxvyvz[:,1]*sinphi
    vT= vxvyvz[:,0]*sinphi+vxvyvz[:,1]*cosphi
    for rr in range(len(XYZ[:,0])):
        rot= numpy.array([[cosphi[rr],sinphi[rr]],
                          [-sinphi[rr],cosphi[rr]]])
        sxy= cov_vxvyvz[rr,0:2,0:2]
        sRT= numpy.dot(rot,numpy.dot(sxy,rot.T))
        cov_vxvyvz[rr,0:2,0:2]= sRT
    return (R,vR,vT,XYZ[:,2],vxvyvz[:,2],cov_vxvyvz)

def setup_aA(pot,options):
    """Function for setting up the actionAngle object"""
    if options.aAmethod.lower() == 'adiabatic':
        return actionAngleAdiabaticGrid(pot=pot,nR=options.aAnR,
                                        nEz=options.aAnEz,nEr=options.aAnEr,
                                        nLz=options.aAnLz,
                                        zmax=options.aAzmax,
                                        Rmax=options.aARmax)
    
def setup_potential(params,options,npops):
    """Function for setting up the potential"""
    potparams= get_potparams(params,options,npops)
    if options.potential.lower() == 'flatlog':
        return potential.LogarithmicHaloPotential(normalize=1.,q=potparams[1])

def full_optimize(params,fehs,afes,binned,options):
    """Function for optimizing the full set of parameters"""
    return optimize.fmin_powell(mloglike,params,
                                args=(fehs,afes,binned,options))

def indiv_optimize_df(params,fehs,afes,binned,options):
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
                                                                         fehs,afes,binned,options,aA,pot,tmpfiles)),
                                     range(len(fehs)),
                                     numcores=numpy.amin([len(fehs),
                                                          multiprocessing.cpu_count(),
                                                          options.multi]))
            #Now read all of the temporary files
            for ii in range(len(fehs)):
                tmpfile= open(tmpfiles[ii][1],'rb')
                new_dfparams= pickle.load(tmpfile)
                params= set_dfparams(new_dfparams,params,ii,options,log=False)
                tmpfile.close()
        finally:
            for ii in range(len(fehs)):
                os.remove(tmpfiles[ii][1])
    else:
        for ii in range(len(fehs)):
            print ii
            init_dfparams= get_dfparams(params,ii,options,log=False)
            new_dfparams= optimize.fmin_powell(indiv_optimize_df_mloglike,
                                               init_dfparams,
                                           args=(fehs,afes,binned,
                                                 options,pot,aA,
                                                 ii,copy.copy(params)),
                                               callback=cb)
            params= set_dfparams(new_dfparams,params,ii,options,log=False)
    return params

def indiv_optimize_df_single(params,ii,fehs,afes,binned,options,aA,pot,tmpfiles):
    """Function to optimize the DF params for a single population when holding the potential fixed and using multi-processing"""
    print ii
    init_dfparams= get_dfparams(params,ii,options,log=False)
    new_dfparams= optimize.fmin_powell(indiv_optimize_df_mloglike,
                                       init_dfparams,
                                       args=(fehs,afes,binned,
                                             options,pot,aA,
                                             ii,copy.copy(params)),
                                       callback=cb)
    #Now save to temporary pickle
    tmpfile= open(tmpfiles[ii][1],'wb')
    pickle.dump(new_dfparams,tmpfile)
    tmpfile.close()
    return None

def indiv_optimize_potential(params,fehs,afes,binned,options):
    """Function for optimizing the potential w/ individual DFs fixed"""
    init_potparams= numpy.array(get_potparams(params,options,len(fehs)))
    print init_potparams
    new_potparams= optimize.fmin_powell(indiv_optimize_pot_mloglike,
                                        init_potparams,
                                        args=(fehs,afes,binned,options,
                                              copy.copy(params)),
                                        callback=cb)
    params= set_potparams(new_potparams,params,len(fehs))

def initialize(options,fehs,afes):
    """Function to initialize the fit; uses fehs and afes to initialize using MAPS"""
    p= []
    if options.fitro:
        p.append(1.)
    if options.fitvsun:
        p.extend([0.,1.,0.])
    mapfehs= monoAbundanceMW.fehs()
    mapafes= monoAbundanceMW.afes()
    for ii in range(len(fehs)):
        if options.dfmodel.lower() == 'qdf':
            #Find nearest mono-abundance bin that has a measurement
            abindx= numpy.argmin((fehs[ii]-mapfehs)**2./0.01 \
                                     +(afes[ii]-mapafes)**2./0.0025)
            p.extend([numpy.log(2.*monoAbundanceMW.sigmaz(mapfehs[abindx],mapafes[abindx])/_REFV0), #sigmaR
                      numpy.log(monoAbundanceMW.sigmaz(mapfehs[abindx],mapafes[abindx])/_REFV0), #sigmaZ
                      numpy.log(monoAbundanceMW.hr(mapfehs[abindx],mapafes[abindx])/_REFR0), #hR
                      0.,0.]) #hsigR, hsigZ
    if options.potential.lower() == 'flatlog':
        p.extend([1.,.9])
    return p

def get_potparams(p,options,npops):
    """Function that returns the set of potential parameters for these options"""
    startindx= 0
    if options.fitro: startindx+= 1
    if options.fitvsun: startindx+= 3
    ndfparams= get_ndfparams(options)
    startindx+= ndfparams*npops
    if options.potential.lower() == 'flatlog':
        return (p[startindx],p[startindx+1]) #vc, q

def set_potparams(p,params,options,npops):
    """Function that sets the set of potential parameters for these options"""
    startindx= 0
    if options.fitro: startindx+= 1
    if options.fitvsun: startindx+= 3
    ndfparams= get_ndfparams(options)
    startindx+= ndfparams*npops
    if options.potential.lower() == 'flatlog':
        params[startindx]= p[0]
        params[startindx+1]= p[1]
    return params

def get_dfparams(p,indx,options,log=False):
    """Function that returns the set of DF parameters for population indx for these options,
    Returns them such that they can be given to the initialization"""
    startindx= 0
    if options.fitro: startindx+= 1
    if options.fitvsun: startindx+= 3
    ndfparams= get_ndfparams(options)
    startindx+= ndfparams*indx
    if options.dfmodel.lower() == 'qdf':
        if log:
            return [p[startindx],
                    p[startindx+1],
                    p[startindx+2],
                    p[startindx+3],
                    p[startindx+4]]
        else:
            return [numpy.exp(p[startindx]),
                    numpy.exp(p[startindx+1]),
                    numpy.exp(p[startindx+2]),
                    numpy.exp(p[startindx+3]),
                    numpy.exp(p[startindx+4])]
        
def set_dfparams(p,params,indx,options,log=True):
    """Function that sets the set of DF parameters for population indx for these options"""
    startindx= 0
    if options.fitro: startindx+= 1
    if options.fitvsun: startindx+= 3
    ndfparams= get_ndfparams(options)
    startindx+= ndfparams*indx
    if options.dfmodel.lower() == 'qdf':
        for ii in range(ndfparams):
            if log:
                params[startindx+ii]= numpy.log(p[ii])
            else:
                params[startindx+ii]= p[ii]
    return params

def get_ndfparams(options):
    """Function that returns the number of DF parameters for a single population"""
    if options.dfmodel.lower() == 'qdf':
        return 5

def get_ro(p,options):
    """Function that returns R0 for these options"""
    if options.fitro:
        return p[0]
    else:
        return 1.

def get_vsun(p,options):
    """Function to return motion of the Sun in the Galactocentric reference frame"""
    if options.fitvsun:
        return (p[1],p[2],p[3])
    else:
        return (-11.1/_REFV0,245./_REFV0,7.25/_REFV0) #BOVY:ADJUST?

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
    parser.add_option("--minndata",dest='minndata',default=100,type='int',
                      help="Minimum number of objects in a bin to perform a fit")   
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
    parser.add_option("--aAmethod",dest='aAmethod',default='adiabatic',
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
    #Fit options
    parser.add_option("--fitro",action="store_true", dest="fitro",
                      default=False,
                      help="If set, fit for R_0")
    parser.add_option("--fitvsun",action="store_true", dest="fitvsun",
                      default=False,
                      help="If set, fit for v_sun")
    parser.add_option("--ninit",dest='ninit',default=1,type='int',
                      help="Number of initial optimizations to perform (indiv DF + potential w/ fixed DF")
    #Sample?
    parser.add_option("--mcsample",action="store_true", dest="mcsample",
                      default=False,
                      help="If set, sample around the best fit, save in args[1]")
    parser.add_option("--nsamples",dest='nsamples',default=1000,type='int',
                      help="Number of MCMC samples to obtain")
    parser.add_option("-m","--multi",dest='multi',default=None,type='int',
                      help="number of cpus to use")
    return parser
  
if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    pixelFitDynamics(options,args)

