import os, os.path
import sys
import math
import numpy
from scipy import optimize
import cPickle as pickle
from optparse import OptionParser
import multi
from galpy.util import bovy_coords, bovy_plot, save_pickles
from galpy import potential
from galpy.actionAngle_src.actionAngleAdiabaticGrid import  actionAngleAdiabaticGrid
from galpy.df_src.quasiisothermaldf import quasiisothermaldf
import bovy_mcmc
import monoAbundanceMW
from segueSelect import read_gdwarfs, read_kdwarfs, _GDWARFFILE, _KDWARFFILE, \
    segueSelect
from fitSigz import _ZSUN
from pixelFitDens import pixelAfeFeh
_REFR0= 8. #kpc
_REFV0= 220. #km/s
def pixelFitDynamics(options,args):
    #Read the data
    if options.sample.lower() == 'g':
        if options.select.lower() == 'program':
            raw= read_gdwarfs(_GDWARFFILE,logg=True,ebv=True,sn=options.snmin)
        else:
            raw= read_gdwarfs(logg=True,ebv=True,sn=options.snmin)
    elif options.sample.lower() == 'k':
        if options.select.lower() == 'program':
            raw= read_kdwarfs(_KDWARFFILE,logg=True,ebv=True,sn=options.snmin)
        else:
            raw= read_kdwarfs(logg=True,ebv=True,sn=options.snmin)
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
        params= indiv_optimize_df(params,fehs,afes,binned,options)
        params= indiv_optimize_potential(params,fehs,afes,binned,options)
    #Optimize full model
    params= full_optimize(params,fehs,afes,binned,options)
    #Save
    print "BOVY: SAVE"
    #Sample?
    return None

def loglike(params,fehs,afes):
    """log likelihood"""
    #Transform coordinates
    #Set up potential and actionAngle
    pot= setup_potential(params,options,len(fehs))
    aA= setup_aA(pot,options)
    #Evaluate individual DFs
    qdf= quasiisothermaldf(1./3.,0.4,0.2,1.,1.,pot=lp,aA=aA)

    return None

def mloglike(*args,**kwargs):
    """minus log likelihood"""
    return -loglike(*args,**kwargs)

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
    return params

def indiv_optimize_df(params,fehs,afes,binned,options):
    """Function for optimizing individual DFs with potential fixed"""
    return params

def indiv_optimize_potential(params,fehs,afes,binned,options):
    """Function for optimizing the potential w/ individual DFs fixed"""
    return params

def initialize(options,fehs,afes):
    """Function to initialize the fit; uses fehs and afes to initialize using MAPS"""
    p= []
    if options.fitro:
        p.append(1.)
    if options.fitvsun:
        p.extend([0.,1.,0.])
    for ii in range(len(fehs)):
        if options.dfmodel.lower() == 'qdf':
            p.extend([numpy.log(2.*monoAbundanceMW.sigmaz(fehs[ii],afes[ii])/_REFV0), #sigmaR
                      numpy.log(monoAbundanceMW.sigmaz(fehs[ii],afes[ii])/_REFV0), #sigmaZ
                      numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii])/_REFR0), #hR
                      1.,1.]) #hsigR, hsigZ
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
    if options.potential.lower == 'flatlog':
        return (p[startindx],p[startindx+1]) #vc, q

def get_dfparams(p,indx,options):
    """Function that returns the set of DF parameters for population indx for these options"""
    startindx= 0
    if options.fitro: startindx+= 1
    if options.fitvsun: startindx+= 3
    ndfparams= get_ndfparams(options)
    if options.dfmodel.lower() == 'qdf':
        return (p[startindx],p[startindx+1],p[startindx+2],p[startindx+3],
                p[startindx+4])

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
    parser.add_option("--dfeh",dest='dfeh',default=0.05,type='float',
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
    return parser
  
if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    pixelFitDynamics(options,args)

