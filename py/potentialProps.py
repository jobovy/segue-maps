import numpy
from scipy import optimize, integrate
from galpy import potential
from pixelFitDF import setup_potential, _REFV0, _REFR0
from calcDFResults import setup_options
ro= 1.
def surfzprofile4surfz(rs,surfz,rd,vo,zh,dlnvcdlnr):
    #First determine fh
    fh= fh4surfz(surfz,rd,vo,zh,dlnvcdlnr)
    potparams= [numpy.log(rd/8.),vo,numpy.log(zh/8000.),fh,dlnvcdlnr]
    options= setup_options(None)
    options.potential= 'dpdiskplhalofixbulgeflatwgasalt'
    options.fitdvt= False
    try:
        pot= setup_potential(potparams,options,0,returnrawpot=True)
    except RuntimeError:
        raise
    #Now calculate the surface profile
    out= numpy.array([2.*integrate.quad((lambda zz: potential.evaluateDensities(r,zz,pot)),0.,options.height/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro for r in rs])
    return out

def surfzdiskprofile4surfz(rs,surfz,rd,vo,zh,dlnvcdlnr):
    #First determine fh
    fh= fh4surfz(surfz,rd,vo,zh,dlnvcdlnr)
    potparams= [numpy.log(rd/8.),vo,numpy.log(zh/8000.),fh,dlnvcdlnr]
    options= setup_options(None)
    options.potential= 'dpdiskplhalofixbulgeflatwgasalt'
    options.fitdvt= False
    try:
        pot= setup_potential(potparams,options,0,returnrawpot=True)
    except RuntimeError:
        raise
    #Now calculate the surface profile
    out= numpy.array([2.*pot[0].dens(r,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.*zh*ro for r in rs])
    return out

def rc4surfz(rs,surfz,rd,vo,zh,dlnvcdlnr):
    #First determine fh
    fh= fh4surfz(surfz,rd,vo,zh,dlnvcdlnr)
    potparams= [numpy.log(rd/8.),vo,numpy.log(zh/8000.),fh,dlnvcdlnr]
    options= setup_options(None)
    options.potential= 'dpdiskplhalofixbulgeflatwgasalt'
    options.fitdvt= False
    try:
        pot= setup_potential(potparams,options,0,returnrawpot=True)
    except RuntimeError:
        raise
    #Now calculate the rotation curve
    out= numpy.array([potential.vcirc(pot,r) for r in rs])
    return out

def rcdisk4surfz(rs,surfz,rd,vo,zh,dlnvcdlnr):
    #First determine fh
    fh= fh4surfz(surfz,rd,vo,zh,dlnvcdlnr)
    potparams= [numpy.log(rd/8.),vo,numpy.log(zh/8000.),fh,dlnvcdlnr]
    options= setup_options(None)
    options.potential= 'dpdiskplhalofixbulgeflatwgasalt'
    options.fitdvt= False
    try:
        pot= setup_potential(potparams,options,0,returnrawpot=True)
    except RuntimeError:
        raise
    #Now calculate the rotation curve
    out= numpy.array([potential.vcirc(pot[0],r) for r in rs])
    return out

def rchalo4surfz(rs,surfz,rd,vo,zh,dlnvcdlnr):
    #First determine fh
    fh= fh4surfz(surfz,rd,vo,zh,dlnvcdlnr)
    potparams= [numpy.log(rd/8.),vo,numpy.log(zh/8000.),fh,dlnvcdlnr]
    options= setup_options(None)
    options.potential= 'dpdiskplhalofixbulgeflatwgasalt'
    options.fitdvt= False
    try:
        pot= setup_potential(potparams,options,0,returnrawpot=True)
    except RuntimeError:
        raise
    #Now calculate the rotation curve
    out= numpy.array([potential.vcirc(pot[1],r) for r in rs])
    return out

def rhodmprofile4surfz(rs,surfz,rd,vo,zh,dlnvcdlnr):
    #First determine fh
    fh= fh4surfz(surfz,rd,vo,zh,dlnvcdlnr)
    potparams= [numpy.log(rd/8.),vo,numpy.log(zh/8000.),fh,dlnvcdlnr]
    options= setup_options(None)
    options.potential= 'dpdiskplhalofixbulgeflatwgasalt'
    options.fitdvt= False
    try:
        pot= setup_potential(potparams,options,0,returnrawpot=True)
    except RuntimeError:
        raise
    #Now calculate the surface profile
    out= numpy.array([pot[1].dens(r,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3. for r in rs])
    return out

def dmalpha4surfz(surfz,rd,vo,zh,dlnvcdlnr):
    #First determine fh
    fh= fh4surfz(surfz,rd,vo,zh,dlnvcdlnr)
    potparams= [numpy.log(rd/8.),vo,numpy.log(zh/8000.),fh,dlnvcdlnr]
    options= setup_options(None)
    options.potential= 'dpdiskplhalofixbulgeflatwgasalt'
    options.fitdvt= False
    try:
        pot= setup_potential(potparams,options,0,returnrawpot=True)
    except RuntimeError:
        raise
    return pot[1].alpha

def fh4surfz(surfz,rd,vo,zh,dlnvcdlnr):
    """Return fh that gives a certain surface density"""
    options= setup_options(None)
    options.potential= 'dpdiskplhalofixbulgeflatwgasalt'
    options.fitdvt= False
    opt= optimize.brent(fh4surfzOpt,args=(surfz,rd,vo,zh,dlnvcdlnr,options),
                        brack=(0.,1.))
    return opt

def fh4surfzOpt(fh,surfz,rd,vo,zh,dlnvcdlnr,options):
    potparams= [numpy.log(rd/8.),vo,numpy.log(zh/8000.),fh,dlnvcdlnr]
    try:
        pot= setup_potential(potparams,options,0,returnrawpot=True)
    except RuntimeError:
        return numpy.finfo(numpy.dtype(numpy.float64)).max
    return (2.*integrate.quad((lambda zz: potential.evaluateDensities(1.,zz,pot)),0.,options.height/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro-surfz)**2.
