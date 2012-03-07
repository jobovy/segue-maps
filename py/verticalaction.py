import math
import numpy
from scipy import stats, integrate
from galpy.potential import MWPotential, evaluatePotentials
_R0= 8.
_V0= 235.
def jzdist(R=_R0,hz=250.,sz=20.,nsamples=1):
    """
    NAME:
       jzdist
    PURPOSE:
       generate samples from the Jz distribution for an exponential disk w/ 
       Gaussian vertical velocities
    INPUT:
       R= Galactocentric radius to create the Jz distribution at [kpc]
       hz = scale height [pc]
       sz= vertical velocity dispersion at R (not at R0) [km/s]
    OUTPUT:
       list of samples
    HISTORY:
       2012-03-07 - Written - Bovy (IAS)
    """
    #Convert to proper units
    sz/= _V0
    R/= _R0
    hz/= _R0*1000.
    #Sample
    out= []
    for ii in range(nsamples):
        #Sample z
        thisz= stats.expon.rvs()*hz
        #Sample vz
        thisvz= stats.norm.rvs()*sz
        #Calculate the action
        #print thisz, thisvz
        out.append(jz(thisz,thisvz,R,pot=MWPotential))
    return out

def jz(z,vz,R,pot=MWPotential):
    """
    NAME:
       jz
    PURPOSE:
       calculate the vertical action
    INPUT:
       z - height (/R0)
       vz - velocity (/V0)
       R - Galactocentric radius (/Ro)
       pot= potential to use
    OUTPUT:
    HISTORY:
       2012-03-07 - Written - Bovy (IAS)
    """
    #Calculate Ez, /vz for stability
    potzero= evaluatePotentials(R,0.,pot)/vz**2.
    Ez= 0.5+evaluatePotentials(R,z,pot)/vz**2.-potzero
    #Integrate to get Jz
    out= 2.*numpy.fabs(vz)/math.pi*integrate.quad(_jzIntegrand,0.,numpy.inf,
                                                  args=(R,pot,Ez,potzero,vz))[0]
    #if out == 0.: print Ez, z, vz
    return out

def _jzIntegrand(z,R,pot,Ez,potzero,vz):
    vz2= 2.*Ez-evaluatePotentials(R,z,pot)/vz**2.+potzero
    if vz2 < 0.: return 0. #Such that we don't have to specify the upper limit
    else: return numpy.sqrt(vz2)
