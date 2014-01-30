import sys
import numpy
from scipy import optimize, integrate
from galpy import potential
from galpy.util import bovy_plot, bovy_conversion
from matplotlib import pyplot
from matplotlib.patches import FancyArrowPatch
def m10mvir(plotfilename):
    pass

def pm10mvirwerr(m10,dm10,mvir,mvirprior=False):
    """p(Mvir | M(<10kpc)) using c(M) relation from Maccio et al."""
    out= integrate.fixed_quad(lambda x: pm10mvirvec(x*dm10+m10,
                                                    mvir,mvirprior=mvirprior)\
                                  *numpy.exp(-0.5*x**2.),
                              -2.99,3.,args=(),n=20)[0]
    print out
    return out

def pm10mvirvec(m10,mvir,mvirprior=False):
    return numpy.array([pm10mvir(m,mvir,mvirprior=mvirprior) for m in m10])
def pm10mvir(m10,mvir,mvirprior=False):
    """p(Mvir | M(<10kpc)) using c(M) relation from Maccio et al."""
    #First determine the concentration
    logc= numpy.log10(concm10mvir(m10,mvir))
    if numpy.isnan(logc): return 0.
    #Calculate the Jacobian d m10 / d c
    logdc= numpy.log10(concm10mvir(m10+0.001,mvir))
    jac= numpy.fabs(0.001/(logdc-logc))
    out= numpy.exp(-0.5*(logc-1.051+0.099*numpy.log10(mvir))**2./0.12**2.)/jac
    if mvirprior:
        #Halo mass function ~ m^-1.9
        out*= mvir**-1.9
    return out

def concm10mvir(m10,mvir):
    """concentration(m10,mvir), m10 in 10^10 Msol, mvir in 10^12 Msol"""
    p= optimize.fmin_powell(_conc_opt,[-0.5,0.7],(m10,mvir),disp=False)
    vo= numpy.exp(p[0])
    a= numpy.exp(p[1])
    nfw= potential.NFWPotential(normalize=1.,a=a)
    try:
        rvir= nfw._rvir(220.*vo,8.,wrtcrit=True,overdens=96.7)
    except ValueError:
        return numpy.nan
    return rvir/a

def _conc_opt(p,m10,mvir):
    vo= numpy.exp(p[0])
    a= numpy.exp(p[1])
    nfw= potential.NFWPotential(normalize=1.,a=a)
    try:
        rvir= nfw._rvir(220.*vo,8.,wrtcrit=True,overdens=96.7)
    except ValueError:
        return numpy.nan
    mass1= nfw.mass(10./8.)*bovy_conversion.mass_in_1010msol(220.*vo,8.)
    mass2= nfw.mass(rvir)*bovy_conversion.mass_in_1010msol(220.*vo,8.)/100.
    return 0.5*((mass1-m10)**2.+(mass2-mvir)**2.)
    

if __name__ == '__main__':
    m10mvir(sys.argv[1])
