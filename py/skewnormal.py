###############################################################################
#   SKEW-NORMAL distribution
###############################################################################
import numpy
from scipy.stats import norm
from scipy import linalg, optimize, integrate
def skewnormal(x,m=0,s=1,a=0):
    """
    NAME:
        skewnormal
    PURPOSE:
        one-D skew-normal distribution
    INPUT:
       x - value(s) to evaluate
       m= mean
       s= std
       a= shape parameter
    OUTPUT:
       value(s)
    HISTORY:
       2011-12-05 - Written - Bovy (IAS)
    """
    t = (x-m) / s
    return 2. / s * norm.pdf(t) * norm.cdf(a*t)

def logskewnormal(x,m=0,s=1,a=0):
    """
    NAME:
        logskewnormal
    PURPOSE:
        logarithm of one-D skew-normal distribution
    INPUT:
       x - value(s) to evaluate
       m= mean
       s= std
       a= shape parameter
    OUTPUT:
       value(s)
    HISTORY:
       2011-12-05 - Written - Bovy (IAS)
    """
    return numpy.log(skewnormal(x,m=m,s=s,a=a))

def multiskewnormal(x,m=None,V=None,a=None):
    """
    NAME:
        multiskewnormal
    PURPOSE:
        multi-D skew-normal distribution
    INPUT:
       x - value(s) to evaluate [D,N]
       m= mean [D]
       V= variance [D,D]
       a= shape parameter [D]
    OUTPUT:
       value(s)
    HISTORY:
       2011-12-05 - Written - Bovy (IAS)
    """
    Vinv= linalg.inv(V)
    L= linalg.cholesky(Vinv,lower=True)
    v= numpy.sqrt(numpy.diag(V))
    xm= numpy.zeros(x.shape)
    xmv= numpy.zeros(xm.shape)
    for ii in range(x.shape[0]):
        xm[ii,:]= x[ii,:]-m[ii]
    xmv= numpy.dot(L,xm)
    t= numpy.sum((xm)*numpy.dot(Vinv,(xm)),axis=0)
    z= numpy.dot(a,xmv)
    mnorm= (2.*numpy.pi)**(-0.5*len(m))
    return 2.*mnorm*numpy.sqrt(linalg.det(Vinv))*numpy.exp(-0.5*t)*norm.cdf(z)

def logmultiskewnormal(x,m=None,V=None,a=None):
    """
    NAME:
        logmultiskewnormal
    PURPOSE:
        logarithm of multi-D skew-normal distribution
    INPUT:
       x - value(s) to evaluate [D,N]
       m= mean [D]
       V= variance [D,D]
       a= shape parameter [D]
    OUTPUT:
       value(s)
    HISTORY:
       2011-12-05 - Written - Bovy (IAS)
    """
    return numpy.log(multiskewnormal(x,m=m,V=V,a=a))

def alphaskew(skew):
    """
    NAME:
       alphaskew
    PURPOSE:
       estimate alpha (shape parameter) based on the skew of a distribution
    INPUT:
       skew - skew
    OUTPUT:
       alpha estimate
    HISTORY:
       2011-12-07 - Written - Bovy (IAS)
    """
    delta2= optimize.brentq(_alphaskewEq,
                            0.,1.,args=((2.*skew/(4.-numpy.pi))**2.,))
    #delta2= numpy.pi/2./(1.+((4.-numpy.pi)/(2.*numpy.fabs(skew)))**2./3.)
    if skew > 0.:
        return numpy.sqrt(delta2/(1.-delta2))
    if skew <= 0.:
        return -numpy.sqrt(delta2/(1.-delta2))

def _alphaskewEq(d2,s2):
    """skew^2 includes 2/4-pi"""
    return s2*(1-d2)**3.-d2

def convskewnormal(x,m=0,s=1,a=0,e=0.):
    """
    NAME:
        convskewnormal
    PURPOSE:
        one-D skew-normal distribution convolved with Gaussian uncertainty
    INPUT:
       x - value(s) to evaluate
       m= mean
       s= std
       a= shape parameter
       e= uncertainty
    OUTPUT:
       value(s)
    HISTORY:
       2011-12-08 - Written - Bovy (IAS)
    """
    out= integrate.quad(_convskewnormalIntegrand,
                        -numpy.Inf,
                        numpy.Inf,
                        args=(x,m,s,a,e))
    return out[0]

def _convskewnormalIntegrand(t,x,m,s,a,e):
    return skewnormal(t,m=m,s=s,a=a) / e * norm.pdf((x-t)/e)
