###############################################################################
#   SKEW-NORMAL distribution
###############################################################################
import numpy
from scipy.stats import norm
from scipy import linalg
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
    return 2 / s * norm.pdf(t) * norm.cdf(a*t)

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
    t= numpy.sum((x-m)*numpy.dot(Vinv,(x-m)),axis=0)
    v= numpy.sqrt(numpy.diag(V))
    z= numpy.dot(a,(x-m)/v)
    mnorm= (2.*numpy.pi)**(-0.5*len(m))
    return mnorm*numpy.sqrt(linalg.det(Vinv))*numpy.exp(-0.5*t**2.)

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
