import numpy
from scipy import special
from monoAbundanceMW import *
_DFEH= 0.1
_DAFE= 0.05
_SQRTTWO= numpy.sqrt(2.)
#Pre-define
sig2zs= numpy.array([sigmaz(results['feh'][ii],results['afe'][ii])**2. for ii in range(len(results))])
fehs= results['feh']
afes= results['afe']
fehmax= fehs+_DFEH/2.
fehmin= fehs-_DFEH/2.
afemax= afes+_DAFE/2.
afemin= afes-_DAFE/2.
def sigmazObs(z,feh,afe,dfeh,dafe):
    if isinstance(z,(list,numpy.ndarray)):
        pass
    else:
        pab= numpy.array([abundanceDist(results['feh'][ii],
                                        results['afe'][ii],z=z) for ii in range(len(results))])
        pab*= _integrateFehAfeDist(feh,afe,dfeh,dafe)
        pab/= numpy.sum(pab)
        print pab*sig2zs
        return numpy.sqrt(numpy.sum(pab*sig2zs))
        
def _integrateFehAfeDist(feh,afe,dfeh,dafe):
    return 0.25*(special.erf((fehmax-feh)/_SQRTTWO/dfeh)\
                     -special.erf((fehmin-feh)/_SQRTTWO/dfeh))\
                     *(special.erf((afemax-afe)/_SQRTTWO/dafe)\
                           -special.erf((afemin-afe)/_SQRTTWO/dafe))

