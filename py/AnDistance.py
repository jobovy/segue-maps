import numpy
import isodist
import monoAbundanceMW
from segueSelect import _mr_gi, _gi_gr
#Load An isochrones
a= isodist.AnIsochrone()
anfehs= [-0.1,-0.2,-0.3,-0.5,-1.,-1.5,-2.,-3.,0.,0.1,0.2,0.4]
def AnDistance(gr,feh,k=False):
    """Return the relative distance factor between An and Ivezic distances for a sample of stars"""
    #First find nearest feh with an isochrone
    mfeh= numpy.mean(feh)
    indx= numpy.argmin(numpy.fabs(mfeh-anfehs))
    #Load the relevant isochrone
    iso= a(numpy.log10(10.),feh=anfehs[indx])
    #Get dwarfs
    if not k:
        rindx= (iso['g']-iso['r'] <= 0.75)*(iso['g']-iso['r'] >= 0.55)\
            *(iso['logg'] > 4.1)
    else:
        rindx= (iso['g']-iso['r'] > 0.75)*(iso['g']-iso['r'] <= 0.95)\
            *(iso['logg'] > 4.1)
    y= -1.*(iso['r'][rindx]-_mr_gi(_gi_gr(iso['g'][rindx]
                                         -iso['r'][rindx]),anfehs[indx]))
    isogr= iso['g'][rindx]-iso['r'][rindx]
    out= 0.
    for ii in range(len(gr)):
        tindx= numpy.argmin(numpy.fabs(gr[ii]-isogr))
        out+= y[tindx]
    return 10.**(out/5./len(gr))
