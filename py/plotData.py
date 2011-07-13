import math
import numpy
from galpy.util import bovy_plot
_DEGTORAD= math.pi/180.
def plotDensz(data,sf,xrange=[0.,2.],normed=True,overplot=False,bins=20,
              log=True,dR=1.,db=None,noweights=False,color='k'):
    """
    NAME:
       plotDensz
    PURPOSE:
       Plot a 'smart' representation of the data
    INPUT:
       data - data structure (recarray)
       sf - segueSelect instance with selection function
       xrange= the x range
       normed= if False, don't normalize (*not* what you want)
       bins= number of bins
       log= if False, plot density, otherwise plot log density
       dR= range/2. in Galactocentric radius to consider
       db= range/2. from pole to consider
       noweights= if True, don't reweight
       color= color to plot
    OUTPUT:
       plot to output
    HISTORY:
       2011-07-13 - Written - Bovy (NYU)
    """
    if not overplot:
        bovy_plot.bovy_print()
    dx= (xrange[1]-xrange[0])/(bins+1)
    xs= numpy.linspace(xrange[0]+dx,xrange[1]-dx,bins)
    hist= numpy.zeros(bins)
    R= ((8.-data.xc)**2.+data.yc**2.)**(0.5)
    for ii in range(len(data.ra)):
        if math.fabs(R[ii] - 8.) > dR: continue
        if not db is None \
                and data[ii].b < (90.-db) and data[ii].b > (-90.+db): continue
        if noweights:
            thissf= 1.
        else:
            #Get the weight
            thissf= sf(data[ii].plate)
        jac= data[ii].dist**2. #*numpy.fabs(numpy.cos(data[ii].b*_DEGTORAD))
        jac/= numpy.fabs(numpy.sin(data[ii].b*_DEGTORAD))
        jac*= numpy.exp(-(R[ii]-8.)/2.75)
        #bin number
        thisbin= int(math.floor((numpy.fabs(data[ii].zc-xrange[0]))/dx))
        if thisbin > (bins-1): continue
        hist[thisbin]+= 1./jac/thissf
    #Normalize
    hist/= numpy.sum(hist)*dx
    if log: hist= numpy.log(hist)
    bovy_plot.bovy_plot(xs,hist,color+'D',overplot=overplot)
