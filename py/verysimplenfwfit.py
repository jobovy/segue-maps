import sys
import numpy
from scipy import optimize
from galpy import potential
from galpy.util import bovy_plot, bovy_conversion
from matplotlib import pyplot
from matplotlib.patches import FancyArrowPatch
import bovy_mcmc
def verysimplenfwfit(plotfilename):
    #Fit
    p= optimize.fmin_powell(chi2,[-0.5,0.7])
    print mvir(p)
    vo= numpy.exp(p[0])
    a= numpy.exp(p[1])
    nfw= potential.NFWPotential(normalize=1.,a=a)
    rs= numpy.linspace(0.01,350.,1001)
    masses= numpy.array([nfw.mass(r/8.) for r in rs])
    bovy_plot.bovy_print(fig_width=6.)
    bovy_plot.bovy_plot(rs,
                        masses*bovy_conversion.mass_in_1010msol(220.*vo,8.)/100.,
                        'k-',loglog=True,
                        xlabel=r'$R\,(\mathrm{kpc})$',
                        ylabel=r'$M(<R)\,(10^{12}\,M_\odot)$',
                        yrange=[0.01,1.2],
                        xrange=[3.,400.])
    pyplot.errorbar([10.,60.],[4.5/100.,3.5/10.],yerr=[1.5/100.,0.7/10.],marker='o',ls='none',color='k')
    pyplot.errorbar([150.],[7.5/10.],[2.5/10.],marker='None',color='k')
    dx= .5
    arr= FancyArrowPatch(posA=(7.,4./100.),posB=(numpy.exp(numpy.log(7.)+dx),numpy.exp(numpy.log(4./100.)+1.5*dx)),arrowstyle='->',connectionstyle='arc3,rad=%4.2f' % (0.),shrinkA=2.0, shrinkB=2.0,mutation_scale=20.0, mutation_aspect=None,fc='k')
    ax = pyplot.gca()
    ax.add_patch(arr)
    #Sample MCMC to get virial mass uncertainty
    samples= bovy_mcmc.markovpy(p,
                                0.05,
                                lnlike,
                                (),
                                isDomainFinite=[[False,False],[False,False]],
                                domain=[[0.,0.],[0.,0.]],
                                nsamples=10000,
                                nwalkers=6)
    mvirs= numpy.array([mvir(x) for x in samples])
    concs= numpy.array([conc(x) for x in samples])
    indx= True-numpy.isnan(mvirs)
    mvirs= mvirs[indx]
    indx= True-numpy.isnan(concs)
    concs= concs[indx]
    bovy_plot.bovy_text(r'$M_{\mathrm{vir}} = %.2f\pm%.2f\times10^{12}\,M_\odot$' % (numpy.median(mvirs),1.4826*numpy.median(numpy.fabs(mvirs-numpy.median(mvirs)))) +'\n'+
                        r'$c = %.1f\pm%.1f$' % (numpy.median(concs),1.4826*numpy.median(numpy.fabs(concs-numpy.median(concs)))),
                        top_left=True,size=18.)
    #Create inset with PDF
    insetAxes= pyplot.axes([0.55,0.22,0.3,0.3])
    pyplot.sca(insetAxes)
    bovy_plot.bovy_hist(mvirs,range=[0.,3.],bins=51,histtype='step',
                        color='k',
                        normed=True,
                        overplot=True)
    insetAxes.set_xlim(0.,3.)
    insetAxes.set_ylim(0.,1.49)
    insetAxes.set_xlabel(r'$M_{\mathrm{vir}}\,(10^{12}\,M_\odot)$')
    bovy_plot._add_ticks()
    bovy_plot.bovy_end_print(plotfilename)

def mvir(p):
    vo= numpy.exp(p[0])
    a= numpy.exp(p[1])  
    nfw= potential.NFWPotential(normalize=1.,a=a)
    try:
        rvir= nfw._rvir(220.*vo,8.,wrtcrit=True,overdens=96.7)
    except ValueError:
        return numpy.nan
    return nfw.mass(rvir)*bovy_conversion.mass_in_1010msol(220.*vo,8.)/100.

def conc(p):
    vo= numpy.exp(p[0])
    a= numpy.exp(p[1])  
    nfw= potential.NFWPotential(normalize=1.,a=a)
    try:
        rvir= nfw._rvir(220.*vo,8.,wrtcrit=True,overdens=96.7)
    except ValueError:
        return numpy.nan
    return rvir/a

def chi2(p):
    """chi2 for the Bovy & Rix and Xue et al. measurements"""
    vo= numpy.exp(p[0])
    a= numpy.exp(p[1])
    if a > 100. or a < 0.01: return 10000000000000.
    nfw= potential.NFWPotential(normalize=1.,a=a)
    try:
        rvir= nfw._rvir(220.*vo,8.,wrtcrit=True,overdens=96.7)
    except ValueError:
        return numpy.nan
    mvir= nfw.mass(rvir)*bovy_conversion.mass_in_1010msol(220.*vo,8.)/100.
    c= rvir/a
    if numpy.fabs(numpy.log10(c)-1.051+0.099*numpy.log10(mvir)) > 0.1111*3.:
        return 10000000000000000.
    mass1= nfw.mass(10./8.)*bovy_conversion.mass_in_1010msol(220.*vo,8.)
    mass2= nfw.mass(60./8.)*bovy_conversion.mass_in_1010msol(220.*vo,8.)
    return 0.5*((mass1-4.5)**2./1.5**2.+(mass2-35)**2./7.**2.)

def lnlike(p):
    return -chi2(p)

if __name__ == '__main__':
    numpy.random.seed(1)
    verysimplenfwfit(sys.argv[1])
