import sys
import numpy
from scipy import optimize
from galpy import potential
from galpy.util import bovy_plot, bovy_conversion
from matplotlib import pyplot
from matplotlib.patches import FancyArrowPatch
import bovy_mcmc
_XUER= numpy.array([22.5, 27.5,32.5,37.5,42.5,47.5,55.])
_XUEVC= numpy.array([164.,183.,143.,183.,203.,166.,180.])
_XUEVC_ERR= numpy.array([20.,20.,22.,39.,35.,30.,35.])
_G= 4.302*10.**-3. #pc / Msolar (km/s)^2
_XUEMASS= _XUER*_XUEVC**2./_G/10.**7.-5
_XUEMASS_ERR= numpy.sqrt(2.)*_XUEVC_ERR/_XUEVC*_XUER*_XUEVC**2./_G/10.**7.
_USE_ALL_XUE= True
def verysimplenfwfit(plotfilename,wcprior=False,wmassprior=False):
    #Fit
    p= optimize.fmin_powell(chi2,[-0.5,0.7],args=(wcprior,wmassprior))
    print mvir(p), conc(p)*numpy.exp(p[1])*8.
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
    if _USE_ALL_XUE:
        plotx= [10.]
        plotx.extend(list(_XUER))
        ploty= [4.5/100.]
        ploty.extend(list(_XUEMASS/100.))
        plotyerr= [1.5/100]
        plotyerr.extend(list(_XUEMASS_ERR/100.))
        pyplot.errorbar(plotx,ploty,yerr=plotyerr,marker='o',ls='none',
                        color='k')
    else:
        pyplot.errorbar([10.,60.],[4.5/100.,3.5/10.],yerr=[1.5/100.,0.7/10.],
                        marker='o',ls='none',color='k')
    pyplot.errorbar([150.],[7.5/10.],[2.5/10.],marker='None',color='k')
    dx= .5
    arr= FancyArrowPatch(posA=(7.,4./100.),posB=(numpy.exp(numpy.log(7.)+dx),numpy.exp(numpy.log(4./100.)+1.5*dx)),arrowstyle='->',connectionstyle='arc3,rad=%4.2f' % (0.),shrinkA=2.0, shrinkB=2.0,mutation_scale=20.0, mutation_aspect=None,fc='k')
    ax = pyplot.gca()
    ax.add_patch(arr)
    #Sample MCMC to get virial mass uncertainty
    samples= bovy_mcmc.markovpy(p,
                                0.05,
                                lnlike,
                                (wcprior,wmassprior),
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
    rvirs= concs[indx]*numpy.array([numpy.exp(x[1]) for x in samples])*8.
    bovy_plot.bovy_text(r'$M_{\mathrm{vir}} = %.2f\pm%.2f\times10^{12}\,M_\odot$' % (numpy.median(mvirs),1.4826*numpy.median(numpy.fabs(mvirs-numpy.median(mvirs)))) +'\n'+
                        r'$r_{\mathrm{vir}} = %i\pm%i\,\mathrm{kpc}$' % (numpy.median(rvirs),1.4826*numpy.median(numpy.fabs(rvirs-numpy.median(rvirs))))+'\n'+
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

def chi2(p,wcprior,wmassprior):
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
    out= 0.
    if wcprior:
        out-= -0.5*(numpy.log10(c)-1.051+0.099*numpy.log10(mvir))**2./0.111**2.
    elif numpy.fabs(numpy.log10(c)-1.051+0.099*numpy.log10(mvir)) > 0.1111*3.:
        return 10000000000000000.
    if wmassprior:
        out-= -0.9*numpy.log10(mvir)
    mass1= nfw.mass(10./8.)*bovy_conversion.mass_in_1010msol(220.*vo,8.)
    if _USE_ALL_XUE:
        out-= -0.5*((mass1-4.5)**2./1.5**2.)
        for ii in range(len(_XUER)):
            tmass= nfw.mass(_XUER[ii]/8.)\
                *bovy_conversion.mass_in_1010msol(220.*vo,8.)
            out-= -0.5*((tmass-_XUEMASS[ii])**2./_XUEMASS_ERR[ii]**2.)
    else:
        mass2= nfw.mass(60./8.)*bovy_conversion.mass_in_1010msol(220.*vo,8.)
        out-= -0.5*((mass1-4.5)**2./1.5**2.+(mass2-35)**2./7.**2.)
    #Jacobian
    out-= numpy.log(jaccmvir(p,numpy.log10(c),numpy.log10(mvir)))
    return out

def jaccmvir(p,logc,logm):
    dx= 0.001
    vo= numpy.exp(p[0]+dx)
    a= numpy.exp(p[1])
    if a > 100. or a < 0.01: return 10000000000000.
    nfw= potential.NFWPotential(normalize=1.,a=a)
    try:
        rvir= nfw._rvir(220.*vo,8.,wrtcrit=True,overdens=96.7)
    except ValueError:
        return numpy.nan
    mvir= nfw.mass(rvir)*bovy_conversion.mass_in_1010msol(220.*vo,8.)/100.
    c= rvir/a
    dlogcdp0= (numpy.log10(c)-logc)/dx
    dlogmdp0= (numpy.log10(mvir)-logm)/dx
    #Change p1
    vo= numpy.exp(p[0])
    a= numpy.exp(p[1]+dx)
    if a > 100. or a < 0.01: return 10000000000000.
    nfw= potential.NFWPotential(normalize=1.,a=a)
    try:
        rvir= nfw._rvir(220.*vo,8.,wrtcrit=True,overdens=96.7)
    except ValueError:
        return numpy.nan
    mvir= nfw.mass(rvir)*bovy_conversion.mass_in_1010msol(220.*vo,8.)/100.
    c= rvir/a
    dlogcdp1= (numpy.log10(c)-logc)/dx
    dlogmdp1= (numpy.log10(mvir)-logm)/dx
    return numpy.fabs((dlogcdp0*dlogmdp1-dlogcdp1*dlogmdp0))

def lnlike(p,wcprior,wmassprior):
    return -chi2(p,wcprior,wmassprior)

if __name__ == '__main__':
    numpy.random.seed(1)
    verysimplenfwfit(sys.argv[1],wcprior=len(sys.argv) > 2,
                     wmassprior=len(sys.argv) > 3)

"""Make plots with
python verysimplenfwfit.py ~/Desktop/massMW.png
python verysimplenfwfit.py ~/Desktop/massMW_wcprior.png wcprior
python verysimplenfwfit.py ~/Desktop/massMW_wcprior_wmassprior.png wcprior wmassprior
"""
