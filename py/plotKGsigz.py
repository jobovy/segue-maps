import numpy
from fitSigz import KGSigmadF
zs= numpy.linspace(0.,3000.,1001)
rhodm= 0.01 #Msolar pc**-3.
D= 390. #pc
sigz_thin= 20.
sigz_thick= 40.
hz_thin= 300.
hz_thick= 850.

from galpy.util import bovy_plot
bovy_plot.bovy_print()

#Calculate expected rho increases
rhof= numpy.sqrt(sigz_thin**2.+2.*hz_thin*rhodm*2.*numpy.pi*4.302*10.**-3.*zs)
bovy_plot.bovy_plot(zs,rhof,'k-',xlabel=r'$Z\ [\mathrm{pc}]$',
                    ylabel=r'$\mathrm{contributions\ to}\ \sigma_z(Z)\ \mathrm{from}\ \rho_{\mathrm{eff}}, \Sigma_0$',
                    xrange=[0.,2700.],yrange=[0.,60.])

rhof= numpy.sqrt(sigz_thick**2.+2.*hz_thick*rhodm*2.*numpy.pi*4.302*10.**-3.*zs)
bovy_plot.bovy_plot(zs,rhof,'k-',overplot=True)

#Calculate Sigma_0 induced increase
f= sigz_thin*numpy.sqrt(numpy.array([KGSigmadF(z/hz_thin,D/hz_thin) for z in zs]))
bovy_plot.bovy_plot(zs,f,'k--',overplot=True)
f= sigz_thick*numpy.sqrt(numpy.array([KGSigmadF(z/hz_thick,D/hz_thick) for z in zs]))
bovy_plot.bovy_plot(zs,f,'k--',overplot=True)

bovy_plot.bovy_end_print('../tex-vel/KGSigmadF.ps')
