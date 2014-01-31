import sys
import pickle
import numpy
from galpy.util import bovy_plot
def plot_seguem10(plotfilename):
    savefile= open('../potential-fits/fitall_10ksamples_derived_whalomass.sav','rb')
    data= pickle.load(savefile)
    savefile.close()
    mh= numpy.array([x[8] for x in data])
    bovy_plot.bovy_print()
    bovy_plot.bovy_hist(mh,range=[0.,10.],
                        bins=71,histtype='step',color='k',
                        xlabel=r'$M_{\mathrm{halo}}(<10\,\mathrm{kpc})\,(10^{10}\,M_\odot)$',
                        normed=True)
    xs= numpy.linspace(0.,10.,1001)
    bovy_plot.bovy_plot(xs,1./numpy.sqrt(2.*numpy.pi*numpy.var(mh[True-numpy.isnan(mh)]))*numpy.exp(-0.5*(xs-numpy.mean(mh[True-numpy.isnan(mh)]))**2./numpy.var(mh[True-numpy.isnan(mh)])),
                        'k-',lw=2,overplot=True)
    bovy_plot.bovy_end_print(plotfilename)

if __name__ == '__main__':
    plot_seguem10(sys.argv[1])

"""plot with
python plot_seguem10.py ~/Desktop/mh.png
"""
