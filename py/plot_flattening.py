from optparse import OptionParser
import numpy
from galpy.potential import MWPotential, flattening
from galpy.util import bovy_plot
from matplotlib import pyplot
def plot_flattening(parser):
    options,args= parser.parse_args()
    Zs= numpy.linspace(0.000001,10./8.,1001)
    Rs= numpy.array([4.,8.,12.,16.])/8.
    plotthis= numpy.array([[flattening(MWPotential,r,z) for z in Zs] for r in Rs])
    plotthisDisk= numpy.array([[flattening(MWPotential[0],r,z) for z in Zs] for r in Rs])
    bovy_plot.bovy_print()
    ii= 0
    bovy_plot.bovy_plot(Zs*8,plotthis[ii,:],'k-',
                        xlabel=r'$|Z|\ [\mathrm{kpc}]$',
                        ylabel=r'$q$',
                        xrange=[0.,10.],
                        yrange=[0.,1.1])
    for ii in range(1,len(Rs)):
        bovy_plot.bovy_plot(Zs*8,plotthis[ii,:],'k-',overplot=True)
    for ii in range(len(Rs)):
        bovy_plot.bovy_plot(Zs*8,plotthisDisk[ii,:],'k--',overplot=True)
    bovy_plot.bovy_text(4.,0.775,r'$R = 4\,\mathrm{kpc}$',fontsize=14.)
    bovy_plot.bovy_text(3.,0.9,r'$R = 16\,\mathrm{kpc}$',fontsize=14.)
    bovy_plot.bovy_text(5.5,0.96,r'$\mathrm{disk+halo+bulge}$',fontsize=14.)
    bovy_plot.bovy_text(6.5,0.74,r'$\mathrm{disk\ only}$',fontsize=14.)
    #Plot Koposov errorbar
    pyplot.errorbar(numpy.array([6.]),
                    numpy.array([0.87]),
                    yerr=numpy.array([0.04,0.07]).reshape((2,1)),
                    color='k',fmt='o',ms=8)
    bovy_plot.bovy_text(6.2,0.83,r'$\mathrm{K10}$',fontsize=12.)
    #Plot Zhang/others errorbar
    zhangz= 0.85
    zhangq= numpy.sqrt(0.85/8.*220.**2./8000./1.75)
    zhangqerr= zhangq*0.15
    btz= 2.8
    btq= numpy.sqrt(2.8/8.*220.**2./8000./2.7)*0.92 #latter is correction to kz
    btqerr= btq*0.15
    pyplot.errorbar(numpy.array([zhangz,btz]),
                    numpy.array([zhangq,btq]),
                    yerr=numpy.array(zhangqerr,btqerr),
                    color='k',fmt='o',ms=8)
    bovy_plot.bovy_text(0.1,0.62,r'$\mathrm{Z12}$',fontsize=12.)
    bovy_plot.bovy_text(1.6,0.82,r'$\mathrm{BT12}$',fontsize=12.)
    bovy_plot.bovy_end_print(options.outfilename)       
        
def get_options():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    #Data options
    parser.add_option("-o",dest='outfilename',default=None,
                      help="Name for an output file")
    return parser
    
if __name__ == '__main__':
    parser= get_options()
    plot_flattening(parser)
