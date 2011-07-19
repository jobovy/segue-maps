import os, os.path
import math
import numpy
from optparse import OptionParser
from galpy.util import bovy_plot
import segueSelect
def selectFigs(parser):
    (options,args)= parser.parse_args()
    if options.type.lower() == 'platesn':
        plot_platesn(options,args)

def plot_platesn(options,args):
    """Plot the platesn vs other sns"""
    sf= segueSelect.segueSelect(sn=True,sample='G') #Unimportant
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(sf.platestr.platesn_r,sf.platestr.sn1_1,'gv',
                        xrange=[0.,300.],yrange=[0.,300],
                        xlabel=r'$\mathrm{plateSN\_r} \equiv (\mathrm{sn1\_1} + \mathrm{sn2\_1})/2$',
                        ylabel=r'$\mathrm{snX\_Y}$')
    bovy_plot.bovy_plot(sf.platestr.platesn_r,sf.platestr.sn2_1,'y^',
                        overplot=True)
    bovy_plot.bovy_plot(sf.platestr.platesn_r,sf.platestr.sn1_0,'bs',
                        overplot=True)
    bovy_plot.bovy_plot(sf.platestr.platesn_r,sf.platestr.sn2_0,'cp',
                        overplot=True)
    bovy_plot.bovy_plot(sf.platestr.platesn_r,sf.platestr.sn1_2,'rh',
                        overplot=True)
    bovy_plot.bovy_plot(sf.platestr.platesn_r,sf.platestr.sn2_2,'mH',
                        overplot=True)
    bovy_plot.bovy_text(25.,280,r'$\mathrm{sn1\_1}:\ r\ \mathrm{band}$',color='g',size=14.)
    bovy_plot.bovy_plot([15.],[285.],'gv',overplot=True)
    bovy_plot.bovy_text(25.,265,r'$\mathrm{sn2\_1}:\ r\ \mathrm{band}$',color='y',size=14.)
    bovy_plot.bovy_plot([15.],[270.],'y^',overplot=True)
    bovy_plot.bovy_text(25.,250,r'$\mathrm{sn1\_0}:\ g\ \mathrm{band}$',color='b',size=14.)
    bovy_plot.bovy_plot([15.],[255.],'bs',overplot=True)
    bovy_plot.bovy_text(25.,235,r'$\mathrm{sn2\_0}:\ g\ \mathrm{band}$',color='c',size=14.)
    bovy_plot.bovy_plot([15.],[240.],'cp',overplot=True)
    bovy_plot.bovy_text(25.,220,r'$\mathrm{sn1\_2}:\ i\ \mathrm{band}$',color='r',size=14.)
    bovy_plot.bovy_plot([15.],[225.],'rh',overplot=True)
    bovy_plot.bovy_text(25.,205,r'$\mathrm{sn2\_2}:\ i\ \mathrm{band}$',color='m',size=14.)
    bovy_plot.bovy_plot([15.],[210.],'mH',overplot=True)
    bovy_plot.bovy_end_print(options.plotfile)
    
def get_options():
    usage = "usage: %prog [options] <savefilename>\n\nsavefilename= name of the file that the fit/samples will be saved to"
    parser = OptionParser(usage=usage)
    parser.add_option("-o",dest='plotfile',
                      help="Name of file for plot")
    parser.add_option("-t",dest='type',
                      help="Type of plot to make")
    #parser.add_option("--plotfunc",action="store_true", dest="plotfunc",
    #                  default=False,
    #                  help="Plot samples from the inferred sigma_z(z) relation at R_0")
    return parser

if __name__ == '__main__':
    selectFigs(get_options())
