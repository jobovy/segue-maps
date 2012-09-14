import os, os.path
import math
import numpy
import cPickle as pickle
from optparse import OptionParser
from galpy.util import bovy_plot, save_pickles
from matplotlib import pyplot, cm
from galpy import potential
from galpy.actionAngle_src.actionAngleAdiabaticGrid import  actionAngleAdiabaticGrid
from galpy.df_src.quasiisothermaldf import quasiisothermaldf
from galpy.actionAngle_src.actionAngleAdiabaticGrid import actionAngleAdiabaticGrid
from galpy.potential import MiyamotoNagaiPotential, LogarithmicHaloPotential, MWPotential
def plot_hrhrvshr(options,args):
    """Plot hr^out/hr^in as a function of hr for various sr"""
    if len(args) == 0.:
        print "Must provide a savefilename ..."
        print "Returning ..."
        return None
    if os.path.exists(args[0]):
        #Load
        savefile= open(args[0],'rb')
        plotthis= pickle.load(savefile)
        hrs= pickle.load(savefile)
        srs= pickle.load(savefile)
        savefile.close()
    else:
        #Grid of models to test
        hrs= numpy.linspace(options.hrmin,options.hrmax,options.nhr)
        srs= numpy.linspace(options.srmin,options.srmax,options.nsr)
        #Tile
        hrs= numpy.tile(hrs,(options.nsr,1)).T
        srs= numpy.tile(srs,(options.nhr,1))
        plotthis= numpy.zeros((options.nhr,options.nsr))
        #Setup potential and aA
        pot= MWPotential
        aA=actionAngleAdiabaticGrid(pot=pot,nR=16,nEz=16,nEr=31,nLz=31,zmax=1.,
                                    Rmax=5.)
        for ii in range(options.nhr):
            for jj in range(options.nsr):
                qdf= quasiisothermaldf(hrs[ii,jj]/8.,srs[ii,jj]/220.,
                                       srs[ii,jj]/2./220.,7./8.,7./8.,
                                       pot=pot,aA=aA)
                plotthis[ii,jj]= qdf.estimate_hr(1.,
                                                 dR=numpy.amin([6./8.,
                                                                3./8.*hrs[ii,jj]]))/hrs[ii,jj]*8.
                print ii*options.nsr+jj+1, options.nsr*options.nhr, \
                    hrs[ii,jj], srs[ii,jj], plotthis[ii,jj]
        #Save
        save_pickles(args[0],plotthis,hrs,srs)
    #Now plot
    bovy_plot.bovy_print()
    indx= 0
    lines= []
    colors= [cm.jet(ii/float(options.nhr)*1.+0.) for ii in range(options.nhr)]
    lss= ['-' for ii in range(options.nhr)]#,'--','-.','..']
    labels= []
    lines.append(bovy_plot.bovy_plot(hrs[:,indx],plotthis[:,indx],
                                     color=colors[indx],ls=lss[indx],
                                     xrange=[0.,8.1],
                                     yrange=[0.8,1.4],
                                     xlabel=r'$h^{\mathrm{in}}_R\ \mathrm{at}\ 8\,\mathrm{kpc}$',
                                     ylabel=r'$h^{\mathrm{out}}_R / h^{\mathrm{in}}_R$'))
    labels.append(r'$\sigma_R = %.0f \,\mathrm{km\,s}^{-1}$' % srs[0,indx])
    for indx in range(1,options.nhr):
        lines.append(bovy_plot.bovy_plot(hrs[:,indx],plotthis[:,indx],
                                         color=colors[indx],ls=lss[indx],
                                         overplot=True))
        labels.append(r'$\sigma_R = %.0f \,\mathrm{km\,s}^{-1}$' % srs[0,indx])
    """
    #Legend
    pyplot.legend(lines,#(line1[0],line2[0],line3[0],line4[0]),
                  labels,#(r'$v_{bc} = 0$',
#                   r'$v_{bc} = 1\,\sigma_{bc}$',
#                   r'$v_{bc} = 2\,\sigma_{bc}$',
#                   r'$v_{bc} = 3\,\sigma_{bc}$'),
                  loc='lower right',#bbox_to_anchor=(.91,.375),
                  numpoints=2,
                  prop={'size':14},
                  frameon=False)
    """
     #Add colorbar
    map = cm.ScalarMappable(cmap=cm.jet)
    map.set_array(srs[0,:])
    map.set_clim(vmin=numpy.amin(srs[0,:]),vmax=numpy.amax(srs[0,:]))
    cbar= pyplot.colorbar(map,fraction=0.15)
    cbar.set_clim(numpy.amin(srs[0,:]),numpy.amax(srs[0,:]))
    cbar.set_label(r'$\sigma_R \,[\mathrm{km\,s}^{-1}]$')
    bovy_plot.bovy_end_print(options.plotfilename)

def get_options():
    usage = "usage: %prog [options] <savefile>\n\nsavefile= name of the file that will hold the data to be plotted"
    parser = OptionParser(usage=usage)
    parser.add_option("-t","--type",dest='type',default=None,
                      help="Type of thing to do")
    parser.add_option("-o",dest='plotfilename',default=None,
                      help="Name for output plot")
    parser.add_option("--rmin",dest='rmin',type='float',
                      default=4.,
                      help="Minimum radius")
    parser.add_option("--rmax",dest='rmax',type='float',
                      default=8.,
                      help="Maximum radius")
    parser.add_option("--hrmin",dest='hrmin',type='float',
                      default=1.,
                      help="Minimum scale length")
    parser.add_option("--hrmax",dest='hrmax',type='float',
                      default=6.,
                      help="Maximum scale length")
    parser.add_option("--srmin",dest='srmin',type='float',
                      default=5.,
                      help="Minimum scale length")
    parser.add_option("--srmax",dest='srmax',type='float',
                      default=80.,
                      help="Maximum scale length")
    parser.add_option("--nr",dest='nr',default=2,type='int',
                      help="Number of r to use")
    parser.add_option("--nhr",dest='nhr',default=2,type='int',
                      help="Number of hr to use")
    parser.add_option("--nsr",dest='nsr',default=2,type='int',
                      help="Number of sr to use")
    return parser

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    numpy.random.seed(1)
    if options.type.lower() == 'hrhrhr':
        plot_hrhrvshr(options,args)
    elif options.type.lower() == 'hrhrr':
        plot_hrhrvsr(options,args)
