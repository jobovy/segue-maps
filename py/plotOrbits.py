import os, os.path
import math
import numpy
import cPickle as pickle
from optparse import OptionParser
from galpy.util import bovy_plot
from pixelFitDens import pixelAfeFeh
def plotOrbits(parser):
    options,args= parser.parse_args()
    if not os.path.exists(args[0]):
        print args[0]+" does not exist ..."
        print "Returning ..."
        return None
    if options.plotfile is None:
        print "-o or --plotfile needs to be set ..."
        print "Returning ..."
        return None
    #Load orbits
    savefile= open(args[0],'rb')
    orbits= pickle.load(savefile)
    savefile.close()
    #Cut to S/N, logg, and EBV
    indx= (orbits.sna > 15.)*(orbits.logga > 4.2)*(orbits.ebv < 0.3)
    orbits= orbits[indx]
    #Load the orbits into the pixel structure
    pix= pixelAfeFeh(orbits,dfeh=options.dfeh,dafe=options.dafe,fehmin=-1.6,
                     fehmax=0.5,afemin=-0.05,afemax=0.55)
    #Run through the pixels and gather
    plotthis= numpy.zeros((pix.npixfeh(),pix.npixafe()))
    for ii in range(pix.npixfeh()):
        for jj in range(pix.npixafe()):
            data= pix(pix.feh(ii),pix.afe(jj))
            if len(data) < options.minndata:
                plotthis[ii,jj]= numpy.nan
                continue
            if options.type == 'rmean':
                vals= data.rmean*8.
            elif options.type == 'e':
                vals= data.e
            elif options.type == 'rap':
                vals= data.rap*8.
            elif options.type == 'rperi':
                vals= data.rperi*8.
            elif options.type == 'zmax':
                vals= data.zmax*8.
            elif options.type == 'vphi':
                vals= data.vyc+220.
            if options.type == 'nstars':
                plotthis[ii,jj]= len(data)
            else:
                if options.mean:
                    plotthis[ii,jj]= numpy.mean(vals)
                else:
                    plotthis[ii,jj]= numpy.median(vals)
    #Set up plot
    if options.type == 'rmean':
        vmin, vmax= 6., 10.
        zlabel=r'$R_{\mathrm{mean}}\ [\mathrm{kpc}]$'
    elif options.type == 'rap':
        vmin, vmax= 6., 10.
        zlabel=r'$R_{\mathrm{apo}}\ [\mathrm{kpc}]$'
    elif options.type == 'rperi':
        vmin, vmax= 6., 10.
        zlabel=r'$R_{\mathrm{peri}}\ [\mathrm{kpc}]$'
    elif options.type == 'zmax':
        vmin, vmax= .7, 4.
        zlabel=r'$Z_{\mathrm{max}}\ [\mathrm{kpc}]$'
    elif options.type == 'vphi':
        vmin, vmax= 140.,250.
        zlabel=r'$v_{\phi}\ [\mathrm{km\ s}^{-1}]$'
    elif options.type == 'e':
        vmin, vmax= 0.,1.
        zlabel=r'$\mathrm{eccentricity}$'
    elif options.type == 'nstars':
        vmin, vmax= 0.,550.
        zlabel=r'$\mathrm{number\ of\ stars}$'
    bovy_plot.bovy_print()
    bovy_plot.bovy_dens2d(plotthis.T,origin='lower',cmap='jet',
                          interpolation='nearest',
                          xrange=[-1.6,0.3],
                          yrange=[-0.05,0.4],
                          xlabel=r'$[\mathrm{Fe/H}]$',
                          ylabel=r'$[\alpha/\mathrm{Fe}]$',
                          zlabel=zlabel,
                          vmin=vmin,vmax=vmax,
                          contours=False,
                          colorbar=True,shrink=0.78)
    bovy_plot.bovy_end_print(options.plotfile)
    return None

def get_options():
    usage = "usage: %prog [options] <savefile>\n\nsavefile= name of the file that contains the orbits"
    parser = OptionParser(usage=usage)
    parser.add_option("-o","--plotfile",dest='plotfile',default=None,
                      help="Name of the file for plot")
    parser.add_option("-t","--type",dest='type',default='rmean',
                      help="Quantity to plot")
    parser.add_option("--sample",dest='sample',default='g',
                      help="Use 'G' or 'K' dwarf sample")
    parser.add_option("--select",dest='select',default='all',
                      help="Select 'all' or 'program' stars")
    parser.add_option("--dfeh",dest='dfeh',default=0.05,type='float',
                      help="FeH bin size")   
    parser.add_option("--dafe",dest='dafe',default=0.025,type='float',
                      help="[a/Fe] bin size")   
    parser.add_option("--minndata",dest='minndata',default=20,type='int',
                      help="Minimum number of objects in a bin to plot")
    parser.add_option("--mean",action="store_true", dest="mean",
                      default=False,
                      help="If set, plot mean value")
    return parser


if __name__ == '__main__':
    plotOrbits(get_options())
