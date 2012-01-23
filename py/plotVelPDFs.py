#Plot example velocity PDFs
import os, os.path
import sys
import math
import numpy
import cPickle as pickle
from scipy import special, linalg
from scipy.stats import gaussian_kde
from optparse import OptionParser
from galpy.util import bovy_coords, bovy_plot, save_pickles
from matplotlib import pyplot, cm
from matplotlib.patches import Ellipse
from segueSelect import read_gdwarfs, read_kdwarfs, _GDWARFFILE, _KDWARFFILE
from pixelFitDens import pixelAfeFeh
from fitSigz import _ZSUN
def plotVelPDFs(options,args):
    if options.sample.lower() == 'g':
        if options.select.lower() == 'program':
            raw= read_gdwarfs(_GDWARFFILE,logg=True,ebv=True,sn=True)
        else:
            raw= read_gdwarfs(logg=True,ebv=True,sn=True)
    elif options.sample.lower() == 'k':
        if options.select.lower() == 'program':
            raw= read_kdwarfs(_KDWARFFILE,logg=True,ebv=True,sn=True)
        else:
            raw= read_kdwarfs(logg=True,ebv=True,sn=True)
    #Bin the data   
    binned= pixelAfeFeh(raw,dfeh=options.dfeh,dafe=options.dafe)
    if options.tighten:
        tightbinned= pixelAfeFeh(raw,dfeh=options.dfeh,dafe=options.dafe,
                                 fehmin=-1.6,fehmax=0.5,afemin=-0.05,
                                 afemax=0.55)
    else:
        tightbinned= binned
    #Savefile1
    if os.path.exists(args[0]):#Load savefile
        savefile= open(args[0],'rb')
        velfits= pickle.load(savefile)
        savefile.close()
    if os.path.exists(args[1]):#Load savefile
        savefile= open(args[1],'rb')
        densfits= pickle.load(savefile)
        savefile.close()
    #Uncertainties are in savefile3 and 4
    if len(args) > 3 and os.path.exists(args[3]):
        savefile= open(args[3],'rb')
        denssamples= pickle.load(savefile)
        savefile.close()
        denserrors= True
    else:
        denssamples= None
        denserrors= False
    if len(args) > 2 and os.path.exists(args[2]):
        savefile= open(args[2],'rb')
        velsamples= pickle.load(savefile)
        savefile.close()
        velerrors= True
    else:
        velsamples= None            
        velerrors= False
    #Now plot
    #Run through the pixels and gather
    if options.type.lower() == 'afe' or options.type.lower() == 'feh' \
            or options.type.lower() == 'fehafe' \
            or options.type.lower() == 'zfunc' \
            or options.type.lower() == 'afefeh':
        plotthis= []
        errors= []
    else:
        plotthis= numpy.zeros((tightbinned.npixfeh(),tightbinned.npixafe()))
    if options.kde: allsamples= []
    sausageFehAfe= [options.feh,options.afe]#-0.15,0.075]#[[-0.85,0.425],[-0.45,0.275],[-0.15,0.075]]
    if options.subtype.lower() == 'sausage':
        sausageSamples= []
    for ii in range(tightbinned.npixfeh()):
        for jj in range(tightbinned.npixafe()):
            data= binned(tightbinned.feh(ii),tightbinned.afe(jj))
            fehindx= binned.fehindx(tightbinned.feh(ii))#Map onto regular binning
            afeindx= binned.afeindx(tightbinned.afe(jj))
            if not (numpy.fabs(tightbinned.feh(ii)-sausageFehAfe[0])< 0.01 and numpy.fabs(tightbinned.afe(jj) - sausageFehAfe[1]) < 0.01):
                continue
            thisdensfit= densfits[afeindx+fehindx*binned.npixafe()]
            thisvelfit= velfits[afeindx+fehindx*binned.npixafe()]
            if options.velmodel.lower() == 'hwr':
                thisplot=[tightbinned.feh(ii),
                              tightbinned.afe(jj),
                              numpy.exp(thisvelfit[1]),
                              numpy.exp(thisvelfit[4]),
                              len(data),
                              thisvelfit[2],
                              thisvelfit[3]]
                #Als find min and max z for this data bin, and median
                zsorted= sorted(numpy.fabs(data.zc+_ZSUN))
                zmin= zsorted[int(numpy.ceil(0.16*len(zsorted)))]
                zmax= zsorted[int(numpy.floor(0.84*len(zsorted)))]
                thisplot.extend([zmin,zmax,numpy.mean(numpy.fabs(data.zc+_ZSUN))])
                #Errors
                if velerrors:
                    thesesamples= velsamples[afeindx+fehindx*binned.npixafe()]
                break
    #Now plot
    if options.type.lower() == 'slopequad':
        plotx= numpy.array([thesesamples[ii][3] for ii in range(len(thesesamples))])
        ploty= numpy.array([thesesamples[ii][2] for ii in range(len(thesesamples))])
        xrange= [-10.,10.]
        yrange= [-20.,20.]
        xlabel=r'$\frac{\mathrm{d}^2 \sigma_z(z_{1/2})}{\mathrm{d} z^2}\ [\mathrm{km}\ \mathrm{s}^{-1}\ \mathrm{kpc}^{-2}]$'
        ylabel=r'$\frac{\mathrm{d} \sigma_z(z_{1/2})}{\mathrm{d} z}\ [\mathrm{km\ s}^{-1}\ \mathrm{kpc}^{-1}]$'
    elif options.type.lower() == 'slopehsm':
        ploty= numpy.array([thesesamples[ii][2] for ii in range(len(thesesamples))])
        plotx= numpy.exp(-numpy.array([thesesamples[ii][4] for ii in range(len(thesesamples))]))
        xrange= [0.,0.3]
        yrange= [-20.,20.]
        xlabel=r'$h^{-1}_\sigma\ [\mathrm{kpc}^{-1}]$'
        ylabel=r'$\frac{\mathrm{d} \sigma_z(z_{1/2})}{\mathrm{d} z}\ [\mathrm{km\ s}^{-1}\ \mathrm{kpc}^{-1}]$'
    elif options.type.lower() == 'slopesz':
        ploty= numpy.array([thesesamples[ii][2] for ii in range(len(thesesamples))])
        plotx= numpy.exp(numpy.array([thesesamples[ii][1] for ii in range(len(thesesamples))]))
        xrange= [0.,60.]
        yrange= [-20.,20.]
        xlabel=r'$\sigma_z(z_{1/2}) [\mathrm{km\ s}^{-1}$'
        ylabel=r'$\frac{\mathrm{d} \sigma_z(z_{1/2})}{\mathrm{d} z}\ [\mathrm{km\ s}^{-1}\ \mathrm{kpc}^{-1}]$'
    elif options.type.lower() == 'szhsm':
        plotx= numpy.exp(-numpy.array([thesesamples[ii][4] for ii in range(len(thesesamples))]))
        ploty= numpy.exp(numpy.array([thesesamples[ii][1] for ii in range(len(thesesamples))]))
        yrange= [0.,60.]
        xrange= [0.,0.3]
        xlabel=r'$h^{-1}_\sigma\ [\mathrm{kpc}^{-1}]$'
        ylabel=r'$\sigma_z(z_{1/2}) [\mathrm{km\ s}^{-1}$'
    bovy_plot.bovy_print()
    bovy_plot.scatterplot(plotx,ploty,'k,',
                          onedhists=True,
                          bins=31,
                          xlabel=xlabel,ylabel=ylabel,
                          xrange=xrange,
                          yrange=yrange)
    #Label
    bovy_plot.bovy_text(r'$[\mathrm{Fe/H}]\ =\ %.2f$' % options.feh
                        +'\n'
                        +r'$[\alpha/\mathrm{Fe}]\ =\ %.3f$' % options.afe,
                        top_right=True,
                        size=18)                     
    bovy_plot.bovy_end_print(options.plotfile)
            
def get_options():
    usage = "usage: %prog [options] READ THE CODE"
    parser = OptionParser(usage=usage)
    parser.add_option("--sample",dest='sample',default='g',
                      help="Use 'G' or 'K' dwarf sample")
    parser.add_option("--select",dest='select',default='all',
                      help="Select 'all' or 'program' stars")
    parser.add_option("--dfeh",dest='dfeh',default=0.05,type='float',
                      help="FeH bin size")   
    parser.add_option("--dafe",dest='dafe',default=0.05,type='float',
                      help="[a/Fe] bin size")   
    parser.add_option("--feh",dest='feh',default=-0.15,type='float',
                      help="FeH bin")   
    parser.add_option("--afe",dest='afe',default=0.075,type='float',
                      help="[a/Fe] bin")   
    parser.add_option("--minndata",dest='minndata',default=100,type='int',
                      help="Minimum number of objects in a bin to perform a fit")   
    parser.add_option("-o","--plotfile",dest='plotfile',default=None,
                      help="Name of the file for plot")
    parser.add_option("-t","--type",dest='type',default='sz',
                      help="Quantity to plot ('sz2hz', 'afe', 'feh'")
    parser.add_option("--subtype",dest='subtype',default='mz',
                      help="Sub-type of plot: when plotting afe, feh, afefeh, or fehafe, plot this")
    parser.add_option("--tighten",action="store_true", dest="tighten",
                      default=False,
                      help="If set, tighten axes")
    parser.add_option("--velmodel",dest='velmodel',default='hwr',
                      help="Velocity model used")
    parser.add_option("--densmodel",dest='densmodel',default='hwr',
                      help="Density model used")
    parser.add_option("--ploterrors",action="store_true", dest="ploterrors",
                      default=False,
                      help="If set, plot the errorbars")
    parser.add_option("--vr",action="store_true", dest="vr",
                      default=False,
                      help="If set, fit vR instead of vz")
    parser.add_option("--kde",action="store_true", dest="kde",
                      default=False,
                      help="Perform KDE estimate of hs")
    parser.add_option("--envelope",action="store_true", dest="envelope",
                      default=False,
                      help="Plot the error sausage envolope")
    return parser

if __name__ == '__main__':
    numpy.random.seed(1)
    parser= get_options()
    options,args= parser.parse_args()
    plotVelPDFs(options,args)

