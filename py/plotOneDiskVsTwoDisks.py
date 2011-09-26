import os, os.path
import sys
import math
import numpy
import cPickle as pickle
from optparse import OptionParser
from galpy.util import bovy_coords, bovy_plot, save_pickles
from matplotlib import pyplot, cm
from segueSelect import read_gdwarfs, read_kdwarfs, _GDWARFFILE, _KDWARFFILE
from pixelFitDens import pixelAfeFeh
from fitSigz import _ZSUN
def plotOneDiskVsTwoDisks(options,args):
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
                                 fehmin=-2.,fehmax=0.3,afemin=0.,afemax=0.45)
    else:
        tightbinned= binned
    #Savefile1
    if os.path.exists(args[0]):#Load savefile
        savefile= open(args[0],'rb')
        onefits= pickle.load(savefile)
        savefile.close()
    if os.path.exists(args[1]):#Load savefile
        savefile= open(args[1],'rb')
        twofits= pickle.load(savefile)
        savefile.close()
    #Uncertainties are in savefile3 and 4
    if len(args) > 3 and os.path.exists(args[3]):
        savefile= open(args[3],'rb')
        twosamples= pickle.load(savefile)
        savefile.close()
        twoerrors= True
    else:
        twosamples= None
        twoerrors= False
    if len(args) > 2 and os.path.exists(args[2]):
        savefile= open(args[2],'rb')
        onesamples= pickle.load(savefile)
        savefile.close()
        oneerrors= True
    else:
        onesamples= None            
        oneerrors= False
    #Run through the pixels and gather
    plotthis= []
    errors= []
    for ii in range(tightbinned.npixfeh()):
        for jj in range(tightbinned.npixafe()):
            data= binned(tightbinned.feh(ii),tightbinned.afe(jj))
            fehindx= binned.fehindx(tightbinned.feh(ii))#Map onto regular binning
            afeindx= binned.afeindx(tightbinned.afe(jj))
            if afeindx+fehindx*binned.npixafe() >= len(onefits):
                continue
            thisonefit= onefits[afeindx+fehindx*binned.npixafe()]
            thistwofit= twofits[afeindx+fehindx*binned.npixafe()]
            if thisonefit is None:
                    continue
            if len(data) < options.minndata:
                    continue
            #print tightbinned.feh(ii), tightbinned.afe(jj), numpy.exp(thisonefit), numpy.exp(thistwofit)
            #Which is the dominant two-exp component?
            if thistwofit[4] > 0.5: twoIndx= 1
            else: twoIndx= 0
            if options.type == 'hz':
                plotthis.append([numpy.exp(thisonefit[0])*1000.,numpy.exp(thistwofit[twoIndx])*1000.,len(data)])
            elif options.type == 'hr':
                plotthis.append([numpy.exp(thisonefit[1]),numpy.exp(thistwofit[twoIndx+2]),len(data)])
            theseerrors= []
            if oneerrors:
                theseonesamples= onesamples[afeindx+fehindx*binned.npixafe()]
                if options.type == 'hz':
                    xs= numpy.array([s[0] for s in theseonesamples])
                    theseerrors.append(500.*(-numpy.exp(numpy.mean(xs)-numpy.std(xs))+numpy.exp(numpy.mean(xs)+numpy.std(xs))))
                elif options.type == 'hr':
                    xs= numpy.array([s[1] for s in theseonesamples])
                    theseerrors.append(0.5*(-numpy.exp(numpy.mean(xs)-numpy.std(xs))+numpy.exp(numpy.mean(xs)+numpy.std(xs))))
            if twoerrors:
                thesetwosamples= twosamples[afeindx+fehindx*binned.npixafe()]
                if options.type == 'hz':
                    xs= numpy.array([s[twoIndx] for s in thesetwosamples])
                    theseerrors.append(500.*(-numpy.exp(numpy.mean(xs)-numpy.std(xs))+numpy.exp(numpy.mean(xs)+numpy.std(xs))))
                elif options.type == 'hr':
                    xs= numpy.array([s[2+twoIndx] for s in thesetwosamples])
                    theseerrors.append(0.5*(-numpy.exp(numpy.mean(xs)-numpy.std(xs))+numpy.exp(numpy.mean(xs)+numpy.std(xs))))
            errors.append(theseerrors)
    x, y, ndata= [], [], []
    if oneerrors: x_err= []
    if twoerrors: y_err= []
    for ii in range(len(plotthis)):
        x.append(plotthis[ii][0])
        y.append(plotthis[ii][1])
        ndata.append(plotthis[ii][2])
        if oneerrors: x_err.append(errors[ii][0])
        if twoerrors: y_err.append(errors[ii][1])
    x= numpy.array(x)
    y= numpy.array(y)
    if oneerrors: x_err= numpy.array(x_err)
    if twoerrors: y_err= numpy.array(y_err)
    ndata= numpy.array(ndata)
    #Process ndata
    ndata= ndata**.5
    ndata= ndata/numpy.median(ndata)*35.
    #Now plot
    if options.type == 'hz':
        xrange= [150,1200]
        xlabel=r'$\mathrm{single-exponential\ scale\ height}$'
        ylabel=r'$\mathrm{two-exponentials\ scale\ height}$'
    elif options.type == 'hr':
        xrange= [1.2,5.]
        xlabel=r'$\mathrm{single-exponential\ scale\ length}$'
        ylabel=r'$\mathrm{two-exponentials\ scale\ length}$'
    yrange=xrange
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(x,y,color='k',
                        ylabel=ylabel,
                        xlabel=xlabel,
                        xrange=xrange,yrange=yrange,
                        scatter=True,edgecolors='none',
                        colorbar=False,zorder=2)
    bovy_plot.bovy_plot([xrange[0],xrange[1]],[xrange[0],xrange[1]],color='0.5',ls='--',overplot=True)
    if oneerrors:
        #Overplot errors
        for ii in range(len(x)):
            #                if (options.type == 'hr' and x[ii] < 5.) or options.type == 'hz':
            pyplot.errorbar(x[ii],y[ii],xerr=x_err[ii],color='k',
                            elinewidth=1.,capsize=3,zorder=0)
    if twoerrors:
        #Overplot errors
        for ii in range(len(x)):
            #                if (options.type == 'hr' and x[ii] < 5.) or options.type == 'hz':
            pyplot.errorbar(x[ii],y[ii],yerr=y_err[ii],color='k',
                            elinewidth=1.,capsize=3,zorder=0)
    bovy_plot.bovy_end_print(options.plotfile)
    return None   

def get_options():
    usage = "usage: %prog [options] <savefile1> <savefile2>\n\nsavefile1= name of the file that holds the single disk fits\nsavefile2 = name of the file that holds the double disk fits\nsavefile3= name of the file that has the samplings for the single disk\nsavefile4= name of the file that has the samplings for the double disk fits"
    parser = OptionParser(usage=usage)
    parser.add_option("--sample",dest='sample',default='g',
                      help="Use 'G' or 'K' dwarf sample")
    parser.add_option("--select",dest='select',default='all',
                      help="Select 'all' or 'program' stars")
    parser.add_option("--dfeh",dest='dfeh',default=0.05,type='float',
                      help="FeH bin size")   
    parser.add_option("--dafe",dest='dafe',default=0.05,type='float',
                      help="[a/Fe] bin size")   
    parser.add_option("--minndata",dest='minndata',default=100,type='int',
                      help="Minimum number of objects in a bin to perform a fit")   
    parser.add_option("-o","--plotfile",dest='plotfile',default=None,
                      help="Name of the file for plot")
    parser.add_option("-t","--type",dest='type',default='sz',
                      help="Quantity to plot ('hz' or 'hr')")
    parser.add_option("--subtype",dest='subtype',default='mz',
                      help="Sub-type of plot: when plotting afe, feh, afefeh, or fehafe, plot this")
    parser.add_option("--tighten",action="store_true", dest="tighten",
                      default=False,
                      help="If set, tighten axes")
    return parser

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    plotOneDiskVsTwoDisks(options,args)

