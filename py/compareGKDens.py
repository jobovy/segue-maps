import os, os.path
import sys
import math
import numpy
from scipy import optimize
import cPickle as pickle
from optparse import OptionParser
from galpy.util import bovy_coords, bovy_plot
from matplotlib import pyplot, cm
import bovy_mcmc
from fitSigz import _FAKEBIMODALGDWARFFILE, _FAKETHINBIMODALGDWARFFILE, \
    _FAKETHICKBIMODALGDWARFFILE
from segueSelect import read_gdwarfs, read_kdwarfs, \
    _GDWARFFILE, _KDWARFFILE
from selectFigs import _squeeze
from pixelFitDens import pixelAfeFeh
def compareGKDens(parser):
    options,args= parser.parse_args()
    rawg= read_gdwarfs(_GDWARFFILE,logg=True,ebv=True,sn=True)
    rawk= read_kdwarfs(_KDWARFFILE,logg=True,ebv=True,sn=True)
    #Bin the data   
    binned= pixelAfeFeh(rawg,dfeh=options.dfeh,dafe=options.dafe)
    binnedk= pixelAfeFeh(rawk,dfeh=options.dfeh,dafe=options.dafe)
    tightbinned= binned
    #Savefile
    if os.path.exists(args[0]):#Load savefile
        savefile= open(args[0],'rb')
        fits= pickle.load(savefile)
        savefile.close()
    #Uncertainty in savefile2
    if len(args) > 1 and os.path.exists(args[1]):
        savefile= open(args[1],'rb')
        denssamples= pickle.load(savefile)
        savefile.close()
        denserrors= True
    else:
        denssamples= None
        denserrors= False
    #Savefile
    if os.path.exists(args[2]):#Load savefile
        savefile= open(args[2],'rb')
        fitsk= pickle.load(savefile)
        savefile.close()
    #Uncertainty in savefile2
    if len(args) > 3 and os.path.exists(args[3]):
        savefile= open(args[3],'rb')
        denssamplesk= pickle.load(savefile)
        savefile.close()
    else:
        denssamplesk= None
    #Run through the pixels and gather
    plotthis= []
    errors= []
    for ii in range(tightbinned.npixfeh()):
        for jj in range(tightbinned.npixafe()):
            data= binned(tightbinned.feh(ii),tightbinned.afe(jj))
            fehindx= binned.fehindx(tightbinned.feh(ii))#Map onto regular binning
            afeindx= binned.afeindx(tightbinned.afe(jj))
            if afeindx+fehindx*binned.npixafe() >= len(fits):
                continue
            thisfit= fits[afeindx+fehindx*binned.npixafe()]
            if thisfit is None:
                continue
            if len(data) < options.minndata:
                continue
            #K
            thisfitk= fitsk[afeindx+fehindx*binned.npixafe()]
            if thisfitk is None:
                continue          
            plotthis.append([tightbinned.feh(ii),
                             tightbinned.afe(jj),
                             numpy.exp(thisfit[0])*1000.,
                             numpy.exp(thisfit[1]),
                             numpy.exp(thisfitk[0])*1000.,
                             numpy.exp(thisfitk[1]),
                             len(data)])
            if denserrors:
                theseerrors= []
                thesesamples= denssamples[afeindx+fehindx*binned.npixafe()]
                for kk in [0,1]:
                    xs= numpy.array([s[kk] for s in thesesamples])
                    theseerrors.append(0.5*(-numpy.exp(numpy.mean(xs)-numpy.std(xs))+numpy.exp(numpy.mean(xs)+numpy.std(xs))))
                thesesamples= denssamplesk[afeindx+fehindx*binned.npixafe()]
                for kk in [0,1]:
                    xs= numpy.array([s[kk] for s in thesesamples])
                    theseerrors.append(0.5*(-numpy.exp(numpy.mean(xs)-numpy.std(xs))+numpy.exp(numpy.mean(xs)+numpy.std(xs))))
                errors.append(theseerrors)
    #Set up plot
    vmin, vmax= 0.0,.5
    zlabel=r'$[\alpha/\mathrm{Fe}]$'
    bovy_plot.bovy_print(fig_height=3.87,fig_width=5.)
    #Gather hR and hz
    hz_err, hr_err, hz, hr,afe, feh, ndata= [], [], [], [], [], [], []
    hzk_err, hrk_err, hzk, hrk= [], [], [], []
    for ii in range(len(plotthis)):
        if denserrors:
            hz_err.append(errors[ii][0]*1000.)
            hr_err.append(errors[ii][1])
            hzk_err.append(errors[ii][2]*1000.)
            hrk_err.append(errors[ii][3])
        hz.append(plotthis[ii][2])
        hr.append(plotthis[ii][3])
        hzk.append(plotthis[ii][4])
        hrk.append(plotthis[ii][5])
        afe.append(plotthis[ii][1])
        feh.append(plotthis[ii][0])
        ndata.append(plotthis[ii][6])
    if denserrors:
        hz_err= numpy.array(hz_err)
        hr_err= numpy.array(hr_err)
        hzk_err= numpy.array(hzk_err)
        hrk_err= numpy.array(hrk_err)
    hz= numpy.array(hz)
    hr= numpy.array(hr)
    hzk= numpy.array(hzk)
    hrk= numpy.array(hrk)
    afe= numpy.array(afe)
    ndata= numpy.array(ndata)
    ndata= ndata**.5
    ndata= ndata/numpy.median(ndata)*35.
    #Now plot
    plotc= afe
    if options.type.lower() == 'hz':
        yrange= [150,1050]
        xrange= [150,1050]
        bovy_plot.bovy_plot(hz,hzk,s=ndata,c=plotc,
                            cmap='jet',
                            xlabel=r'$\mathrm{G-dwarf\ vertical\ scale\ height\ [pc]}$',
                            ylabel=r'$\mathrm{K-dwarf\ vertical\ scale\ height\ [pc]}$',
                            clabel=zlabel,
                            xrange=xrange,yrange=yrange,
                            vmin=vmin,vmax=vmax,
                            scatter=True,edgecolors='none',
                            colorbar=True,zorder=2)
        bovy_plot.bovy_plot(xrange,yrange,'-',color='0.5',overplot=True)
        #Overplot errors
        colormap = cm.jet
        for ii in range(len(hz)):
            if hr[ii] < 5.:
                pyplot.errorbar(hz[ii],hzk[ii],
                                xerr=hz_err[ii],yerr=hzk_err[ii],
                                color=colormap(_squeeze(plotc[ii],
                                                        numpy.amax([numpy.amin(plotc)]),
                                                        numpy.amin([numpy.amax(plotc)]))),
                                elinewidth=1.,capsize=3,zorder=0)
    elif options.type.lower() == 'hr':
        xrange= [.5,5.]
        yrange= [.5,5.]
        bovy_plot.bovy_plot(hr,hrk,s=ndata,c=plotc,
                            cmap='jet',
                            xlabel=r'$\mathrm{G-dwarf\ radial\ scale\ length\ [kpc]}$',
                            ylabel=r'$\mathrm{K-dwarf\ radial\ scale\ length\ [kpc]}$',
                            clabel=zlabel,
                            xrange=xrange,yrange=yrange,
                            vmin=vmin,vmax=vmax,
                            scatter=True,edgecolors='none',
                            colorbar=True,zorder=2)
        bovy_plot.bovy_plot(xrange,yrange,'-',color='0.5',overplot=True)
        #Overplot errors
        colormap = cm.jet
        for ii in range(len(hz)):
            if hr[ii] < 5.:
                pyplot.errorbar(hr[ii],hrk[ii],
                                xerr=hr_err[ii],yerr=hrk_err[ii],
                                color=colormap(_squeeze(plotc[ii],
                                                        numpy.amax([numpy.amin(plotc)]),
                                                        numpy.amin([numpy.amax(plotc)]))),
                                elinewidth=1.,capsize=3,zorder=0)
    bovy_plot.bovy_end_print(options.plotfile) 

def get_options():
    usage = "usage: %prog [options] <savefile> <savefile>\n\nsavefile= name of the file that the fits will be saved to\nsavefile = name of the file that the samples will be saved to (optional)"
    parser = OptionParser(usage=usage)
    parser.add_option("--dfeh",dest='dfeh',default=0.05,type='float',
                      help="FeH bin size")   
    parser.add_option("--dafe",dest='dafe',default=0.05,type='float',
                      help="[a/Fe] bin size")   
    parser.add_option("--minndata",dest='minndata',default=100,type='int',
                      help="Minimum number of objects in a bin to perform a fit for G")   
    parser.add_option("-o","--plotfile",dest='plotfile',default=None,
                      help="Name of the file for plot")
    parser.add_option("-t","--type",dest='type',default='hr',
                      help="Quantity to plot ('hz', 'hr', 'hzhr'")
    return parser
  
if __name__ == '__main__':
    parser= get_options()
    compareGKDens(parser)
