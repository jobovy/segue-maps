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
_DEGTORAD= numpy.pi/180.
def plotTilt(options,args):
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
    #Uncertainties are in savefile2
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
    for ii in range(tightbinned.npixfeh()):
        for jj in range(tightbinned.npixafe()):
            data= binned(tightbinned.feh(ii),tightbinned.afe(jj))
            fehindx= binned.fehindx(tightbinned.feh(ii))#Map onto regular binning
            afeindx= binned.afeindx(tightbinned.afe(jj))
            if afeindx+fehindx*binned.npixafe() >= len(velfits) \
                    or afeindx+fehindx*binned.npixafe() >= len(velfits):
                if options.type.lower() == 'afe' or options.type.lower() == 'feh' or options.type.lower() == 'fehafe' \
                        or options.type.lower() == 'zfunc' \
                        or options.type.lower() == 'afefeh':
                    continue
                else:
                    plotthis[ii,jj]= numpy.nan
                    continue
            thisvelfit= velfits[afeindx+fehindx*binned.npixafe()]
            if thisvelfit is None:
                if options.type.lower() == 'afe' or options.type.lower() == 'feh' or options.type.lower() == 'fehafe' \
                        or options.type.lower() == 'zfunc' \
                        or options.type.lower() == 'afefeh':
                    continue
                else:
                    plotthis[ii,jj]= numpy.nan
                    continue
            if len(data) < options.minndata:
                if options.type.lower() == 'afe' or options.type.lower() == 'feh' or options.type.lower() == 'fehafe' \
                        or options.type.lower() == 'zfunc' \
                        or options.type.lower() == 'afefeh':
                    continue
                else:
                    plotthis[ii,jj]= numpy.nan
                    continue
            if options.type == 'tilt':
                plotthis[ii,jj]= numpy.arctan(thisvelfit[9])/_DEGTORAD
            elif options.type == 'tiltslope':
                plotthis[ii,jj]= thisvelfit[10]
            elif options.type == 'srz':
                plotthis[ii,jj]= (numpy.exp(2.*thisvelfit[5])-numpy.exp(2.*thisvelfit[1]))*thisvelfit[9]/(1.-thisvelfit[9]**2.)
            elif options.type.lower() == 'afe' \
                    or options.type.lower() == 'feh' \
                    or options.type.lower() == 'fehafe' \
                    or options.type.lower() == 'zfunc' \
                    or options.type.lower() == 'afefeh':
                thisplot=[tightbinned.feh(ii),
                          tightbinned.afe(jj),
                          len(data)]
                thisplot.extend(thisvelfit)
                #Als find min and max z for this data bin, and median
                if options.subtype.lower() == 'rfunc':
                    zsorted= sorted(numpy.sqrt((8.-data.xc)**2.+data.yc**2.))
                else:
                    zsorted= sorted(numpy.fabs(data.zc+_ZSUN))
                zmin= zsorted[int(numpy.ceil(0.16*len(zsorted)))]
                zmax= zsorted[int(numpy.floor(0.84*len(zsorted)))]
                zmin= zsorted[int(numpy.ceil(0.025*len(zsorted)))]
                zmax= zsorted[int(numpy.floor(0.975*len(zsorted)))]
                if options.pivotmean:
                    thisplot.extend([zmin,zmax,numpy.mean(numpy.fabs(data.zc+_ZSUN))])
                else:
                    thisplot.extend([zmin,zmax,numpy.median(numpy.fabs(data.zc+_ZSUN))])
                plotthis.append(thisplot)
                #Errors
                if velerrors:
                    theseerrors= []
                    thesesamples= velsamples[afeindx+fehindx*binned.npixafe()]
                    for kk in [1,4,5,8]:
                        xs= numpy.array([s[kk] for s in thesesamples])
                        theseerrors.append(0.5*(numpy.exp(numpy.mean(xs))-numpy.exp(numpy.mean(xs)-numpy.std(xs))-numpy.exp(numpy.mean(xs))+numpy.exp(numpy.mean(xs)+numpy.std(xs))))
                    for kk in [0,2,3,6,7,9,10,11]:
                        xs= numpy.array([s[kk] for s in thesesamples])
                        theseerrors.append(numpy.std(xs))
                        errors.append(theseerrors)
    #Set up plot
    #print numpy.nanmin(plotthis), numpy.nanmax(plotthis)
    if options.type == 'tilt':
        print numpy.nanmin(plotthis), numpy.nanmax(plotthis)
        vmin, vmax= -20.,20.
        zlabel= r'$\mathrm{tilt\ at}\ Z = 0\ [\mathrm{degree}]$'
    elif options.type == 'tiltslope':
        print numpy.nanmin(plotthis), numpy.nanmax(plotthis)
        vmin, vmax= -2.,2.
        zlabel= r'$\frac{\mathrm{d}\tan \mathrm{tilt}}{\mathrm{d} (Z/R)}$'
    elif options.type == 'srz':
        vmin, vmax= -100.,100.
        zlabel= r'$\sigma^2_{RZ}\ [\mathrm{km}^2\, \mathrm{s}^{-2}]$'
    elif options.type == 'afe':
        vmin, vmax= 0.0,.5
        zlabel=r'$[\alpha/\mathrm{Fe}]$'
    elif options.type == 'feh':
        vmin, vmax= -1.6,0.4
        zlabel=r'$[\mathrm{Fe/H}]$'
    if options.tighten:
        xrange=[-1.6,0.5]
        yrange=[-0.05,0.55]
    else:
        xrange=[-2.,0.6]
        yrange=[-0.1,0.6]
    if options.type.lower() == 'afe' or options.type.lower() == 'feh' \
            or options.type.lower() == 'fehafe' \
            or options.type.lower() == 'afefeh':
        bovy_plot.bovy_print(fig_height=3.87,fig_width=5.)
        #Gather everything
        zmin, zmax, pivot, tilt, tiltp1, tiltp2, afe, feh, ndata= [], [], [], [], [], [], [], [], []
        tilt_err, tiltp1_err, tiltp2_err= [], [], []
        for ii in range(len(plotthis)):
            if velerrors:
                tilt_err.append(errors[ii][9])
                tiltp1_err.append(errors[ii][10])
                tiltp2_err.append(errors[ii][11])
            tilt.append(plotthis[ii][12])
            tiltp1.append(plotthis[ii][13])
            tiltp2.append(plotthis[ii][14])
            afe.append(plotthis[ii][1])
            feh.append(plotthis[ii][0])
            ndata.append(plotthis[ii][4])
            zmin.append(plotthis[ii][15])
            zmax.append(plotthis[ii][16])
            pivot.append(plotthis[ii][17])
        indxarray= numpy.array([True for ii in range(len(plotthis))],dtype='bool')
        tilt= numpy.array(tilt)[indxarray]
        tiltp1= numpy.array(tiltp1)[indxarray]
        tiltp2= numpy.array(tiltp2)[indxarray]
        pivot= numpy.array(pivot)[indxarray]
        zmin= numpy.array(zmin)[indxarray]
        zmax= numpy.array(zmax)[indxarray]
        if velerrors:
            tilt_err= numpy.array(tilt_err)[indxarray]
            tiltp1_err= numpy.array(tiltp1_err)[indxarray]
            tiltp2_err= numpy.array(tiltp2_err)[indxarray]
        afe= numpy.array(afe)[indxarray]
        feh= numpy.array(feh)[indxarray]
        ndata= numpy.array(ndata)[indxarray]
        #Process ndata
        ndata= ndata**.5
        ndata= ndata/numpy.median(ndata)*35.
        #ndata= numpy.log(ndata)/numpy.log(numpy.median(ndata))
        #ndata= (ndata-numpy.amin(ndata))/(numpy.amax(ndata)-numpy.amin(ndata))*25+12.
        if options.type.lower() == 'afe':
            plotc= afe
        elif options.type.lower() == 'feh':
            plotc= feh
        if options.subtype.lower() == 'zfunc':
            from selectFigs import _squeeze
            colormap = cm.jet
            #Set up plot
            yrange= [-30.,30.]
            ylabel=r'$\mathrm{tilt}(Z|R=8\,\mathrm{kpc})\ [\mathrm{degree}]$'
            bovy_plot.bovy_plot([-100.,-100.],[100.,100.],'k,',
                                xrange=[0,2700],yrange=yrange,
                                xlabel=r'$|z|\ [\mathrm{pc}]$',
                                ylabel=ylabel)
            #Calculate and plot all zfuncs
            for ii in numpy.random.permutation(len(afe)):
                if velerrors: #Don't plot if errors > 30%
                    if tilt_err[ii]/tilt[ii] > .2: continue
                ds= numpy.linspace(zmin[ii]*1000.,zmax[ii]*1000.,1001)/8000.
                thiszfunc= numpy.arctan(tilt[ii]+tiltp1[ii]*ds+tiltp2[ii]*ds**2.)/_DEGTORAD
                pyplot.plot(numpy.linspace(zmin[ii]*1000.,1000*zmax[ii],1001),
                            thiszfunc,'-',
                            color=colormap(_squeeze(plotc[ii],vmin,vmax)),
                            lw=ndata[ii]/15.)
                if not options.nofatdots:
                    #Also plot pivot
                    pyplot.plot(1000.*pivot[ii],
                                numpy.arctan(tilt[ii]+tiltp1[ii]*pivot[ii]/8.\
                                    +tiltp2[ii]*(pivot[ii]/8.)**2.)/_DEGTORAD,
                                'o',ms=8.,mec='none',
                                color=colormap(_squeeze(plotc[ii],vmin,vmax)))
            #Add colorbar
            m = cm.ScalarMappable(cmap=cm.jet)
            m.set_array(plotc)
            m.set_clim(vmin=vmin,vmax=vmax)
            cbar= pyplot.colorbar(m,fraction=0.15)
            cbar.set_clim((vmin,vmax))
            cbar.set_label(zlabel)
    else:
        bovy_plot.bovy_print()
        bovy_plot.bovy_dens2d(plotthis.T,origin='lower',cmap='jet',
                              interpolation='nearest',
                              xlabel=r'$[\mathrm{Fe/H}]$',
                              ylabel=r'$[\alpha/\mathrm{Fe}]$',
                              zlabel=zlabel,
                              xrange=xrange,yrange=yrange,
                              vmin=vmin,vmax=vmax,
                              contours=False,
                              colorbar=True,shrink=0.78)
    bovy_plot.bovy_end_print(options.plotfile)
    return None



def get_options():
    usage = "usage: %prog [options] <savefile1> <savefile2>\n\nsavefile1= name of the file that the velocity fits are saved to\nsavefile2 = name of the file that the density fits are saved to"
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
    parser.add_option("--nofatdots",action="store_true", dest="nofatdots",
                      default=False,
                      help="Don't plot the fat dots")
    parser.add_option("--pivotmean",action="store_true", dest="pivotmean",
                      default=False,
                      help="If set, pivot on the mean z rather than on the median")
    return parser

if __name__ == '__main__':
    numpy.random.seed(19)
    parser= get_options()
    options,args= parser.parse_args()
    plotTilt(options,args)

