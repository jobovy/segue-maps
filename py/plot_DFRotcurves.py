import os, os.path
import sys
import copy
import tempfile
import math
import numpy
from scipy import optimize, interpolate, linalg
from scipy.maxentropy import logsumexp
import cPickle as pickle
from optparse import OptionParser
import multi
import multiprocessing
import monoAbundanceMW
from galpy.util import bovy_coords, bovy_plot, save_pickles
from galpy import potential
from galpy.util import bovy_plot
import monoAbundanceMW
from segueSelect import read_gdwarfs, read_kdwarfs, _GDWARFFILE, _KDWARFFILE, \
    segueSelect, _mr_gi, _gi_gr, _ERASESTR, _append_field_recarray, \
    ivezic_dist_gr
from fitDensz import cb, _ZSUN, DistSpline, _ivezic_dist, _NDS
from pixelFitDens import pixelAfeFeh
from pixelFitDF import _REFV0, get_options, read_rawdata, get_potparams, \
    get_dfparams, _REFR0, get_vo, get_ro, setup_potential
from selectFigs import _squeeze
from matplotlib import pyplot, cm
_legendsize= 16
def plot_DFRotcurves(options,args):
    raw= read_rawdata(options)
    #Bin the data
    binned= pixelAfeFeh(raw,dfeh=options.dfeh,dafe=options.dafe)
    if options.tighten:
        tightbinned= pixelAfeFeh(raw,dfeh=options.dfeh,dafe=options.dafe,
                                 fehmin=-1.6,fehmax=0.5,afemin=-0.05,
                                 afemax=0.55)
    else:
        tightbinned= binned
    #Map the bins with ndata > minndata in 1D
    fehs, afes= [], []
    counter= 0
    abindx= numpy.zeros((len(binned.fehedges)-1,len(binned.afeedges)-1),
                        dtype='int')
    for ii in range(len(binned.fehedges)-1):
        for jj in range(len(binned.afeedges)-1):
            data= binned(binned.feh(ii),binned.afe(jj))
            if len(data) < options.minndata:
                continue
            #print binned.feh(ii), binned.afe(jj), len(data)
            fehs.append(binned.feh(ii))
            afes.append(binned.afe(jj))
            abindx[ii,jj]= counter
            counter+= 1
    nabundancebins= len(fehs)
    fehs= numpy.array(fehs)
    afes= numpy.array(afes)
    #Load each solutions
    sols= []
    savename= args[0]
    initname= options.init
    for ii in range(nabundancebins):
        spl= savename.split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        savefilename= newname
        #Read savefile
        try:
            savefile= open(savefilename,'rb')
        except IOError:
            print "WARNING: MISSING ABUNDANCE BIN"
            sols.append(None)
        else:
            sols.append(pickle.load(savefile))
            savefile.close()
        #Load samples as well
        if options.mcsample:
            #Do the same for init
            spl= initname.split('.')
            newname= ''
            for jj in range(len(spl)-1):
                newname+= spl[jj]
                if not jj == len(spl)-2: newname+= '.'
            newname+= '_%i.' % ii
            newname+= spl[-1]
            options.init= newname
    mapfehs= monoAbundanceMW.fehs()
    mapafes= monoAbundanceMW.afes()
    #Now plot
    #Run through the pixels and plot rotation curves
    if options.type == 'afe':
        vmin, vmax= 0.0,.5
        zlabel=r'$[\alpha/\mathrm{Fe}]$'
    elif options.type == 'feh':
        vmin, vmax= -1.6,0.4
        zlabel=r'$[\mathrm{Fe/H}]$'
    overplot= False
    if options.subtype is None or options.subtype.lower() != 'median':
        bovy_plot.bovy_print(fig_height=3.87,fig_width=5.)
    medianvc= []
    medianvc_disk= []
    medianvc_halo= []
    medianvc_bulge= []
    medianrs= numpy.linspace(0.001,2.,1001)
    for ii in range(tightbinned.npixfeh()):
        for jj in range(tightbinned.npixafe()):
            data= binned(tightbinned.feh(ii),tightbinned.afe(jj))
            if len(data) < options.minndata:
                if options.type.lower() == 'afe' or options.type.lower() == 'feh' or options.type.lower() == 'fehafe' \
                        or options.type.lower() == 'afefeh':
                    continue
                else:
                    plotthis[ii,jj]= numpy.nan
                    continue
            #Find abundance indx
            fehindx= binned.fehindx(tightbinned.feh(ii))#Map onto regular binning
            afeindx= binned.afeindx(tightbinned.afe(jj))
            solindx= abindx[fehindx,afeindx]
            monoabindx= numpy.argmin((tightbinned.feh(ii)-mapfehs)**2./0.01 \
                                         +(tightbinned.afe(jj)-mapafes)**2./0.0025)
            if sols[solindx] is None:
                if options.type.lower() == 'afe' or options.type.lower() == 'feh' or options.type.lower() == 'fehafe' \
                        or options.type.lower() == 'afefeh':
                    continue
                else:
                    plotthis[ii,jj]= numpy.nan
                    continue
            s= get_potparams(sols[solindx],options,1)
            #Setup potential
            pot= setup_potential(sols[solindx],options,1)
            vo= get_vo(sols[solindx],options,1)
            ro= get_ro(sols[solindx],options)
            if options.type.lower() == 'afe':
                plotc= tightbinned.afe(jj)
            elif options.type.lower() == 'feh':
                plotc= tightbinned.feh(jj)
            colormap = cm.jet
            if options.subtype is None or options.subtype.lower() == 'full':
                potential.plotRotcurve(pot,Rrange=[0.001,2.],
                                       overplot=overplot,ls='-',
                                       color=colormap(_squeeze(plotc,vmin,vmax)),
                                       yrange=[0.,1.29],
                                       ylabel= r"$V_c(R)/V_c(R_0)$",
                                       zorder=int(numpy.random.uniform()*100))
            elif options.subtype.lower() == 'disk':
                if 'mwpotential' in options.potential.lower():
                    diskpot= pot[0]
                potential.plotRotcurve(diskpot,Rrange=[0.001,2.],
                                       yrange=[0.,1.29],
                                       overplot=overplot,ls='-',
                                       color=colormap(_squeeze(plotc,vmin,vmax)),
                                       ylabel= r"$V_c(R)/V_c(R_0)$",
                                       zorder=int(numpy.random.uniform()*100))
            elif options.subtype.lower() == 'halo':
                if 'mwpotential' in options.potential.lower():
                    halopot= pot[1]
                potential.plotRotcurve(halopot,Rrange=[0.001,2.],
                                       overplot=overplot,ls='-',
                                       yrange=[0.,1.29],
                                       ylabel= r"$V_c(R)/V_c(R_0)$",
                                       color=colormap(_squeeze(plotc,vmin,vmax)),
                                       zorder=int(numpy.random.uniform()*100))
            elif options.subtype.lower() == 'bulge':
                if 'mwpotential' in options.potential.lower():
                    bulgepot= pot[2]
                potential.plotRotcurve(bulgepot,Rrange=[0.001,2.],
                                       overplot=overplot,ls='-',
                                       ylabel= r"$V_c(R)/V_c(R_0)$",
                                       yrange=[0.,1.29],
                                       color=colormap(_squeeze(plotc,vmin,vmax)),
                                       zorder=int(numpy.random.uniform()*100))
            elif options.subtype.lower() == 'median':
                if 'mwpotential' in options.potential.lower():
                    diskpot= pot[0]
                    halopot= pot[1]
                    bulgepot= pot[2]
                vo= get_vo(sols[solindx],options,1)
                medianvc.append(vo*potential.calcRotcurve(pot,medianrs))
                medianvc_disk.append(vo*potential.calcRotcurve(diskpot,medianrs))
                medianvc_halo.append(vo*potential.calcRotcurve(halopot,medianrs))
                medianvc_bulge.append(vo*potential.calcRotcurve(bulgepot,medianrs))
            overplot=True
    if options.subtype is None or options.subtype.lower() != 'median':
    #Add colorbar
        m = cm.ScalarMappable(cmap=cm.jet)
        m.set_array(plotc)
        m.set_clim(vmin=vmin,vmax=vmax)
        cbar= pyplot.colorbar(m,fraction=0.15)
        cbar.set_clim((vmin,vmax))
        cbar.set_label(zlabel)
        if options.subtype is None:
            pass
        elif options.subtype.lower() == 'disk':
            bovy_plot.bovy_text(r'$\mathrm{Disk}$',bottom_right=True,size=_legendsize)
        elif options.subtype.lower() == 'halo':
            bovy_plot.bovy_text(r'$\mathrm{Halo}$',bottom_right=True,size=_legendsize)
        elif options.subtype.lower() == 'bulge':
            bovy_plot.bovy_text(r'$\mathrm{Bulge}$',bottom_right=True,size=_legendsize)
    else:
        #Calc medians
        nbins= len(medianvc)
        vc= numpy.empty((len(medianrs),nbins))
        vc_disk= numpy.empty((len(medianrs),nbins))
        vc_bulge= numpy.empty((len(medianrs),nbins))
        vc_halo= numpy.empty((len(medianrs),nbins))
        for ii in range(nbins):
            vc[:,ii]= medianvc[ii]
            vc_disk[:,ii]= medianvc_disk[ii]
            vc_halo[:,ii]= medianvc_halo[ii]
            vc_bulge[:,ii]= medianvc_bulge[ii]
        vc= numpy.median(vc,axis=1)
        vcro= vc[numpy.argmin(numpy.fabs(medianrs-1.))]
        vc/= vcro
        vc_disk= numpy.median(vc_disk,axis=1)/vcro
        vc_halo= numpy.median(vc_halo,axis=1)/vcro
        vc_bulge= numpy.median(vc_bulge,axis=1)/vcro
        bovy_plot.bovy_print(fig_height=3.87,fig_width=5.)
        bovy_plot.bovy_plot(medianrs,vc,'k-',
                            xlabel=r"$R/R_0$",
                            ylabel= r"$V_c(R)/V_c(R_0)$",
                            yrange=[0.,1.29],
                            xrange=[0.,2.])
        linedisk= bovy_plot.bovy_plot(medianrs,vc_disk,'k--',
                                      overplot=True)
        linedisk[0].set_dashes([5,5])
        bovy_plot.bovy_plot(medianrs,vc_halo,'k:',
                            overplot=True)
        linebulge= bovy_plot.bovy_plot(medianrs,vc_bulge,'k--',
                                       overplot=True)
        linebulge[0].set_dashes([10,4])
        bovy_plot.bovy_text(1.95,0.5,r'$\mathrm{Disk}$',size=_legendsize,
                            horizontalalignment='right')
        bovy_plot.bovy_text(1.95,0.83,r'$\mathrm{Halo}$',size=_legendsize,
                            horizontalalignment='right')
        bovy_plot.bovy_text(1.95,0.1,r'$\mathrm{Bulge}$',size=_legendsize,
                            horizontalalignment='right')
    bovy_plot.bovy_end_print(options.outfilename)

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    numpy.random.seed(options.seed)
    plot_DFRotcurves(options,args)

