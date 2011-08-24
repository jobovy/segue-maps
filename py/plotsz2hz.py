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
def plotsz2hz(options,args):
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
        velfits= pickle.load(savefile)
        savefile.close()
    if os.path.exists(args[1]):#Load savefile
        savefile= open(args[1],'rb')
        densfits= pickle.load(savefile)
        savefile.close()
    #Now plot
    #Run through the pixels and gather
    if options.type.lower() == 'afe' or options.type.lower() == 'feh' \
            or options.type.lower() == 'fehafe' \
            or options.type.lower() == 'zfunc' \
            or options.type.lower() == 'afefeh':
        plotthis= []
    else:
        plotthis= numpy.zeros((tightbinned.npixfeh(),tightbinned.npixafe()))
    for ii in range(tightbinned.npixfeh()):
        for jj in range(tightbinned.npixafe()):
            data= binned(tightbinned.feh(ii),tightbinned.afe(jj))
            fehindx= binned.fehindx(tightbinned.feh(ii))#Map onto regular binning
            afeindx= binned.afeindx(tightbinned.afe(jj))
            if afeindx+fehindx*binned.npixafe() >= len(densfits) \
                    or afeindx+fehindx*binned.npixafe() >= len(velfits):
                if options.type.lower() == 'afe' or options.type.lower() == 'feh' or options.type.lower() == 'fehafe' \
                        or options.type.lower() == 'zfunc' \
                        or options.type.lower() == 'afefeh':
                    continue
                else:
                    plotthis[ii,jj]= numpy.nan
                    continue
            thisdensfit= densfits[afeindx+fehindx*binned.npixafe()]
            thisvelfit= velfits[afeindx+fehindx*binned.npixafe()]
            if thisdensfit is None or thisvelfit is None:
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
            if options.velmodel.lower() == 'hwr':
                if options.type == 'sz2hz':
                    numerator= numpy.exp(2.*thisvelfit[1])
                elif options.type.lower() == 'afe' \
                        or options.type.lower() == 'feh' \
                        or options.type.lower() == 'fehafe' \
                        or options.type.lower() == 'zfunc' \
                        or options.type.lower() == 'afefeh':
                    thisplot=[tightbinned.feh(ii),
                              tightbinned.afe(jj),
                              numpy.exp(thisvelfit[1]),
                              numpy.exp(thisvelfit[3]),
                              len(data),
                              thisvelfit[1],
                              thisvelfit[2]]
                    #Als find min and max z for this data bin, and median
                    zsorted= sorted(numpy.fabs(data.zc+_ZSUN))
                    zmin= zsorted[int(numpy.ceil(0.16*len(zsorted)))]
                    zmax= zsorted[int(numpy.floor(0.84*len(zsorted)))]
                    thisplot.extend([zmin,zmax,numpy.mean(numpy.fabs(data.zc+_ZSUN))])
            if options.densmodel.lower() == 'hwr':
                if options.type == 'sz2hz':
                    denominator= numpy.exp(thisdensfit[0])*1000.
                    plotthis[ii,jj]= numerator/denominator
                elif options.type.lower() == 'afe' \
                        or options.type.lower() == 'feh' \
                        or options.type.lower() == 'fehafe' \
                        or options.type.lower() == 'afefeh':
                    thisplot.extend([numpy.exp(thisdensfit[0])*1000.,
                                     numpy.exp(thisdensfit[1]),
                                     numpy.median(numpy.fabs(data.zc)-_ZSUN)])
                    plotthis.append(thisplot)
    #Set up plot
    #print numpy.nanmin(plotthis), numpy.nanmax(plotthis)
    if options.type == 'sz2hz':
        plotthis/= 2.*numpy.pi*4.302*10.**-3 #2piG
        print numpy.nanmin(plotthis), numpy.nanmax(plotthis)
        vmin, vmax= 15.,100.
        zlabel= r'$\sigma_z^2(z=1000\ \mathrm{pc}) / h_z\ [M_\odot\ \mathrm{pc}^{-2}]$'
    elif options.type == 'afe':
        vmin, vmax= 0.05,.4
        zlabel=r'$[\alpha/\mathrm{Fe}]$'
    elif options.type == 'feh':
        vmin, vmax= -1.5,0.
        zlabel=r'$[\mathrm{Fe/H}]$'
    elif options.type == 'fehafe':
        vmin, vmax= -.7,.7
        zlabel=r'$[\mathrm{Fe/H}]-[\mathrm{Fe/H}]_{1/2}([\alpha/\mathrm{Fe}])$'
    elif options.type == 'afefeh':
        vmin, vmax= -.15,.15
        zlabel=r'$[\alpha/\mathrm{Fe}]-[\alpha/\mathrm{Fe}]_{1/2}([\mathrm{Fe/H}])$'
    if options.tighten:
        xrange=[-2.,0.3]
        yrange=[0.,0.45]
    else:
        xrange=[-2.,0.6]
        yrange=[-0.1,0.6]
    if options.type.lower() == 'afe' or options.type.lower() == 'feh' \
            or options.type.lower() == 'fehafe' \
            or options.type.lower() == 'afefeh':
        bovy_plot.bovy_print(fig_height=3.87,fig_width=5.)
        #Gather everything
        pivot, zmin, zmax, mz, p1, p2, sz, hs, hz, hr,afe, feh, ndata= [], [], [], [], [], [], [], [], [], [], [], [], []
        for ii in range(len(plotthis)):
            sz.append(plotthis[ii][2])
            hs.append(plotthis[ii][3])
            hz.append(plotthis[ii][10])
            hr.append(plotthis[ii][11])
            afe.append(plotthis[ii][1])
            feh.append(plotthis[ii][0])
            ndata.append(plotthis[ii][4])
            p1.append(plotthis[ii][5])
            p2.append(plotthis[ii][6])         
            mz.append(plotthis[ii][11])
            zmin.append(plotthis[ii][7])
            zmax.append(plotthis[ii][8])
            pivot.append(plotthis[ii][9])
        pivot= numpy.array(pivot)
        zmin= numpy.array(zmin)
        zmax= numpy.array(zmax)
        sz= numpy.array(sz)
        mz= numpy.array(mz)*1000.
        hs= numpy.array(hs)
        hz= numpy.array(hz)
        hr= numpy.array(hr)
        afe= numpy.array(afe)
        feh= numpy.array(feh)
        ndata= numpy.array(ndata)
        #Process ndata
        ndata= ndata**.5
        ndata= ndata/numpy.median(ndata)*35.
        #ndata= numpy.log(ndata)/numpy.log(numpy.median(ndata))
        #ndata= (ndata-numpy.amin(ndata))/(numpy.amax(ndata)-numpy.amin(ndata))*25+12.
        if options.type.lower() == 'afe':
            plotc= afe
        elif options.type.lower() == 'feh':
            plotc= feh
        elif options.type.lower() == 'afefeh':
            #Go through the bins to determine whether feh is high or low for this alpha
            plotc= numpy.zeros(len(afe))
            for ii in range(tightbinned.npixfeh()):
                fehbin= ii
                data= tightbinned.data[(tightbinned.data.feh > tightbinned.fehedges[fehbin])\
                                           *(tightbinned.data.feh <= tightbinned.fehedges[fehbin+1])]
                medianafe= numpy.median(data.afe)
                for jj in range(len(afe)):
                    if feh[jj] == tightbinned.feh(ii):
                        plotc[jj]= afe[jj]-medianafe
        else:
            #Go through the bins to determine whether feh is high or low for this alpha
            plotc= numpy.zeros(len(feh))
            for ii in range(tightbinned.npixafe()):
                afebin= ii
                data= tightbinned.data[(tightbinned.data.afe > tightbinned.afeedges[afebin])\
                                           *(tightbinned.data.afe <= tightbinned.afeedges[afebin+1])]
                medianfeh= numpy.median(data.feh)
                for jj in range(len(feh)):
                    if afe[jj] == tightbinned.afe(ii):
                        plotc[jj]= feh[jj]-medianfeh
        if not options.subtype == 'zfunc':
            if options.subtype == 'mz':
                yrange= [0,130.]
                xrange= [0,1500]
                plotx= mz
                ploty= (sz+(mz/1000.-pivot)*p1+p2*(mz/1000.-pivot)**2.)**2./hz/2./numpy.pi/4.302/10**-3.
                xlabel=r'$\mathrm{median\ \ height}\ z_{1/2}\ \mathrm{[pc]}$'
                ylabel=r'$\sigma_z^2(z = z_{1/2}) / h_z\ [M_\odot\ \mathrm{pc}^{-2}]$'
            elif options.subtype == 'hz':
                yrange= [0,130.]
                xrange= [0,1200]
                plotx= hz
                ploty= (sz+(hz/1000.-pivot)*p1+p2*(hz/1000.-pivot)**2.)**2./hz/2./numpy.pi/4.302/10**-3.
                ylabel=r'$\sigma_z^2(z = h_z) / h_z\ [M_\odot\ \mathrm{pc}^{-2}]$'
                xlabel=r'$\mathrm{vertical\ scale\ height}\ h_z\ \mathrm{[pc]}$'
            bovy_plot.bovy_plot(plotx,ploty,
                                s=ndata,c=plotc,
                                cmap='jet',
                                xlabel=xlabel,ylabel=ylabel,
                                clabel=zlabel,
                                xrange=xrange,yrange=yrange,
                                vmin=vmin,vmax=vmax,
                                scatter=True,edgecolors='none',
                                colorbar=True)
            #Add local density
            bovy_plot.bovy_plot([0.,2000],[0.,200],'--',color='0.5',overplot=True)
            #add Siebert
            bovy_plot.bovy_plot(numpy.linspace(0.,2000.,1001),
                                90.*numpy.linspace(0.,2000.,1001)/numpy.sqrt(numpy.linspace(0.,2000.,1001)**2.+637**2.)+0.01*numpy.linspace(0.,2000.,1001)*2.,
                                '-.',color='0.5',overplot=True)
        else:
            from selectFigs import _squeeze
            colormap = cm.jet
            #Set up plot
            bovy_plot.bovy_plot([-100.,-100.],[100.,100.],'k,',
                                xrange=[0,2700],yrange=[0.,70.],
                                xlabel=r'$|Z|\ [\mathrm{pc}]$',
                                ylabel=r'$\sigma_z(z)\ [\mathrm{km\ s}^{-1}]$')
            #Calculate and plot all zfuncs
            for ii in numpy.random.permutation(len(afe)):
                ds= numpy.linspace(zmin[ii]*1000.,zmax[ii]*1000.,1001)/1000.-pivot[ii]
                thiszfunc= sz[ii]+p1[ii]*ds+p2[ii]*ds**2.
                pyplot.plot(numpy.linspace(zmin[ii]*1000.,1000*zmax[ii],1001),
                            thiszfunc,'-',
                            color=colormap(_squeeze(plotc[ii],vmin,vmax)),
                            lw=ndata[ii]/15.)
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
    return parser

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    plotsz2hz(options,args)
