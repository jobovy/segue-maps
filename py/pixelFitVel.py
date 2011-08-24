import os, os.path
import sys
import math
import numpy
from scipy import optimize
import cPickle as pickle
from optparse import OptionParser
from galpy.util import bovy_coords, bovy_plot, save_pickles
from segueSelect import read_gdwarfs, read_kdwarfs, _GDWARFFILE, _KDWARFFILE
from fitSigz import _IsothermLikeMinus, _HWRLikeMinus, _ZSUN
from pixelFitDens import pixelAfeFeh
def pixelFitVel(options,args):
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
    #Savefile
    if os.path.exists(args[0]):#Load savefile
        savefile= open(args[0],'rb')
        fits= pickle.load(savefile)
        ii= pickle.load(savefile)
        jj= pickle.load(savefile)
        savefile.close()
    else:
        fits= []
        ii, jj= 0, 0
    #Model
    if options.model.lower() == 'hwr':
        like_func= _HWRLikeMinus
    elif options.model.lower() == 'isotherm':
        like_func= _IsothermLikeMinus
    #Run through the bins
    while ii < len(binned.fehedges)-1:
        while jj < len(binned.afeedges)-1:
            data= binned(binned.feh(ii),binned.afe(jj))
            if len(data) < options.minndata:
                fits.append(None)
                jj+= 1
                if jj == len(binned.afeedges)-1: 
                    jj= 0
                    ii+= 1
                    break
                continue               
            print binned.feh(ii), binned.afe(jj), len(data)
            #Create XYZ and R, vxvyvz, cov_vxvyvz
            R= ((8.-data.xc)**2.+data.yc**2.)**0.5
            XYZ= numpy.zeros((len(data),3))
            XYZ[:,0]= data.xc
            XYZ[:,1]= data.yc
            XYZ[:,2]= data.zc+_ZSUN
            d= (XYZ[:,2]-numpy.median(numpy.fabs(XYZ[:,2])))
            vxvyvz= numpy.zeros((len(data),3))
            vxvyvz[:,0]= data.vxc
            vxvyvz[:,1]= data.vyc
            vxvyvz[:,2]= data.vzc
            cov_vxvyvz= numpy.zeros((len(data),3,3))
            cov_vxvyvz[:,0,0]= data.vxc_err**2.
            cov_vxvyvz[:,1,1]= data.vyc_err**2.
            cov_vxvyvz[:,2,2]= data.vzc_err**2.
            cov_vxvyvz[:,0,1]= data.vxvyc_rho*data.vxc_err*data.vyc_err
            cov_vxvyvz[:,0,2]= data.vxvzc_rho*data.vxc_err*data.vzc_err
            cov_vxvyvz[:,1,2]= data.vyvzc_rho*data.vyc_err*data.vzc_err
            #Fit this data
            #Initial condition
            if options.model.lower() == 'hwr':
                params= numpy.array([0.02,numpy.log(30.),0.,0.,numpy.log(6.)])
            elif options.model.lower() == 'isotherm':
                params= numpy.array([0.02,numpy.log(30.),numpy.log(6.)])
            #Optimize likelihood
            params= optimize.fmin_powell(like_func,params,
                                         args=(XYZ,vxvyvz,cov_vxvyvz,R,d))
            print numpy.exp(params)
            fits.append(params)
            jj+= 1
            if jj == len(binned.afeedges)-1: 
                jj= 0
                ii+= 1
            save_pickles(args[0],fits,ii,jj)
            if jj == 0: #this means we've reset the counter 
                break
    save_pickles(args[0],fits,ii,jj)
    return None

def plotPixelFitVel(options,args):
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
    #Savefile
    if os.path.exists(args[0]):#Load savefile
        savefile= open(args[0],'rb')
        fits= pickle.load(savefile)
        savefile.close()
    #Now plot
    #Run through the pixels and gather
    if options.type.lower() == 'afe' or options.type.lower() == 'feh' \
            or options.type.lower() == 'fehafe' \
            or options.type.lower() == 'afefeh':
        plotthis= []
    else:
        plotthis= numpy.zeros((tightbinned.npixfeh(),tightbinned.npixafe()))
    for ii in range(tightbinned.npixfeh()):
        for jj in range(tightbinned.npixafe()):
            data= binned(tightbinned.feh(ii),tightbinned.afe(jj))
            fehindx= binned.fehindx(tightbinned.feh(ii))#Map onto regular binning
            afeindx= binned.afeindx(tightbinned.afe(jj))
            if afeindx+fehindx*binned.npixafe() >= len(fits):
                if options.type.lower() == 'afe' or options.type.lower() == 'feh' or options.type.lower() == 'fehafe' \
                        or options.type.lower() == 'afefeh':
                    continue
                else:
                    plotthis[ii,jj]= numpy.nan
                    continue
            thisfit= fits[afeindx+fehindx*binned.npixafe()]
            if thisfit is None:
                if options.type.lower() == 'afe' or options.type.lower() == 'feh' or options.type.lower() == 'fehafe' \
                        or options.type.lower() == 'afefeh':
                    continue
                else:
                    plotthis[ii,jj]= numpy.nan
                    continue
            if len(data) < options.minndata:
                if options.type.lower() == 'afe' or options.type.lower() == 'feh' or options.type.lower() == 'fehafe' \
                        or options.type.lower() == 'afefeh':
                    continue
                else:
                    plotthis[ii,jj]= numpy.nan
                    continue
            if options.model.lower() == 'hwr':
                if options.type == 'sz':
                    plotthis[ii,jj]= numpy.exp(thisfit[1])
                elif options.type == 'sz2':
                    plotthis[ii,jj]= numpy.exp(2.*thisfit[1])
                elif options.type == 'hs':
                    plotthis[ii,jj]= numpy.exp(thisfit[4])
                elif options.type.lower() == 'afe' \
                        or options.type.lower() == 'feh' \
                        or options.type.lower() == 'fehafe' \
                        or options.type.lower() == 'afefeh':
                    plotthis.append([tightbinned.feh(ii),
                                     tightbinned.afe(jj),
                                     numpy.exp(thisfit[1]),
                                     numpy.exp(thisfit[3]),
                                     len(data)])
    #Set up plot
    #print numpy.nanmin(plotthis), numpy.nanmax(plotthis)
    if options.type == 'sz':
        vmin, vmax= 15.,60.
        zlabel= r'$\sigma_z(z=1000\ \mathrm{pc})\ [\mathrm{km\ s}^{-1}]$'
    elif options.type == 'sz2':
        vmin, vmax= 15.**2.,50.**2.
        zlabel= r'$\sigma_z^2(z=1000\ \mathrm{pc})\ [\mathrm{km\ s}^{-1}]$'
    elif options.type == 'hs':
        vmin, vmax= 3.,15.
        zlabel= r'$R_\sigma\ [\mathrm{kpc}]$'
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
        print "Update!!"
        return None
        bovy_plot.bovy_print(fig_height=3.87,fig_width=5.)
        #Gather hR and hz
        hz, hr,afe, feh, ndata= [], [], [], [], []
        for ii in range(len(plotthis)):
            hz.append(plotthis[ii][2])
            hr.append(plotthis[ii][3])
            afe.append(plotthis[ii][1])
            feh.append(plotthis[ii][0])
            ndata.append(plotthis[ii][4])
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
        yrange= [150,1200]
        xrange= [1.2,5.]
        bovy_plot.bovy_plot(hr,hz,s=ndata,c=plotc,
                            cmap='jet',
                            ylabel=r'$\mathrm{vertical\ scale\ height\ [pc]}$',
                            xlabel=r'$\mathrm{radial\ scale\ length\ [kpc]}$',
                            clabel=zlabel,
                            xrange=xrange,yrange=yrange,
                            vmin=vmin,vmax=vmax,
                            scatter=True,edgecolors='none',
                            colorbar=True)
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
    usage = "usage: %prog [options] <savefile>\n\nsavefile= name of the file that the fits will be saved to"
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
    parser.add_option("--model",dest='model',default='hwr',
                      help="Model to fit")
    parser.add_option("-o","--plotfile",dest='plotfile',default=None,
                      help="Name of the file for plot")
    parser.add_option("-t","--type",dest='type',default='sz',
                      help="Quantity to plot ('sz', 'hs', 'afe', 'feh'")
    parser.add_option("--plot",action="store_true", dest="plot",
                      default=False,
                      help="If set, plot, otherwise, fit")
    parser.add_option("--tighten",action="store_true", dest="tighten",
                      default=False,
                      help="If set, tighten axes")
    return parser
  
if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    if options.plot:
        plotPixelFitVel(options,args)
    else:
        pixelFitVel(options,args)
