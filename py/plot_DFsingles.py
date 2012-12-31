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
#from galpy.actionAngle_src.actionAngleAdiabaticGrid import  actionAngleAdiabaticGrid
#from galpy.df_src.quasiisothermaldf import quasiisothermaldf
#import bovy_mcmc
#import acor
from galpy.util import bovy_plot
import monoAbundanceMW
from segueSelect import read_gdwarfs, read_kdwarfs, _GDWARFFILE, _KDWARFFILE, \
    segueSelect, _mr_gi, _gi_gr, _ERASESTR, _append_field_recarray, \
    ivezic_dist_gr
from fitDensz import cb, _ZSUN, DistSpline, _ivezic_dist, _NDS
from pixelFitDens import pixelAfeFeh
from pixelFitDF import _REFV0, get_options, read_rawdata, get_potparams, \
    get_dfparams, _REFR0
def plot_DFsingles(options,args):
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
    #Run through the pixels and gather
    if options.type.lower() == 'afe' or options.type.lower() == 'feh' \
            or options.type.lower() == 'fehafe' \
            or options.type.lower() == 'afefeh':
        plotthis= []
        errors= []
    else:
        plotthis= numpy.zeros((tightbinned.npixfeh(),tightbinned.npixafe()))
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
            if options.type.lower() == 'q':
                s= get_potparams(sols[solindx],options,1)
                plotthis[ii,jj]= s[0]
                if not options.flatten is None:
                    plotthis[ii,jj]/= options.flatten
            elif options.type.lower() == 'vc':
                s= get_potparams(sols[solindx],options,1)
                plotthis[ii,jj]= s[1]
            elif options.type.lower() == 'rd':
                s= get_potparams(sols[solindx],options,1)
                plotthis[ii,jj]= numpy.exp(s[0])
            elif options.type.lower() == 'zh':
                s= get_potparams(sols[solindx],options,1)
                plotthis[ii,jj]= numpy.exp(s[2])
            elif options.type.lower() == 'kz':
                s= get_potparams(sols[solindx],options,1)
                plotthis[ii,jj]= (s[1]**2./s[0]**2.)/(1./options.flatten**2.)
            elif options.type.lower() == 'ndata':
                plotthis[ii,jj]= len(data)
            elif options.type.lower() == 'hr':
                s= get_dfparams(sols[solindx],0,options)
                plotthis[ii,jj]= s[0]*_REFR0
                if options.relative:
                    thishr= monoAbundanceMW.hr(mapfehs[monoabindx],mapafes[monoabindx])
                    plotthis[ii,jj]/= thishr
            elif options.type.lower() == 'sz':
                s= get_dfparams(sols[solindx],0,options)
                plotthis[ii,jj]= s[2]*_REFV0
                if options.relative:
                    thissz= monoAbundanceMW.sigmaz(mapfehs[monoabindx],mapafes[monoabindx])
                    plotthis[ii,jj]/= thissz
            elif options.type.lower() == 'sr':
                s= get_dfparams(sols[solindx],0,options)
                plotthis[ii,jj]= s[1]*_REFV0
                if options.relative:
                    thissr= monoAbundanceMW.sigmaz(mapfehs[monoabindx],mapafes[monoabindx])*2.
                    plotthis[ii,jj]/= thissr
            elif options.type.lower() == 'afe' or options.type.lower() == 'feh' or options.type.lower() == 'fehafe' \
                    or options.type.lower() == 'afefeh':
                thisplot=[tightbinned.feh(ii),
                          tightbinned.afe(jj),
                          len(data)]
                if options.subtype.lower() == 'qvc':
                    s= get_potparams(sols[solindx],options,1)
                    thisq= s[0]
                    if not options.flatten is None:
                        thisq/= options.flatten
                    thisvc= s[1]
                    thisplot.extend([thisq,thisvc])
                plotthis.append(thisplot)
    #Set up plot
    if options.type.lower() == 'q':
        if not options.flatten is None:
            vmin, vmax= 0.9, 1.1
            zlabel=r'$\mathrm{flattening}\ q / %.1f$' % options.flatten
        elif 'real' in options.outfilename.lower():
            vmin, vmax= 0.9, 1.1
            medianq= numpy.median(plotthis[numpy.isfinite(plotthis)])
            plotthis/= medianq
            zlabel=r'$\mathrm{flattening}\ q / %.2f$' % medianq
        else:
            vmin, vmax= 0.5, 1.2
            zlabel=r'$\mathrm{flattening}\ q$'
    elif options.type.lower() == 'vc':
        vmin, vmax= 0.95, 1.05
        zlabel=r'$V_c / %i\ \mathrm{km\,s}^{-1}$' % int(_REFV0)
        if 'real' in options.outfilename.lower():
           medianvc= numpy.median(plotthis[numpy.isfinite(plotthis)])
           plotthis/= medianvc
           zlabel=r'$V_c / %i\ \mathrm{km\,s}^{-1}$' % int(_REFV0*medianvc)
    elif options.type.lower() == 'rd':
        vmin, vmax= 0.2, 0.6
        zlabel=r'$R_d / R_0$'
    elif options.type.lower() == 'zh':
        vmin, vmax= 0.0125, 0.075
        zlabel=r'$z_h / R_0$'
    elif options.type.lower() == 'kz':
        vmin, vmax= 0.5, 1.5
        zlabel=r'$K_z / K_z^{\mathrm{true}}$'
    elif options.type.lower() == 'ndata':
        vmin, vmax= numpy.nanmin(plotthis), numpy.nanmax(plotthis)
        zlabel=r'$N_\mathrm{data}$'
    elif options.type == 'hr':
        if options.relative:
            vmin, vmax= 0.8, 1.2
            zlabel=r'$\mathrm{input / output\ radial\ scale\ length}$'
        else:
            vmin, vmax= 1.35,4.5
            zlabel=r'$\mathrm{model\ radial\ scale\ length\ [kpc]}$'
    elif options.type == 'sz':
        if options.relative:
            vmin, vmax= 0.8, 1.2
            zlabel= r'$\mathrm{input / output\ model}\ \sigma_z$'
        else:
            vmin, vmax= 10.,60.
            zlabel= r'$\mathrm{model}\ \sigma_z\ [\mathrm{km\ s}^{-1}]$'
    elif options.type == 'sr':
        if options.relative:
            vmin, vmax= 0.8, 1.2
            zlabel= r'$\mathrm{input/output\ model}\ \sigma_R$'
        else:
            vmin, vmax= 20.,100.
            zlabel= r'$\mathrm{model}\ \sigma_R\ [\mathrm{km\ s}^{-1}]$'
    elif options.type == 'afe':
        vmin, vmax= 0.0,.5
        zlabel=r'$[\alpha/\mathrm{Fe}]$'
    elif options.type == 'feh':
        vmin, vmax= -1.6,0.4
        zlabel=r'$[\mathrm{Fe/H}]$'
    elif options.type == 'fehafe':
        vmin, vmax= -.7,.7
        zlabel=r'$[\mathrm{Fe/H}]-[\mathrm{Fe/H}]_{1/2}([\alpha/\mathrm{Fe}])$'
    elif options.type == 'afefeh':
        vmin, vmax= -.15,.15
        zlabel=r'$[\alpha/\mathrm{Fe}]-[\alpha/\mathrm{Fe}]_{1/2}([\mathrm{Fe/H}])$'
    if options.tighten:
        xrange=[-1.6,0.5]
        yrange=[-0.05,0.55]
    else:
        xrange=[-2.,0.5]
        yrange=[-0.2,0.6]
    #Now plot
    if options.type.lower() == 'afe' or options.type.lower() == 'feh' \
            or options.type.lower() == 'fehafe':
        #Gather everything
        afe, feh, ndata, x, y= [], [], [], [], []
        for ii in range(len(plotthis)):
            afe.append(plotthis[ii][1])
            feh.append(plotthis[ii][0])
            ndata.append(plotthis[ii][2])
            x.append(plotthis[ii][3])
            y.append(plotthis[ii][4])
        afe= numpy.array(afe)
        feh= numpy.array(feh)
        ndata= numpy.array(ndata)
        x= numpy.array(x)
        y= numpy.array(y)
        #Process ndata
        ndata= ndata**.5
        ndata= ndata/numpy.median(ndata)*35.
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
        if options.subtype.lower() == 'qvc':
            if not options.flatten is None:
                xrange= [0.9,1.1]
                xlabel=r'$\mathrm{flattening}\ q / %.1f$' % options.flatten
            elif 'real' in options.outfilename.lower():
                xrange= [0.9,1.1]
                medianq= numpy.median(x[numpy.isfinite(x)])
                x/= medianq
                xlabel=r'$\mathrm{flattening}\ q / %.2f$' % medianq
            else:
                xrange= [0.5, 1.2]
                xlabel=r'$\mathrm{flattening}\ q$'
            yrange= [0.95, 1.05]
            ylabel=r'$V_c / %i\ \mathrm{km\,s}^{-1}$' % int(_REFV0)
            if 'real' in options.outfilename.lower():
                medianvc= numpy.median(y[numpy.isfinite(y)])
                y/= medianvc
                ylabel=r'$V_c / %i\ \mathrm{km\,s}^{-1}$' % int(_REFV0*medianvc)
        bovy_plot.bovy_print(fig_height=3.87,fig_width=5.)
        bovy_plot.bovy_plot(x,y,
                            s=ndata,c=plotc,
                            cmap='jet',
                            xlabel=xlabel,ylabel=ylabel,
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
        if options.type.lower() == 'q' or options.type.lower() == 'vc' \
                or options.relative or options.type.lower() == 'rd':
            bovy_plot.bovy_text(r'$\mathrm{median} = %.2f \pm %.2f$' % (numpy.median(plotthis[numpy.isfinite(plotthis)]),
                                                                        1.4826*numpy.median(numpy.fabs(plotthis[numpy.isfinite(plotthis)]-numpy.median(plotthis[numpy.isfinite(plotthis)])))),
                                bottom_left=True,size=14.)
        if options.type.lower() == 'zh':
            bovy_plot.bovy_text(r'$\mathrm{median} = %.4f \pm %.4f$' % (numpy.median(plotthis[numpy.isfinite(plotthis)]),
                                                                        1.4826*numpy.median(numpy.fabs(plotthis[numpy.isfinite(plotthis)]-numpy.median(plotthis[numpy.isfinite(plotthis)])))),
                                bottom_left=True,size=14.)
    bovy_plot.bovy_end_print(options.outfilename)
    return None
                

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    numpy.random.seed(options.seed)
    plot_DFsingles(options,args)

