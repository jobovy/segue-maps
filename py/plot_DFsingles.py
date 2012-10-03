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
    get_dfparams
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
            if sols[solindx] is None:
                if options.type.lower() == 'afe' or options.type.lower() == 'feh' or options.type.lower() == 'fehafe' \
                        or options.type.lower() == 'afefeh':
                    continue
                else:
                    plotthis[ii,jj]= numpy.nan
                    continue
            if options.type.lower() == 'q':
                s= get_potparams(sols[solindx],options,1)
                plotthis[ii,jj]= s[1]
                if not options.flatten is None:
                    plotthis[ii,jj]/= options.flatten
            elif options.type.lower() == 'vc':
                s= get_potparams(sols[solindx],options,1)
                plotthis[ii,jj]= s[0]
            elif options.type.lower() == 'kz':
                s= get_potparams(sols[solindx],options,1)
                plotthis[ii,jj]= (s[0]**2./s[1]**2.)/(1./options.flatten**2.)
            elif options.type.lower() == 'ndata':
                plotthis[ii,jj]= len(data)
    #Set up plot
    if options.type.lower() == 'q':
        if not options.flatten is None:
            vmin, vmax= 0.5, 1.5
            zlabel=r'$\mathrm{flattening}\ q / %.1f$' % options.flatten
        else:
            vmin, vmax= 0.5, 1.2
            zlabel=r'$\mathrm{flattening}\ q$'
    elif options.type.lower() == 'vc':
        vmin, vmax= 0.95, 1.05
        zlabel=r'$V_c / %i\ \mathrm{km\,s}^{-1}$' % int(_REFV0)
    elif options.type.lower() == 'kz':
        vmin, vmax= 0.5, 1.5
        zlabel=r'$K_z / K_z^{\mathrm{true}}$'
    elif options.type.lower() == 'ndata':
        vmin, vmax= numpy.nanmin(plotthis), numpy.nanmax(plotthis)
        zlabel=r'$N_\mathrm{data}$'
    if options.tighten:
        xrange=[-1.6,0.5]
        yrange=[-0.05,0.55]
    else:
        xrange=[-2.,0.6]
        yrange=[-0.1,0.6]
    #Now plot
    if options.type.lower() == 'afe' or options.type.lower() == 'feh' \
            or options.type.lower() == 'fehafe':
        bovy_plot.bovy_print(fig_height=3.87,fig_width=5.)
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
        if not options.type.lower() == 'ndata':
            bovy_plot.bovy_text(r'$\mathrm{median} = %.2f \pm %.2f$' % (numpy.median(plotthis[numpy.isfinite(plotthis)]),
                                                                        1.4826*numpy.median(numpy.fabs(plotthis[numpy.isfinite(plotthis)]-numpy.median(plotthis[numpy.isfinite(plotthis)])))),
                                bottom_left=True,size=14.)
    bovy_plot.bovy_end_print(options.outfilename)
    return None
                

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    numpy.random.seed(options.seed)
    plot_DFsingles(options,args)

