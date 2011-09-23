import os, os.path
import sys
import math
import numpy
import cPickle as pickle
from optparse import OptionParser
from galpy.util import bovy_coords, bovy_plot, save_pickles
from matplotlib import pyplot, cm
from segueSelect import read_gdwarfs, read_kdwarfs, _gi_gr, _mr_gi, \
    segueSelect, _GDWARFFILE, _KDWARFFILE
from selectFigs import _squeeze
from fitDensz import _TwoDblExpDensity, _HWRLikeMinus, _ZSUN, DistSpline, \
    _ivezic_dist, _NDS, cb, _HWRDensity, _HWRLike
from pixelFitDens import pixelAfeFeh
from predictBellHalo import predictDiskMass
_NGR= 11
_NFEH=11
def calcMass(options,args):
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
        mass= pickle.load(savefile)
        ii= pickle.load(savefile)
        jj= pickle.load(savefile)
        savefile.close()
    else:
        mass= []
        ii, jj= 0, 0
    #parameters
    if os.path.exists(args[1]):#Load initial
        savefile= open(args[1],'rb')
        fits= pickle.load(savefile)
        savefile.close()
    else:
        print "Error: must provide parameters of best fits"
        print "Returning ..."
        return None
    #Sample?
    if options.mcsample:
        if ii < len(binned.fehedges)-1 and jj < len(binned.afeedges)-1:
            print "First do all of the best-fit mass estimates ..."
            print "Returning ..."
            return None
        if os.path.exists(args[2]): #Load savefile
            savefile= open(args[2],'rb')
            masssamples= pickle.load(savefile)
            ii= pickle.load(savefile)
            jj= pickle.load(savefile)
            savefile.close()
        else:
            masssamples= []
            ii, jj= 0, 0
        if os.path.exists(args[3]): #Load savefile
            savefile= open(args[3],'rb')
            denssamples= pickle.load(savefile)
            savefile.close()
        else:
            print "If mcsample you need to provide the file with the density samples ..."
            print "Returning ..."
            return None
    #Set up model etc.
    if options.model.lower() == 'hwr':
        densfunc= _HWRDensity
    elif options.model.lower() == 'twodblexp':
        densfunc= _TwoDblExpDensity
    like_func= _HWRLikeMinus
    pdf_func= _HWRLike
    if options.sample.lower() == 'g':
        colorrange=[0.48,0.55]
    elif options.sample.lower() == 'k':
        colorrange=[0.55,0.75]
    #Load selection function
    plates= numpy.array(list(set(list(raw.plate))),dtype='int') #Only load plates that we use
    print "Using %i plates, %i stars ..." %(len(plates),len(raw))
    sf= segueSelect(plates=plates,type_faint='tanhrcut',
                    sample=options.sample,type_bright='tanhrcut',
                    sn=True,select=options.select)
    platelb= bovy_coords.radec_to_lb(sf.platestr.ra,sf.platestr.dec,
                                     degree=True)
    indx= [not 'faint' in name for name in sf.platestr.programname]
    platebright= numpy.array(indx,dtype='bool')
    indx= ['faint' in name for name in sf.platestr.programname]
    platefaint= numpy.array(indx,dtype='bool')
    if options.sample.lower() == 'g':
        grmin, grmax= 0.48, 0.55
        rmin,rmax= 14.5, 20.2
    #Run through the bins
    while ii < len(binned.fehedges)-1:
        while jj < len(binned.afeedges)-1:
            data= binned(binned.feh(ii),binned.afe(jj))
            if len(data) < options.minndata:
                if options.mcsample: masssamples.append(None)
                else: mass.append(None)
                jj+= 1
                if jj == len(binned.afeedges)-1: 
                    jj= 0
                    ii+= 1
                    break
                continue               
            print binned.feh(ii), binned.afe(jj), len(data)
            fehindx= binned.fehindx(binned.feh(ii))
            afeindx= binned.afeindx(binned.afe(jj))
            #set up feh and color
            feh= binned.feh(ii)
            fehrange= [binned.fehedges[ii],binned.fehedges[ii+1]]
            #FeH
            fehdist= DistSpline(*numpy.histogram(data.feh,bins=5,
                                                 range=fehrange),
                                 xrange=fehrange,dontcuttorange=False)
            #Color
            colordist= DistSpline(*numpy.histogram(data.dered_g\
                                                       -data.dered_r,
                                                   bins=9,range=colorrange),
                                   xrange=colorrange)
            
            #Age marginalization
            afe= binned.afe(jj)
            if afe > 0.25: agemin, agemax= 7.,10.
            else: agemin,agemax= 1.,8.
            if options.mcsample:
                #Loop over samples
                thissamples= denssamples[afeindx+fehindx*binned.npixafe()]
                if options.nsamples < len(thissamples):
                    #Random permutation
                    thissamples= numpy.random.permutation(thissamples)[0:options.nsamples]
                thismasssamples= []
                for kk in range(len(thissamples)):
                    thisparams= thissamples[kk]
                    thismasssamples.append(predictDiskMass(densfunc,
                                                           thisparams,sf,
                                                           colordist,fehdist,
                                                           fehrange[0],
                                                           fehrange[1],feh,
                                                           data,colorrange[0],
                                                           colorrange[1],
                                                           agemin,agemax,
                                                           normalize=options.normalize))
                #Print some stuff
                print numpy.std(numpy.array(thismasssamples))
                masssamples.append(thismasssamples)
            else:
                thisparams= fits[afeindx+fehindx*binned.npixafe()]
                mass.append(predictDiskMass(densfunc,
                                            thisparams,sf,
                                            colordist,fehdist,
                                            fehrange[0],
                                            fehrange[1],feh,
                                            data,colorrange[0],
                                            colorrange[1],
                                            agemin,agemax,
                                            normalize=options.normalize))
                print mass[-1]
            jj+= 1
            if jj == len(binned.afeedges)-1: 
                jj= 0
                ii+= 1
            if options.mcsample: save_pickles(args[2],masssamples,ii,jj)
            else: save_pickles(args[0],mass,ii,jj)
            if jj == 0: #this means we've reset the counter 
                break
    if options.mcsample: save_pickles(args[2],masssamples,ii,jj)
    else: save_pickles(args[0],mass,ii,jj)
    return None

def plotMass(options,args):
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
        mass= pickle.load(savefile)
        ii= pickle.load(savefile)
        jj= pickle.load(savefile)
        savefile.close()
    else:
        fits= []
        ii, jj= 0, 0
    #parameters
    if os.path.exists(args[1]):#Load initial
        savefile= open(args[1],'rb')
        fits= pickle.load(savefile)
        savefile.close()
    else:
        print "Error: must provide parameters of best fits"
        print "Returning ..."
        return None
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
            thismass= mass[afeindx+fehindx*binned.npixafe()]
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
            if options.type == 'mass':
                if options.logmass:
                    plotthis[ii,jj]= numpy.log10(thismass/10**6.)
                else:
                    plotthis[ii,jj]= thismass/10**6.
            elif options.model.lower() == 'hwr':
                if options.type == 'hz':
                    plotthis[ii,jj]= numpy.exp(thisfit[0])*1000.
                elif options.type == 'hr':
                    plotthis[ii,jj]= numpy.exp(thisfit[1])
                elif options.type.lower() == 'afe' \
                        or options.type.lower() == 'feh' \
                        or options.type.lower() == 'fehafe' \
                        or options.type.lower() == 'afefeh':
                    plotthis.append([tightbinned.feh(ii),
                                         tightbinned.afe(jj),
                                     numpy.exp(thisfit[0])*1000.,
                                     numpy.exp(thisfit[1]),
                                     len(data),
                                     thismass/10.**6.])
    #Set up plot
    #print numpy.nanmin(plotthis), numpy.nanmax(plotthis)
    if options.type == 'mass':
        if options.logmass:
            vmin, vmax= numpy.log10(0.001), numpy.log10(2.)
            zlabel=r'$\log_{10} \Sigma(R_0)\ [M_{\odot}\ \mathrm{pc}^{-2}]$'
        else:
            vmin, vmax= 0.,2.
            zlabel=r'$\Sigma(R_0)\ [M_{\odot}\ \mathrm{pc}^{-2}]$'
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
        #Gather hR and hz
        mass, hz, hr,afe, feh, ndata= [], [], [], [], [], []
        for ii in range(len(plotthis)):
            mass.append(plotthis[ii][5])
            hz.append(plotthis[ii][2])
            hr.append(plotthis[ii][3])
            afe.append(plotthis[ii][1])
            feh.append(plotthis[ii][0])
            ndata.append(plotthis[ii][4])
        mass= numpy.array(mass)
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
        xrange= [150,1200]
        if options.cumul:
            ids= numpy.argsort(hz)
            plotc= plotc[ids]
            ndata= ndata[ids]
            mass= mass[ids]
            mass= numpy.cumsum(mass)
            hz.sort()
            ylabel=r'$\mathrm{cumulative}\ \Sigma(R_0)\ [M_{\odot}\ \mathrm{pc}^{-2}]$'
            if options.logmass:
                yrange= [0.01,30.]
            else:
                yrange= [-0.1,30.]
        else:
            if options.logmass:
                yrange= [0.01,2.]
            else:
                yrange= [-0.1,2.]
            ylabel=r'$\Sigma(R_0)\ [M_{\odot}\ \mathrm{pc}^{-2}]$'
        bovy_plot.bovy_plot(hz,mass,
                            s=ndata,c=plotc,
                            cmap='jet',
                            xlabel=r'$\mathrm{vertical\ scale\ height\ [pc]}$',
                            ylabel=ylabel,
                            clabel=zlabel,
                            xrange=xrange,yrange=yrange,
                            vmin=vmin,vmax=vmax,
                            scatter=True,edgecolors='none',
                            colorbar=True,zorder=2,
                            semilogy=options.logmass)
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
    usage = "usage: %prog [options] <savefile> <savefile>\n\nsavefile= name of the file that the mass will be saved to\nsavefile = name of the file that has the best fits"
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
    parser.add_option("--model",dest='model',default='twodblexp',
                      help="Model to fit")
    parser.add_option("-o","--plotfile",dest='plotfile',default=None,
                      help="Name of the file for plot")
    parser.add_option("-t","--type",dest='type',default='mass',
                      help="Quantity to plot ('mass', 'afe', 'feh'")
    parser.add_option("--normalize",dest='normalize',default='Z',
                      help="Normalize over Z or over space")
    parser.add_option("--plot",action="store_true", dest="plot",
                      default=False,
                      help="If set, plot, otherwise, fit")
    parser.add_option("--tighten",action="store_true", dest="tighten",
                      default=False,
                      help="If set, tighten axes")
    parser.add_option("--ploterrors",action="store_true", dest="ploterrors",
                      default=False,
                      help="If set, plot the errorbars")
    parser.add_option("--mcsample",action="store_true", dest="mcsample",
                      default=False,
                      help="If set, sample around the best fit, save in args[1]")
    parser.add_option("--nsamples",dest='nsamples',default=10,type='int',
                      help="Number of Mass samples to obtain")
    parser.add_option("--logmass",action="store_true", dest="logmass",
                      default=False,
                      help="If set, plot the log of the mass")
    parser.add_option("--cumul",action="store_true", dest="cumul",
                      default=False,
                      help="If set, plot cumulative mass as a function of h_z")
    return parser
  
if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    if options.plot:
        plotMass(options,args)
    else:
        calcMass(options,args)

