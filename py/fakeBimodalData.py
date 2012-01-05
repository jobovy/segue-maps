#Create a fake 'bimodal' G dwarf data set
import sys
import os, os.path
import cPickle as pickle
from optparse import OptionParser
import pyfits
import fitsio
import math
import numpy
from extreme_deconvolution import extreme_deconvolution
from galpy.util import bovy_plot
from fitDensz import _DblExpDensity, DistSpline
from segueSelect import read_gdwarfs, _append_field_recarray, _ERASESTR, \
    segueSelect
from pixelFitDens import pixelAfeFeh
from compareDataModel import _predict_rdist_plate
def fakeBimodalData(parser):
    (options,args)= parser.parse_args()
    if len(args) == 0:
        parser.print_help()
        return
    #First deconvolve the abundance distribution
    deconvolveAbundances(options,args)
    #Then assign stars to the two components
    raw, comps= reassignStars(options,args)
    #And re-assign magnitudes
    resampleMags(raw,comps,options,args)
def resampleMags(raw,comps,options,args):
    #For each data point's line-of-sight and color and feh 
    #calculate the "thin" and "thick" distributions
    model_thick= _DblExpDensity
    model_thin= _DblExpDensity
    params_thick= numpy.array([numpy.log(options.hz_thick/1000.),
                              numpy.log(options.hr_thick)])
    params_thin= numpy.array([numpy.log(options.hz_thin/1000.),
                             numpy.log(options.hr_thin)])
    #Load sf
    sf= segueSelect(sample=options.sample,sn=True,
                    type_bright='tanhrcut',
                    type_faint='tanhrcut')
    #Cut out bright stars on faint plates and vice versa
    cutbright= False
    if cutbright:
        indx= []
        for ii in range(len(raw.feh)):
            if sf.platebright[str(raw[ii].plate)] and raw[ii].dered_r >= 17.8:
                indx.append(False)
            elif not sf.platebright[str(raw[ii].plate)] and raw[ii].dered_r < 17.8:
                indx.append(False)
            else:
                indx.append(True)
        indx= numpy.array(indx,dtype='bool')
        raw= raw[indx]
    #Metallicity and color
    fehrange= [-1.,0.5]
    if options.sample.lower() == 'g':
        colorrange=[0.48,0.55]
        rmax= 20.2
    elif options.sample.lower() == 'k':
        colorrange=[0.55,0.75]
        rmax= 19.
    #Load model distributions
    #FeH
    fehdist= DistSpline(*numpy.histogram(raw.feh,bins=11,range=fehrange),
                         xrange=fehrange)
    #Color
    cdist= DistSpline(*numpy.histogram(raw.dered_g-raw.dered_r,
                                       bins=9,range=colorrange),
                       xrange=colorrange)
    rs_bright= numpy.linspace(14.50000001,17.8,1001)#Avoid sample cut
    rs_faint= numpy.linspace(17.8,rmax,1001)
    for ii in range(len(raw)):
        sys.stdout.write('\r'+"Resampling star %i / %i" % (ii+1,len(raw)))
        sys.stdout.flush()
        #Predict the r-distribution for this plate
        if sf.platebright[str(raw.plate[ii])]:
            thisrmin, thisrmax= 14.5, 17.8
            rs= rs_bright
        else: #Faint
            thisrmin, thisrmax= 17.8, rmax
            rs= rs_faint
        #Thick or thin?
        if comps[ii] == 0: #thin
            model= model_thin
            params= params_thin
        else:
            model= model_thick
            params= params_thick
        rdist= _predict_rdist_plate(rs,model,params,thisrmin,thisrmax,
                                    raw.l[ii],raw.b[ii],
                                    colorrange[0],colorrange[1],
                                    raw.feh[ii],raw.feh[ii],
                                    -.5,cdist,fehdist,
                                    sf,int(raw.plate[ii]),
                                    dontmarginalizecolorfeh=False,
                                    ngr=1,nfeh=1,
                                    dontmultiplycolorfehdist=True)
        #Sample from this
        rdist[(numpy.isnan(rdist))]= 0.
        crdist= numpy.cumsum(rdist)/numpy.sum(rdist)
        kk= 0
        randnum= numpy.random.uniform()
        while crdist[kk] < randnum and kk < (len(rs)-1): kk+= 1
        oldgr= raw.dered_g[ii]-raw.dered_r[ii]
        oldr= raw.dered_r[ii]
        raw.dered_r[ii]= rs[kk]
        raw.dered_g[ii]= oldgr+raw.dered_r[ii]
        #print oldr-raw.dered_r[ii], oldgr-(raw.dered_g[ii]-raw.dered_r[ii])
        sys.stdout.write('\r'+_ERASESTR+'\r')
    sys.stdout.flush()
    #Dump raw
    fitsio.write(args[0],raw)
def reassignStars(options,args):
    #Restore deconvolution
    outfile= open(options.xdfile,'rb')
    xamp= pickle.load(outfile)
    xmean= pickle.load(outfile)
    xcovar= pickle.load(outfile)
    outfile.close()
    #Add errors again to xcovar
    xcovar[:,0,0]+= options.dfeh**2.
    xcovar[:,1,1]+= options.dafe**2.
    invcovar1= numpy.linalg.inv(xcovar[0,:,:])
    invcovar2= numpy.linalg.inv(xcovar[1,:,:])
    idet1= numpy.sqrt(numpy.linalg.det(invcovar1))
    idet2= numpy.sqrt(numpy.linalg.det(invcovar2))
    #Re-assign
    raw= readRealData(options,args)
    comps= numpy.zeros(len(raw),dtype='int')
    for ii in range(len(raw)):
        #Calculate probabilities
        ab= numpy.array([raw[ii].feh,raw[ii].afe])
        p1= idet1*numpy.exp(-0.5*numpy.dot(ab-xmean[0,:],numpy.dot(invcovar1,ab-xmean[0,:])))
        p2= idet2*numpy.exp(-0.5*numpy.dot(ab-xmean[1,:],numpy.dot(invcovar2,ab-xmean[1,:])))
        p1/= (p1+p2)
        if numpy.random.uniform() < p1: comps[ii]= 0
        else: comps[ii]= 1
    #Make informative figure
    if not options.compsplotfile is None:
        plotComps(options,args,comps)
    return (raw,comps)
def plotComps(options,args,comps):
    raw= readRealData(options,args)
    raw= _append_field_recarray(raw,'comps',comps)
    pix= pixelAfeFeh(raw,dfeh=0.1,dafe=0.05,fehmin=-1.6,
                     fehmax=0.5,afemin=-0.05,afemax=0.55)
    plotthis= numpy.zeros((pix.npixfeh(),pix.npixafe()))
    for ii in range(pix.npixfeh()):
        for jj in range(pix.npixafe()):
            data= pix(pix.feh(ii),pix.afe(jj))
            if len(data) < 100:
                plotthis[ii,jj]= numpy.nan
                continue
            plotthis[ii,jj]= numpy.mean(data.comps)
    #Plot
    bovy_plot.bovy_print()
    bovy_plot.bovy_dens2d(plotthis.T,origin='lower',cmap='gist_yarg',
                          interpolation='nearest',
                          xrange=[-1.6,0.5],
                          yrange=[-0.05,0.55],
                          xlabel=r'$[\mathrm{Fe/H}]$',
                          ylabel=r'$[\alpha/\mathrm{Fe}]$',
                          zlabel=r'$\mathrm{fraction\ of\ stars\ in\ thick\ component}$',
#                          vmin=vmin,vmax=vmax,
                          colorbar=True,shrink=0.78,
                          contours=False)
    #Overplot contours of the underlying components
    #Restore deconvolution
    outfile= open(options.xdfile,'rb')
    xamp= pickle.load(outfile)
    xmean= pickle.load(outfile)
    xcovar= pickle.load(outfile)
    outfile.close()
    from matplotlib import pyplot
    from matplotlib.patches import Ellipse
    eigs1= numpy.linalg.eig(xcovar[0,:,:])
    eigs2= numpy.linalg.eig(xcovar[1,:,:])
    angle1= math.atan(-eigs1[1][0,1]/eigs1[1][1,1])/numpy.pi*180.
    angle2= math.atan(-eigs2[1][0,1]/eigs2[1][1,1])/numpy.pi*180.
    thisellipse= Ellipse(xmean[0,:],2.*numpy.sqrt(eigs1[0][0]),
                         2*numpy.sqrt(eigs1[0][1]),angle1)
    ells= [thisellipse]
    ells.append(Ellipse(xmean[0,:],4.*numpy.sqrt(eigs1[0][0]),
                        4*numpy.sqrt(eigs1[0][1]),angle1))
    ells.append(Ellipse(xmean[0,:],6.*numpy.sqrt(eigs1[0][0]),
                        6*numpy.sqrt(eigs1[0][1]),angle1))
    ells.append(Ellipse(xmean[1,:],2.*numpy.sqrt(eigs2[0][0]),
                        2*numpy.sqrt(eigs2[0][1]),angle2))
    ells.append(Ellipse(xmean[1,:],4.*numpy.sqrt(eigs2[0][0]),
                        4*numpy.sqrt(eigs2[0][1]),angle2))
    ells.append(Ellipse(xmean[1,:],6.*numpy.sqrt(eigs2[0][0]),
                        6*numpy.sqrt(eigs2[0][1]),angle2))
    ax= pyplot.gca()
    xlims= ax.get_xlim()
    ylims= ax.get_ylim()
    for e in ells:
        ax.add_artist(e)
        e.set_facecolor('none')
        e.set_edgecolor('0.85')
        e.set_linestyle('dashed')
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    bovy_plot.bovy_end_print(options.compsplotfile)
def readRealData(options,args):
    if options.sample.lower() == 'g':
        if options.select.lower() == 'program':
            raw= read_gdwarfs(_GDWARFFILE,logg=True,ebv=True,sn=True,nocoords=True)
        else:
            raw= read_gdwarfs(logg=True,ebv=True,sn=True,nocoords=True)
    elif options.sample.lower() == 'k':
        if options.select.lower() == 'program':
            raw= read_kdwarfs(_KDWARFFILE,logg=True,ebv=True,sn=True,nocoords=True)
        else:
            raw= read_kdwarfs(logg=True,ebv=True,sn=True,nocoords=True)
    #Cut
    raw= raw[(raw.feh > -1.5)*(raw.feh < 0.5)\
                 *(raw.afe > -0.05)*(raw.afe < 0.55)]
    return raw
def deconvolveAbundances(options,args):
    if options.xdfile is None:
        print "'xdfile' option needs to be set ..."
        print "Returning ..."
        return
    if os.path.exists(options.xdfile):
        return #Nothing to do
        #Load data
    raw= readRealData(options,args)
    #Deconvolve using XD, setup data
    ydata= numpy.zeros((len(raw),2))
    ycovar= numpy.zeros((len(raw),2))
    ydata[:,0]= raw.feh
    ydata[:,1]= raw.afe
    ycovar[:,0]= options.dfeh**2.
    ycovar[:,0]= options.dafe**2.
    #setup initial cond
    xamp= numpy.ones(2)/2.
    xmean= numpy.zeros((2,2))
    xmean[0,0]= 0. #Solar abundances
    xmean[0,1]= 0.
    xmean[1,]= -0.6 #"thick" abundances
    xmean[1,1]= 0.35
    xcovar= numpy.zeros((2,2,2))
    xcovar[0,0,0]= 0.2**2.
    xcovar[1,0,0]= 0.2**2.
    xcovar[0,1,1]= 0.1**2.
    xcovar[1,1,1]= 0.1**2.
    #Run XD
    extreme_deconvolution(ydata,ycovar,xamp,xmean,xcovar)
    #Save
    outfile= open(options.xdfile,'wb')
    pickle.dump(xamp,outfile)
    pickle.dump(xmean,outfile)
    pickle.dump(xcovar,outfile)
    outfile.close()

def get_options():
    usage = "usage: %prog [options] <savefilename>\n\nsavefilename= name of the file that the fake data will be saved to"
    parser = OptionParser(usage=usage)
    parser.add_option("--sample",dest='sample',default='g',
                      help="Use 'G' or 'K' dwarf sample")
    parser.add_option("--select",dest='select',default='all',
                      help="'all' or 'program' to select all or program stars")
    parser.add_option("--dfeh",dest='dfeh',default=0.2,type='float',
                      help="Assumed unertainty in FeH")
    parser.add_option("--dafe",dest='dafe',default=0.15,type='float',
                      help="Assumed unertainty in aFe")
    parser.add_option("--xdfile",dest='xdfile',default=None,
                      help="File to save the XD deconvolution to")
    parser.add_option("--plotcomps",dest='compsplotfile',default=None,
                      help="File to save a plot of the assigned components to")
    parser.add_option("--hz_thick",dest='hz_thick',default=850.,type='float',
                      help="'thick' disk scale height (pc)")
    parser.add_option("--hz_thin",dest='hz_thin',default=300.,type='float',
                      help="'thin' disk scale height (pc)")
    parser.add_option("--hr_thick",dest='hr_thick',default=2.,type='float',
                      help="'thick' disk scale length (kpc)")
    parser.add_option("--hr_thin",dest='hr_thin',default=3.5,type='float',
                      help="'thin' disk scale length (kpc)")
    return parser

if __name__ == '__main__':
    numpy.random.seed(1)
    fakeBimodalData(get_options())