import os, os.path
import sys
import math
import numpy
from scipy import optimize
import cPickle as pickle
from optparse import OptionParser
from galpy.util import bovy_coords, bovy_plot
from segueSelect import read_gdwarfs, read_kdwarfs, _gi_gr, _mr_gi, \
    segueSelect
from fitDensz import _TwoDblExpDensity, _HWRLikeMinus, _ZSUN, DistSpline, \
    _ivezic_dist, _NDS, cb, _HWRDensity
_NGR= 11
_NFEH=11
class pixelAfeFeh:
    """Class that pixelizes the data in afe and feh"""
    def __init__(self,data,dfeh=0.05,dafe=0.05):
        """
        NAME:
           __init__
        PURPOSE:
           initialize
        INPUT:
           data as recarray, has feh and afe
           dfeh, dafe= bin widths
        OUTPUT:
           object
        HISTORY:
            2011-08-16 - Written - Bovy (NYU)
        """
        self.data= data
        self.dfeh= dfeh
        self.dafe= dafe
        #These are somewhat ridiculous to be sure to contain the data
        self.fehmin=-2.
        self.fehmax= .6
        self.afemin= -.1
        self.afemax= .6
        #edges in feh
        fehedges= list(numpy.arange(0.,self.fehmax+0.01,dfeh))
        fehedges.extend(list(numpy.arange(0,self.fehmin-0.01,-dfeh)))
        self.fehedges= numpy.array(sorted(list(set(fehedges))))
        #edges in afe
        afeedges= list(numpy.arange(0.,self.afemax+0.01,dafe))
        afeedges.extend(list(numpy.arange(0,self.afemin-0.01,-dafe)))
        self.afeedges= numpy.array(sorted(list(set(afeedges))))
        return None

    def __call__(self,*args,**kwargs):
        """
        NAME:
           __call__
        PURPOSE:
           return the part of the sample in a afe and feh pixel
        INPUT:
           feh, afe
        OUTPUT:
           returns data recarray in the bin that feh and afe are in
        HISTORY:
           2011-08-16 - Written - Bovy (NYU)
        """
        #Find bin
        fehbin= int(math.floor((args[0]-self.fehmin)/self.dfeh))
        afebin= int(math.floor((args[1]-self.afemin)/self.dafe))
        #Return data
        return self.data[(self.data.feh > self.fehedges[fehbin])\
                             *(self.data.feh <= self.fehedges[fehbin+1])\
                             *(self.data.afe > self.afeedges[afebin])\
                             *(self.data.afe <= self.afeedges[afebin+1])]
    def feh(self,i):
        """
        NAME:
           feh
        PURPOSE:
           return the i-th bin's feh center
        INPUT:
           i - bin
        OUTPUT:
           bin's central feh
        HISTORY:
           2011-08-16 - Written - Bovy (NYU)
        """
        return 0.5*(self.fehedges[i]+self.fehedges[i+1])

    def afe(self,i):
        """
        NAME:
           afe
        PURPOSE:
           return the i-th bin's central afe
        INPUT:
           i - bin
        OUTPUT:
           bin's central afe
        HISTORY:
           2011-08-16 - Written - Bovy (NYU)
        """
        return 0.5*(self.afeedges[i]+self.afeedges[i+1])

    def npixfeh(self):
        """Return the number of FeH pixels"""
        return len(self.fehedges)-1

    def npixafe(self):
        """Return the number of AFe pixels"""
        return len(self.afeedges)-1
    
def pixelFitDens(parser):
    options,args= parser.parse_args()
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
    #Set up model etc.
    if options.model.lower() == 'hwr':
        densfunc= _HWRDensity
    elif options.model.lower() == 'twodblexp':
        densfunc= _TwoDblExpDensity
    like_func= _HWRLikeMinus
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
                fits.append(None)
                jj+= 1
                if jj == len(binned.afeedges)-1: 
                    jj= 0
                    ii+= 1
                    break
                continue               
            print binned.feh(ii), binned.afe(jj), len(data)
            #Create XYZ and R
            R= ((8.-data.xc)**2.+data.yc**2.)**0.5
            XYZ= numpy.zeros((len(data),3))
            XYZ[:,0]= data.xc
            XYZ[:,1]= data.yc
            XYZ[:,2]= data.zc+_ZSUN
            #Fit this data, set up feh and color
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
            #Initial condition
            if options.model.lower() == 'hwr':
                params= numpy.array([numpy.log(0.5),numpy.log(3.),0.05])
            elif options.model.lower() == 'twodblexp':
                params= numpy.array([numpy.log(0.3),numpy.log(1.),numpy.log(2.5),numpy.log(2.5),0.5])
           #Integration grid when binning
            grs= numpy.linspace(grmin,grmax,_NGR)
            fehs= numpy.linspace(fehrange[0],fehrange[1],_NFEH)
            rhogr= numpy.array([colordist(gr) for gr in grs])
            rhofeh= numpy.array([fehdist(feh) for feh in fehs])
            mr= numpy.zeros((_NGR,_NFEH))
            for kk in range(_NGR):
                for ll in range(_NFEH):
                    mr[kk,ll]= _mr_gi(_gi_gr(grs[kk]),fehs[ll])
            #determine dmin and dmax
            allbright, allfaint= True, True
            #dmin and dmax for this rmin, rmax
            for p in sf.plates:
                #l and b?
                pindx= (sf.plates == p)
                plateb= platelb[pindx,1][0]
                if 'faint' in sf.platestr[pindx].programname[0]:
                    allbright= False
                else:
                    allfaint= False
            if allbright:
                thisrmin, thisrmax= rmin, 17.8
            elif allfaint:
                thisrmin, thisrmax= 17.8, rmax
            else:
                thisrmin, thisrmax= rmin, rmax
            _THISNGR, _THISNFEH= 51, 51
            thisgrs= numpy.zeros((_THISNGR,_THISNFEH))
            thisfehs= numpy.zeros((_THISNGR,_THISNFEH))
            for kk in range(_THISNGR):
                thisfehs[kk,:]= numpy.linspace(fehrange[0],fehrange[1],
                                               _THISNFEH)
            for kk in range(_THISNFEH):
                thisgrs[:,kk]= numpy.linspace(grmin,grmax,_THISNGR)
            dmin= numpy.amin(_ivezic_dist(thisgrs,thisrmin,thisfehs))
            dmax= numpy.amax(_ivezic_dist(thisgrs,thisrmax,thisfehs))
            ds= numpy.linspace(dmin,dmax,_NDS)
            #Optimize likelihood
            params= optimize.fmin_powell(like_func,params,
                                         args=(XYZ,R,
                                               sf,sf.plates,platelb[:,0],
                                               platelb[:,1],platebright,
                                               platefaint,1.,
                                               grmin,grmax,rmin,rmax,
                                               fehrange[0],fehrange[1],
                                               feh,colordist,densfunc,
                                               fehdist,False,
                                               False,1.,
                                               grs,fehs,rhogr,rhofeh,mr,
                                               False,dmin,dmax,ds),
                                         callback=cb)
            print numpy.exp(params)
            fits.append(params)
            jj+= 1
            if jj == len(binned.afeedges)-1: 
                jj= 0
                ii+= 1
            save_pickles(fits,ii,jj,args[0])
            if jj == 0: #this means we've reset the counter 
                break
    save_pickles(fits,ii,jj,args[0])
    if options.plotfile is None: return None
    #Now plot
    #Run through the pixels and gather
    plotthis= numpy.zeros((binned.npixfeh(),binned.npixafe()))
    for ii in range(binned.npixfeh()):
        for jj in range(binned.npixafe()):
            #print binned.feh(ii), binned.afe(jj), fits[jj+ii*binned.npixafe()]
            #print jj+ii*binned.npixafe(), len(fits)
            thisfit= fits[jj+ii*binned.npixafe()]
            if thisfit is None:
                plotthis[ii,jj]= numpy.nan
                continue
            if options.model.lower() == 'hwr':
                if options.type == 'hz':
                    plotthis[ii,jj]= numpy.exp(thisfit[0])*1000.
                elif options.type == 'hr':
                    plotthis[ii,jj]= numpy.exp(thisfit[1])
    #Set up plot
    if options.type == 'hz':
        vmin, vmax= 200,1200
        zlabel=r'$\mathrm{vertical\ scale\ height\ [pc]}$'
    elif options.type == 'hr':
        vmin, vmax= 1.5,5.
        zlabel=r'$\mathrm{radial\ scale\ length\ [kpc]}$'
    bovy_plot.bovy_print()
    bovy_plot.bovy_dens2d(plotthis.T,origin='lower',cmap='jet',
                          interpolation='nearest',
                          xrange=[-2.,0.6],
                          yrange=[-0.1,0.6],
                          xlabel=r'$[\mathrm{Fe/H}]$',
                          ylabel=r'$[\alpha/\mathrm{Fe}]$',
                          zlabel=zlabel,
                          vmin=vmin,vmax=vmax,
                          contours=False,
                          colorbar=True,shrink=0.78)
    bovy_plot.bovy_end_print(options.plotfile)
    return None

def save_pickles(fits,ii,jj,savefilename):
    saving= True
    interrupted= False
    tmp_savefilename= savefilename+'.tmp'
    while saving:
        try:
            savefile= open(tmp_savefilename,'wb')
            pickle.dump(fits,savefile)
            pickle.dump(ii,savefile)
            pickle.dump(jj,savefile)
            savefile.close()
            os.rename(tmp_savefilename,savefilename)
            saving= False
            if interrupted:
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            if not saving:
                raise
            print "KeyboardInterrupt ignored while saving pickle ..."
            interrupted= True

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
    parser.add_option("--model",dest='model',default='twodblexp',
                      help="Model to fit")
    parser.add_option("-o","--plotfile",dest='plotfile',default=None,
                      help="Name of the file for plot")
    parser.add_option("-t","--type",dest='type',default='hr',
                      help="Quantity to plot")
    parser.add_option("--old",action="store_true", dest="old",
                      default=False,
                      help="If set, old, messed up fits list")
    return parser
  
if __name__ == '__main__':
    pixelFitDens(get_options())
