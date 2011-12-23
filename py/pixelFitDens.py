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
from segueSelect import read_gdwarfs, read_kdwarfs, _gi_gr, _mr_gi, \
    segueSelect, _GDWARFFILE, _KDWARFFILE
from fitSigz import _FAKEBIMODALGDWARFFILE
from selectFigs import _squeeze
from fitDensz import _TwoDblExpDensity, _HWRLikeMinus, _ZSUN, DistSpline, \
    _ivezic_dist, _NDS, cb, _HWRDensity, _HWRLike, _KGDensity
_NGR= 11
_NFEH=11
class pixelAfeFeh:
    """Class that pixelizes the data in afe and feh"""
    def __init__(self,data,dfeh=0.05,dafe=0.05,fehmin=-2.,fehmax=0.6,
                 afemin=-0.1,afemax=.6):
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
        self.fehmin= fehmin
        self.fehmax= fehmax
        self.afemin= afemin
        self.afemax= afemax
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

    def fehindx(self,feh):
        """
        NAME:
           fehindx
        PURPOSE:
           return the index corresponding to a FeH value
        INPUT:
           feh
        OUTPUT:
           index
        HISTORY:
           2011-08-19 - Written - Bovy (NYU)
        """
        return int(math.floor((feh-self.fehmin)/self.dfeh))

    def afeindx(self,afe):
        """
        NAME:
           afeindx
        PURPOSE:
           return the index corresponding to a AFe value
        INPUT:
           afe
        OUTPUT:
           index
        HISTORY:
           2011-08-19 - Written - Bovy (NYU)
        """
        return int(math.floor((afe-self.afemin)/self.dafe))

    def npixfeh(self):
        """Return the number of FeH pixels"""
        return len(self.fehedges)-1

    def npixafe(self):
        """Return the number of AFe pixels"""
        return len(self.afeedges)-1
    
def pixelFitDens(options,args):
    if options.sample.lower() == 'g':
        if options.select.lower() == 'program':
            raw= read_gdwarfs(_GDWARFFILE,logg=True,ebv=True,sn=True)
        elif options.select.lower() == 'fakebimodal':
            raw= read_gdwarfs(_FAKEBIMODALGDWARFFILE,
                              logg=True,ebv=True,sn=True)
            options.select= 'all'
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
    #Initial conditions?
    if not options.init is None and os.path.exists(options.init):#Load initial
        savefile= open(options.init,'rb')
        initfits= pickle.load(savefile)
        savefile.close()
    else:
        initfits= None

    #Sample?
    if options.mcsample:
        if ii < len(binned.fehedges)-1 and jj < len(binned.afeedges)-1:
            print "First do all of the fits ..."
            print "Returning ..."
            return None
        if os.path.exists(args[1]): #Load savefile
            savefile= open(args[1],'rb')
            samples= pickle.load(savefile)
            ii= pickle.load(savefile)
            jj= pickle.load(savefile)
            savefile.close()
        else:
            samples= []
            ii, jj= 0, 0
    #Set up model etc.
    if options.model.lower() == 'hwr':
        densfunc= _HWRDensity
        isDomainFinite=[[False,True],[False,True],[True,True]]
        domain=[[0.,4.6051701859880918],[0.,4.6051701859880918],[0.,1.]]
    if options.model.lower() == 'kg':
        densfunc= _KGDensity
        isDomainFinite=[[True,True],[True,True],[False,True],
                        [False,True]]
        domain=[[numpy.log(0.0027),numpy.log(54.)],
                [numpy.log(0.00027),numpy.log(3.*10.**2.)],
                [0.,4.6051701859880918],
                [0.,4.6051701859880918]]
    elif options.model.lower() == 'twodblexp':
        densfunc= _TwoDblExpDensity
        isDomainFinite=[[False,True],[False,True],[False,True],
                        [False,True],[True,True],[False,False]]
        domain=[[0.,4.6051701859880918],[0.,4.6051701859880918],
                [0.,4.6051701859880918],
                [0.,4.6051701859880918],[0.,1.],[0.,0.]]
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
            if not initfits is None:
                fehindx= binned.fehindx(binned.feh(ii))
                afeindx= binned.afeindx(binned.afe(jj))
                thisinitfit= initfits[afeindx+fehindx*binned.npixafe()]
            #Create XYZ and R
            R= ((8.-data.xc)**2.+data.yc**2.)**0.5
            #Confine to R-range?
            if not options.rmin is None and not options.rmax is None:
                dataindx= (R >= options.rmin)*\
                    (R < options.rmax)
                data= data[dataindx]
                R= R[dataindx]
            XYZ= numpy.zeros((len(data),3))
            XYZ[:,0]= data.xc
            XYZ[:,1]= data.yc
            XYZ[:,2]= data.zc+_ZSUN
            if len(data) < options.minndata:
                if options.mcsample: samples.append(None)
                else: fits.append(None)
                jj+= 1
                if jj == len(binned.afeedges)-1: 
                    jj= 0
                    ii+= 1
                    break
                continue               
            print binned.feh(ii), binned.afe(jj), len(data)
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
                if not initfits is None:
                    params= thisinitfit
                else:
                    params= numpy.array([numpy.log(0.5),numpy.log(3.),0.05])
            elif options.model.lower() == 'kg':
                if not initfits is None:
                    params= thisinitfit
                else:
                    sz= 30.
                    params= numpy.array([numpy.log(1350./sz**2.),
                                         numpy.log(270/sz**2.),
                                         numpy.log(0.5),
                                         numpy.log(2.75)])
            elif options.model.lower() == 'twodblexp':
                if not initfits is None:
                    params= numpy.array([thisinitfit[0],numpy.log(0.5),thisinitfit[1],numpy.log(2.5),0.001])
                else:
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
            if not options.mcsample:
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
                                                   False,dmin,dmax,ds,options),
                                             callback=cb)
                print numpy.exp(params)
                fits.append(params)
            else:
                #Load best-fit params
                params= fits[jj+ii*binned.npixafe()]
                thesesamples= bovy_mcmc.markovpy(params,
                                                 0.01,
                                                 pdf_func,
                                                 (XYZ,R,
                                                  sf,sf.plates,platelb[:,0],
                                                  platelb[:,1],platebright,
                                                  platefaint,1.,
                                                  grmin,grmax,rmin,rmax,
                                                  fehrange[0],fehrange[1],
                                                  feh,colordist,densfunc,
                                                  fehdist,False,
                                                  False,1.,
                                                  grs,fehs,rhogr,rhofeh,mr,
                                                  False,dmin,dmax,ds,options),
                                                 isDomainFinite=isDomainFinite,
                                                 domain=domain,
                                                 nsamples=options.nsamples)
                #Print some helpful stuff
                printthis= []
                for kk in range(len(params)):
                    xs= numpy.array([s[kk] for s in thesesamples])
                    printthis.append(0.5*(numpy.exp(numpy.mean(xs))-numpy.exp(numpy.mean(xs)-numpy.std(xs))-numpy.exp(numpy.mean(xs))+numpy.exp(numpy.mean(xs)+numpy.std(xs))))
                print printthis
                samples.append(thesesamples)
            jj+= 1
            if jj == len(binned.afeedges)-1: 
                jj= 0
                ii+= 1
            if options.mcsample: save_pickles(samples,ii,jj,args[1])
            else: save_pickles(fits,ii,jj,args[0])
            if jj == 0: #this means we've reset the counter 
                break
    if options.mcsample: save_pickles(samples,ii,jj,args[1])
    else: save_pickles(fits,ii,jj,args[0])
    return None

def plotPixelFit(options,args):
    if options.sample.lower() == 'g':
        if options.select.lower() == 'program':
            raw= read_gdwarfs(_GDWARFFILE,logg=True,ebv=True,sn=True)
        elif select.lower() == 'fakebimodal':
            raw= read_gdwarfs(_FAKEBIMODALGDWARFFILE,
                              logg=True,ebv=True,sn=True)
            options.select= 'all'
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
    #Savefile
    if os.path.exists(args[0]):#Load savefile
        savefile= open(args[0],'rb')
        fits= pickle.load(savefile)
        savefile.close()
    #Uncertainties are in savefile3 and 4
    if len(args) > 1 and os.path.exists(args[1]):
        savefile= open(args[1],'rb')
        denssamples= pickle.load(savefile)
        savefile.close()
        denserrors= True
    else:
        denssamples= None
        denserrors= False
    #If --mass is set to a filename, load the masses from that file
    #and use those for the symbol size
    if not options.mass is None and os.path.exists(options.mass):
        savefile= open(options.mass,'rb')
        mass= pickle.load(savefile)
        savefile.close()
        ndata= mass
        masses= True
    else:
        masses= False
    #Exclude bins better fit by two disks
    if not options.exclude is None and os.path.exists(options.exclude):#Load two double exponential fits
        savefile= open(options.exclude,'rb')
        twofits= pickle.load(savefile)
        savefile.close()
    else:
        twofits= None
    if not options.exclude_errs is None and os.path.exists(options.exclude_errs):#Load two double exponential fits errors
        savefile= open(options.exclude_errs,'rb')
        twosamples= pickle.load(savefile)
        savefile.close()
        twoerrors= True
    else:
        twosamples= None
        twoerrors= False
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
            if not twofits is None:
                #Exclude the bin if it is better fit by two exponential disks
                thistwofit= twofits[afeindx+fehindx*binned.npixafe()]
                if thistwofit[4] > 0.5: twoIndx= 1
                else: twoIndx= 0
                hz1= numpy.exp(thisfit[0])*1000.
                hz2= numpy.exp(thistwofit[twoIndx])*1000.
                if twoerrors:
                    thesetwosamples= twosamples[afeindx+fehindx*binned.npixafe()]
                    xs= numpy.array([s[twoIndx] for s in thesetwosamples])
                    hz2_err= 500.*(-numpy.exp(numpy.mean(xs)-numpy.std(xs))+numpy.exp(numpy.mean(xs)+numpy.std(xs)))
                    if numpy.fabs(hz1-hz2)/hz2_err > 1.:
                        continue
                elif numpy.fabs(hz1-hz2)/hz1 > 0.15:
                    continue
            if options.model.lower() == 'hwr':
                if options.type == 'hz':
                    plotthis[ii,jj]= numpy.exp(thisfit[0])*1000.
                elif options.type == 'hr':
                    plotthis[ii,jj]= numpy.exp(thisfit[1])
                elif options.type.lower() == 'afe' \
                        or options.type.lower() == 'feh' \
                        or options.type.lower() == 'fehafe' \
                        or options.type.lower() == 'afefeh':
                    if masses:
                        plotthis.append([tightbinned.feh(ii),
                                         tightbinned.afe(jj),
                                         numpy.exp(thisfit[0])*1000.,
                                         numpy.exp(thisfit[1]),
                                         mass[afeindx+fehindx*binned.npixafe()]])
                    else:
                        plotthis.append([tightbinned.feh(ii),
                                         tightbinned.afe(jj),
                                         numpy.exp(thisfit[0])*1000.,
                                         numpy.exp(thisfit[1]),
                                         len(data)])
                    if denserrors:
                        theseerrors= []
                        thesesamples= denssamples[afeindx+fehindx*binned.npixafe()]
                        if options.model.lower() == 'hwr':
                            for kk in [0,1]:
                                xs= numpy.array([s[kk] for s in thesesamples])
                                theseerrors.append(0.5*(-numpy.exp(numpy.mean(xs)-numpy.std(xs))+numpy.exp(numpy.mean(xs)+numpy.std(xs))))
                        errors.append(theseerrors)
    #Set up plot
    #print numpy.nanmin(plotthis), numpy.nanmax(plotthis)
    if options.type == 'hz':
        vmin, vmax= 180,1200
        zlabel=r'$\mathrm{vertical\ scale\ height\ [pc]}$'
    elif options.type == 'hr':
        vmin, vmax= 1.35,4.5
        zlabel=r'$\mathrm{radial\ scale\ length\ [kpc]}$'
    elif options.type == 'afe':
        vmin, vmax= 0.0,.5
        zlabel=r'$[\alpha/\mathrm{Fe}]$'
    elif options.type == 'feh':
        vmin, vmax= -1.5,0.2
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
        xrange=[-2.,0.6]
        yrange=[-0.1,0.6]
    if options.type.lower() == 'afe' or options.type.lower() == 'feh' \
            or options.type.lower() == 'fehafe' \
            or options.type.lower() == 'afefeh':
        bovy_plot.bovy_print(fig_height=3.87,fig_width=5.)
        #Gather hR and hz
        hz_err, hr_err, hz, hr,afe, feh, ndata= [], [], [], [], [], [], []
        for ii in range(len(plotthis)):
            if denserrors:
                hz_err.append(errors[ii][0]*1000.)
                hr_err.append(errors[ii][1])
            hz.append(plotthis[ii][2])
            hr.append(plotthis[ii][3])
            afe.append(plotthis[ii][1])
            feh.append(plotthis[ii][0])
            ndata.append(plotthis[ii][4])
        if denserrors:
            hz_err= numpy.array(hz_err)
            hr_err= numpy.array(hr_err)
        hz= numpy.array(hz)
        hr= numpy.array(hr)
        afe= numpy.array(afe)
        feh= numpy.array(feh)
        ndata= numpy.array(ndata)
        #Process ndata
        if not masses:
            ndata= ndata**.5
            ndata= ndata/numpy.median(ndata)*35.
        else:
#            ndata= numpy.log(ndata)
            ndata= _squeeze(ndata,numpy.amin(ndata),numpy.amax(ndata))
            ndata= ndata*200.+10.
        #ndata= numpy.log(ndata)/numpy.log(numpy.median(ndata))
        #ndata= (ndata-numpy.amin(ndata))/(numpy.amax(ndata)-numpy.amin(ndata))*25+12.
        print ndata
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
        yrange= [150,1250]
        xrange= [1.2,5.]
        bovy_plot.bovy_plot(hr,hz,s=ndata,c=plotc,
                            cmap='jet',
                            ylabel=r'$\mathrm{vertical\ scale\ height\ [pc]}$',
                            xlabel=r'$\mathrm{radial\ scale\ length\ [kpc]}$',
                            clabel=zlabel,
                            xrange=xrange,yrange=yrange,
                            vmin=vmin,vmax=vmax,
                            scatter=True,edgecolors='none',
                            colorbar=True,zorder=2)
        #Overplot errors
        if options.ploterrors:
            colormap = cm.jet
            for ii in range(len(hz)):
                if hr[ii] < 5.:
                    if (hz[ii]-((800.-520.)/(4.-2.5)*(hr[ii]-4.)+800.))**2./100.**2. < 1. and hr[ii] > 2.3:
                        print hr[ii], hz[ii], ndata[ii]/numpy.sum(ndata)
                        pyplot.errorbar(hr[ii],hz[ii],xerr=hr_err[ii],yerr=hz_err[ii],
                                        color=colormap(_squeeze(plotc[ii],
                                                                numpy.amax([numpy.amin(plotc)]),
                                                                numpy.amin([numpy.amax(plotc)]))),
                                        elinewidth=1.,capsize=3,zorder=0,elinestyle='--')
                        """
                        Might need to add this to axes.py
                    if elinestyle:
                        lines_kw['linestyle'] = elinestyle
                    else:
                        if 'linestyle' in kwargs:
                           lines_kw['linestyle']=kwargs['linstyle']
                        if 'ls' in kwargs:
                           lines_kw['ls']=kwargs['ls']
                         """
                    else:
                        pyplot.errorbar(hr[ii],hz[ii],xerr=hr_err[ii],yerr=hz_err[ii],
                                        color=colormap(_squeeze(plotc[ii],
                                                                numpy.amax([numpy.amin(plotc)]),
                                                                numpy.amin([numpy.amax(plotc)]))),
                                        elinewidth=1.,capsize=3,zorder=0)
        #Overplot upper limits in hR
        colormap = cm.jet
        for jj in range(len(hr)):
            if hr[jj] < 5.: continue
            pyplot.errorbar(4.8,hz[jj],xerr=0.1,xuplims=True,
                                    color=colormap(_squeeze(plotc[jj],
                                                            numpy.amax([numpy.amin(plotc)]),
                                                            numpy.amin([numpy.amax(plotc)]))),
#                            color=colormap(_squeeze(plotc[jj],vmin,vmax)),
                            elinewidth=1.,capsize=3)
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
    usage = "usage: %prog [options] <savefile> <savefile>\n\nsavefile= name of the file that the fits will be saved to\nsavefile = name of the file that the samples will be saved to (optional)"
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
                      help="Quantity to plot ('hz', 'hr', 'afe', 'feh'")
    parser.add_option("--plot",action="store_true", dest="plot",
                      default=False,
                      help="If set, plot, otherwise, fit")
    parser.add_option("--tighten",action="store_true", dest="tighten",
                      default=False,
                      help="If set, tighten axes")
    parser.add_option("--mcsample",action="store_true", dest="mcsample",
                      default=False,
                      help="If set, sample around the best fit, save in args[1]")
    parser.add_option("--ploterrors",action="store_true", dest="ploterrors",
                      default=False,
                      help="If set, plot the errorbars")
    parser.add_option("--nsamples",dest='nsamples',default=1000,type='int',
                      help="Number of MCMC samples to obtain")
    parser.add_option("--init",dest='init',default=None,
                      help="Initial conditions for fit from this file (same gridding and format as output file, assumed to be a single-exponential fit for double-exponential)")
    parser.add_option("--exclude",dest='exclude',default=None,
                      help="Exclude bins better fit with two exponential disks, the two disk-fits are in this file")
    parser.add_option("--exclude_errs",dest='exclude_errs',default=None,
                      help="Exclude bins better fit with two exponential disks, the two disk-fits are in this file; errors")
    parser.add_option("--mass",dest='mass',default=None,
                      help="If set, use the masses from this file as the symbol size")
    parser.add_option("--zmin",dest='zmin',type='float',
                      default=None,
                      help="Minimum height")
    parser.add_option("--zmax",dest='zmax',type='float',
                      default=None,
                      help="Maximum height")
    parser.add_option("--rmin",dest='rmin',type='float',
                      default=None,
                      help="Minimum radius")
    parser.add_option("--rmax",dest='rmax',type='float',
                      default=None,
                      help="Maximum radius")
    return parser
  
if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    if options.plot:
        plotPixelFit(options,args)
    else:
        pixelFitDens(options,args)
