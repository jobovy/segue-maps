import os, os.path
import sys
import copy
import math
import numpy
import cPickle as pickle
from optparse import OptionParser
from extreme_deconvolution import extreme_deconvolution
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
            if options.simpleage:
                agemin, agemax= 0.5, 10.
            else:
                if afe > 0.25: agemin, agemax= 7.,10.
                else: agemin,agemax= 1.,8.
            if options.mcsample:
                #Loop over samples
                thissamples= denssamples[afeindx+fehindx*binned.npixafe()]
                if options.nsamples < len(thissamples):
                    #Random permutation
                    thissamples= numpy.random.permutation(thissamples)[0:options.nsamples]
                thismasssamples= []
                print "WARNING: DISK MASS IN CALCMASS ONLY FOR G COLORS"
                for kk in range(len(thissamples)):
                    thisparams= thissamples[kk]
                    thismasssamples.append(predictDiskMass(densfunc,
                                                           thisparams,sf,
                                                           colordist,fehdist,
                                                           fehrange[0],
                                                           fehrange[1],feh,
                                                           data,0.45,
                                                           0.58,
                                                           agemin,agemax,
                                                           normalize=options.normalize))
                #Print some stuff
                print numpy.mean(numpy.array(thismasssamples)), numpy.std(numpy.array(thismasssamples))
                masssamples.append(thismasssamples)
            else:
                thisparams= fits[afeindx+fehindx*binned.npixafe()]
                print "WARNING: DISK MASS IN CALCMASS ONLY FOR G COLORS"
                mass.append(predictDiskMass(densfunc,
                                            thisparams,sf,
                                            colordist,fehdist,
                                            fehrange[0],
                                            fehrange[1],feh,
                                            data,0.45,
                                            0.58,
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
                                 fehmin=-1.6,fehmax=0.5,afemin=-0.05,
                                 afemax=0.55)
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
    #Mass uncertainties are in savefile3
    if len(args) > 2 and os.path.exists(args[2]):
        savefile= open(args[2],'rb')
        masssamples= pickle.load(savefile)
        savefile.close()
        masserrors= True
    else:
        masssamples= None
        masserrors= False
    if len(args) > 3 and os.path.exists(args[3]): #Load savefile
        savefile= open(args[3],'rb')
        denssamples= pickle.load(savefile)
        savefile.close()
        denserrors= True
    else:
        denssamples= None
        denserrors= False
    #Now plot
    #Run through the pixels and gather
    if options.type.lower() == 'afe' or options.type.lower() == 'feh' \
            or options.type.lower() == 'fehafe' \
            or options.type.lower() == 'afefeh':
        plotthis= []
    else:
        plotthis= numpy.zeros((tightbinned.npixfeh(),tightbinned.npixafe()))
    if denserrors: errors= []
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
            if masserrors:
                thismasssamples= masssamples[afeindx+fehindx*binned.npixafe()]
            else:
                thismasssamples= None
            thisfit= fits[afeindx+fehindx*binned.npixafe()]
            if denserrors:
                thisdenssamples= denssamples[afeindx+fehindx*binned.npixafe()]
            else:
                thisdenssamples= None
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
                if not options.distzmin is None or not options.distzmax is None:
                    if options.distzmin is None and not options.distzmax is None:
                        thismass*= (1.-numpy.exp(-options.distzmax/numpy.exp(thisfit[0])/1000.))
                    elif not options.distzmin is None and options.distzmax is None:
                        thismass*= numpy.exp(-options.distzmin/numpy.exp(thisfit[0])/1000.)
                    else:
                        thismass*= (numpy.exp(-options.distzmin/numpy.exp(thisfit[0])/1000.)-numpy.exp(-options.distzmax/numpy.exp(thisfit[0])/1000.))
                if options.logmass:
                    plotthis[ii,jj]= numpy.log10(thismass/10**6.)
                else:
                    plotthis[ii,jj]= thismass/10**6.
            elif options.type == 'nstars':
                if options.logmass:
                    plotthis[ii,jj]= numpy.log10(len(data))
                else:
                    plotthis[ii,jj]= len(data)
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
                                     thismass/10.**6.,
                                     thismasssamples])
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
    if options.type == 'mass':
        if options.logmass:
            vmin, vmax= numpy.log10(0.01), numpy.log10(2.)
            zlabel=r'$\log_{10} \Sigma(R_0)\ [M_{\odot}\ \mathrm{pc}^{-2}]$'
        else:
            vmin, vmax= 0.,1.
            zlabel=r'$\Sigma(R_0)\ [M_{\odot}\ \mathrm{pc}^{-2}]$'
            if not options.distzmin is None or not options.distzmax is None:
                vmin, vmax= None, None
        title=r'$\mathrm{mass\ weighted}$'
    elif options.type == 'nstars':
        if options.logmass:
            vmin, vmax= 2., 3.
            zlabel=r'$\log_{10} \mathrm{raw\ number\ of\ G}$-$\mathrm{type\ dwarfs}$'
        else:
            vmin, vmax= 100.,1000.
            zlabel=r'$\mathrm{raw\ number\ of\ G}$-$\mathrm{type\ dwarfs}$'
        title= r'$\mathrm{raw\ sample\ counts}$'
    elif options.type == 'afe':
        vmin, vmax= 0.0,.5
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
        hz_err, hr_err, mass_err, mass, hz, hr,afe, feh, ndata= [], [], [], [], [], [], [], [], []
        for ii in range(len(plotthis)):
            if denserrors:
                hz_err.append(errors[ii][0]*1000.)
                hr_err.append(errors[ii][1])
            mass.append(plotthis[ii][5])
            hz.append(plotthis[ii][2])
            hr.append(plotthis[ii][3])
            afe.append(plotthis[ii][1])
            feh.append(plotthis[ii][0])
            ndata.append(plotthis[ii][4])
            if masserrors:
                mass_err.append(numpy.std(numpy.array(plotthis[ii][6])/10.**6.))
                """
                if options.logmass:
                    mass_err.append(numpy.std(numpy.log10(numpy.array(plotthis[ii][6])/10.**6.)))
                else:
                    mass_err.append(numpy.std(numpy.array(plotthis[ii][6])/10.**6.))
                """
        if denserrors:
            hz_err= numpy.array(hz_err)
            hr_err= numpy.array(hr_err)
        mass= numpy.array(mass)
        if masserrors:
            mass_err= numpy.array(mass_err)
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
            #Print total surface mass and uncertainty
            totmass= numpy.sum(mass)
            toterr= numpy.sqrt(numpy.sum(mass_err**2.))
            print "Total surface-mass density: %4.1f +/- %4.2f" %(totmass,toterr)
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
                yrange= [0.005,2.]
            else:
                yrange= [-0.1,10.]
            ylabel=r'$\Sigma_{R_0}(h_z)\ [M_{\odot}\ \mathrm{pc}^{-2}]$'
        if not options.vstructure and not options.hzhr:
            if options.hr:
                ploth= hr
                plotherr= hr_err
                xlabel=r'$\mathrm{radial\ scale\ length\ [kpc]}$'
                xrange= [1.2,4.]
            else:
                ploth= hz
                plotherr= hz_err
                xlabel=r'$\mathrm{vertical\ scale\ height}\ h_z\ \mathrm{[pc]}$'
                xrange= [150,1200]
            bovy_plot.bovy_plot(ploth,mass,
                                s=ndata,c=plotc,
                                cmap='jet',
                                xlabel=xlabel,
                                ylabel=ylabel,
                                clabel=zlabel,
                                xrange=xrange,yrange=yrange,
                                vmin=vmin,vmax=vmax,
                                scatter=True,edgecolors='none',
                                colorbar=True,zorder=2,
                                semilogy=options.logmass)
            if not options.cumul and masserrors and options.ploterrors:
                colormap = cm.jet
                for ii in range(len(hz)):
                    pyplot.errorbar(ploth[ii],mass[ii],yerr=mass_err[ii],
                                    color=colormap(_squeeze(plotc[ii],
                                                            numpy.amax([vmin,
                                                                        numpy.amin(plotc)]),
                                                            numpy.amin([vmax,
                                                                        numpy.amax(plotc)]))),
                                    elinewidth=1.,capsize=3,zorder=0)
            if not options.cumul and denserrors and options.ploterrors:
                colormap = cm.jet
                for ii in range(len(hz)):
                    pyplot.errorbar(ploth[ii],mass[ii],xerr=plotherr[ii],
                                    color=colormap(_squeeze(plotc[ii],
                                                            numpy.amax([vmin,
                                                                        numpy.amin(plotc)]),
                                                            numpy.amin([vmax,
                                                                        numpy.amax(plotc)]))),
                                    elinewidth=1.,capsize=3,zorder=0)
            #Add binsize label
            bovy_plot.bovy_text(r'$\mathrm{points\ use}\ \Delta [\mathrm{Fe/H}] = 0.1,$'+'\n'+r'$\Delta [\alpha/\mathrm{Fe}] = 0.05\ \mathrm{bins}$',
                                bottom_left=True)
            #Overplot histogram
            ax2 = pyplot.twinx()
            pyplot.hist(ploth,range=xrange,weights=mass,color='k',histtype='step',
                        normed=True,bins=10,lw=3.,zorder=10)
            #Also XD?
            if options.xd:
                #Set up data
                ydata= numpy.zeros((len(hz),1))
                ydata[:,0]= numpy.log(hz)
                ycovar= numpy.zeros((len(hz),1))
                ycovar[:,0]= hz_err**2./hz**2.
                #Set up initial conditions
                xamp= numpy.ones(options.k)/float(options.k)
                xmean= numpy.zeros((options.k,1))
                for kk in range(options.k):
                    xmean[kk,:]= numpy.mean(ydata,axis=0)\
                        +numpy.random.normal()*numpy.std(ydata,axis=0)
                xcovar= numpy.zeros((options.k,1,1))
                for kk in range(options.k):
                    xcovar[kk,:,:]= numpy.cov(ydata.T)
                #Run XD
                print extreme_deconvolution(ydata,ycovar,xamp,xmean,xcovar,
                                      weight=mass)*len(hz)
                print xamp, xmean, xcovar
                #Plot
                xs= numpy.linspace(xrange[0],xrange[1],1001)
                xdys= numpy.zeros(len(xs))
                for kk in range(options.k):
                    xdys+= xamp[kk]/numpy.sqrt(2.*numpy.pi*xcovar[kk,0,0])\
                        *numpy.exp(-0.5*(numpy.log(xs)-xmean[kk,0])**2./xcovar[kk,0,0])
                xdys/= xs
                bovy_plot.bovy_plot(xs,xdys,'-',color='0.5',overplot=True)
            ax2.set_yscale('log')
            ax2.set_yticklabels('')        
            if options.hr:
                pyplot.ylim(10**-2.,10.**0.)
            else:
                pyplot.ylim(10**-5.5,10.**-1.5)
            pyplot.xlim(xrange[0],xrange[1])
        elif options.hzhr:
            #Make density plot in hR and hz
            bovy_plot.scatterplot(hr,hz,'k,',
                                  levels=[1.01],#HACK such that outliers aren't plotted
                                  cmap='gist_yarg',
                                  bins=11,
                                  xrange=[1.,7.],
                                  yrange=[150.,1200.],
                                  ylabel=r'$\mathrm{vertical\ scale\ height\ [pc]}$',
                                  xlabel=r'$\mathrm{radial\ scale\ length\ [kpc]}$',
                                  onedhists=False,
                                  weights=mass)
        else:
            #Make an illustrative plot of the vertical structure
            nzs= 1001
            zs= numpy.linspace(200.,3000.,nzs)
            total= numpy.zeros(nzs)
            for ii in range(len(hz)):
                total+= mass[ii]/2./hz[ii]*numpy.exp(-zs/hz[ii])
            bovy_plot.bovy_plot(zs,total,color='k',ls='-',lw=3.,
                                semilogy=True,
                                xrange=[0.,3200.],
                                yrange=[0.000001,0.02],
                                xlabel=r'$\mathrm{vertical\ height}\ Z$',
                                ylabel=r'$\rho_*(R=R_0,Z)\ [\mathrm{M}_\odot\ \mathrm{pc}^{-3}]$',
                                zorder=10)
            if options.vbinned:
                #Bin
                mhist, edges= numpy.histogram(hz,range=xrange,
                                              weights=mass,bins=10)
                stotal= numpy.zeros(nzs)
                for ii in range(len(mhist)):
                    hz= (edges[ii+1]+edges[ii])/2.
                    if options.vcumul:
                        if ii == 0.:
                            pstotal= numpy.zeros(nzs)+0.0000001
                        else:
                            pstotal= copy.copy(stotal)
                        stotal+= mhist[ii]/2./hz*numpy.exp(-zs/hz)
                        pyplot.fill_between(zs,stotal,pstotal,
                                            color='%.6f' % (0.25+0.5/(len(mhist)-1)*ii))
                    else:
                        bovy_plot.bovy_plot([zs[0],zs[-1]],
                                            1.*numpy.array([mhist[ii]/2./hz*numpy.exp(-zs[0]/hz),
                                                            mhist[ii]/2./hz*numpy.exp(-zs[-1]/hz)]),
                                            color='0.5',ls='-',overplot=True,
                                            zorder=0)
            else:
                colormap = cm.jet
                for ii in range(len(hz)):
                    bovy_plot.bovy_plot([zs[0],zs[-1]],
                                        100.*numpy.array([mass[ii]/2./hz[ii]*numpy.exp(-zs[0]/hz[ii]),
                                                          mass[ii]/2./hz[ii]*numpy.exp(-zs[-1]/hz[ii])]),
                                        ls='-',overplot=True,alpha=0.5,
                                        zorder=0,
                                        color=colormap(_squeeze(plotc[ii],
                                                                numpy.amax([vmin,
                                                                        numpy.amin(plotc)]),
                                                                numpy.amin([vmax,
                                                                            numpy.amax(plotc)]))))
    else:
        bovy_plot.bovy_print()
        bovy_plot.bovy_dens2d(plotthis.T,origin='lower',cmap='gist_yarg',
                              interpolation='nearest',
                              xlabel=r'$[\mathrm{Fe/H}]$',
                              ylabel=r'$[\alpha/\mathrm{Fe}]$',
                              zlabel=zlabel,
                              xrange=xrange,yrange=yrange,
                              vmin=vmin,vmax=vmax,
                              onedhists=True,
                              contours=False)
        bovy_plot.bovy_text(title,top_right=True,fontsize=16)
        if not options.distzmin is None or not options.distzmax is None:
            if options.distzmin is None:
                distlabel= r'$|Z| < %i\ \mathrm{pc}$' % int(options.distzmax)
            elif options.distzmax is None:
                distlabel= r'$|Z| > %i\ \mathrm{pc}$' % int(options.distzmin)
            else:
                distlabel= r'$%i < |Z| < %i\ \mathrm{pc}$' % (int(options.distzmin),int(options.distzmax))
            bovy_plot.bovy_text(distlabel,bottom_left=True,fontsize=16)
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
    parser.add_option("--vstructure",action="store_true", dest="vstructure",
                      default=False,
                      help="If set, plot the vertical structure of the disk")
    parser.add_option("--vbinned",action="store_true", dest="vbinned",
                      default=False,
                      help="If set, plot the vertical structure in bins in hz when plotting vstructure")
    parser.add_option("--vcumul",action="store_true", dest="vcumul",
                      default=False,
                      help="If set, plot the vertical structure cumulatively")
    parser.add_option("--xd",action="store_true", dest="xd",
                      default=False,
                      help="If set, also fit XD to the distribution and plot")
    parser.add_option("-k",dest='k',default=4,type='int',
                      help="Number of XD Gaussians")   
    parser.add_option("--simpleage",action="store_true", dest="simpleage",
                      default=False,
                      help="If set, use a simple age prescription (all the same marginalization)")
    parser.add_option("--hr",action="store_true", dest="hr",
                      default=False,
                      help="If set, plot hR rather than hz")
    parser.add_option("--hzhr",action="store_true", dest="hzhr",
                      default=False,
                      help="If set, plot both hR and hz")
    parser.add_option("--distzmin",dest='distzmin',type='float',
                      default=None,
                      help="Plot the mass-weighted distriution from this zmin (pc)")
    parser.add_option("--distzmax",dest='distzmax',type='float',
                      default=None,
                      help="Plot the mass-weighted distriution to this zmax (pc)")  
    return parser
  
if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    if options.plot:
        plotMass(options,args)
    else:
        calcMass(options,args)

