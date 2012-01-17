import os, os.path
import sys
import math
import numpy
import cPickle as pickle
from scipy import special, linalg
from scipy.stats import gaussian_kde
from optparse import OptionParser
from galpy.util import bovy_coords, bovy_plot, save_pickles
from matplotlib import pyplot, cm
from matplotlib.patches import Ellipse
from segueSelect import read_gdwarfs, read_kdwarfs, _GDWARFFILE, _KDWARFFILE
from pixelFitDens import pixelAfeFeh
from fitSigz import _ZSUN
class kde_mult:
    def __init__(self,kde_list):
        """Input: list of kde instances, function is product of all"""
        self._kde_list= kde_list
    def __call__(self,x):
        return numpy.prod(numpy.array([l(x) for l in self._kde_list]))
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
                                 fehmin=-1.6,fehmax=0.5,afemin=-0.05,
                                 afemax=0.55)
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
    #Uncertainties are in savefile3 and 4
    if len(args) > 3 and os.path.exists(args[3]):
        savefile= open(args[3],'rb')
        denssamples= pickle.load(savefile)
        savefile.close()
        denserrors= True
    else:
        denssamples= None
        denserrors= False
    if len(args) > 2 and os.path.exists(args[2]):
        savefile= open(args[2],'rb')
        velsamples= pickle.load(savefile)
        savefile.close()
        velerrors= True
    else:
        velsamples= None            
        velerrors= False
    #Now plot
    #Run through the pixels and gather
    if options.type.lower() == 'afe' or options.type.lower() == 'feh' \
            or options.type.lower() == 'fehafe' \
            or options.type.lower() == 'zfunc' \
            or options.type.lower() == 'afefeh':
        plotthis= []
        errors= []
    else:
        plotthis= numpy.zeros((tightbinned.npixfeh(),tightbinned.npixafe()))
    if options.kde: allsamples= []
    sausageFehAfe= [[-0.85,0.425],[-0.45,0.275],[-0.15,0.075]]
    if options.subtype.lower() == 'sausage':
        sausageSamples= []
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
                if options.type == 'sz2hz' or options.type.lower() == 'asz' \
                       or options.type.lower() == 'bsz':
                    numerator= numpy.exp(2.*thisvelfit[1])
                elif options.type == 'slopes': 
                    if velerrors: #Don't plot if errors > 20%
                        sz= numpy.exp(thisvelfit[1])
                        thesesamples= velsamples[afeindx+fehindx*binned.npixafe()]
                        xs= numpy.array([s[1] for s in thesesamples])
                        sz_err= 0.5*(numpy.exp(numpy.mean(xs))-numpy.exp(numpy.mean(xs)-numpy.std(xs))-numpy.exp(numpy.mean(xs))+numpy.exp(numpy.mean(xs)+numpy.std(xs)))
                        if sz_err/sz > .2:
                            plotthis[ii,jj]= numpy.nan
                        else:
                            plotthis[ii,jj]= thisvelfit[2]
                elif options.type.lower() == 'afe' \
                        or options.type.lower() == 'feh' \
                        or options.type.lower() == 'fehafe' \
                        or options.type.lower() == 'zfunc' \
                        or options.type.lower() == 'afefeh':
                    thisplot=[tightbinned.feh(ii),
                              tightbinned.afe(jj),
                              numpy.exp(thisvelfit[1]),
                              numpy.exp(thisvelfit[4]),
                              len(data),
                              thisvelfit[2],
                              thisvelfit[3]]
                    if thisvelfit[2] > 10.:
                        print "Strange point: ", tightbinned.feh(ii), \
                            tightbinned.afe(jj)
                    #Als find min and max z for this data bin, and median
                    zsorted= sorted(numpy.fabs(data.zc+_ZSUN))
                    zmin= zsorted[int(numpy.ceil(0.16*len(zsorted)))]
                    zmax= zsorted[int(numpy.floor(0.84*len(zsorted)))]
                    thisplot.extend([zmin,zmax,numpy.mean(numpy.fabs(data.zc+_ZSUN))])
                    #Errors
                    if velerrors:
                        theseerrors= []
                        thesesamples= velsamples[afeindx+fehindx*binned.npixafe()]
                        for kk in [1,4]:
                            xs= numpy.array([s[kk] for s in thesesamples])
                            theseerrors.append(0.5*(numpy.exp(numpy.mean(xs))-numpy.exp(numpy.mean(xs)-numpy.std(xs))-numpy.exp(numpy.mean(xs))+numpy.exp(numpy.mean(xs)+numpy.std(xs))))
                        xs= numpy.array([s[4] for s in thesesamples])
                        theseerrors.append(0.5*(numpy.exp(-numpy.mean(xs))-numpy.exp(numpy.mean(-xs)-numpy.std(-xs))-numpy.exp(numpy.mean(-xs))+numpy.exp(numpy.mean(-xs)+numpy.std(-xs))))
                        #theseerrors.append(0.5*(numpy.exp(-numpy.mean(xs))-numpy.exp(numpy.mean(-xs)-numpy.std(-xs))-numpy.exp(numpy.mean(-xs))+numpy.exp(numpy.mean(-xs)+numpy.std(-xs))))
                        if options.kde and \
                                (options.subtype.lower() == 'hs' \
                                     or options.subtype.lower() == 'hsm'): \
                            allsamples.append(numpy.exp(-xs))
                        elif options.kde and (options.subtype.lower() == 'slopequad' \
                                                  or options.subtype.lower() == 'slopequadquantiles'):
                            xs= numpy.array([s[3] for s in thesesamples])
                            ys= numpy.array([s[2] for s in thesesamples])
                            allsamples.append(numpy.array([xs,ys]))
                        xs= numpy.array([s[2] for s in thesesamples])
                        theseerrors.append(numpy.std(xs))
                        xs= numpy.array([s[3] for s in thesesamples])
                        theseerrors.append(numpy.std(xs))
                        if options.subtype.lower() == 'slopehsm':
                            xs= numpy.array([s[2] for s in thesesamples])
                            ys= numpy.exp(-numpy.array([s[4] for s in thesesamples]))
                            theseerrors.append(numpy.corrcoef(xs,ys)[0,1])
                        elif (options.subtype.lower() == 'slopequad' \
                                  or options.subtype.lower() == 'slopequadquantiles'):
                            xs= numpy.array([s[2] for s in thesesamples])
                            ys= numpy.array([s[3] for s in thesesamples])
                            theseerrors.append(numpy.corrcoef(xs,ys)[0,1])
                        elif options.subtype.lower() == 'slopesz':
                            xs= numpy.array([s[2] for s in thesesamples])
                            ys= numpy.exp(numpy.array([s[1] for s in thesesamples]))
                            theseerrors.append(numpy.corrcoef(xs,ys)[0,1])
                        elif options.subtype.lower() == 'szhsm':
                            xs= numpy.exp(-numpy.array([s[4] for s in thesesamples]))
                            ys= numpy.exp(numpy.array([s[1] for s in thesesamples]))
                            theseerrors.append(numpy.corrcoef(xs,ys)[0,1])
                        errors.append(theseerrors)
                        if options.kde and \
                                     options.subtype.lower() == 'slope':
                                 allsamples.append(xs)
                        if thisvelfit[2] > 10.:
                            xs= numpy.array([s[2] for s in thesesamples])
                            strerr= 0.5*(numpy.exp(numpy.mean(xs))-numpy.exp(numpy.mean(xs)-numpy.std(xs))-numpy.exp(numpy.mean(xs))+numpy.exp(numpy.mean(xs)+numpy.std(xs)))
                            print "Strange errors: ", thisvelfit[2], strerr
                            #Remove
                            errors.pop()
                            continue
                        if options.subtype.lower() == 'sausage':
                            indx= [True for kk in range(len(sausageFehAfe)) if numpy.fabs(tightbinned.feh(ii)-sausageFehAfe[kk][0])< 0.01 and numpy.fabs(tightbinned.afe(jj) - sausageFehAfe[kk][1]) < 0.01]
                            if len(indx) == 1:
                                print tightbinned.feh(ii), tightbinned.afe(jj)
                                sausageSamples.append(thesesamples)
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
            elif options.densmodel.lower() == 'kg':
                if options.type.lower() == 'asz':
                    plotthis[ii,jj]= numerator*numpy.exp(thisdensfit[0])/1000.
                elif options.type.lower() == 'bsz':
                    plotthis[ii,jj]= numerator*numpy.exp(thisdensfit[1])/10.**6.
                elif options.type.lower() == 'afe' \
                        or options.type.lower() == 'feh' \
                        or options.type.lower() == 'fehafe' \
                        or options.type.lower() == 'afefeh':
                    thisplot.extend([numpy.exp(thisdensfit[0])/1000.,
                                     numpy.exp(thisdensfit[1])/10.**6.,
                                     numpy.median(numpy.fabs(data.zc)-_ZSUN)])
                    plotthis.append(thisplot)
    #Set up plot
    #print numpy.nanmin(plotthis), numpy.nanmax(plotthis)
    if options.type == 'sz2hz':
        plotthis/= 2.*numpy.pi*4.302*10.**-3 #2piG
        print numpy.nanmin(plotthis), numpy.nanmax(plotthis)
        vmin, vmax= 15.,100.
        zlabel= r'$\sigma_z^2(z=1000\ \mathrm{pc}) / h_z\ [M_\odot\ \mathrm{pc}^{-2}]$'
    elif options.type == 'asz':
        plotthis/= 2.*numpy.pi*4.302*10.**-3 #2piG
        print numpy.nanmin(plotthis), numpy.nanmax(plotthis)
        vmin, vmax= 15.,100.
        zlabel= r'$\Sigma_{\mathrm{disk}}\ [M_\odot\ \mathrm{pc^{-2}}]$'
    elif options.type == 'bsz':
        plotthis/= 2.*numpy.pi*4.302*10.**-3 #2piG
        print numpy.nanmin(plotthis), numpy.nanmax(plotthis)
        vmin, vmax= 0.,.02
        zlabel= r'$\rho_{\mathrm{DM}}(z=0)\ [M_\odot\ \mathrm{pc^{-3}}]$'
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
        xrange=[-2.,0.6]
        yrange=[-0.1,0.6]
    if options.type.lower() == 'afe' or options.type.lower() == 'feh' \
            or options.type.lower() == 'fehafe' \
            or options.type.lower() == 'afefeh':
        bovy_plot.bovy_print(fig_height=3.87,fig_width=5.)
        #Gather everything
        p2_err, p1hsm_corr, p1_err, hsm_err, hs_err, sz_err, pivot, zmin, zmax, mz, p1, p2, sz, hs, hz, hr,afe, feh, ndata= [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        for ii in range(len(plotthis)):
            if velerrors:
                sz_err.append(errors[ii][0])
                hs_err.append(errors[ii][1])
                hsm_err.append(errors[ii][2])
                p1_err.append(errors[ii][3])
                p2_err.append(errors[ii][4])
                if options.subtype.lower() == 'slopehsm' \
                        or options.subtype.lower() == 'slopequad' \
                        or options.subtype.lower() == 'slopequadquantiles' \
                        or options.subtype.lower() == 'slopesz' \
                        or options.subtype.lower() == 'szhsm':
                    p1hsm_corr.append(errors[ii][5])
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
        p1= numpy.array(p1)
        p2= numpy.array(p2)
        zmin= numpy.array(zmin)
        zmax= numpy.array(zmax)
        if velerrors:
            sz_err= numpy.array(sz_err)
            hs_err= numpy.array(hs_err)
            hsm_err= numpy.array(hsm_err)
            p1_err= numpy.array(p1_err)
            p2_err= numpy.array(p2_err)
            if options.subtype.lower() == 'slopehsm' \
                    or options.subtype.lower() == 'slopequad' \
                    or options.subtype.lower() == 'slopequadquantiles' \
                    or options.subtype.lower() == 'slopesz' \
                    or options.subtype.lower() == 'szhsm':
                p1hsm_corr= numpy.array(p1hsm_corr)
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
        if not options.subtype == 'zfunc' \
               and not options.subtype.lower() == 'hs' \
               and not options.subtype.lower() == 'sausage' \
               and not options.subtype.lower() == 'hsm' \
               and not options.subtype.lower() == 'slope' \
               and not options.subtype.lower() == 'slopehsm' \
               and not options.subtype.lower() == 'slopequad' \
               and not options.subtype.lower() == 'slopequadquantiles' \
               and not options.subtype.lower() == 'szhsm' \
               and not options.subtype.lower() == 'slopesz' \
               and not options.subtype.lower() == 'asz' \
               and not options.subtype.lower() == 'bsz':
            if options.subtype == 'mz':
                if velerrors: #Don't plot if errors > 20%
                    indx= (sz_err/sz <= .2)
                    sz= sz[indx]
                    mz= mz[indx]
                    hz= hz[indx]
                    pivot= pivot[indx]
                    ndata= ndata[indx]
                    plotc= plotc[indx]
                    p1= p1[indx]
                    p2= p2[indx]
                yrange= [0,130.]
                xrange= [0,1500]
                plotx= mz
                ploty= (sz+(mz/1000.-pivot)*p1+p2*(mz/1000.-pivot)**2.)**2./hz/2./numpy.pi/4.302/10**-3.
                xlabel=r'$\mathrm{median\ \ height}\ z_{1/2}\ \mathrm{[pc]}$'
                ylabel=r'$\sigma_z^2(z = z_{1/2}) / h_z\ [M_\odot\ \mathrm{pc}^{-2}]$'
            elif options.subtype == 'hz':
                if velerrors: #Don't plot if errors > 30%
                    indx= (sz_err/sz <= .2)
                    sz= sz[indx]
                    hz= hz[indx]
                    pivot= pivot[indx]
                    ndata= ndata[indx]
                    plotc= plotc[indx]
                    p1= p1[indx]
                    p2= p2[indx]
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
        elif options.subtype.lower() == 'zfunc':
            from selectFigs import _squeeze
            colormap = cm.jet
            #Set up plot
            if options.vr:
                yrange= [30.,80.]
                ylabel=r'$\sigma_R(z)\ [\mathrm{km\ s}^{-1}]$'
            else:
                yrange= [0.,60.]
                ylabel=r'$\sigma_z(z)\ [\mathrm{km\ s}^{-1}]$'
            bovy_plot.bovy_plot([-100.,-100.],[100.,100.],'k,',
                                xrange=[0,2700],yrange=yrange,
                                xlabel=r'$|z|\ [\mathrm{pc}]$',
                                ylabel=ylabel)
            #Calculate and plot all zfuncs
            for ii in numpy.random.permutation(len(afe)):
                if velerrors: #Don't plot if errors > 30%
                    if sz_err[ii]/sz[ii] > .2: continue
                    #if sz_err[ii] > 20.: continue
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
        elif options.subtype.lower() == 'sausage':
            from selectFigs import _squeeze
            colormap = cm.jet
            #Set up plot
            if options.vr:
                yrange= [30.,80.]
                ylabel=r'$\sigma_R(z)\ [\mathrm{km\ s}^{-1}]$'
            else:
                yrange= [0.,60.]
                ylabel=r'$\sigma_z(z)\ [\mathrm{km\ s}^{-1}]$'
            bovy_plot.bovy_plot([-100.,-100.],[100.,100.],'k,',
                                xrange=[0,2700],yrange=yrange,
                                xlabel=r'$|z|\ [\mathrm{pc}]$',
                                ylabel=ylabel)
            #Calculate and plot all zfuncs
            nsigs= 1
            nsamples= 12
            for kk in range(len(sausageFehAfe)):
                #Find index corresponding to this kk
                indx= (numpy.fabs(feh-sausageFehAfe[kk][0]) < 0.01)*(numpy.fabs(afe - sausageFehAfe[kk][1]) < 0.01)
                ii= (list(indx)).index(True)
                if velerrors: #Don't plot if errors > 30%
                    if sz_err[ii]/sz[ii] > .2: 
                        print "sausage %i has large errors" % ii
                        continue
                    #if sz_err[ii] > 20.: continue
                ds= numpy.linspace(zmin[ii]*1000.,zmax[ii]*1000.,1001)/1000.-pivot[ii]
                thiszfunc= sz[ii]+p1[ii]*ds+p2[ii]*ds**2.
                zs= numpy.linspace(zmin[ii]*1000.,1000*zmax[ii],1001)
                pyplot.plot(numpy.linspace(zmin[ii]*1000.,1000*zmax[ii],1001),
                            thiszfunc,'-',
                            color=colormap(_squeeze(plotc[ii],vmin,vmax)),
                            lw=ndata[ii]/15.)
                #Plot some samples
                thesesamples= sausageSamples[kk]
                randIndx= numpy.random.permutation(len(thesesamples))
                for ll in range(nsamples):
                    thissample= thesesamples[randIndx[ll]]
                    thiszfunc= numpy.exp(thissample[1])+thissample[2]*ds+thissample[3]*ds**2.
                    pyplot.plot(numpy.linspace(zmin[ii]*1000.,1000*zmax[ii],1001),
                                thiszfunc,'-',
                                color=colormap(_squeeze(plotc[ii],vmin,vmax)),
                                lw=0.3,
                                zorder=0)
                if options.envelope:
                    #Also calculate envelope
                    fs= numpy.zeros((len(ds),len(thesesamples)))
                    for ll in range(len(thesesamples)):
                        fs[:,ll]= numpy.exp(thesesamples[ll][1])\
                            +thesesamples[ll][2]*ds\
                            +thesesamples[ll][3]*ds**2.
                    zsigs= numpy.zeros((len(ds),2*nsigs))
                    for ll in range(nsigs):
                        for jj in range(len(ds)):
                            thisf= sorted(fs[jj,:])
                            thiscut= 0.5*special.erfc((ll+1.)/math.sqrt(2.))
                            zsigs[jj,2*ll]= thisf[int(math.floor(thiscut*len(thesesamples)))]
                            thiscut= 1.-thiscut
                            zsigs[jj,2*ll+1]= thisf[int(math.floor(thiscut*len(thesesamples)))]
                    pyplot.fill_between(zs,zsigs[:,0],zsigs[:,1],
                                        color=colormap(_squeeze(plotc[ii],vmin,vmax)))
                    nsigma= nsigs
                    cc= 1
                    while nsigma > 1:
                        pyplot.fill_between(zs,zsigs[:,cc+1],zsigs[:,cc-1],
                                            color=colormap(_squeeze(plotc[ii],vmin,vmax)))
                        pyplot.fill_between(zs,zsigs[:,cc],zsigs[:,cc+2],
                                            color=colormap(_squeeze(plotc[ii],vmin,vmax)))
                        cc+= 1.
                        nsigma-= 1
                        pyplot.plot(numpy.linspace(zmin[ii]*1000.,1000*zmax[ii],1001),
                                    thiszfunc,'-',
                                    color=colormap(_squeeze(plotc[ii],vmin,vmax)),
                                    lw=ndata[ii]/15.,
                                    zorder=2.)
            #Add colorbar
            m = cm.ScalarMappable(cmap=cm.jet)
            m.set_array(plotc)
            m.set_clim(vmin=vmin,vmax=vmax)
            cbar= pyplot.colorbar(m,fraction=0.15)
            cbar.set_clim((vmin,vmax))
            cbar.set_label(zlabel)
        elif options.subtype.lower() == 'hs':
            from selectFigs import _squeeze
            colormap = cm.jet
            #plot hs vs. afe/feh, colorcoded using feh/afe
            if options.type.lower() == 'afe' or options.type.lower() == 'afefeh':
                plotx= feh+numpy.random.random(len(hs))*0.05 #jitter
                xrange= [-1.6,0.4]
                xlabel=r'$[\mathrm{Fe/H}]$'
            elif options.type.lower() == 'feh' or options.type.lower() == 'fehafe':
                plotx= afe+numpy.random.random(len(hs))*0.025 #jitter
                xrange= [-0.05,0.55]
                xlabel=r'$[\alpha\mathrm{/H}]$'
            yrange= [0.,20.]
            bovy_plot.bovy_plot(plotx,hs,
                                s=ndata,c=plotc,
                                cmap='jet',
                                xlabel=xlabel,
                                ylabel=r'$h_\sigma\ [\mathrm{kpc}]$',
                                clabel=zlabel,
                                xrange=xrange,yrange=yrange,
                                vmin=vmin,vmax=vmax,
                                scatter=True,edgecolors='none',
                                colorbar=True)
            #Overplot errorbars
            if options.ploterrors:
                for ii in range(len(hs)):
                    pyplot.errorbar(plotx[ii],
                                    hs[ii],yerr=hs_err[ii],
                                    color=colormap(_squeeze(plotc[ii],
                                                            numpy.amax([numpy.amin(plotc)]),
                                                            numpy.amin([numpy.amax(plotc)]))),
                                    elinewidth=1.,capsize=3,zorder=0)  
            #KDE estimate?
            #Form KDE estimates of all PDFs, multiply and display
            if options.kde:
                kde_list= []
                hss= numpy.linspace(0.,20.,101)
                for ii in range(len(hs)):
                    kde_list.append(gaussian_kde(1./allsamples[ii].reshape((1,len(allsamples[ii])))))
                kde_est= kde_mult(kde_list)
                hss= numpy.linspace(0.,20.,1001)
                pdf= numpy.zeros(len(hss))
                for ii in range(len(hss)):
                    pdf[ii]= kde_est(hss[ii])
                hs_m= numpy.sum(hss*pdf)/numpy.sum(pdf)
                hs_std= numpy.sqrt(numpy.sum(hss**2.*pdf)/numpy.sum(pdf)-hs_m**2.)
                #Rescale
                pdf*= 0.4/numpy.amax(pdf)
                pdf-= 1.6
                #Overplot
                bovy_plot.bovy_plot(pdf,hss,'k-',overplot=True)
            #Overplot weighted mean + stddev
            print hs_m, hs_std
            bovy_plot.bovy_plot(xrange,[hs_m,hs_m],'k-',overplot=True)
            bovy_plot.bovy_plot(xrange,[hs_m-hs_std,hs_m-hs_std],'-',
                                color='0.5',overplot=True)
            bovy_plot.bovy_plot(xrange,[hs_m+hs_std,hs_m+hs_std],'-',
                                color='0.5',overplot=True)
        elif options.subtype.lower() == 'hsm':
            from selectFigs import _squeeze
            colormap = cm.jet
            #plot hs vs. afe/feh, colorcoded using feh/afe
            if options.type.lower() == 'afe' or options.type.lower() == 'afefeh':
                plotx= feh+numpy.random.random(len(hs))*0.05 #jitter
                xrange= [-1.6,0.4]
                xlabel=r'$[\mathrm{Fe/H}]$'
            elif options.type.lower() == 'feh' or options.type.lower() == 'fehafe':
                plotx= afe+numpy.random.random(len(hs))*0.025 #jitter
                xrange= [-0.05,0.55]
                xlabel=r'$[\alpha\mathrm{/H}]$'
            yrange= [0.,.3]
            bovy_plot.bovy_plot(plotx,1./hs,
                                s=ndata,c=plotc,
                                cmap='jet',
                                xlabel=xlabel,
                                ylabel=r'$h^{-1}_\sigma\ [\mathrm{kpc}^{-1}]$',
                                clabel=zlabel,
                                xrange=xrange,yrange=yrange,
                                vmin=vmin,vmax=vmax,
                                scatter=True,edgecolors='none',
                                colorbar=True)
            #Overplot errorbars
            if options.ploterrors:
                for ii in range(len(hs)):
                    pyplot.errorbar(plotx[ii],
                                    1./hs[ii],yerr=hsm_err[ii],
                                    color=colormap(_squeeze(plotc[ii],
                                                            numpy.amax([numpy.amin(plotc)]),
                                                            numpy.amin([numpy.amax(plotc)]))),
                                    elinewidth=1.,capsize=3,zorder=0)  
            #KDE estimate?
            #Form KDE estimates of all PDFs, multiply and display
            if options.kde:
                kde_list= []
                for ii in range(len(hs)):
                    kde_list.append(gaussian_kde(1./allsamples[ii].reshape((1,len(allsamples[ii])))))
                kde_est= kde_mult(kde_list)
                hsms= numpy.linspace(0.01,0.3,1001)
                pdf= numpy.zeros(len(hsms))
                for ii in range(len(hsms)):
                    pdf[ii]= kde_est(1./hsms[ii])*hsms[ii]**-2.
                hs_m= numpy.sum(hsms*pdf)/numpy.sum(pdf)
                hs_std= numpy.sqrt(numpy.sum(hsms**2.*pdf)/numpy.sum(pdf)-hs_m**2.)
                #Rescale
                pdf*= 0.4/numpy.amax(pdf)
                pdf-= 1.6
                #Overplot
                bovy_plot.bovy_plot(pdf,hsms,'k-',overplot=True)
            #Overplot weighted mean + stddev
            print hs_m, hs_std
            bovy_plot.bovy_plot(xrange,[hs_m,hs_m],'k-',overplot=True)
            bovy_plot.bovy_plot(xrange,[hs_m-hs_std,hs_m-hs_std],'-',
                                color='0.5',overplot=True)
            bovy_plot.bovy_plot(xrange,[hs_m+hs_std,hs_m+hs_std],'-',
                                color='0.5',overplot=True)
        elif options.subtype.lower() == 'slope':
            from selectFigs import _squeeze
            colormap = cm.jet
            #plot hs vs. afe/feh, colorcoded using feh/afe
            if options.type.lower() == 'afe' or options.type.lower() == 'afefeh':
                plotx= feh+numpy.random.random(len(hs))*0.05 #jitter
                xrange= [-1.6,0.4]
                xlabel=r'$[\mathrm{Fe/H}]$'
            elif options.type.lower() == 'feh' or options.type.lower() == 'fehafe':
                plotx= afe+numpy.random.random(len(hs))*0.025 #jitter
                xrange= [-0.05,0.55]
                xlabel=r'$[\alpha\mathrm{/H}]$'
            yrange= [-20.,20.]
            bovy_plot.bovy_plot(plotx,p1,
                                s=ndata,c=plotc,
                                cmap='jet',
                                xlabel=xlabel,
                                ylabel=r'$\frac{\mathrm{d} \sigma_z(z_{1/2})}{\mathrm{d} z}\ [\mathrm{km\ s}^{-1}\ \mathrm{kpc}^{-1}]$',
                                clabel=zlabel,
                                xrange=xrange,yrange=yrange,
                                vmin=vmin,vmax=vmax,
                                scatter=True,edgecolors='none',
                                colorbar=True)
            #Overplot errorbars
            if options.ploterrors:
                for ii in range(len(hs)):
                    pyplot.errorbar(plotx[ii],
                                    p1[ii],yerr=p1_err[ii],
                                    color=colormap(_squeeze(plotc[ii],
                                                            numpy.amax([numpy.amin(plotc)]),
                                                            numpy.amin([numpy.amax(plotc)]))),
                                    elinewidth=1.,capsize=3,zorder=0)  
            #KDE estimate?
            #Form KDE estimates of all PDFs, multiply and display
            if options.kde:
                kde_list= []
                for ii in range(len(p1)):
                    kde_list.append(gaussian_kde(allsamples[ii].reshape((1,len(allsamples[ii])))))
                kde_est= kde_mult(kde_list)
                p1s= numpy.linspace(-5.,5.,1001)
                pdf= numpy.zeros(len(p1s))
                for ii in range(len(p1s)):
                    pdf[ii]= kde_est(p1s[ii])
                hs_m= numpy.sum(p1s*pdf)/numpy.sum(pdf)
                hs_std= numpy.sqrt(numpy.sum(p1s**2.*pdf)/numpy.sum(pdf)-hs_m**2.)
                #Rescale
                pdf*= 0.4/numpy.amax(pdf)
                pdf-= 1.6
                #Overplot
                bovy_plot.bovy_plot(pdf,p1s,'k-',overplot=True)
            #Overplot weighted mean + stddev
            print hs_m, hs_std
            bovy_plot.bovy_plot(xrange,[hs_m,hs_m],'k-',overplot=True)
            bovy_plot.bovy_plot(xrange,[hs_m-hs_std,hs_m-hs_std],'-',
                                color='0.5',overplot=True)
            bovy_plot.bovy_plot(xrange,[hs_m+hs_std,hs_m+hs_std],'-',
                                color='0.5',overplot=True)
        elif options.subtype.lower() == 'slopehsm':
            from selectFigs import _squeeze
            colormap = cm.jet
            xrange= [0.,.3]
            yrange= [-20.,20.]
            bovy_plot.bovy_plot(1./hs,p1,
                                s=ndata,c=plotc,
                                cmap='jet',
                                xlabel=r'$h^{-1}_\sigma\ [\mathrm{kpc}^{-1}]$',                                ylabel=r'$\frac{\mathrm{d} \sigma_z(z_{1/2})}{\mathrm{d} z}\ [\mathrm{km\ s}^{-1}\ \mathrm{kpc}^{-1}]$',
                                clabel=zlabel,
                                xrange=xrange,yrange=yrange,
                                vmin=vmin,vmax=vmax,
                                scatter=True,edgecolors='none',
                                colorbar=True)
            #Overplot errorbars
            ax= pyplot.gca()
            if options.ploterrors:
                for ii in range(len(hs)):
                    #Calculate the eigenvalues and the rotation angle
                    ycovar= numpy.zeros((2,2))
                    ycovar[0,0]= hsm_err[ii]**2.
                    ycovar[1,1]= p1_err[ii]**2.
                    ycovar[0,1]= p1hsm_corr[ii]*math.sqrt(ycovar[0,0]*ycovar[1,1])
                    ycovar[1,0]= ycovar[0,1]
                    eigs= linalg.eig(ycovar)
                    angle= math.atan(-eigs[1][0,1]/eigs[1][1,1])/math.pi*180.
                    e= Ellipse(numpy.array([1./hs[ii],p1[ii]]),
                               2*math.sqrt(eigs[0][0]),
                               2*math.sqrt(eigs[0][1]),angle)
                    ax.add_artist(e)
                    e.set_facecolor('none')
                    e.set_linewidth(ndata[ii]/50.)
                    e.set_zorder(-10.)
                    e.set_edgecolor(colormap(_squeeze(plotc[ii],
                                                      numpy.amax([numpy.amin(plotc)]),
                                                      numpy.amin([numpy.amax(plotc)]))))
        elif options.subtype.lower() == 'slopequad':
            from selectFigs import _squeeze
            colormap = cm.jet
            xrange= [-10.,10.]
            yrange= [-20.,20.]
            bovy_plot.bovy_plot(p2,p1,
                                s=ndata,c=plotc,
                                cmap='jet',
                                xlabel=r'$\frac{\mathrm{d}^2 \sigma_z(z_{1/2})}{\mathrm{d} z^2}\ [\mathrm{km}^2\ \mathrm{s}^{-2}\ \mathrm{kpc}^{-1}]$',
                                ylabel=r'$\frac{\mathrm{d} \sigma_z(z_{1/2})}{\mathrm{d} z}\ [\mathrm{km\ s}^{-1}\ \mathrm{kpc}^{-1}]$',
                                clabel=zlabel,
                                xrange=xrange,yrange=yrange,
                                vmin=vmin,vmax=vmax,
                                scatter=True,edgecolors='none',
                                colorbar=True,zorder=100)
            #Overplot errorbars
            ax= pyplot.gca()
            if options.ploterrors:
                for ii in range(len(hs)):
                    #Calculate the eigenvalues and the rotation angle
                    ycovar= numpy.zeros((2,2))
                    ycovar[0,0]= p2_err[ii]**2.
                    ycovar[1,1]= p1_err[ii]**2.
                    ycovar[0,1]= p1hsm_corr[ii]*math.sqrt(ycovar[0,0]*ycovar[1,1])
                    ycovar[1,0]= ycovar[0,1]
                    eigs= linalg.eig(ycovar)
                    angle= math.atan(-eigs[1][0,1]/eigs[1][1,1])/math.pi*180.
                    e= Ellipse(numpy.array([p2[ii],p1[ii]]),
                               2*math.sqrt(eigs[0][0]),
                               2*math.sqrt(eigs[0][1]),angle)
                    ax.add_artist(e)
                    e.set_facecolor('none')
                    e.set_linewidth(ndata[ii]/50.)
                    #e.set_linestyle('dashed')
                    e.set_zorder(-10.)
                    e.set_edgecolor(colormap(_squeeze(plotc[ii],
                                                      numpy.amax([numpy.amin(plotc)]),
                                                      numpy.amin([numpy.amax(plotc)]))))
            #KDE estimate?
            #Form KDE estimates of all PDFs, multiply and display
            if options.kde:
                kde_list= []
                print "Forming KDE ..."
                for ii in range(len(p1)):
                    kde_list.append(gaussian_kde(allsamples[ii]))
                kde_est= kde_mult(kde_list)
                p1s= numpy.linspace(-5.,5.,51)
                p2s= numpy.linspace(-5.,5.,51)
                pdf= numpy.zeros((len(p2s),len(p1s)))
                print "Evaluating KDE ..."
                for ii in range(len(p2s)):
                    for jj in range(len(p1s)):
                        pdf[ii,jj]= kde_est([p2s[ii],p1s[jj]])
                p2m= numpy.dot(p2s**1.,numpy.dot(pdf,p1s**0.))/numpy.sum(pdf)
                p1m= numpy.dot(p2s**0.,numpy.dot(pdf,p1s**1.))/numpy.sum(pdf)
                s22= numpy.dot(p2s**2.,numpy.dot(pdf,p1s**0.))/numpy.sum(pdf)-p2m**2.
                s11= numpy.dot(p2s**0.,numpy.dot(pdf,p1s**2.))/numpy.sum(pdf)-p1m**2.
                s12= numpy.dot(p2s**1.,numpy.dot(pdf,p1s**1.))/numpy.sum(pdf)-p1m*p2m
                print p2m, p1m, s22, s12, s11
                pyplot.plot(p2m,p1m,'kx',markersize=12.,zorder=100,mew=3.)
                #Overplot joint PDF ellipse
                nsigs= 3
                for ii in range(nsigs):
                    #Calculate the eigenvalues and the rotation angle
                    ycovar= numpy.zeros((2,2))
                    ycovar[0,0]= (ii+1.)*s22
                    ycovar[1,1]= (ii+1.)*s11
                    ycovar[0,1]= (ii+1.)*s12
                    ycovar[1,0]= ycovar[0,1]
                    eigs= linalg.eig(ycovar)
                    angle= math.atan(-eigs[1][0,1]/eigs[1][1,1])/math.pi*180.
                    print angle
                    e= Ellipse(numpy.array([p2m,p1m]),
                               2*math.sqrt(eigs[0][0]),
                               2*math.sqrt(eigs[0][1]),angle)
                    ax.add_artist(e)
                    e.set_facecolor('none')
                    e.set_linewidth(ndata)
                    e.set_zorder(99.)
                    e.set_edgecolor('k')
        elif options.subtype.lower() == 'slopequadquantiles':
            from selectFigs import _squeeze
            colormap = cm.jet
            #KDE estimate?
            #Form KDE estimates of all PDFs, multiply, calculate quantiles
            if options.kde:
                if os.path.exists(options.savekde):
                    savekdefile= open(options.savekde,'rb')
                    nps= pickle.load(savekdefile)
                    p1s= pickle.load(savekdefile)
                    p2s= pickle.load(savekdefile)
                    pdf= pickle.load(savekdefile)
                    ip1s= pickle.load(savekdefile)
                    ip2s= pickle.load(savekdefile)
                    ipdf= pickle.load(savekdefile)
                    savekdefile.close()
                else:
                    kde_list= []
                    print "Forming KDE ..."
                    for ii in range(len(p1)):
                        kde_list.append(gaussian_kde(allsamples[ii]))
                    kde_est= kde_mult(kde_list)
                    nps= 101
                    p1s= numpy.linspace(-5.,5.,nps)
                    p2s= numpy.linspace(-5.,5.,nps)
                    ip1s= numpy.linspace(-20.,20.,nps)
                    ip2s= numpy.linspace(-10.,10.,nps)
                    pdf= numpy.zeros((len(p2s),len(p1s)))
                    ipdf= numpy.zeros((len(p1),len(p2s),len(p1s)))
                    print "Evaluating KDE ..."
                    for ii in range(len(p2s)):
                        for jj in range(len(p1s)):
                            pdf[ii,jj]= kde_est([p2s[ii],p1s[jj]])
                            for kk in range(len(p1)):
                                ipdf[kk,ii,jj]= kde_list[kk]([ip2s[ii],ip1s[jj]])
                    save_pickles(options.savekde,nps,p1s,p2s,pdf,ip1s,ip2s,ipdf)
                p2m= numpy.dot(p2s**1.,numpy.dot(pdf,p1s**0.))/numpy.sum(pdf)
                p1m= numpy.dot(p2s**0.,numpy.dot(pdf,p1s**1.))/numpy.sum(pdf)
                s22= numpy.dot(p2s**2.,numpy.dot(pdf,p1s**0.))/numpy.sum(pdf)-p2m**2.
                s11= numpy.dot(p2s**0.,numpy.dot(pdf,p1s**2.))/numpy.sum(pdf)-p1m**2.
                s12= numpy.dot(p2s**1.,numpy.dot(pdf,p1s**1.))/numpy.sum(pdf)-p1m*p2m
                print p2m, p1m, s22, s12, s11
                #Also calculate p1m if p2==
                p1mp20= numpy.sum(pdf[int((nps-1)/2),:]*p1s)/numpy.sum(pdf[int((nps-1)/2),:])
                s11p20= numpy.sqrt(numpy.sum(pdf[int((nps-1)/2),:]*p1s**2.)/numpy.sum(pdf[int((nps-1)/2),:])-p1mp20**2.)
                print p1mp20, s11p20
                #Calculate quantiles
                quants= numpy.zeros(len(p1))
                #Find bin of best fit
                mjj, mii= int((p1m+20.)/40.*(nps-1)), int((p2m+10.)/20.*(nps-1))
                for kk in range(len(p1)):
                    #Sum from the top down!
                    X= ipdf[kk,:,:]
                    X[numpy.isnan(X)]= 0.
                    sortindx= numpy.argsort(X.flatten())[::-1]
                    cumul= numpy.cumsum(numpy.sort(X.flatten())[::-1])/numpy.sum(X.flatten())
                    cntrThis= numpy.zeros(numpy.prod(X.shape))
                    cntrThis[sortindx]= cumul
                    cntrThis= numpy.reshape(cntrThis,X.shape)
                    quants[kk]= cntrThis[mii,mjj]
                #Overplot data points
                bovy_plot.bovy_plot(quants,numpy.random.uniform(size=len(p1))*0.5+0.25,
                                    c=plotc,s=ndata,overplot=False,
                                    scatter=True,
                                    xlabel=r'$\mathrm{quantile\ of}\ (\hat{p}_2,\hat{p}_3)\ \mathrm{in\ individual\ PDF}$',
                                    xrange=[0.,1.],
                                    yrange=[0.,1.4],
                                    colorbar=True,
                                    vmin=vmin,vmax=vmax,clabel=zlabel,
                                    edgecolors='none')
                bovy_plot.bovy_hist(quants,range=[0.,1.],bins=7,overplot=True,
                                    histtype='step',normed=True,color='k',lw=2.)
                #Also overplot KDE estimate of the histogram
                histkde= gaussian_kde(quants)
                qs= numpy.linspace(0.,1.,1001)
                hqs= numpy.zeros(len(qs))
                for ii in range(len(qs)): hqs[ii]= histkde(qs[ii])
                bovy_plot.bovy_plot(qs,hqs,'k-',overplot=True)
        elif options.subtype.lower() == 'slopesz':
            from selectFigs import _squeeze
            colormap = cm.jet
            xrange= [0.,60.]
            yrange= [-20.,20.]
            bovy_plot.bovy_plot(sz,p1,
                                s=ndata,c=plotc,
                                cmap='jet',
                                xlabel=r'$\sigma_z(z_{1/2}) [\mathrm{km\ s}^{-1}$',
                                ylabel=r'$\frac{\mathrm{d} \sigma_z(z_{1/2})}{\mathrm{d} z}\ [\mathrm{km\ s}^{-1}\ \mathrm{kpc}^{-1}]$',
                                clabel=zlabel,
                                xrange=xrange,yrange=yrange,
                                vmin=vmin,vmax=vmax,
                                scatter=True,edgecolors='none',
                                colorbar=True,zorder=100)
            #Overplot errorbars
            ax= pyplot.gca()
            if options.ploterrors:
                for ii in range(len(hs)):
                    #Calculate the eigenvalues and the rotation angle
                    ycovar= numpy.zeros((2,2))
                    ycovar[0,0]= sz_err[ii]**2.
                    ycovar[1,1]= p1_err[ii]**2.
                    ycovar[0,1]= p1hsm_corr[ii]*math.sqrt(ycovar[0,0]*ycovar[1,1])
                    ycovar[1,0]= ycovar[0,1]
                    eigs= linalg.eig(ycovar)
                    angle= math.atan(-eigs[1][0,1]/eigs[1][1,1])/math.pi*180.
                    e= Ellipse(numpy.array([sz[ii],p1[ii]]),
                               2*math.sqrt(eigs[0][0]),
                               2*math.sqrt(eigs[0][1]),angle)
                    ax.add_artist(e)
                    e.set_facecolor('none')
                    e.set_linewidth(ndata[ii]/50.)
                    e.set_zorder(-10.)
                    e.set_edgecolor(colormap(_squeeze(plotc[ii],
                                                      numpy.amax([numpy.amin(plotc)]),
                                                      numpy.amin([numpy.amax(plotc)]))))
        elif options.subtype.lower() == 'szhsm':
            from selectFigs import _squeeze
            colormap = cm.jet
            xrange= [0.,0.3]
            yrange= [0.,60.]
            bovy_plot.bovy_plot(1./hs,sz,
                                s=ndata,c=plotc,
                                cmap='jet',
                                xlabel=r'$h^{-1}_\sigma\ [\mathrm{kpc}^{-1}]$',
                                ylabel=r'$\sigma_z(z_{1/2}) [\mathrm{km\ s}^{-1}$',
                                clabel=zlabel,
                                xrange=xrange,yrange=yrange,
                                vmin=vmin,vmax=vmax,
                                scatter=True,edgecolors='none',
                                colorbar=True,zorder=100)
            #Overplot errorbars
            ax= pyplot.gca()
            if options.ploterrors:
                for ii in range(len(hs)):
                    #Calculate the eigenvalues and the rotation angle
                    ycovar= numpy.zeros((2,2))
                    ycovar[0,0]= hsm_err[ii]**2.
                    ycovar[1,1]= sz_err[ii]**2.
                    ycovar[0,1]= p1hsm_corr[ii]*math.sqrt(ycovar[0,0]*ycovar[1,1])
                    ycovar[1,0]= ycovar[0,1]
                    eigs= linalg.eig(ycovar)
                    angle= math.atan(-eigs[1][0,1]/eigs[1][1,1])/math.pi*180.
                    e= Ellipse(numpy.array([1./hs[ii],sz[ii]]),
                               2*math.sqrt(eigs[0][0]),
                               2*math.sqrt(eigs[0][1]),angle)
                    ax.add_artist(e)
                    e.set_facecolor('none')
                    e.set_linewidth(.5)
                    e.set_zorder(-10.)
                    e.set_edgecolor(colormap(_squeeze(plotc[ii],
                                                      numpy.amax([numpy.amin(plotc)]),
                                                      numpy.amin([numpy.amax(plotc)]))))
        elif options.subtype.lower() == 'asz':
            from selectFigs import _squeeze
            colormap = cm.jet
            #plot hs vs. afe/feh, colorcoded using feh/afe
            if options.type.lower() == 'afe' or options.type.lower() == 'afefeh':
                plotx= feh+numpy.random.random(len(hs))*0.05 #jitter
                xrange= [-1.6,0.4]
                xlabel=r'$[\mathrm{Fe/H}]$'
            elif options.type.lower() == 'feh' or options.type.lower() == 'fehafe':
                plotx= afe+numpy.random.random(len(hs))*0.025 #jitter
                xrange= [-0.05,0.55]
                xlabel=r'$[\alpha\mathrm{\alpha/Fe}]$'
            yrange= [15.,100.]
            #print numpy.mean(ndata**2.*hz*sz**2./(2.*numpy.pi*4.302*10.**-3))/numpy.mean(ndata**2.)
            bovy_plot.bovy_plot(plotx,
                                hz*sz**2./(2.*numpy.pi*4.302*10.**-3), #2piG
                                s=ndata,c=plotc,
                                cmap='jet',
                                xlabel=xlabel,
                                ylabel=r'$\Sigma_{\mathrm{disk}}\ [M_\odot\ \mathrm{pc}^{-2}]$',
                                clabel=zlabel,
                                xrange=xrange,yrange=yrange,
                                vmin=vmin,vmax=vmax,
                                scatter=True,edgecolors='none',
                                colorbar=True)
            #Overplot weighted mean + stddev
            print hs_m, hs_std
            bovy_plot.bovy_plot(xrange,[hs_m,hs_m],'k-',overplot=True)
            bovy_plot.bovy_plot(xrange,[hs_m-hs_std,hs_m-hs_std],'-',
                                color='0.5',overplot=True)
            bovy_plot.bovy_plot(xrange,[hs_m+hs_std,hs_m+hs_std],'-',
                                color='0.5',overplot=True)
        elif options.subtype.lower() == 'bsz':
            from selectFigs import _squeeze
            colormap = cm.jet
            #plot hs vs. afe/feh, colorcoded using feh/afe
            if options.type.lower() == 'afe' or options.type.lower() == 'afefeh':
                plotx= feh+numpy.random.random(len(hs))*0.05 #jitter
                xrange= [-1.6,0.4]
                xlabel=r'$[\mathrm{Fe/H}]$'
            elif options.type.lower() == 'feh' or options.type.lower() == 'fehafe':
                plotx= afe+numpy.random.random(len(hs))*0.025 #jitter
                xrange= [-0.05,0.55]
                xlabel=r'$[\alpha\mathrm{\alpha/Fe}]$'
            yrange= [0.,0.02]
            #print numpy.mean(ndata**2.*hr*sz**2./(2.*numpy.pi*4.302*10.**-3))/numpy.mean(ndata**2.)
            bovy_plot.bovy_plot(plotx,
                                hr*sz**2./(2.*numpy.pi*4.302*10.**-3), #2piG
                                s=ndata,c=plotc,
                                cmap='jet',
                                xlabel=xlabel,
                                ylabel=r'$\rho_{\mathrm{DM}}\ [M_\odot\ \mathrm{pc}^{-3}]$',
                                clabel=zlabel,
                                xrange=xrange,yrange=yrange,
                                vmin=vmin,vmax=vmax,
                                scatter=True,edgecolors='none',
                                colorbar=True)
            #Overplot errorbars
            if options.ploterrors:
                print "UPDATE!!"
                for ii in range(len(hs)):
                    pyplot.errorbar(plotx[ii],
                                    hs[ii],yerr=hs_err[ii],
                                    color=colormap(_squeeze(plotc[ii],
                                                            numpy.amax([numpy.amin(plotc)]),
                                                            numpy.amin([numpy.amax(plotc)]))),
                                    elinewidth=1.,capsize=3,zorder=0,elinestyle='-')  
                #Overplot weighted mean + stddev
                hs_m= numpy.sum(hs/hs_err**2.)/numpy.sum(1./hs_err**2.)
                hs_std= numpy.sqrt(1./numpy.sum(1./hs_err**2.))
                print hs_m, hs_std
                bovy_plot.bovy_plot(xrange,[hs_m,hs_m],'k-',overplot=True)
                bovy_plot.bovy_plot(xrange,[hs_m-hs_std,hs_m-hs_std],'-',
                                    color='0.5',overplot=True)
                bovy_plot.bovy_plot(xrange,[hs_m+hs_std,hs_m+hs_std],'-',
                                    color='0.5',overplot=True)
    elif options.type.lower() == 'slopes':
        bovy_plot.bovy_print()
        bovy_plot.bovy_hist(plotthis.flatten(),
                            range=[-5.,5.],
                            bins=11,
                            histtype='step',
                            color='k',
                            xlabel=r'$\sigma_z(z)\ \mathrm{slope\ [km\ s}^{-1}\ \mathrm{kpc}^{-1}]$')
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
    parser.add_option("--ploterrors",action="store_true", dest="ploterrors",
                      default=False,
                      help="If set, plot the errorbars")
    parser.add_option("--vr",action="store_true", dest="vr",
                      default=False,
                      help="If set, fit vR instead of vz")
    parser.add_option("--kde",action="store_true", dest="kde",
                      default=False,
                      help="Perform KDE estimate of hs")
    parser.add_option("--envelope",action="store_true", dest="envelope",
                      default=False,
                      help="Plot the error sausage envolope")
    parser.add_option("--savekde",dest='savekde',default=None,
                      help="Name of the file to save KDE estimates")
    return parser

if __name__ == '__main__':
    numpy.random.seed(1)
    parser= get_options()
    options,args= parser.parse_args()
    plotsz2hz(options,args)

