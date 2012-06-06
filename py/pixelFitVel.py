import os, os.path
import sys
import math
import numpy
from scipy import optimize
import cPickle as pickle
from optparse import OptionParser
from galpy.util import bovy_coords, bovy_plot, save_pickles
import bovy_mcmc
from segueSelect import read_gdwarfs, read_kdwarfs, _GDWARFFILE, _KDWARFFILE
from fitSigz import _IsothermLikeMinus, _HWRLikeMinus, _ZSUN, \
    _HWRLike, _IsothermLike
from pixelFitDens import pixelAfeFeh
def pixelFitVel(options,args):
    if options.sample.lower() == 'g':
        if options.select.lower() == 'program':
            raw= read_gdwarfs(_GDWARFFILE,logg=True,ebv=True,sn=options.snmin,
                              distfac=options.distfac)
        else:
            raw= read_gdwarfs(logg=True,ebv=True,sn=options.snmin,
                              distfac=options.distfac)
    elif options.sample.lower() == 'k':
        if options.select.lower() == 'program':
            raw= read_kdwarfs(_KDWARFFILE,logg=True,ebv=True,sn=options.snmin,
                              distfac=options.distfac)
        else:
            raw= read_kdwarfs(logg=True,ebv=True,sn=options.snmin,
                              distfac=options.distfac)
    if not options.bmin is None:
        #Cut on |b|
        raw= raw[(numpy.fabs(raw.b) > options.bmin)]
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
    #Model
    if options.model.lower() == 'hwr':
        like_func= _HWRLikeMinus
        pdf_func= _HWRLike
        step= [0.01,0.3,0.3,0.3,0.3]
        create_method=['full','step_out','step_out',
                       'step_out','step_out']
        isDomainFinite=[[True,True],[True,True],
                        [True,True],[True,True],
                        [False,True]]
        domain=[[0.,1.],[-10.,10.],[-100.,100.],[-100.,100.],
                [0.,4.6051701859880918]]
    elif options.model.lower() == 'isotherm':
        like_func= _IsothermLikeMinus
        pdf_func= _IsothermLike
        step= [0.01,0.05,0.3]
        create_method=['full','step_out','step_out']
        isDomainFinite=[[True,True],[False,False],
                        [False,True]]
        domain=[[0.,1.],[0.,0.],[0.,4.6051701859880918]]
    #Run through the bins
    while ii < len(binned.fehedges)-1:
        while jj < len(binned.afeedges)-1:
            data= binned(binned.feh(ii),binned.afe(jj))
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
            #Create XYZ and R, vxvyvz, cov_vxvyvz
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
            d= numpy.fabs((XYZ[:,2]-numpy.median(numpy.fabs(XYZ[:,2]))))
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
            if options.vr:
                #Rotate vxvyvz to vRvTvz
                cosphi= (8.-XYZ[:,0])/R
                sinphi= XYZ[:,1]/R
                vR= vxvyvz[:,0]*cosphi+vxvyvz[:,1]*sinphi
                vT= -vxvyvz[:,0]*sinphi+vxvyvz[:,1]*cosphi
                vxvyvz[:,0]= vR
                vxvyvz[:,1]= vT
                for rr in range(len(XYZ[:,0])):
                    rot= numpy.array([[cosphi[rr],sinphi[rr]],
                                      [-sinphi[rr],cosphi[rr]]])
                    sxy= cov_vxvyvz[rr,0:2,0:2]
                    sRT= numpy.dot(rot,numpy.dot(sxy,rot.T))
                    cov_vxvyvz[rr,0:2,0:2]= sRT
            #Fit this data
            #Initial condition
            if options.model.lower() == 'hwr':
                params= numpy.array([0.02,numpy.log(30.),0.,0.,numpy.log(6.)])
            elif options.model.lower() == 'isotherm':
                params= numpy.array([0.02,numpy.log(30.),numpy.log(6.)])
            if not options.mcsample:
                #Optimize likelihood
                params= optimize.fmin_powell(like_func,params,
                                             args=(XYZ,vxvyvz,cov_vxvyvz,R,d,
                                                   options.vr))
                print numpy.exp(params)
                fits.append(params)
            else:
                #Load best-fit params
                params= fits[jj+ii*binned.npixafe()]
                print numpy.exp(params)
                thesesamples= bovy_mcmc.markovpy(params,
                #thesesamples= bovy_mcmc.slice(params,
                                                 #step,
                                                 0.01,
                                                 pdf_func,
                                                 (XYZ,vxvyvz,cov_vxvyvz,R,d,
                                                  options.vr),
                                                 #create_method=create_method,
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
            if options.mcsample: save_pickles(args[1],samples,ii,jj)
            else: save_pickles(args[0],fits,ii,jj)
            if jj == 0: #this means we've reset the counter 
                break
    if options.mcsample: save_pickles(args[1],samples,ii,jj)
    else: save_pickles(args[0],fits,ii,jj)
    return None

def plotPixelFitVel(options,args):
    if options.sample.lower() == 'g':
        if options.select.lower() == 'program':
            raw= read_gdwarfs(_GDWARFFILE,logg=True,ebv=True,sn=options.snmin,
                              distfac=options.distfac)
        else:
            raw= read_gdwarfs(logg=True,ebv=True,sn=options.snmin,
                              distfac=options.distfac)
    elif options.sample.lower() == 'k':
        if options.select.lower() == 'program':
            raw= read_kdwarfs(_KDWARFFILE,logg=True,ebv=True,sn=options.snmin,
                              distfac=options.distfac)
        else:
            raw= read_kdwarfs(logg=True,ebv=True,sn=options.snmin,
                              distfac=options.distfac)
    if not options.bmin is None:
        #Cut on |b|
        raw= raw[(numpy.fabs(raw.b) > options.bmin)]
    #print len(raw)
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
    #Now plot
    #Run through the pixels and gather
    if options.type.lower() == 'afe' or options.type.lower() == 'feh' \
            or options.type.lower() == 'fehafe' \
            or options.type.lower() == 'afefeh':
        plotthis= []
    else:
        plotthis= numpy.zeros((tightbinned.npixfeh(),tightbinned.npixafe()))
    #ndata= 0
    #maxndata= 0
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
            #if len(data) > maxndata: maxndata= len(data)
            #ndata+= len(data)
            if options.model.lower() == 'hwr':
                if options.type == 'sz':
                    plotthis[ii,jj]= numpy.exp(thisfit[1])
                elif options.type == 'sz2':
                    plotthis[ii,jj]= numpy.exp(2.*thisfit[1])
                elif options.type == 'hs':
                    plotthis[ii,jj]= numpy.exp(thisfit[4])
                elif options.type == 'hsm':
                    plotthis[ii,jj]= numpy.exp(-thisfit[4])
                elif options.type == 'pbad':
                    plotthis[ii,jj]= thisfit[0]
                elif options.type == 'slopes':
                    plotthis[ii,jj]= thisfit[2]
                elif options.type == 'slope':
                    plotthis[ii,jj]= thisfit[2]
                elif options.type == 'zmedian':
                    plotthis[ii,jj]= numpy.median(numpy.fabs(XYZ[:,2]))))
                elif options.type.lower() == 'afe' \
                        or options.type.lower() == 'feh' \
                        or options.type.lower() == 'fehafe' \
                        or options.type.lower() == 'afefeh':
                    plotthis.append([tightbinned.feh(ii),
                                     tightbinned.afe(jj),
                                     numpy.exp(thisfit[1]),
                                     numpy.exp(thisfit[3]),
                                     len(data)])
    #print ndata
    #print maxndata
    #Set up plot
    #print numpy.nanmin(plotthis), numpy.nanmax(plotthis)
    if options.type == 'sz':
        if options.vr:
            vmin, vmax= 40.,80.
            zlabel= r'$\sigma_R(z = \langle z \rangle)\ [\mathrm{km\ s}^{-1}]$'
        else:
            vmin, vmax= 10.,60.
            zlabel= r'$\sigma_z(z_{1/2})\ [\mathrm{km\ s}^{-1}]$'
    elif options.type == 'sz2':
        if options.vr:
            vmin, vmax= 40.**2.,80.**2.
            zlabel= r'$\sigma_R^2(z= \langle z \rangle)\ [\mathrm{km\ s}^{-1}]$'
        else:
            vmin, vmax= 15.**2.,50.**2.
            zlabel= r'$\sigma_z^2(z= \langle z \rangle)\ [\mathrm{km\ s}^{-1}]$'
    elif options.type == 'hs':
        if options.vr:
            vmin, vmax= 3.,25.
            zlabel= r'$R_\sigma\ [\mathrm{kpc}]$'
        else:
            vmin, vmax= 3.,15.
            zlabel= r'$h_\sigma\ [\mathrm{kpc}]$'
    elif options.type == 'hsm':
        if options.vr:
            vmin, vmax= 0.,0.3
            zlabel= r'$R^{-1}_\sigma\ [\mathrm{kpc}^{-1}]$'
        else:
            vmin, vmax= 0.,0.3
            zlabel= r'$R^{-1}_\sigma\ [\mathrm{kpc}^{-1}]$'
    elif options.type == 'slope':
        vmin, vmax= -5.,5.
        zlabel= r'$\frac{\mathrm{d} \sigma_z}{\mathrm{d} z}(z_{1/2})\ [\mathrm{km\ s}^{-1}\ \mathrm{kpc}^{-1}]$'
    elif options.type == 'pbad':
        vmin, vmax= 0.,0.1
        zlabel= r'$P_{\mathrm{bad}}$'
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
        xrange=[-1.6,0.5]
        yrange=[-0.05,0.55]
    else:
        xrange=[-2.,0.6]
        yrange=[-0.1,0.6]
    if options.type.lower() == 'afe' or options.type.lower() == 'feh' \
            or options.type.lower() == 'fehafe' \
            or options.type.lower() == 'afefeh':
        print "Update!! Never used until now (afe etc. type fitting is in plotsz2hz"
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
    if options.observed:
        bovy_plot.bovy_text(r'$\mathrm{observed}$',
                            top_right=True,size=18.)
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
    parser.add_option("--bmin",dest='bmin',type='float',
                      default=None,
                      help="Minimum Galactic latitude")
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
    parser.add_option("--mcsample",action="store_true", dest="mcsample",
                      default=False,
                      help="If set, sample around the best fit, save in args[1]")
    parser.add_option("--nsamples",dest='nsamples',default=1000,type='int',
                      help="Number of MCMC samples to obtain")
    parser.add_option("--rmin",dest='rmin',type='float',
                      default=None,
                      help="Minimum radius")
    parser.add_option("--rmax",dest='rmax',type='float',
                      default=None,
                      help="Maximum radius")
    parser.add_option("--snmin",dest='snmin',type='float',
                      default=15.,
                      help="Minimum S/N")
    parser.add_option("--vr",action="store_true", dest="vr",
                      default=False,
                      help="If set, fit vR instead of vz")
    parser.add_option("--observed",action="store_true", dest="observed",
                      default=False,
                      help="If set, write observed on it")
    parser.add_option("--distfac",dest="distfac",
                      default=None,type='float',
                      help="If set, apply a distance factor of this value")
    return parser
  
if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    if options.plot:
        plotPixelFitVel(options,args)
    else:
        pixelFitVel(options,args)

