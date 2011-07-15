import sys
import os, os.path
import math
import numpy
import cPickle as pickle
from matplotlib import pyplot
from optparse import OptionParser
from scipy import optimize, special
from galpy.util import bovy_coords, bovy_plot, bovy_quadpack
import bovy_mcmc
from segueSelect import ivezic_dist_gr, segueSelect, _gi_gr, _mr_gi
from fitSigz import readData
from plotData import plotDensz
_ERASESTR= "                                                                                "
_VERBOSE=True
_DEBUG=True
_INTEGRATEPLATESEP= True
_EPSREL=0.1
_EPSABS=0.0
_DEGTORAD=math.pi/180.
_ZSUN=0.025 #Sun's offset from the plane toward the NGP in kpc
_DZ=6.
_DR=5.
def fitDensz(parser):
    (options,args)= parser.parse_args()
    if len(args) == 0:
        parser.print_help()
        return
    if options.model.lower() == 'hwr':
        densfunc= _HWRDensity
    elif options.model.lower() == 'flare':
        densfunc= _FlareDensity
    elif options.model.lower() == 'twovertical':
        densfunc= _TwoVerticalDensity
    #First read the data
    if _VERBOSE:
        print "Reading and parsing data ..."
    XYZ,vxvyvz,cov_vxvyvz,rawdata= readData(metal=options.metal,
                                            sample=options.sample)
    #Load selection function
    if _VERBOSE:
        print "Loading selection function ..."
    plates= numpy.array(list(set(list(rawdata.plate))),dtype='int') #Only load plates that we use
    sf= segueSelect(plates=plates,type=options.sel)
    platelb= bovy_coords.radec_to_lb(sf.platestr.ra,sf.platestr.dec,
                                     degree=True)
    indx= [not 'faint' in name for name in sf.platestr.programname]
    platebright= numpy.array(indx,dtype='bool')
    indx= ['faint' in name for name in sf.platestr.programname]
    platefaint= numpy.array(indx,dtype='bool')
    if options.bright or options.faint:
        indx= []
        for ii in range(len(rawdata.ra)):
            pindx= (sf.plates == rawdata[ii].plate)
            if options.bright \
                    and not 'faint' in sf.platestr[pindx].programname[0]:
                indx.append(True)
            elif options.faint \
                    and 'faint' in sf.platestr[pindx].programname:
                indx.append(True)
            else:
                indx.append(False)
        indx= numpy.array(indx,dtype='bool')
        rawdata= rawdata[indx]
    if options.bright or options.faint:
        #Reload selection function
        plates= numpy.array(list(set(list(rawdata.plate))),dtype='int') #Only load plates that we use
        print plates
        sf= segueSelect(plates=plates,type=options.sel)
        platelb= bovy_coords.radec_to_lb(sf.platestr.ra,sf.platestr.dec,
                                         degree=True)
        indx= [not 'faint' in name for name in sf.platestr.programname]
        platebright= numpy.array(indx,dtype='bool')
        indx= ['faint' in name for name in sf.platestr.programname]
        platefaint= numpy.array(indx,dtype='bool')
    Ap= math.pi*2.*(1.-numpy.cos(1.49*_DEGTORAD)) #SEGUE PLATE=1.49 deg radius
    if options.sample.lower() == 'g':
        grmin, grmax= 0.48, 0.55
        rmin,rmax= 14.5, 20.2
    if options.metal.lower() == 'rich':
        feh= -0.15
    elif options.metal.lower() == 'poor':
        feh= -0.65
    else:
        feh= -0.5 
    colordist= _const_colordist
    if os.path.exists(args[0]):#Load savefile
        savefile= open(args[0],'rb')
        params= pickle.load(savefile)
        samples= pickle.load(savefile)
        savefile.close()
        if _DEBUG:
            print "Printing mean and std dev of samples ..."
            for ii in range(len(params)):
                xs= numpy.array([s[ii] for s in samples])
                print numpy.mean(xs), numpy.std(xs)
    else:
        #Subsample
        if not options.subsample is None:
            randindx= numpy.random.permutation(len(rawdata.ra))
            randindx= randindx[0:options.subsample]
            XYZ= XYZ[randindx,:]
            vxvyvz= vxvyvz[randindx,:]
            cov_vxvyvz= cov_vxvyvz[randindx,:,:]
            rawdata= rawdata[randindx]
        XYZ= XYZ.astype(numpy.float64)
        R= ((8.-XYZ[:,0])**2.+XYZ[:,1]**2.)**(0.5)
        XYZ[:,2]+= _ZSUN
        #BOVY: Z here and other places
        like_func= _HWRLikeMinus
        pdf_func= _HWRLike
        if options.model.lower() == 'hwr':
            if options.metal == 'rich':
                params= numpy.array([numpy.log(0.3),numpy.log(2.5),0.0])
            elif options.metal == 'poor':
                params= numpy.array([numpy.log(1.2),numpy.log(2.5),0.0])
            else:
                params= numpy.array([numpy.log(0.3),numpy.log(2.5),0.0])
            densfunc= _HWRDensity
            #Slice sampling keywords
            step= [0.3,0.3,0.02]
            create_method=['step_out','step_out','step_out']
            isDomainFinite=[[False,True],[False,True],[True,True]]
            domain=[[0.,4.6051701859880918],[0.,4.6051701859880918],[0.,1.]]
        elif options.model.lower() == 'flare':
            if options.metal == 'rich':
                params= numpy.array([numpy.log(0.3),numpy.log(2.5),numpy.log(2.5)])
            elif options.metal == 'poor':
                params= numpy.array([numpy.log(1.),numpy.log(2.5),numpy.log(2.5)])
            else:
                params= numpy.array([numpy.log(0.3),numpy.log(2.5),numpy.log(2.5)])
            densfunc= _FlareDensity
            #Slice sampling keywords
            step= [0.3,0.3,0.3]
            create_method=['step_out','step_out','step_out']
            isDomainFinite=[[False,True],[False,True],[False,True]]
            domain=[[0.,4.6051701859880918],[0.,4.6051701859880918],
                    [0.,4.6051701859880918]]
        elif options.model.lower() == 'twovertical':
            if options.metal == 'rich':
                params= numpy.array([numpy.log(0.3),numpy.log(1.),numpy.log(2.5),0.025])
            elif options.metal == 'poor':
                params= numpy.array([numpy.log(1.),numpy.log(2.),numpy.log(2.5),0.025])
            else:
                params= numpy.array([numpy.log(0.3),numpy.log(1.),numpy.log(2.5),0.025])
            densfunc= _TwoVerticalDensity
            #Slice sampling keywords
            step= [0.3,0.3,0.3,0.025]
            create_method=['step_out','step_out','step_out','step_out']
            isDomainFinite=[[False,True],[False,True],[False,True],[True,True]]
            domain=[[0.,4.6051701859880918],[0.,4.6051701859880918],
                    [0.,4.6051701859880918],[0.,1.]]
        #Optimize likelihood
        if _VERBOSE:
            print "Optimizing the likelihood ..."
        params= optimize.fmin_powell(like_func,params,
                                     args=(XYZ,R,
                                           sf,plates,platelb[:,0],
                                           platelb[:,1],platebright,
                                           platefaint,Ap,
                                           grmin,grmax,rmin,rmax,
                                           feh,colordist,densfunc))
        if _VERBOSE:
            print "Optimal likelihood:", params
        #Now sample
        if _VERBOSE:
            print "Sampling the likelihood ..."
            samples= bovy_mcmc.slice(params,
                                     step,
                                     pdf_func,
                                     (XYZ,R,
                                      sf,plates,platelb[:,0],
                                      platelb[:,1],platebright,
                                      platefaint,Ap,
                                      grmin,grmax,rmin,rmax,
                                      feh,colordist,densfunc),
                                     create_method=create_method,
                                     isDomainFinite=isDomainFinite,
                                     domain=domain,
                                     nsamples=options.nsamples)
        if _DEBUG:
            print "Printing mean and std dev of samples ..."
            for ii in range(len(params)):
                xs= numpy.array([s[ii] for s in samples])
                print numpy.mean(xs), numpy.std(xs)
        if _VERBOSE:
            print "Saving ..."
        savefile= open(args[0],'wb')
        pickle.dump(params,savefile)
        pickle.dump(samples,savefile)
        savefile.close()
    #Plot
    if options.plotzfunc:
        zs= numpy.linspace(0.3,2.,1001)
        #Plot the mean and std-dev from the posterior
        zmean= numpy.zeros(len(zs))
        nsigs= 3
        zsigs= numpy.zeros((len(zs),2*nsigs))
        fs= numpy.zeros((len(zs),len(samples)))
        for ii in range(len(samples)):
            thisparams= samples[ii]
            fs[:,ii]= numpy.log(densfunc(8.,zs,thisparams))
        #Record mean and std-devs
        zmean[:]= numpy.mean(fs,axis=1)
        norm= numpy.log(numpy.sum(numpy.exp(zmean)*(zs[1]-zs[0])))
        zmean-= norm
        if options.xmin is None or options.xmax is None:
            xrange= [numpy.amin(zs)-0.2,numpy.amax(zs)+0.1]
        else:
            xrange= [options.xmin,options.xmax]
        if options.ymin is None or options.ymax is None:
            yrange= [numpy.amin(zmean)-1.,numpy.amax(zmean)+1.]
        else:
            yrange= [options.ymin,options.ymax]
        bovy_plot.bovy_print()
        bovy_plot.bovy_plot(zs,zmean,'k-',xrange=xrange,yrange=yrange,
                            xlabel=options.xlabel,
                            ylabel=options.ylabel)
        for ii in range(nsigs):
            for jj in range(len(zs)):
                thisf= sorted(fs[jj,:])
                thiscut= 0.5*special.erfc((ii+1.)/math.sqrt(2.))
                zsigs[jj,2*ii]= thisf[int(math.ceil(thiscut*len(samples)))]
                thiscut= 1.-thiscut
                zsigs[jj,2*ii+1]= thisf[int(math.floor(thiscut*len(samples)))]
        colord, cc= (1.-0.75)/nsigs, 1
        nsigma= nsigs
        pyplot.fill_between(zs,zsigs[:,0]-norm,zsigs[:,1]-norm,color='0.75')
        while nsigma > 1:
            pyplot.fill_between(zs,zsigs[:,cc+1]-norm,zsigs[:,cc-1]-norm,
                                color='%f' % (.75+colord*cc))
            pyplot.fill_between(zs,zsigs[:,cc]-norm,zsigs[:,cc+2]-norm,
                                color='%f' % (.75+colord*cc))
            cc+= 1.
            nsigma-= 1
        bovy_plot.bovy_plot(zs,zmean,'k-',overplot=True)
        #Plot the data
        plotDensz(rawdata,sf,xrange=[zs[0],zs[-1]],normed=True,overplot=True)
        plotDensz(rawdata,sf,xrange=[zs[0],zs[-1]],normed=True,overplot=True,
                  noweights=True,color='r')
        plotDensz(rawdata,sf,xrange=[zs[0],zs[-1]],normed=True,overplot=True,
                  color='b',db=15.)
        bovy_plot.bovy_end_print(options.plotfile)
    if options.plotrfunc:
        zs= numpy.linspace(rmin,rmax,1001)
        #Plot the mean and std-dev from the posterior
        zmean= numpy.zeros(len(zs))
        thisparams= []
        for ii in range(len(params)):
            xs= numpy.array([s[ii] for s in samples])
            thisparams.append(numpy.mean(xs))
        zmean= _predict_rdist(zs,densfunc,thisparams,rmin,rmax,platelb,
                              grmin,grmax,feh,sf,colordist)
        zmean_const= _predict_rdist(zs,_ConstDensity,
                                    thisparams,rmin,rmax,platelb,
                                    grmin,grmax,feh,sf,colordist)
        if options.metal == 'rich':
            zmean_twofifty= _predict_rdist(zs,densfunc,
                                           [numpy.log(.25),
                                            thisparams[1],thisparams[2]],
                                           rmin,rmax,platelb,
                                           grmin,grmax,feh,sf,colordist)
        elif options.metal == 'poor':
            zmean_twofifty= _predict_rdist(zs,densfunc,
                                           [numpy.log(1.25),
                                            thisparams[1],thisparams[2]],
                                           rmin,rmax,platelb,
                                           grmin,grmax,feh,sf,colordist)
        """
        nsigs= 3
        zsigs= numpy.zeros((len(zs),2*nsigs))
        fs= numpy.zeros((len(zs),len(samples)))
        for ii in range(len(samples)):
            thisparams= samples[ii]
            fs[:,ii]= _predict_rdist(zs,densfunc,thisparams,rmin,rmax,platelb,
                                     grmin,grmax,feh,sf,colordist)
        #Record mean and std-devs
        zmean[:]= numpy.mean(fs,axis=1)
        """
        norm= numpy.nansum(zmean*(zs[1]-zs[0]))
        zmean/= norm
        from scipy import ndimage
        ndimage.filters.gaussian_filter1d(zmean,0.2/(zs[1]-zs[0]),output=zmean)
        norm_twofifty= numpy.nansum(zmean_twofifty*(zs[1]-zs[0]))
        zmean_twofifty/= norm_twofifty
        zmean_const/= zmean_const[0]/zmean[0]
        if options.xmin is None or options.xmax is None:
            xrange= [numpy.amin(zs)-0.2,numpy.amax(zs)+0.1]
        else:
            xrange= [options.xmin,options.xmax]
        if options.ymin is None or options.ymax is None:
            yrange= [0.,1.2*numpy.amax(zmean)]
        else:
            yrange= [options.ymin,options.ymax]
        bovy_plot.bovy_print()
        bovy_plot.bovy_plot(zs,zmean,'k-',xrange=xrange,yrange=yrange,
                            xlabel=options.xlabel,
                            ylabel=options.ylabel)
        bovy_plot.bovy_plot(zs,zmean_const,'k--',overplot=True)
        bovy_plot.bovy_plot(zs,zmean_twofifty,'r--',overplot=True)
        """
        for ii in range(nsigs):
            for jj in range(len(zs)):
                thisf= sorted(fs[jj,:])
                thiscut= 0.5*special.erfc((ii+1.)/math.sqrt(2.))
                zsigs[jj,2*ii]= thisf[int(math.ceil(thiscut*len(samples)))]
                thiscut= 1.-thiscut
                zsigs[jj,2*ii+1]= thisf[int(math.floor(thiscut*len(samples)))]
        colord, cc= (1.-0.75)/nsigs, 1
        nsigma= nsigs
        pyplot.fill_between(zs,zsigs[:,0]/norm,zsigs[:,1]/norm,color='0.75')
        while nsigma > 1:
            pyplot.fill_between(zs,zsigs[:,cc+1]/norm,zsigs[:,cc-1]/norm,
                                color='%f' % (.75+colord*cc))
            pyplot.fill_between(zs,zsigs[:,cc]/norm,zsigs[:,cc+2]/norm,
                                color='%f' % (.75+colord*cc))
            cc+= 1.
            nsigma-= 1
        bovy_plot.bovy_plot(zs,zmean,'k-',overplot=True)
        """
        #Plot the data
        hist= bovy_plot.bovy_hist(rawdata.dered_r,normed=True,bins=31,ec='k',
                                  histtype='step',
                                  overplot=True,range=xrange)
        bovy_plot.bovy_end_print(options.plotfile)

def _predict_rdist(rs,densfunc,params,rmin,rmax,platelb,grmin,grmax,
                   feh,sf,colordist):
    """Predict the r distribution for the sample"""
    plates= sf.plates
    out= numpy.zeros(len(rs))
    for ii in range(len(plates)):
        if 'faint' in sf.platestr[ii].programname:
            out+= sf(numpy.array([plates[ii] for jj in range(len(rs))]),
                     r=rs)*_predict_rdist_plate(rs,densfunc,params,17.8,
                                                rmax,platelb[ii,0],
                                                platelb[ii,1],
                                                grmin,grmax,
                                                feh,colordist)
        else:
            out+= sf(numpy.array([plates[ii] for jj in range(len(rs))]),
                     r=rs)*_predict_rdist_plate(rs,densfunc,params,rmin,
                                                      17.8,platelb[ii,0],
                                                      platelb[ii,1],
                                                      grmin,grmax,
                                                      feh,colordist)
    return out

def _predict_rdist_plate(rs,densfunc,params,rmin,rmax,l,b,grmin,grmax,
                         feh,colordist):
    """Predict the r distribution for a plate"""
    #BOVY: APPROXIMATELY INTEGRATE OVER GR
    ngr= 11
    grs= numpy.linspace(grmin,grmax,ngr)
    out= numpy.zeros(len(rs))
    norm= 0.
    for jj in range(ngr):
       #Calculate distances
        ds= _ivezic_dist(grs[jj],rs,feh)
        #Calculate (R,z)s
        XYZ= bovy_coords.lbd_to_XYZ(numpy.array([l for ii in range(len(ds))]),
                                    numpy.array([b for ii in range(len(ds))]),
                                    ds,degree=True)
        XYZ= XYZ.astype(numpy.float64)
        R= ((8.-XYZ[:,0])**2.+XYZ[:,1]**2.)**(0.5)
        XYZ[:,2]+= _ZSUN
        out+= ds**3.*densfunc(R,XYZ[:,2],params)*colordist(grs[jj])
        norm+= colordist(grs[jj])
    out/= norm
    out[(rs < rmin)]= 0.
    out[(rs > rmax)]= 0.
    return out

def _NormInt(params,XYZ,R,
             sf,plates,platel,plateb,platebright,platefaint,Ap,
             grmin,grmax,rmin,rmax,feh,
             colordist,densfunc):
    out= 0.
    if _INTEGRATEPLATESEP:
        for ii in range(len(plates)):
        #if _DEBUG: print plates[ii], sf(plates[ii])
            if platebright[ii]:
                thisrmin= rmin
                thisrmax= 17.8
            else:
                thisrmin= 17.8
                thisrmax= rmax
            out+= bovy_quadpack.dblquad(_HWRLikeNormInt,grmin,grmax,
                                        lambda x: _ivezic_dist(x,thisrmin,feh),
                                        lambda x: _ivezic_dist(x,thisrmax,feh),
                                        args=(colordist,platel[ii],plateb[ii],
                                              params,densfunc,sf,plates[ii],
                                              feh),
                                        epsrel=_EPSREL,epsabs=_EPSABS)[0]
    else:
        #First bright plates
        brightplates= plates[platebright]
        thisrmin= rmin
        thisrmax= 17.8
        out+= bovy_quadpack.dblquad(_HWRLikeNormIntAll,grmin,grmax,
                                    lambda x: _ivezic_dist(x,thisrmin,feh),
                                    lambda x: _ivezic_dist(x,thisrmax,feh),
                                    args=(colordist,platel[platebright],
                                          plateb[platebright],
                                          params,brightplates,sf,densfunc,
                                          feh),
                                    epsrel=_EPSREL,epsabs=_EPSABS)[0]
        #then faint plates
        faintplates= plates[platefaint]
        thisrmin= 17.8
        thisrmax= rmax
        out+= bovy_quadpack.dblquad(_HWRLikeNormIntAll,grmin,grmax,
                                    lambda x: _ivezic_dist(x,thisrmin,feh),
                                    lambda x: _ivezic_dist(x,thisrmax,feh),
                                    args=(colordist,platel[platefaint],
                                          plateb[platefaint],
                                          params,faintplates,sf,densfunc),
                                    epsrel=_EPSREL,epsabs=_EPSABS)[0]
    out*= Ap
    return out

def _HWRLike(params,XYZ,R,
             sf,plates,platel,plateb,platebright,platefaint,Ap,#selection,platelist,l,b,area of plates
             grmin,grmax,rmin,rmax,feh,#sample definition
             colordist,densfunc): #function that describes the color-distribution and the density
    """log likelihood for the HWR model"""
    return -_HWRLikeMinus(params,XYZ,R,sf,plates,platel,plateb,platebright,
                          platefaint,Ap,
                          grmin,grmax,rmin,rmax,feh,
                          colordist,densfunc)

def _HWRLikeMinus(params,XYZ,R,
                  sf,plates,platel,plateb,platebright,platefaint,Ap,#selection,platelist,l,b,area of plates
                  grmin,grmax,rmin,rmax,feh,#sample definition
                  colordist,densfunc): #function that describes the color-distribution and function that describes the density
    """Minus log likelihood for all models"""
    if densfunc == _HWRDensity:
        if params[0] > 4.6051701859880918 \
                or params[1] > 4.6051701859880918 \
                or params[2] < 0. or params[2] > 1.:
            return numpy.finfo(numpy.dtype(numpy.float64)).max
    elif densfunc == _FlareDensity:
        if params[0] > 4.6051701859880918 \
                or params[1] > 4.6051701859880918 \
                or params[2] > 4.6051701859880918:
            return numpy.finfo(numpy.dtype(numpy.float64)).max       
    elif densfunc == _TwoVerticalDensity:
        if params[0] > 4.6051701859880918 \
                or params[1] > 4.6051701859880918 \
                or params[2] > 4.6051701859880918 \
                or params[3] < 0. or params[3] > 1.:
            return numpy.finfo(numpy.dtype(numpy.float64)).max       
    #First calculate the normalizing integral
    out= _NormInt(params,XYZ,R,
                  sf,plates,platel,plateb,platebright,platefaint,Ap,
                  grmin,grmax,rmin,rmax,feh,
                  colordist,densfunc)
    out= len(R)*numpy.log(out)
    #Then evaluate the individual densities
    out+= -numpy.sum(numpy.log(densfunc(R,XYZ[:,2],params)))
    if _DEBUG: print out, numpy.exp(params)
    return out

def _HWRLikeNormInt(d,gr,colordist,l,b,params,densfunc,sf,plate,feh):
    #Go back to r
    mr= _mr_gi(_gi_gr(gr),feh)
    r= 5.*numpy.log10(d)+10.+mr
    select= sf(plate,r=r)
    #Color density
    rhogr= colordist(gr)
    #Spatial density
    XYZ= bovy_coords.lbd_to_XYZ(l,b,d,degree=True)
    R= ((8.-XYZ[0])**2.+XYZ[1]**2.)**(0.5)
    Z= XYZ[2]+_ZSUN
    dens= densfunc(R,XYZ[2],params)
    #dens= numpy.exp(params[2]-(R-8.)/numpy.exp(params[1])
    #                -numpy.fabs(Z)/numpy.exp(params[0]))
    #Jacobian
    jac= d**2.
    return rhogr*dens*jac*select

def _HWRLikeNormIntAll(d,gr,colordist,l,b,params,plates,sf,densfunc,feh):
    out= 0.
    #Go back to r
    mr= _mr_gi(_gi_gr(gr),feh)
    r= 5.*numpy.log10(d)+10.+mr
    for ii in range(len(plates)):
        #Color density
        rhogr= colordist(gr)
        #Spatial density
        XYZ= bovy_coords.lbd_to_XYZ(l[ii],b[ii],d,degree=True)
        Z= XYZ[2]+_ZSUN
        R= ((8.-XYZ[0])**2.+XYZ[1]**2.)**(0.5)
        dens= densfunc(R,Z,params)
        #Jacobian
        select= sf(plates[ii],r=r)
        jac= d**2. #*numpy.fabs(numpy.cos(b[ii]*_DEGTORAD)) #/R
        out+= rhogr*dens*jac*select
    return out

###############################################################################
#            DENSITY MODELS
###############################################################################
def _HWRDensity(R,Z,params):
    """Double exponential disk + constant,
    params= [hz,hR,Pbad]"""
    hR= numpy.exp(params[1])
    hz= numpy.exp(params[0])
    return ((1.-params[2])/(2.*hz*hR)\
                *numpy.exp(-(R-8.)/hR
                            -numpy.fabs(Z)/numpy.exp(params[0]))\
                +params[2]/(_DZ*8.))
    
def _TwoVerticalDensity(R,Z,params):
    """Double exponential disk with two vertical scale-heights
    params= [hz1,hz2,hR,Pbad]"""
    hR= numpy.exp(params[2])
    hz1= numpy.exp(params[0])
    hz2= numpy.exp(params[1])
    return numpy.exp(-(R-8.)/hR)*\
        ((1.-params[3])/hz1*numpy.exp(-numpy.fabs(Z)/hz1)
         +params[3]/hz2*numpy.exp(-numpy.fabs(Z)/hz2))

def _FlareDensity(R,Z,params):
    """Double exponential disk with flaring scale-height
    params= [hz1,hflare,hR]"""
    hR= numpy.exp(params[2])
    hz= numpy.exp(params[0])
    hf= hz*numpy.exp((R-8.)/numpy.exp(params[1]))
    return numpy.exp(-(R-8.)/hR)/hf*numpy.exp(-numpy.fabs(Z)/hf)
    
def _ConstDensity(R,Z,params):
    """Constant density"""
    return 1.
    
def _const_colordist(gr):
    return 1./.07

def _ivezic_dist(gr,r,feh):
    d,derr= ivezic_dist_gr(gr+r,r,feh)
    return d

def get_options():
    usage = "usage: %prog [options] <savefilename>\n\nsavefilename= name of the file that the fit/samples will be saved to"
    parser = OptionParser(usage=usage)
    parser.add_option("-o",dest='plotfile',
                      help="Name of file for plot")
    parser.add_option("--model",dest='model',default='HWR',
                      help="Model to fit")
    parser.add_option("--sample",dest='sample',default='g',
                      help="Use 'G' or 'K' dwarf sample")
    parser.add_option("--metal",dest='metal',default='rich',
                      help="Use metal-poor or rich sample ('poor', 'rich' or 'all')")
    parser.add_option("--sel",dest='sel',default='r',
                      help="Selection function to use ('constant', 'r')")
    parser.add_option("-n","--nsamples",dest='nsamples',type='int',
                      default=100,
                      help="Number of MCMC samples to use")
    parser.add_option("--subsample",dest='subsample',type='int',
                      default=None,
                      help="If set, use a random subset of this size instead of all of the data")
    parser.add_option("--d1",dest='d1',type='int',default=1,
                      help="First dimension to plot")
    parser.add_option("--d2",dest='d2',type='int',default=4,
                      help="Second dimension to plot")
    parser.add_option("--expd1",action="store_true", dest="expd1",
                      default=False,
                      help="Plot exp() of d1")
    parser.add_option("--expd2",action="store_true", dest="expd2",
                      default=False,
                      help="Plot exp() of d2")
    parser.add_option("--xmin",dest='xmin',type='float',default=None,
                      help="xrange[0]")
    parser.add_option("--xmax",dest='xmax',type='float',default=None,
                      help="xrange[1]")
    parser.add_option("--ymin",dest='ymin',type='float',default=None,
                      help="yrange[0]")
    parser.add_option("--ymax",dest='ymax',type='float',default=None,
                      help="yrange[1]")
    parser.add_option("--xlabel",dest='xlabel',default=None,
                      help="xlabel")
    parser.add_option("--ylabel",dest='ylabel',default=None,
                      help="ylabel")
    parser.add_option("--plotzfunc",action="store_true", dest="plotzfunc",
                      default=False,
                      help="Plot the inferred rho(z) relation")
    parser.add_option("--plotRfunc",action="store_true", dest="plotRfunc",
                      default=False,
                      help="Plot the inferred rho(R) relation")
    parser.add_option("--plotrfunc",action="store_true", dest="plotrfunc",
                      default=False,
                      help="Plot the inferred distribution of rs")
    parser.add_option("--bright",action="store_true", dest="bright",
                      default=False,
                      help="Fit just the bright plates")
    parser.add_option("--faint",action="store_true", dest="faint",
                      default=False,
                      help="Fit just the faint plates")
    return parser

if __name__ == '__main__':
    fitDensz(get_options())
