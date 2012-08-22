import os, os.path
import sys
import copy
import tempfile
import math
import numpy
from scipy import optimize
#from scipy.maxentropy import logsumexp
import cPickle as pickle
from optparse import OptionParser
import multi
import multiprocessing
from galpy.util import bovy_coords, bovy_plot, save_pickles
from galpy import potential
from galpy.actionAngle_src.actionAngleAdiabaticGrid import  actionAngleAdiabaticGrid
from galpy.df_src.quasiisothermaldf import quasiisothermaldf
import bovy_mcmc
import monoAbundanceMW
from segueSelect import read_gdwarfs, read_kdwarfs, _GDWARFFILE, _KDWARFFILE, \
    segueSelect, _mr_gi, _gi_gr
from fitDensz import cb, _ZSUN, DistSpline, _ivezic_dist, _NDS
from pixelFitDens import pixelAfeFeh
_REFR0= 8. #kpc
_REFV0= 220. #km/s
_NGR= 11
_NFEH=11
def pixelFitDynamics(options,args):
    #Read the data
    print "Reading the data ..."
    if options.sample.lower() == 'g':
        if options.select.lower() == 'program':
            raw= read_gdwarfs(_GDWARFFILE,logg=True,ebv=True,sn=options.snmin,nosolar=True)
        else:
            raw= read_gdwarfs(logg=True,ebv=True,sn=options.snmin,nosolar=True)
    elif options.sample.lower() == 'k':
        if options.select.lower() == 'program':
            raw= read_kdwarfs(_KDWARFFILE,logg=True,ebv=True,sn=options.snmin,nosolar=True)
        else:
            raw= read_kdwarfs(logg=True,ebv=True,sn=options.snmin,nosolar=True)
    if not options.bmin is None:
        #Cut on |b|
        raw= raw[(numpy.fabs(raw.b) > options.bmin)]
    #Bin the data
    binned= pixelAfeFeh(raw,dfeh=options.dfeh,dafe=options.dafe)
    #Map the bins with ndata > minndata in 1D
    fehs, afes= [], []
    for ii in range(len(binned.fehedges)-1):
        for jj in range(len(binned.afeedges)-1):
            data= binned(binned.feh(ii),binned.afe(jj))
            if len(data) < options.minndata:
                continue
            fehs.append(binned.feh(ii))
            afes.append(binned.afe(jj))
    nabundancebins= len(fehs)
    fehs= numpy.array(fehs)
    afes= numpy.array(afes)
    #Setup everything for the selection function
    normintstuff= setup_normintstuff(options,raw,binned,fehs,afes)
    #First initialization
    params= initialize(options,fehs,afes)
    #Optimize DF w/ fixed potential and potential w/ fixed DF
    for cc in range(options.ninit):
        print "Iteration %i  / %i ..." % (cc+1,options.ninit)
        print "Optimizing individual DFs with fixed potential ...."
        #params= indiv_optimize_df(params,fehs,afes,binned,options,normintstuff)
        print "Optimizing potential with individual DFs fixed ..."
        params= indiv_optimize_potential(params,fehs,afes,binned,options,normintstuff)
    #Optimize full model
    params= full_optimize(params,fehs,afes,binned,options,normintstuff)
    #Save
    print "BOVY: SAVE"
    #Sample?
    return None

def mloglike(*args,**kwargs):
    """minus log likelihood"""
    return -loglike(*args,**kwargs)

def indiv_optimize_df_mloglike(params,fehs,afes,binned,options,pot,aA,
                               indx,_bigparams,normintstuff):
    """Minus log likelihood when optimizing the parameters of a single DF"""
    #_bigparams is a hack to propagate the parameters to the overall like
    theseparams= set_dfparams(params,_bigparams,indx,options,log=False)
    ml= -indiv_logdf(theseparams,indx,pot,aA,fehs,afes,binned,normintstuff)
    print params, ml
    return ml

def indiv_optimize_pot_mloglike(params,fehs,afes,binned,options,
                                _bigparams,normintstuff):
    """Minus log likelihood when optimizing the parameters of a single DF"""
    #_bigparams is a hack to propagate the parameters to the overall like
    theseparams= set_potparams(params,_bigparams,options,len(fehs))
    ml= mloglike(theseparams,fehs,afes,binned,options,normintstuff)
    print params, ml
    return ml#oglike(theseparams,fehs,afes,binned,options)

def loglike(params,fehs,afes,binned,options,normintstuff):
    """log likelihood"""
    #Set up potential and actionAngle
    pot= setup_potential(params,options,len(fehs))
    aA= setup_aA(pot,options)
    return logdf(params,pot,aA,fehs,afes,binned,normintstuff)

def logdf(params,pot,aA,fehs,afes,binned,normintstuff):
    logl= numpy.zeros(len(fehs))
    #Evaluate individual DFs
    args= (pot,aA,fehs,afes,binned,normintstuff)
    if not options.multi is None:
        logl= multi.parallel_map((lambda x: indiv_logdf(params,x,
                                                        *args)),
                                 range(len(fehs)),
                                 numcores=numpy.amin([len(fehs),
                                                      multiprocessing.cpu_count(),
                                                      options.multi]))
    else:
        for ii in range(len(fehs)):
            print ii
            logl[ii]= indiv_logdf(params,ii,*args)
    return numpy.sum(logl)

def indiv_logdf(params,indx,pot,aA,fehs,afes,binned,normintstuff):
    """Individual population log likelihood"""
    dfparams= get_dfparams(params,indx,options)
    qdf= quasiisothermaldf(*dfparams,pot=pot,aA=aA)
    #Get data ready
    R,vR,vT,z,vz,covv= prepare_coordinates(params,indx,fehs,afes,binned)
    data_lndf= numpy.zeros(len(R))
    for ii in range(len(R)):
        #print R[ii], vR[ii], vT[ii], z[ii], vz[ii]
        data_lndf[ii]= qdf(R[ii],vR[ii],vT[ii],z[ii],vz[ii],log=True)
        if data_lndf[ii] == -numpy.finfo(numpy.dtype(numpy.float64)).max:
            print "Warning; data likelihood is -inf"
            data_lndf[ii]= 0.
    #Normalize
    normalization= calc_normint(qdf,indx,normintstuff)
    return numpy.sum(data_lndf)

def calc_normint(qdf,indx,normintstuff):
    """Calculate the normalization integratl"""
    thisnormintstuff= normintstuff[indx]
    sf, plates,platel,plateb,platebright,platefaint,grmin,grmax,rmin,rmax,fehmin,fehmax,feh,colordist,fehdist,gr,rhogr,rhofeh,mr,dmin,dmax,ds= unpack_thisnormintstuff(thisnorminstuff)
    for ii in range(len(plates)):
        #if _DEBUG: print plates[ii], sf(plates[ii])
        if sf.platebright[str(plates[ii])] and not sf.type_bright.lower() == 'sharprcut':
            thisrmin= rmin
            thisrmax= 17.8
        elif sf.platebright[str(plates[ii])] and sf.type_bright.lower() == 'sharprcut':
            thisrmin= rmin
            thisrmax= numpy.amin([sf.rcuts[str(plates[ii])],17.8])
        elif not sf.type_faint.lower() == 'sharprcut':
            thisrmin= 17.8
            thisrmax= rmax
        elif sf.type_faint.lower() == 'sharprcut':
            thisrmin= 17.8
            thisrmax= numpy.amin([sf.rcuts[str(plates[ii])],rmax])
        #Compute integral by binning everything in distance
        thisout= numpy.zeros(_NDS)
        for kk in range(_NGR):
            for jj in range(_NFEH):
                #What rs do these ds correspond to
                rs= 5.*numpy.log10(ds)+10.+mr[kk,jj]
                thisout+= sf(plates[ii],r=rs)*rhogr[kk]*rhofeh[jj]
        #Calculate (R,z)s
        XYZ= bovy_coords.lbd_to_XYZ(numpy.array([platel[ii] for dd in range(len(ds))]),
                                    numpy.array([plateb[ii] for dd in range(len(ds))]),
                                    ds,degree=True)
        R= ((8.-XYZ[:,0])**2.+XYZ[:,1]**2.)**(0.5)
        XYZ[:,2]+= _ZSUN
        if not options.zmin is None and not options.zmax is None:
            indx= (numpy.fabs(XYZ[:,2]) < options.zmin)
            thisout[indx]= 0.
            indx= (numpy.fabs(XYZ[:,2]) >= options.zmax)
            thisout[indx]= 0.
        if not options.rmin is None and not options.rmax is None:
            indx= (R < options.rmin)
            thisout[indx]= 0.
            indx= (R >= options.rmax)
            thisout[indx]= 0.
        thisout*= ds**2.*numpy.exp(-(R-8.)/3.-numpy.fabs(XYZ[:,2])/0.3) #BOVY:EDIT
        out+= numpy.sum(thisout)
    return out

def setup_normintstuff(options,raw,binned,fehs,afes):
    """Gather everything necessary for calculating the normalization integral"""
    #Load selection function
    plates= numpy.array(list(set(list(raw.plate))),dtype='int') #Only load plates that we use
    print "Using %i plates, %i stars ..." %(len(plates),len(raw))
    sf= segueSelect(plates=plates,type_faint='tanhrcut',
                    sample=options.sample,type_bright='tanhrcut',
                    sn=options.snmin,select=options.select,
                    indiv_brightlims=options.indiv_brightlims)
    platelb= bovy_coords.radec_to_lb(sf.platestr.ra,sf.platestr.dec,
                                     degree=True)
    indx= [not 'faint' in name for name in sf.platestr.programname]
    platebright= numpy.array(indx,dtype='bool')
    indx= ['faint' in name for name in sf.platestr.programname]
    platefaint= numpy.array(indx,dtype='bool')
    if options.sample.lower() == 'g':
        grmin, grmax= 0.48, 0.55
        rmin,rmax= 14.5, 20.2
    if options.sample.lower() == 'k':
        grmin, grmax= 0.55, 0.75
        rmin,rmax= 14.5, 19.
    colorrange=[grmin,grmax]
    out= []
    for ii in range(len(fehs)):
        data= binned(fehs[ii],afes[ii])
        thisnormintstuff= normintstuffClass() #Empty object to act as container
        #Fit this data, set up feh and color
        feh= fehs[ii]
        fehrange= [feh-options.dfeh/2.,feh+options.dfeh/2.]
        #FeH
        fehdist= DistSpline(*numpy.histogram(data.feh,bins=5,
                                             range=fehrange),
                             xrange=fehrange,dontcuttorange=False)
        #Color
        colordist= DistSpline(*numpy.histogram(data.dered_g\
                                                   -data.dered_r,
                                               bins=9,range=colorrange),
                               xrange=colorrange)
        #Integration grid when binning
        grs= numpy.linspace(grmin,grmax,_NGR)
        fehsgrid= numpy.linspace(fehrange[0],fehrange[1],_NFEH)
        rhogr= numpy.array([colordist(gr) for gr in grs])
        rhofeh= numpy.array([fehdist(feh) for feh in fehsgrid])
        mr= numpy.zeros((_NGR,_NFEH))
        for kk in range(_NGR):
            for ll in range(_NFEH):
                mr[kk,ll]= _mr_gi(_gi_gr(grs[kk]),fehsgrid[ll])
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
        if not (options.sample.lower() == 'k' \
                    and options.indiv_brightlims):
            if allbright:
                thisrmin, thisrmax= rmin, 17.8
            elif allfaint:
                thisrmin, thisrmax= 17.8, rmax
            else:
                thisrmin, thisrmax= rmin, rmax
        else:
            thisrmin, thisrmax= rmin, rmax
        _THISNGR, _THISNFEH= 51, 51
        thisgrs= numpy.zeros((_THISNGR,_THISNFEH))
        thisfehs= numpy.zeros((_THISNGR,_THISNFEH))
        for kk in range(_THISNGR):
            thisfehs[kk,:]= numpy.linspace(fehrange[0],fehrange[1],_THISNFEH)
        for kk in range(_THISNFEH):
            thisgrs[:,kk]= numpy.linspace(grmin,grmax,_THISNGR)
        dmin= numpy.amin(_ivezic_dist(thisgrs,thisrmin,thisfehs))
        dmax= numpy.amax(_ivezic_dist(thisgrs,thisrmax,thisfehs))
        ds= numpy.linspace(dmin,dmax,_NDS)
        #Load into thisnormintstuff
        thisnormintstuff.sf= sf
        thisnormintstuff.plates= plates
        thisnormintstuff.platel= platelb[:,0]
        thisnormintstuff.plateb= platelb[:,1]
        thisnormintstuff.platebright= platebright
        thisnormintstuff.platefaint= platefaint
        thisnormintstuff.grmin= grmin
        thisnormintstuff.grmax= grmax
        thisnormintstuff.rmin= rmin
        thisnormintstuff.rmax= rmax
        thisnormintstuff.fehmin= fehrange[0]
        thisnormintstuff.fehmax= fehrange[1]
        thisnormintstuff.feh= feh
        thisnormintstuff.colordist= colordist
        thisnormintstuff.fehdist= fehdist
        thisnormintstuff.grs= grs
        thisnormintstuff.rhogr= rhogr
        thisnormintstuff.mr= mr
        thisnormintstuff.dmin= dmin
        thisnormintstuff.dmax= dmax
        thisnormintstuff.ds= ds
        out.append(thisnormintstuff)
    return out

def unpack_normintstuff(normintstuff):
    return (normintstuff.sf,
            normintstuff.plates,
            thisnormintstuff.platel,
            thisnormintstuff.plateb,
            thisnormintstuff.platebright,
            thisnormintstuff.platefaint,
            thisnormintstuff.grmin,
            thisnormintstuff.grmax,
            thisnormintstuff.rmin,
            thisnormintstuff.rmax,
            thisnormintstuff.fehmin,
            thisnormintstuff.fehmax,
            thisnormintstuff.feh,
            thisnormintstuff.colordist,
            thisnormintstuff.fehdist,
            thisnormintstuff.grs,
            thisnormintstuff.rhogr,
            thisnormintstuff.mr,
            thisnormintstuff.dmin,
            thisnormintstuff.dmax,
            thisnormintstuff.ds)

class normintstuffClass:
    """Empty class to hold normalization integral necessities"""
    pass

def prepare_coordinates(params,indx,fehs,afes,binned):
    vc= get_potparams(params,options,len(fehs))[0] #Always zero-th
    ro= get_ro(params,options)
    vsun= get_vsun(params,options)
    #Create XYZ and R, vxvyvz, cov_vxvyvz
    data= copy.copy(binned(fehs[indx],afes[indx]))
    R= ((1.-data.xc/_REFR0/ro)**2.+(data.yc/_REFR0/ro)**2.)**0.5
    #Confine to R-range?
    if not options.rmin is None and not options.rmax is None:
        dataindx= (R >= options.rmin/_REFR0/ro)*\
            (R < options.rmax/_REFR0/ro)
        data= data[dataindx]
        R= R[dataindx]
    #Confine to z-range?
    if not options.zmin is None and not options.zmax is None:
        dataindx= ((data.zc+_ZSUN) >= options.zmin)*\
            ((data.zc+_ZSUN) < options.zmax)
        data= data[dataindx]
        R= R[dataindx]
    XYZ= numpy.zeros((len(data),3))
    XYZ[:,0]= data.xc/_REFR0/ro
    XYZ[:,1]= data.yc/_REFR0/ro
    XYZ[:,2]= (data.zc+_ZSUN)/_REFR0/ro
    vxvyvz= numpy.zeros((len(data),3))
    vxvyvz[:,0]= data.vxc/_REFV0/vc-vsun[0]
    vxvyvz[:,1]= data.vyc/_REFV0/vc+vsun[1]
    vxvyvz[:,2]= data.vzc/_REFV0/vc+vsun[2]
    cov_vxvyvz= numpy.zeros((len(data),3,3))
    cov_vxvyvz[:,0,0]= data.vxc_err**2./_REFV0/_REFV0/vc/vc
    cov_vxvyvz[:,1,1]= data.vyc_err**2./_REFV0/_REFV0/vc/vc
    cov_vxvyvz[:,2,2]= data.vzc_err**2./_REFV0/_REFV0/vc/vc
    cov_vxvyvz[:,0,1]= data.vxvyc_rho*data.vxc_err*data.vyc_err/_REFV0/_REFV0/vc/vc
    cov_vxvyvz[:,0,2]= data.vxvzc_rho*data.vxc_err*data.vzc_err/_REFV0/_REFV0/vc/vc
    cov_vxvyvz[:,1,2]= data.vyvzc_rho*data.vyc_err*data.vzc_err/_REFV0/_REFV0/vc/vc
    #Rotate to Galactocentric frame
    cosphi= (1.-XYZ[:,0])/R
    sinphi= XYZ[:,1]/R
    vR= -vxvyvz[:,0]*cosphi+vxvyvz[:,1]*sinphi
    vT= vxvyvz[:,0]*sinphi+vxvyvz[:,1]*cosphi
    for rr in range(len(XYZ[:,0])):
        rot= numpy.array([[cosphi[rr],sinphi[rr]],
                          [-sinphi[rr],cosphi[rr]]])
        sxy= cov_vxvyvz[rr,0:2,0:2]
        sRT= numpy.dot(rot,numpy.dot(sxy,rot.T))
        cov_vxvyvz[rr,0:2,0:2]= sRT
    return (R,vR,vT,XYZ[:,2],vxvyvz[:,2],cov_vxvyvz)

def setup_aA(pot,options):
    """Function for setting up the actionAngle object"""
    if options.aAmethod.lower() == 'adiabatic':
        return actionAngleAdiabaticGrid(pot=pot,nR=options.aAnR,
                                        nEz=options.aAnEz,nEr=options.aAnEr,
                                        nLz=options.aAnLz,
                                        zmax=options.aAzmax,
                                        Rmax=options.aARmax)
    
def setup_potential(params,options,npops):
    """Function for setting up the potential"""
    potparams= get_potparams(params,options,npops)
    if options.potential.lower() == 'flatlog':
        return potential.LogarithmicHaloPotential(normalize=1.,q=potparams[1])

def full_optimize(params,fehs,afes,binned,options,normintstuff):
    """Function for optimizing the full set of parameters"""
    return optimize.fmin_powell(mloglike,params,
                                args=(fehs,afes,binned,options,normintstuff))

def indiv_optimize_df(params,fehs,afes,binned,options,normintstuff):
    """Function for optimizing individual DFs with potential fixed"""
    #Set up potential and actionAngle
    pot= setup_potential(params,options,len(fehs))
    aA= setup_aA(pot,options)
    if not options.multi is None:
        #Generate list of temporary files
        tmpfiles= []
        for ii in range(len(fehs)): tmpfiles.append(tempfile.mkstemp())
        try:
            logl= multi.parallel_map((lambda x: indiv_optimize_df_single(params,x,
                                                                         fehs,afes,binned,options,aA,pot,normintstuff,tmpfiles)),
                                     range(len(fehs)),
                                     numcores=numpy.amin([len(fehs),
                                                          multiprocessing.cpu_count(),
                                                          options.multi]))
            #Now read all of the temporary files
            for ii in range(len(fehs)):
                tmpfile= open(tmpfiles[ii][1],'rb')
                new_dfparams= pickle.load(tmpfile)
                params= set_dfparams(new_dfparams,params,ii,options,log=False)
                tmpfile.close()
        finally:
            for ii in range(len(fehs)):
                os.remove(tmpfiles[ii][1])
    else:
        for ii in range(len(fehs)):
            print ii
            init_dfparams= get_dfparams(params,ii,options,log=False)
            new_dfparams= optimize.fmin_powell(indiv_optimize_df_mloglike,
                                               init_dfparams,
                                           args=(fehs,afes,binned,
                                                 options,pot,aA,
                                                 ii,copy.copy(params),
                                                 normintstuff),
                                               callback=cb)
            params= set_dfparams(new_dfparams,params,ii,options,log=False)
    return params

def indiv_optimize_df_single(params,ii,fehs,afes,binned,options,aA,pot,normintstuff,tmpfiles):
    """Function to optimize the DF params for a single population when holding the potential fixed and using multi-processing"""
    print ii
    init_dfparams= get_dfparams(params,ii,options,log=False)
    new_dfparams= optimize.fmin_powell(indiv_optimize_df_mloglike,
                                       init_dfparams,
                                       args=(fehs,afes,binned,
                                             options,pot,aA,
                                             ii,copy.copy(params),
                                             normintstuff),
                                       callback=cb)
    #Now save to temporary pickle
    tmpfile= open(tmpfiles[ii][1],'wb')
    pickle.dump(new_dfparams,tmpfile)
    tmpfile.close()
    return None

def indiv_optimize_potential(params,fehs,afes,binned,options,normintstuff):
    """Function for optimizing the potential w/ individual DFs fixed"""
    init_potparams= numpy.array(get_potparams(params,options,len(fehs)))
    print init_potparams
    new_potparams= optimize.fmin_powell(indiv_optimize_pot_mloglike,
                                        init_potparams,
                                        args=(fehs,afes,binned,options,
                                              copy.copy(params),
                                              normintstuff),
                                        callback=cb)
    params= set_potparams(new_potparams,params,len(fehs))

def initialize(options,fehs,afes):
    """Function to initialize the fit; uses fehs and afes to initialize using MAPS"""
    p= []
    if options.fitro:
        p.append(1.)
    if options.fitvsun:
        p.extend([0.,1.,0.])
    mapfehs= monoAbundanceMW.fehs()
    mapafes= monoAbundanceMW.afes()
    for ii in range(len(fehs)):
        if options.dfmodel.lower() == 'qdf':
            #Find nearest mono-abundance bin that has a measurement
            abindx= numpy.argmin((fehs[ii]-mapfehs)**2./0.01 \
                                     +(afes[ii]-mapafes)**2./0.0025)
            p.extend([numpy.log(2.*monoAbundanceMW.sigmaz(mapfehs[abindx],mapafes[abindx])/_REFV0), #sigmaR
                      numpy.log(monoAbundanceMW.sigmaz(mapfehs[abindx],mapafes[abindx])/_REFV0), #sigmaZ
                      numpy.log(monoAbundanceMW.hr(mapfehs[abindx],mapafes[abindx])/_REFR0), #hR
                      0.,0.]) #hsigR, hsigZ
    if options.potential.lower() == 'flatlog':
        p.extend([1.,.9])
    return p

def get_potparams(p,options,npops):
    """Function that returns the set of potential parameters for these options"""
    startindx= 0
    if options.fitro: startindx+= 1
    if options.fitvsun: startindx+= 3
    ndfparams= get_ndfparams(options)
    startindx+= ndfparams*npops
    if options.potential.lower() == 'flatlog':
        return (p[startindx],p[startindx+1]) #vc, q

def set_potparams(p,params,options,npops):
    """Function that sets the set of potential parameters for these options"""
    startindx= 0
    if options.fitro: startindx+= 1
    if options.fitvsun: startindx+= 3
    ndfparams= get_ndfparams(options)
    startindx+= ndfparams*npops
    if options.potential.lower() == 'flatlog':
        params[startindx]= p[0]
        params[startindx+1]= p[1]
    return params

def get_dfparams(p,indx,options,log=False):
    """Function that returns the set of DF parameters for population indx for these options,
    Returns them such that they can be given to the initialization"""
    startindx= 0
    if options.fitro: startindx+= 1
    if options.fitvsun: startindx+= 3
    ndfparams= get_ndfparams(options)
    startindx+= ndfparams*indx
    if options.dfmodel.lower() == 'qdf':
        if log:
            return [p[startindx],
                    p[startindx+1],
                    p[startindx+2],
                    p[startindx+3],
                    p[startindx+4]]
        else:
            return [numpy.exp(p[startindx]),
                    numpy.exp(p[startindx+1]),
                    numpy.exp(p[startindx+2]),
                    numpy.exp(p[startindx+3]),
                    numpy.exp(p[startindx+4])]
        
def set_dfparams(p,params,indx,options,log=True):
    """Function that sets the set of DF parameters for population indx for these options"""
    startindx= 0
    if options.fitro: startindx+= 1
    if options.fitvsun: startindx+= 3
    ndfparams= get_ndfparams(options)
    startindx+= ndfparams*indx
    if options.dfmodel.lower() == 'qdf':
        for ii in range(ndfparams):
            if log:
                params[startindx+ii]= numpy.log(p[ii])
            else:
                params[startindx+ii]= p[ii]
    return params

def get_ndfparams(options):
    """Function that returns the number of DF parameters for a single population"""
    if options.dfmodel.lower() == 'qdf':
        return 5

def get_ro(p,options):
    """Function that returns R0 for these options"""
    if options.fitro:
        return p[0]
    else:
        return 1.

def get_vsun(p,options):
    """Function to return motion of the Sun in the Galactocentric reference frame"""
    if options.fitvsun:
        return (p[1],p[2],p[3])
    else:
        return (-11.1/_REFV0,245./_REFV0,7.25/_REFV0) #BOVY:ADJUST?

def get_options():
    usage = "usage: %prog [options] <savefile>\n\nsavefile= name of the file that the fits will be saved to"
    parser = OptionParser(usage=usage)
    #Data options
    parser.add_option("--sample",dest='sample',default='g',
                      help="Use 'G' or 'K' dwarf sample")
    parser.add_option("--select",dest='select',default='all',
                      help="Select 'all' or 'program' stars")
    parser.add_option("--dfeh",dest='dfeh',default=0.1,type='float',
                      help="FeH bin size")   
    parser.add_option("--dafe",dest='dafe',default=0.05,type='float',
                      help="[a/Fe] bin size")   
    parser.add_option("--minndata",dest='minndata',default=100,type='int',
                      help="Minimum number of objects in a bin to perform a fit")   
    parser.add_option("--bmin",dest='bmin',type='float',
                      default=None,
                      help="Minimum Galactic latitude")
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
    parser.add_option("--snmin",dest='snmin',type='float',
                      default=15.,
                      help="Minimum S/N")
    parser.add_option("--indiv_brightlims",action="store_true", 
                      dest="indiv_brightlims",
                      default=False,
                      help="indiv_brightlims keyword for segueSelect")
    #Potential model
    parser.add_option("--potential",dest='potential',default='flatlog',
                      help="Potential model to fit")
    #DF model
    parser.add_option("--dfmodel",dest='dfmodel',default='qdf',#Quasi-isothermal
                      help="DF model to fit")
    #Action-angle options
    parser.add_option("--aAmethod",dest='aAmethod',default='adiabatic',
                      help="action angle method to use")
    parser.add_option("--aAnR",dest='aAnR',default=16,type='int',
                      help="Number of radii for Ez grid in aA")
    parser.add_option("--aAnEz",dest='aAnEz',default=16,type='int',
                      help="Number of Ez grid points in aA")
    parser.add_option("--aAnEr",dest='aAnEr',default=31,type='int',
                      help="Number of Er grid points in aA")
    parser.add_option("--aAnLz",dest='aAnLz',default=31,type='int',
                      help="Number of Lz grid points in aA")
    parser.add_option("--aAzmax",dest='aAzmax',default=1.,type='float',
                      help="zmax in aA")
    parser.add_option("--aARmax",dest='aARmax',default=5.,type='float',
                      help="Rmax in aA")
    #Fit options
    parser.add_option("--fitro",action="store_true", dest="fitro",
                      default=False,
                      help="If set, fit for R_0")
    parser.add_option("--fitvsun",action="store_true", dest="fitvsun",
                      default=False,
                      help="If set, fit for v_sun")
    parser.add_option("--ninit",dest='ninit',default=1,type='int',
                      help="Number of initial optimizations to perform (indiv DF + potential w/ fixed DF")
    #Sample?
    parser.add_option("--mcsample",action="store_true", dest="mcsample",
                      default=False,
                      help="If set, sample around the best fit, save in args[1]")
    parser.add_option("--nsamples",dest='nsamples',default=1000,type='int',
                      help="Number of MCMC samples to obtain")
    parser.add_option("-m","--multi",dest='multi',default=None,type='int',
                      help="number of cpus to use")
    return parser
  
if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    pixelFitDynamics(options,args)

