#for testing: python fakeDFData.py --dfeh=0.5 --dafe=0.25 --singlefeh=-0.2 --singleafe=0.2 -p 1880 --minndata=1 testFakeDFData.sav
#
#for running: python fakeDFData.py --dfeh=0.25 --dafe=0.2 --aAmethod=staeckel -q 0.7 ../fakeDF/fakeDF_dfeh0.25_dafe0.2_q0.7_staeckel.fits --nmcv=100
#
# for adiabatic: python fakeDFData.py --dfeh=0.25 --dafe=0.2 ../fakeDF/fakeDF_dfeh0.25_dafe0.2_q0.6_aa.fits  -q 0.6 --nmcv=100
#
import sys
import os, os.path
import cPickle as pickle
import copy
import numpy
from scipy.maxentropy import logsumexp
from scipy import interpolate
import fitsio
import multi
import multiprocessing
from galpy.util import bovy_coords, bovy_plot
from galpy.df_src.quasiisothermaldf import quasiisothermaldf
import monoAbundanceMW
from segueSelect import _ERASESTR, read_gdwarfs, read_kdwarfs, segueSelect, \
    ivezic_dist_gr, _load_fits, \
    _GDWARFFILE, _KDWARFFILE
from pixelFitDens import pixelAfeFeh
from pixelFitDF import get_options,fidDens, get_dfparams, get_ro, get_vo, \
    _REFR0, _REFV0, setup_potential, setup_aA, initialize, \
    outDens, _SRHALO, _SZHALO, _SPHIHALO, get_outfrac, \
    get_potparams, set_potparams, \
    _VRSUN, _VTSUN, _VZSUN, \
    _SURFNRS, _SURFNZS, _SURFSUBTRACTEXPON
from fitDensz import _ZSUN, DistSpline, _ivezic_dist
from compareDataModel import _predict_rdist_plate
_SRHALOFAKE=100. #not the same as in pixelFitDF
_SPHIHALOFAKE=100.
_SZHALOFAKE=100.
_NMIN= 1000
_DEBUG= True
def generate_fakeDFData(options,args):
    #Check whether the savefile already exists
    if os.path.exists(args[0]):
        savefile= open(args[0],'rb')
        print "Savefile already exists, not re-sampling and overwriting ..."
        return None
    #Read the data
    print "Reading the data ..."
    if options.sample.lower() == 'g':
        if options.select.lower() == 'program':
            raw= read_gdwarfs(_GDWARFFILE,logg=True,ebv=True,sn=options.snmin,nosolar=True,nocoords=True)
        else:
            raw= read_gdwarfs(logg=True,ebv=True,sn=options.snmin,nosolar=True,nocoords=True)
    elif options.sample.lower() == 'k':
        if options.select.lower() == 'program':
            raw= read_kdwarfs(_KDWARFFILE,logg=True,ebv=True,sn=options.snmin,nosolar=True,nocoords=True)
        else:
            raw= read_kdwarfs(logg=True,ebv=True,sn=options.snmin,nosolar=True,
                              nocoords=True)
    if not options.bmin is None:
        #Cut on |b|
        raw= raw[(numpy.fabs(raw.b) > options.bmin)]
    if not options.fehmin is None:
        raw= raw[(raw.feh >= options.fehmin)]
    if not options.fehmax is None:
        raw= raw[(raw.feh < options.fehmax)]
    if not options.afemin is None:
        raw= raw[(raw.afe >= options.afemin)]
    if not options.afemax is None:
        raw= raw[(raw.afe < options.afemax)]
    if not options.plate is None and not options.loo:
        raw= raw[(raw.plate == options.plate)]
    elif not options.plate is None:
        raw= raw[(raw.plate != options.plate)]
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
    if not options.singlefeh is None:
        if options.loo:
            pass
        else:
            #Set up single feh
            indx= binned.callIndx(options.singlefeh,options.singleafe)
            if numpy.sum(indx) == 0:
                raise IOError("Bin corresponding to singlefeh and singleafe is empty ...")
            data= copy.copy(binned.data[indx])
            print "Using %i data points ..." % (len(data))
            #Bin again
            binned= pixelAfeFeh(data,dfeh=options.dfeh,dafe=options.dafe)
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
    #Setup the selection function
    #Load selection function
    plates= numpy.array(list(set(list(raw.plate))),dtype='int') #Only load plates that we use
    print "Using %i plates, %i stars ..." %(len(plates),len(raw))
    sf= segueSelect(plates=plates,type_faint='tanhrcut',
                    sample=options.sample,type_bright='tanhrcut',
                    sn=options.snmin,select=options.select,
                    indiv_brightlims=options.indiv_brightlims)
    platelb= bovy_coords.radec_to_lb(sf.platestr.ra,sf.platestr.dec,
                                     degree=True)
    if options.sample.lower() == 'g':
        grmin, grmax= 0.48, 0.55
        rmin,rmax= 14.50001, 20.199999 #so we don't go out of the range
    if options.sample.lower() == 'k':
        grmin, grmax= 0.55, 0.75
        rmin,rmax= 14.50001, 18.999999
    colorrange=[grmin,grmax]
    mapfehs= monoAbundanceMW.fehs()
    mapafes= monoAbundanceMW.afes()
    #Setup params
    if not options.init is None:
        #Load initial parameters from file
        savefile= open(options.init,'rb')
        tparams= pickle.load(savefile)
        savefile.close()
        #Setup the correct form
        params= initialize(options,fehs,afes)
        params[0:6]= get_dfparams(tparams,options.index,options,log=True)
        params[6:11]= tparams[-5:len(tparams)]
    else:
        params= initialize(options,fehs,afes)
    #Setup potential
    if (options.potential.lower() == 'flatlog' or options.potential.lower() == 'flatlogdisk') \
            and not options.flatten is None:
        #Set flattening
        potparams= list(get_potparams(params,options,len(fehs)))
        potparams[0]= options.flatten
        params= set_potparams(potparams,params,options,len(fehs))
    pot= setup_potential(params,options,len(fehs))
    aA= setup_aA(pot,options)
    if not options.multi is None:
        binned= fakeDFData_abundance_singles(binned,options,args,fehs,afes)
    else:
        for ii in range(len(fehs)):
            print "Working on population %i / %i ..." % (ii+1,len(fehs))
            #Setup qdf
            dfparams= get_dfparams(params,ii,options,log=False)
            vo= get_vo(params,options,len(fehs))
            ro= get_ro(params,options)
            if options.dfmodel.lower() == 'qdf':
                #Normalize
                hr= dfparams[0]/ro
                sr= dfparams[1]/vo
                sz= dfparams[2]/vo
                hsr= dfparams[3]/ro
                hsz= dfparams[4]/ro
            print hr, sr, sz, hsr, hsz
            qdf= quasiisothermaldf(hr,sr,sz,hsr,hsz,pot=pot,aA=aA,cutcounter=True)
            #Some more selection stuff
            data= binned(fehs[ii],afes[ii])
            #feh and color
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
            #Re-sample
            binned= fakeDFData(binned,qdf,ii,params,fehs,afes,options,
                               rmin,rmax,
                               platelb,
                               grmin,grmax,
                               fehrange,
                               colordist,
                               fehdist,feh,sf,
                               mapfehs,mapafes,
                               ro=None,vo=None)
    #Save to new file
    fitsio.write(args[0],binned.data)
    return None

def fakeDFData(binned,qdf,ii,params,fehs,afes,options,
               rmin,rmax,
               platelb,
               grmin,grmax,
               fehrange,
               colordist,
               fehdist,feh,sf,
               mapfehs,mapafes,
               ro=None,vo=None,
               ndata=None,#If set, supersedes binned, only to be used w/ returnlist=True
               returnlist=False): #last one useful for pixelFitDF normintstuff
    if ro is None:
        ro= get_ro(params,options)
    if vo is None:
        vo= get_vo(params,options,len(fehs))
    thishr= qdf.estimate_hr(1.,z=0.125)*_REFR0*ro #qdf._hr*_REFR0*ro
    thishz= qdf.estimate_hz(1.,z=0.125)*_REFR0*ro
    if thishr < 0.: thishr= 10. #Probably close to flat
    if thishz  < 0.1: thishz= 0.2
    thissr= qdf._sr*_REFV0*vo
    thissz= qdf._sz*_REFV0*vo
    thishsr= qdf._hsr*_REFR0*ro
    thishsz= qdf._hsz*_REFR0*ro
    if True:
        if options.aAmethod.lower() == 'staeckel':
            #Make everything 10% larger
            thishr*= 1.2
            thishz*= 1.2
            thishsr*= 1.2
            thishsz*= 1.2
            thissr*= 2.
            thissz*= 2.
        else:
            #Make everything 20% larger
            thishr*= 1.2
            thishz*= 1.2
            thishsr*= 1.2
            thishsz*= 1.2
            thissr*= 2.
            thissz*= 2.
    #Find nearest mono-abundance bin that has a measurement
    abindx= numpy.argmin((fehs[ii]-mapfehs)**2./0.01 \
                             +(afes[ii]-mapafes)**2./0.0025)
    #Calculate the r-distribution for each plate
    nrs= 1001
    ngr, nfeh= 11, 11 #BOVY: INCREASE?
    tgrs= numpy.linspace(grmin,grmax,ngr)
    tfehs= numpy.linspace(fehrange[0]+0.00001,fehrange[1]-0.00001,nfeh)
    #Calcuate FeH and gr distriutions
    fehdists= numpy.zeros(nfeh)
    for jj in range(nfeh): fehdists[jj]= fehdist(tfehs[jj])
    fehdists= numpy.cumsum(fehdists)
    fehdists/= fehdists[-1]
    colordists= numpy.zeros(ngr)
    for jj in range(ngr): colordists[jj]= colordist(tgrs[jj])
    colordists= numpy.cumsum(colordists)
    colordists/= colordists[-1]
    rs= numpy.linspace(rmin,rmax,nrs)
    rdists= numpy.zeros((len(sf.plates),nrs,ngr,nfeh))
    #outlier model that we want to sample (not the one to aid in the sampling)
    srhalo= _SRHALO/vo/_REFV0
    sphihalo= _SPHIHALO/vo/_REFV0
    szhalo= _SZHALO/vo/_REFV0
    logoutfrac= numpy.log(get_outfrac(params,ii,options))
    loghalodens= numpy.log(ro*outDens(1.,0.,None))
    #Calculate surface(R=1.) for relative outlier normalization
    logoutfrac+= numpy.log(qdf.surfacemass_z(1.,ngl=options.ngl))
    if options.mcout:
        fidoutfrac= get_outfrac(params,ii,options)
        rdistsout= numpy.zeros((len(sf.plates),nrs,ngr,nfeh))
    lagoutfrac= 0.15 #.0000000000000000000000001 #seems good
    #Setup density model
    use_real_dens= True
    if use_real_dens:
        #nrs, nzs= 101, 101
        nrs, nzs= 64, 64
        thisRmin, thisRmax= 4./_REFR0, 15./_REFR0
        thiszmin, thiszmax= 0., .8
        Rgrid= numpy.linspace(thisRmin,thisRmax,nrs)
        zgrid= numpy.linspace(thiszmin,thiszmax,nzs)
        surfgrid= numpy.empty((nrs,nzs))
        for ll in range(nrs):
            for jj in range(nzs):
                sys.stdout.write('\r'+"Working on grid-point %i/%i" % (jj+ll*nzs+1,nzs*nrs))
                sys.stdout.flush()
                surfgrid[ll,jj]= qdf.density(Rgrid[ll],zgrid[jj],
                                             nmc=options.nmcv,
                                             ngl=options.ngl)
        sys.stdout.write('\r'+_ERASESTR+'\r')
        sys.stdout.flush()
        if _SURFSUBTRACTEXPON:
            Rs= numpy.tile(Rgrid,(nzs,1)).T
            Zs= numpy.tile(zgrid,(nrs,1))
            ehr= qdf.estimate_hr(1.,z=0.125)
#            ehz= qdf.estimate_hz(1.,zmin=0.5,zmax=0.7)#Get large z behavior right
            ehz= qdf.estimate_hz(1.,z=0.125)
            surfInterp= interpolate.RectBivariateSpline(Rgrid,zgrid,
                                                        numpy.log(surfgrid)
                                                        +Rs/ehr+numpy.fabs(Zs)/ehz,
                                                        kx=3,ky=3,
                                                        s=0.)
#                                                        s=10.*float(nzs*nrs))
        else:
            surfInterp= interpolate.RectBivariateSpline(Rgrid,zgrid,
                                                        numpy.log(surfgrid),
                                                        kx=3,ky=3,
                                                        s=0.)
#                                                        s=10.*float(nzs*nrs))
        if _SURFSUBTRACTEXPON:
            compare_func= lambda x,y,du: numpy.exp(surfInterp.ev(x/ro/_REFR0,numpy.fabs(y)/ro/_REFR0)-x/ro/_REFR0/ehr-numpy.fabs(y)/ehz/ro/_REFR0)
        else:
            compare_func= lambda x,y,du: numpy.exp(surfInterp.ev(x/ro/_REFR0,numpy.fabs(y)/ro/_REFR0))
    else:
        compare_func= lambda x,y,z: fidDens(x,y,thishr,thishz,z)
    for jj in range(len(sf.plates)):
        p= sf.plates[jj]
        sys.stdout.write('\r'+"Working on plate %i (%i/%i)" % (p,jj+1,len(sf.plates)))
        sys.stdout.flush()
        rdists[jj,:,:,:]= _predict_rdist_plate(rs,
                                               compare_func,
                                               None,rmin,rmax,
                                               platelb[jj,0],platelb[jj,1],
                                               grmin,grmax,
                                               fehrange[0],fehrange[1],feh,
                                               colordist,
                                               fehdist,sf,sf.plates[jj],
                                               dontmarginalizecolorfeh=True,
                                               ngr=ngr,nfeh=nfeh)
        if options.mcout:
            rdistsout[jj,:,:,:]= _predict_rdist_plate(rs,
                                                      lambda x,y,z: outDens(x,y,z),
                                                      None,rmin,rmax,
                                                      platelb[jj,0],platelb[jj,1],
                                                      grmin,grmax,
                                                      fehrange[0],fehrange[1],feh,
                                                      colordist,
                                                      fehdist,sf,sf.plates[jj],
                                                      dontmarginalizecolorfeh=True,
                                                      ngr=ngr,nfeh=nfeh)
    sys.stdout.write('\r'+_ERASESTR+'\r')
    sys.stdout.flush()
    numbers= numpy.sum(rdists,axis=3)
    numbers= numpy.sum(numbers,axis=2)
    numbers= numpy.sum(numbers,axis=1)
    numbers= numpy.cumsum(numbers)
    if options.mcout:
        totfid= numbers[-1]
    numbers/= numbers[-1]
    rdists= numpy.cumsum(rdists,axis=1)
    for ll in range(len(sf.plates)):
        for jj in range(ngr):
            for kk in range(nfeh):
                rdists[ll,:,jj,kk]/= rdists[ll,-1,jj,kk]
    if options.mcout:
        numbersout= numpy.sum(rdistsout,axis=3)
        numbersout= numpy.sum(numbersout,axis=2)
        numbersout= numpy.sum(numbersout,axis=1)
        numbersout= numpy.cumsum(numbersout)
        totout= fidoutfrac*numbersout[-1]
        totnumbers= totfid+totout
        totfid/= totnumbers
        totout/= totnumbers
        if _DEBUG:
            print totfid, totout
        numbersout/= numbersout[-1]
        rdistsout= numpy.cumsum(rdistsout,axis=1)
        for ll in range(len(sf.plates)):
            for jj in range(ngr):
                for kk in range(nfeh):
                    rdistsout[ll,:,jj,kk]/= rdistsout[ll,-1,jj,kk]
    #Now sample
    thisout= []
    newrs= []
    newls= []
    newbs= []
    newplate= []
    newgr= []
    newfeh= []
    newds= []
    newzs= []
    newvas= []
    newRs= []
    newphi= []
    newvr= []
    newvt= []
    newvz= []
    newlogratio= []
    newfideval= []
    newqdfeval= []
    newpropeval= []
    if ndata is None:
        thisdata= binned(fehs[ii],afes[ii])
        thisdataIndx= binned.callIndx(fehs[ii],afes[ii])
        ndata= len(thisdata)
    #First sample from spatial density
    for ll in range(ndata):
        #First sample a plate
        ran= numpy.random.uniform()
        kk= 0
        while numbers[kk] < ran: kk+= 1
        #Also sample a FeH and a color
        ran= numpy.random.uniform()
        ff= 0
        while fehdists[ff] < ran: ff+= 1
        ran= numpy.random.uniform()
        cc= 0
        while colordists[cc] < ran: cc+= 1
        #plate==kk, feh=ff,color=cc; now sample from the rdist of this plate
        ran= numpy.random.uniform()
        jj= 0
        if options.mcout and numpy.random.uniform() < totout: #outlier
            while rdistsout[kk,jj,cc,ff] < ran: jj+= 1
            thisoutlier= True
        else:
            while rdists[kk,jj,cc,ff] < ran: jj+= 1
            thisoutlier= False
        #r=jj
        newrs.append(rs[jj])
        newls.append(platelb[kk,0])
        newbs.append(platelb[kk,1])
        newplate.append(sf.plates[kk])
        newgr.append(tgrs[cc])
        newfeh.append(tfehs[ff])
        dist= _ivezic_dist(tgrs[cc],rs[jj],tfehs[ff])
        newds.append(dist)
        #calculate R,z
        XYZ= bovy_coords.lbd_to_XYZ(platelb[kk,0],platelb[kk,1],
                                    dist,degree=True)
        R= ((_REFR0-XYZ[0])**2.+XYZ[1]**2.)**(0.5)
        newRs.append(R)
        phi= numpy.arcsin(XYZ[1]/R)
        if (_REFR0-XYZ[0]) < 0.:
            phi= numpy.pi-phi
        newphi.append(phi)
        z= XYZ[2]+_ZSUN
        newzs.append(z)
    newrs= numpy.array(newrs)
    newls= numpy.array(newls)
    newbs= numpy.array(newbs)
    newplate= numpy.array(newplate)
    newgr= numpy.array(newgr)
    newfeh= numpy.array(newfeh)
    newds= numpy.array(newds)
    newRs= numpy.array(newRs)
    newzs= numpy.array(newzs)
    newphi= numpy.array(newphi)
    #Add mock velocities
    newvr= numpy.empty_like(newrs)
    newvt= numpy.empty_like(newrs)
    newvz= numpy.empty_like(newrs)
    use_sampleV= True
    if use_sampleV:
        for kk in range(ndata):
            newv= qdf.sampleV(newRs[kk]/_REFR0,newzs[kk]/_REFR0,n=1)
            newvr[kk]= newv[0,0]*_REFV0*vo
            newvt[kk]= newv[0,1]*_REFV0*vo
            newvz[kk]= newv[0,2]*_REFV0*vo
    else:
        accept_v= numpy.zeros(ndata,dtype='bool')
        naccept= numpy.sum(accept_v)
        sigz= thissz*numpy.exp(-(newRs-_REFR0)/thishsz)
        sigr= thissr*numpy.exp(-(newRs-_REFR0)/thishsr)
        va= numpy.empty_like(newrs)
        sigphi= numpy.empty_like(newrs)
        maxqdf= numpy.empty_like(newrs)
        nvt= 101
        tvt= numpy.linspace(0.1,1.2,nvt)
        for kk in range(ndata):
            #evaluate qdf for vt
            pvt= qdf(newRs[kk]/ro/_REFR0+numpy.zeros(nvt),
                     numpy.zeros(nvt),
                     tvt,
                     newzs[kk]/ro/_REFR0+numpy.zeros(nvt),
                     numpy.zeros(nvt),log=True)
            pvt_maxindx= numpy.argmax(pvt)
            va[kk]= (1.-tvt[pvt_maxindx])*_REFV0*vo
            if options.aAmethod.lower() == 'adiabaticgrid' and options.flatten >= 0.9:
                maxqdf[kk]= pvt[pvt_maxindx]+numpy.log(250.)
            elif options.aAmethod.lower() == 'adiabaticgrid' and options.flatten >= 0.8:
                maxqdf[kk]= pvt[pvt_maxindx]+numpy.log(250.)
            else:
                maxqdf[kk]= pvt[pvt_maxindx]+numpy.log(40.)
            sigphi[kk]= _REFV0*vo*4.*numpy.sqrt(numpy.sum(numpy.exp(pvt)*tvt**2.)/numpy.sum(numpy.exp(pvt))-(numpy.sum(numpy.exp(pvt)*tvt)/numpy.sum(numpy.exp(pvt)))**2.)
        ntries= 0
        ngtr1= 0
        while naccept < ndata:
            sys.stdout.write('\r %i %i %i \r' % (ntries,naccept,ndata))
            sys.stdout.flush()
            #print ntries, naccept, ndata
            ntries+= 1
            accept_v_comp= True-accept_v
            prop_vr= numpy.random.normal(size=ndata)*sigr
            prop_vt= numpy.random.normal(size=ndata)*sigphi+vo*_REFV0-va
            prop_vz= numpy.random.normal(size=ndata)*sigz
            qoverp= numpy.zeros(ndata)-numpy.finfo(numpy.dtype(numpy.float64)).max
            qoverp[accept_v_comp]= (qdf(newRs[accept_v_comp]/ro/_REFR0,
                                        prop_vr[accept_v_comp]/vo/_REFV0,
                                        prop_vt[accept_v_comp]/vo/_REFV0,
                                        newzs[accept_v_comp]/ro/_REFR0,
                                        prop_vz[accept_v_comp]/vo/_REFV0,log=True)
                                    -maxqdf[accept_v_comp] #normalize max to 1
                                    -(-0.5*(prop_vr[accept_v_comp]**2./sigr[accept_v_comp]**2.+prop_vz[accept_v_comp]**2./sigz[accept_v_comp]**2.+(prop_vt[accept_v_comp]-_REFV0*vo+va[accept_v_comp])**2./sigphi[accept_v_comp]**2.)))
            if numpy.any(qoverp > 0.):
                ngtr1+= numpy.sum((qoverp > 0.))
                if ngtr1 > 5:
                    qindx= (qoverp > 0.)
                    print naccept, ndata, newRs[qindx], newzs[qindx], prop_vr[qindx], va[qindx], sigphi[qindx], prop_vt[qindx], prop_vz[qindx], qoverp[qindx]
                    raise RuntimeError("max qoverp = %f > 1, but shouldn't be" % (numpy.exp(numpy.amax(qoverp))))
            accept_these= numpy.log(numpy.random.uniform(size=ndata))
            #print accept_these, (accept_these < qoverp)
            accept_these= (accept_these < qoverp)        
            newvr[accept_these]= prop_vr[accept_these]
            newvt[accept_these]= prop_vt[accept_these]
            newvz[accept_these]= prop_vz[accept_these]
            accept_v[accept_these]= True
            naccept= numpy.sum(accept_v)
        sys.stdout.write('\r'+_ERASESTR+'\r')
        sys.stdout.flush()
    """
    ntot= 0
    nsamples= 0
    itt= 0
    fracsuccess= 0.
    fraccomplete= 0.
    while fraccomplete < 1.:
        if itt == 0:
            nthis= numpy.amax([ndata,_NMIN])
        else:
            nthis= int(numpy.ceil((1-fraccomplete)/fracsuccess*ndata))
        itt+= 1
        count= 0
        while count < nthis:
            count+= 1
            sigz= thissz*numpy.exp(-(R-_REFR0)/thishsz)
            sigr= thissr*numpy.exp(-(R-_REFR0)/thishsr)
            sigphi= sigr #/numpy.sqrt(2.) #BOVY: FOR NOW
            #Estimate asymmetric drift
            va= sigr**2./2./_REFV0/vo\
                *(-.5+R*(1./thishr+2./thishsr))+10.*numpy.fabs(z)
            newvas.append(va)
            if options.mcout and thisoutlier:
                #Sample from outlier gaussian
                newvz.append(numpy.random.normal()*_SZHALOFAKE*2.)
                newvr.append(numpy.random.normal()*_SRHALOFAKE*2.)
                newvt.append(numpy.random.normal()*_SPHIHALOFAKE*2.)
            elif numpy.random.uniform() < lagoutfrac:
                #Sample from lagging gaussian
                newvz.append(numpy.random.normal()*_SZHALOFAKE)
                newvr.append(numpy.random.normal()*_SRHALOFAKE)
                newvt.append(numpy.random.normal()*_SPHIHALOFAKE*2.+_REFV0*vo/4.)
            else:
                #Sample from disk gaussian
                newvz.append(numpy.random.normal()*sigz)
                newvr.append(numpy.random.normal()*sigr)
                newvt.append(numpy.random.normal()*sigphi+_REFV0*vo-va)
            newlogratio= list(newlogratio)
            fidlogeval= numpy.log(1.-lagoutfrac)\
                     -numpy.log(sigr)-numpy.log(sigphi)-numpy.log(sigz)-0.5*(newvr[-1]**2./sigr**2.+newvz[-1]**2./sigz**2.+(newvt[-1]-_REFV0*vo+va)**2./sigphi**2.)
            lagoutlogeval= numpy.log(lagoutfrac)\
                     -numpy.log(_SRHALOFAKE)\
                     -numpy.log(_SPHIHALOFAKE*2.)\
                     -numpy.log(_SZHALOFAKE)\
                     -0.5*(newvr[-1]**2./_SRHALOFAKE**2.+newvz[-1]**2./_SZHALOFAKE**2.+(newvt[-1]-_REFV0*vo/4.)**2./_SPHIHALOFAKE**2./4.)
            if use_real_dens:
                fidlogeval+= numpy.log(compare_func(R,z,None)[0])
                lagoutlogeval+= numpy.log(compare_func(R,z,None)[0])
            else:
                fidlogeval+= numpy.log(fidDens(R,z,thishr,thishz,None))
                lagoutlogeval+= numpy.log(fidDens(R,z,thishr,thishz,None))
            newfideval.append(fidlogeval)
            if options.mcout:
                fidoutlogeval= numpy.log(fidoutfrac)\
                    +numpy.log(outDens(R,z,None))\
                    -numpy.log(_SRHALOFAKE*2.)\
                    -numpy.log(_SPHIHALOFAKE*2.)\
                    -numpy.log(_SZHALOFAKE*2.)\
                    -0.5*(newvr[-1]**2./_SRHALOFAKE**2./4.+newvz[-1]**2./_SZHALOFAKE**2./4.+newvt[-1]**2./_SPHIHALOFAKE**2./4.)
                newpropeval.append(logsumexp([fidoutlogeval,fidlogeval,
                                              lagoutlogeval]))
            else:
                newpropeval.append(logsumexp([lagoutlogeval,fidlogeval]))
            qdflogeval= qdf(R/ro/_REFR0,newvr[-1]/vo/_REFV0,newvt[-1]/vo/_REFV0,z/ro/_REFR0,newvz[-1]/vo/_REFV0,log=True)
            if isinstance(qdflogeval,(list,numpy.ndarray)):
                qdflogeval= qdflogeval[0]
            if options.mcout:
                outlogeval= logoutfrac+loghalodens\
                    -numpy.log(srhalo)-numpy.log(sphihalo)-numpy.log(szhalo)\
                    -0.5*((newvr[-1]/vo/_REFV0)**2./srhalo**2.+(newvz[-1]/vo/_REFV0)**2./szhalo**2.+(newvt[-1]/vo/_REFV0)**2./sphihalo**2.)\
                    -1.5*numpy.log(2.*numpy.pi)
                newqdfeval.append(logsumexp([qdflogeval,outlogeval]))
            else:
                newqdfeval.append(qdflogeval)
            newlogratio.append(qdflogeval
                               -newpropeval[-1])#logsumexp([fidlogeval,fidoutlogeval]))
        newlogratio= numpy.array(newlogratio)
        thisnewlogratio= copy.copy(newlogratio)
        maxnewlogratio= numpy.amax(thisnewlogratio)
        if False:
            argsort_thisnewlogratio= numpy.argsort(thisnewlogratio)[::-1]
            thisnewlogratio-= thisnewlogratio[argsort_thisnewlogratio[2]] #3rd largest
        else:
            thisnewlogratio-= numpy.amax(thisnewlogratio)
        thisnewratio= numpy.exp(thisnewlogratio)
        if len(thisnewratio.shape) > 1 and thisnewratio.shape[1] == 1:
            thisnewratio= numpy.reshape(thisnewratio,(thisnewratio.shape[0]))
        #Rejection sample
        accept= numpy.random.uniform(size=len(thisnewratio))
        accept= (accept < thisnewratio)
        fraccomplete= float(numpy.sum(accept))/ndata
        fracsuccess= float(numpy.sum(accept))/len(thisnewratio)
        if _DEBUG:
            print fraccomplete, fracsuccess, ndata
            print numpy.histogram(thisnewratio,bins=16)
            indx= numpy.argmax(thisnewratio)
            print numpy.array(newvr)[indx], \
                numpy.array(newvt)[indx], \
                numpy.array(newvz)[indx], \
                numpy.array(newrs)[indx], \
                numpy.array(newds)[indx], \
                numpy.array(newls)[indx], \
                numpy.array(newbs)[indx], \
                numpy.array(newfideval)[indx]
            bovy_plot.bovy_print()
            bovy_plot.bovy_plot(numpy.array(newvt),
                                numpy.exp(numpy.array(newqdfeval)),'b,',
                                xrange=[-300.,500.],yrange=[0.,1.])
            bovy_plot.bovy_plot(newvt,
                                numpy.exp(numpy.array(newpropeval+maxnewlogratio)),
                                'g,',
                                overplot=True)
            bovy_plot.bovy_plot(numpy.array(newvt),
                                numpy.exp(numpy.array(newlogratio-maxnewlogratio)),
                                'b,',
                                xrange=[-300.,500.],
                                #                            xrange=[0.,20.],
                                #                            xrange=[0.,3.],
                                #                            xrange=[6.,9.],
                                yrange=[0.001,1.],semilogy=True)
            bovy_plot.bovy_end_print('/home/bovy/public_html/segue-local/test.png')
    #Now collect the samples
    newrs= numpy.array(newrs)[accept][0:ndata]
    newls= numpy.array(newls)[accept][0:ndata]
    newbs= numpy.array(newbs)[accept][0:ndata]
    newplate= numpy.array(newplate)[accept][0:ndata]
    newgr= numpy.array(newgr)[accept][0:ndata]
    newfeh= numpy.array(newfeh)[accept][0:ndata]
    newvr= numpy.array(newvr)[accept][0:ndata]
    newvt= numpy.array(newvt)[accept][0:ndata]
    newvz= numpy.array(newvz)[accept][0:ndata]
    newphi= numpy.array(newphi)[accept][0:ndata]
    newds= numpy.array(newds)[accept][0:ndata]
    newqdfeval= numpy.array(newqdfeval)[accept][0:ndata]
    """
    vx, vy, vz= bovy_coords.galcencyl_to_vxvyvz(newvr,newvt,newvz,newphi,
                                                vsun=[_VRSUN,_VTSUN,_VZSUN])
    vrpmllpmbb= bovy_coords.vxvyvz_to_vrpmllpmbb(vx,vy,vz,newls,newbs,newds,
                                                 XYZ=False,degree=True)
    pmrapmdec= bovy_coords.pmllpmbb_to_pmrapmdec(vrpmllpmbb[:,1],
                                                 vrpmllpmbb[:,2],
                                                 newls,newbs,
                                                 degree=True)
    #Dump everything for debugging the coordinate transformation
    from galpy.util import save_pickles
    save_pickles('dump.sav',
                 newds,newls,newbs,newphi,
                 newvr,newvt,newvz,
                 vx, vy, vz,
                 vrpmllpmbb,
                 pmrapmdec)
    if returnlist:
        out= []
        for ii in range(ndata):
            out.append([newrs[ii],
                        newgr[ii],
                        newfeh[ii],
                        newls[ii],
                        newbs[ii],
                        newplate[ii],
                        newds[ii],
                        False, #outlier?
                        vrpmllpmbb[ii,0],
                        vrpmllpmbb[ii,1],
                        vrpmllpmbb[ii,2]])#,
#                        newqdfeval[ii]])
        return out
    #Load into data
    binned.data.feh[thisdataIndx]= newfeh
    oldgr= thisdata.dered_g-thisdata.dered_r
    oldr= thisdata.dered_r
    if options.noerrs:
        binned.data.dered_r[thisdataIndx]= newrs
    else:
        binned.data.dered_r[thisdataIndx]= newrs\
            +numpy.random.normal(size=numpy.sum(thisdataIndx))\
            *ivezic_dist_gr(oldgr,0., #g-r is all that counts
                            binned.data.feh[thisdataIndx],
                            dg=binned.data[thisdataIndx].g_err,
                            dr=binned.data[thisdataIndx].r_err,
                            dfeh=binned.data[thisdataIndx].feh_err,
                            return_error=True,
                            _returndmr=True)
    binned.data.dered_r[(binned.data.dered_r >= rmax)]= rmax #tweak to make sure everything stays within the observed range
    if False:
        binned.data.dered_r[(binned.data.dered_r <= rmin)]= rmin
    binned.data.dered_g[thisdataIndx]= oldgr+binned.data[thisdataIndx].dered_r
    #Also change plate and l and b
    binned.data.plate[thisdataIndx]= newplate
    radec= bovy_coords.lb_to_radec(newls,newbs,degree=True)
    binned.data.ra[thisdataIndx]= radec[:,0]
    binned.data.dec[thisdataIndx]= radec[:,1]
    binned.data.l[thisdataIndx]= newls
    binned.data.b[thisdataIndx]= newbs
    if options.noerrs:
        binned.data.vr[thisdataIndx]= vrpmllpmbb[:,0]
        binned.data.pmra[thisdataIndx]= pmrapmdec[:,0]
        binned.data.pmdec[thisdataIndx]= pmrapmdec[:,1]
    else:
        binned.data.vr[thisdataIndx]= vrpmllpmbb[:,0]+numpy.random.normal(size=numpy.sum(thisdataIndx))*binned.data.vr_err[thisdataIndx]
        binned.data.pmra[thisdataIndx]= pmrapmdec[:,0]+numpy.random.normal(size=numpy.sum(thisdataIndx))*binned.data.pmra_err[thisdataIndx]
        binned.data.pmdec[thisdataIndx]= pmrapmdec[:,1]+numpy.random.normal(size=numpy.sum(thisdataIndx))*binned.data.pmdec_err[thisdataIndx]
    return binned

##RUNNING SINGLE BINS IN A SINGLE CALL
def fakeDFData_abundance_singles(binned,options,args,fehs,afes):
    savename= args[0]
    if True:
        if not options.multi is None:
            dummy= multi.parallel_map((lambda x: fakeDFData_abundance_singles_single(options,args,fehs,afes,x,
                                                                                     savename)),
                                      range(len(fehs)),
                                      numcores=numpy.amin([len(fehs),
                                                           multiprocessing.cpu_count(),
                                                       options.multi]))
        else:
            for ii in range(len(fehs)):
                fakeDFData_abundance_singles_single(options,args,fehs,afes,ii,
                                                    savename)
    #Combine
    return combine_abundance_singles(binned,options,savename,fehs,afes)

def combine_abundance_singles(binned,options,savename,fehs,afes):
    for ii in range(len(fehs)):
        thisdata= binned(fehs[ii],afes[ii])
        thisdataIndx= binned.callIndx(fehs[ii],afes[ii])
        #Read from temporary file
        spl= savename.split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        newdata= _load_fits(newname)
        #Load into binned
        binned.data.feh[thisdataIndx]= newdata.feh
        binned.data.dered_r[thisdataIndx]= newdata.dered_r
        binned.data.dered_g[thisdataIndx]= newdata.dered_g
        binned.data.plate[thisdataIndx]= newdata.plate
        binned.data.ra[thisdataIndx]= newdata.ra
        binned.data.dec[thisdataIndx]= newdata.dec
        binned.data.l[thisdataIndx]= newdata.l
        binned.data.b[thisdataIndx]= newdata.b
        binned.data.vr[thisdataIndx]= newdata.vr
        binned.data.pmra[thisdataIndx]= newdata.pmra
        binned.data.pmdec[thisdataIndx]= newdata.pmdec
        #Remove temporary file
        os.remove(newname)
    return binned

def fakeDFData_abundance_singles_single(options,args,fehs,afes,ii,savename):
    #Prepare args and options
    spl= savename.split('.')
    newname= ''
    for jj in range(len(spl)-1):
        newname+= spl[jj]
        if not jj == len(spl)-2: newname+= '.'
    newname+= '_%i.' % ii
    newname+= spl[-1]
    args[0]= newname
    options.singlefeh= fehs[ii]
    options.singleafe= afes[ii]
    #Now run
    toptions= copy.copy(options)
    toptions.multi= None
    toptions.index= ii
    generate_fakeDFData(toptions,args)

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    numpy.random.seed(options.seed)
    generate_fakeDFData(options,args)

