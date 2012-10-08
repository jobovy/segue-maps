#for testing: python fakeDFData.py --dfeh=0.5 --dafe=0.25 --singlefeh=-0.2 --singleafe=0.2 -p 1880 --minndata=1 testFakeDFData.sav
#
import sys
import os, os.path
import copy
import numpy
from scipy.maxentropy import logsumexp
import fitsio
from galpy.util import bovy_coords, bovy_plot
from galpy.df_src.quasiisothermaldf import quasiisothermaldf
import monoAbundanceMW
from segueSelect import _ERASESTR, read_gdwarfs, read_kdwarfs, segueSelect, \
    ivezic_dist_gr
from pixelFitDens import pixelAfeFeh
from pixelFitDF import get_options,fidDens, get_dfparams, get_ro, get_vo, \
    _REFR0, _REFV0, setup_potential, setup_aA, initialize, \
    outDens, _SRHALO, _SZHALO, _SPHIHALO, get_outfrac, \
    get_potparams, set_potparams, \
    _VRSUN, _VTSUN, _VZSUN
from fitDensz import _ZSUN, DistSpline, _ivezic_dist
from compareDataModel import _predict_rdist_plate
_SRHALOFAKE=100. #not the same as in pixelFitDF
_SPHIHALOFAKE=100.
_SZHALOFAKE=100.
_NMIN= 1000
_DEBUG= False
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
        params= pickle.load(savefile)
        savefile.close()
    else:
        params= initialize(options,fehs,afes)
    #Setup potential
    if options.potential.lower() == 'flatlog' and not options.flatten is None:
        #Set flattening
        potparams= list(get_potparams(params,options,len(fehs)))
        potparams[1]= options.flatten
        params= set_potparams(potparams,params,options,len(fehs))
    pot= setup_potential(params,options,len(fehs))
    aA= setup_aA(pot,options)
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
    thishr= qdf.estimate_hr(1.)*_REFR0*ro #qdf._hr*_REFR0*ro
    thishz= qdf.estimate_hz(1.,zmin=0.1,zmax=0.3,nz=11)*_REFR0*ro
    thissr= qdf._sr*_REFV0*vo
    thissz= qdf._sz*_REFV0*vo
    thishsr= qdf._hsr*_REFR0*ro
    thishsz= qdf._hsz*_REFR0*ro
    if True:
        #Make everything 20% larger
        thishr*= 1.2
        thishz*= 1.2
        thishsr*= 1.2
        thishsz*= 1.2
        thissr*= 1.4
        thissz*= 1.2
    #Find nearest mono-abundance bin that has a measurement
    abindx= numpy.argmin((fehs[ii]-mapfehs)**2./0.01 \
                             +(afes[ii]-mapafes)**2./0.0025)
    #Calculate the r-distribution for each plate
    nrs= 1001
    ngr, nfeh= 11, 11 #BOVY: INCREASE?
    tgrs= numpy.linspace(grmin,grmax,ngr)
    tfehs= numpy.linspace(fehrange[0],fehrange[1],nfeh)
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
    logoutfrac+= numpy.log(qdf.surfacemass_z(1.))
    if options.mcout:
        fidoutfrac= get_outfrac(params,ii,options)
        rdistsout= numpy.zeros((len(sf.plates),nrs,ngr,nfeh))
    lagoutfrac= 0.2 #.0000000000000000000000001 #seems good
    for jj in range(len(sf.plates)):
        p= sf.plates[jj]
        sys.stdout.write('\r'+"Working on plate %i (%i/%i)" % (p,jj+1,len(sf.plates)))
        sys.stdout.flush()
        rdists[jj,:,:,:]= _predict_rdist_plate(rs,
                                               lambda x,y,z: fidDens(x,y,thishr,thishz,z),
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
            #First sample a plate
            ran= numpy.random.uniform()
            kk= 0
            while numbers[kk] < ran: kk+= 1
            #Also sample a Feh and a color
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
            #Add mock velocities
            #First calculate all R
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
            sigz= thissz*numpy.exp(-(R-_REFR0)/thishsz)
            sigr= thissr*numpy.exp(-(R-_REFR0)/thishsr)
            sigphi= sigr/numpy.sqrt(2.) #BOVY: FOR NOW
            #Estimate asymmetric drift
            va= sigr**2./2./_REFV0/vo\
                *(-.5+R*(1./thishr+2./thishsr))
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
                newvt.append(numpy.random.normal()*_SPHIHALOFAKE+_REFV0*vo/2.)
            else:
                #Sample from disk gaussian
                newvz.append(numpy.random.normal()*sigz)
                newvr.append(numpy.random.normal()*sigr)
                newvt.append(numpy.random.normal()*sigphi+_REFV0*vo-va)
            newlogratio= list(newlogratio)
            fidlogeval= numpy.log(1.-lagoutfrac)\
                +numpy.log(fidDens(R,z,thishr,thishz,None))\
                -numpy.log(sigr)-numpy.log(sigphi)-numpy.log(sigz)-0.5*(newvr[-1]**2./sigr**2.+newvz[-1]**2./sigz**2.+(newvt[-1]-_REFV0*vo+va)**2./sigphi**2.)
            lagoutlogeval= numpy.log(lagoutfrac)\
                +numpy.log(fidDens(R,z,thishr,thishz,None))\
                -numpy.log(_SRHALOFAKE)\
                -numpy.log(_SPHIHALOFAKE)\
                -numpy.log(_SZHALOFAKE)\
                -0.5*(newvr[-1]**2./_SRHALOFAKE**2.+newvz[-1]**2./_SZHALOFAKE**2.+(newvt[-1]-_REFV0*vo/2)**2./_SPHIHALOFAKE**2.)
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
            if options.mcout:
                outlogeval= logoutfrac+loghalodens\
                    -numpy.log(srhalo)-numpy.log(sphihalo)-numpy.log(szhalo)\
                    -0.5*((newvr[-1]/vo/_REFV0)**2./srhalo**2.+(newvz[-1]/vo/_REFV0)**2./szhalo**2.+(newvt[-1]/vo/_REFV0)**2./sphihalo**2.)\
                    -1.5*numpy.log(2.*math.pi)
                newqdfeval.append(logsumexp([qdflogeval,outlogeval]))
            else:
                newqdfeval.append(qdflogeval)
            newlogratio.append(qdflogeval
                               -newpropeval[-1])#logsumexp([fidlogeval,fidoutlogeval]))
        newlogratio= numpy.array(newlogratio)
        thisnewlogratio= copy.copy(newlogratio)
        maxnewlogratio= numpy.amax(thisnewlogratio)
        thisnewlogratio-= numpy.amax(thisnewlogratio)
        thisnewratio= numpy.exp(thisnewlogratio)
        #Rejection sample
        accept= numpy.random.uniform(size=len(thisnewratio))
        accept= (accept < thisnewratio)
        fraccomplete= float(numpy.sum(accept))/ndata
        fracsuccess= float(numpy.sum(accept))/len(thisnewratio)
        if _DEBUG:
            print fraccomplete, fracsuccess
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
                                yrange=[0.,1.])
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
                        vrpmllpmbb[ii,2],
                        newqdfeval[ii]])
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

if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    numpy.random.seed(options.seed)
    generate_fakeDFData(options,args)

