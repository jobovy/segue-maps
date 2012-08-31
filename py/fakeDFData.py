import sys
from segueSelect import _ERASESTR
from pixelFitDF import get_options,fidDensz, get_dfparams, get_ro, get_vo, \
    _REFR0, _REFV0
_NMIN= 1000
def fakeDFData(binned,qdf,ii,params,fehs,afes,options,
               rmin,rmax,
               platelb,
               grmin,grmax,
               fehrange,
               colordist,
               fehdist,sf,
               ro=None,vo=None):
    if ro is None:
        ro= get_ro(params,options)
    if vo is None:
        vo= get_vo(params,options,len(fehs))
    thishr= qdf._hr*_REFR0*ro
    thishz= qdf.estimate_hz(1.,zmin=0.1,zmax=0.3,nz=11)*_REFR0*ro
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
    for jj in range(len(sf.plates)):
        p= sf.plates[jj]
        sys.stdout.write('\r'+"Working on plate %i (%i/%i)" % (p,jj+1,len(sf.plates)))
        sys.stdout.flush()
        rdists[jj,:,:,:]= _predict_rdist_plate(rs,
                                               lambda x,y,z: ExpDens(x,y,thishr,thishz,z),
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
    numbers/= numbers[-1]
    rdists= numpy.cumsum(rdists,axis=1)
    for ll in range(len(sf.plates)):
        for jj in range(ngr):
            for kk in range(nfeh):
                rdists[ll,:,jj,kk]/= rdists[ll,-1,jj,kk]
    #Now sample
    thisout= []
    newrs= []
    newls= []
    newbs= []
    newplate= []
    newgr= []
    newfeh= []
    newds= []
    newphi= []
    newvr= []
    newvt= []
    newvz= []
    newlogratio= []
    thisdata= binned(fehs[ii],afes[ii])
    thisdataIndx= binned.callIndx(fehs[ii],afes[ii])
    ndata= len(data)
    ntot= 0
    nsamples= 0
    itt= 0
    fracsuccess= 0.
    fraccomplete= 0.
    while fraccomplete < 1.:
        if itt == 0:
            nthis= numpy.amin([ndata,_NMIN])
        else:
            nthis= int(numpy.ceil(1-fraccomplete)/fracsuccess*ndata)
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
            while rdists[kk,jj,cc,ff] < ran: jj+= 1
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
            phi= numpy.arcsin(XYZ[1]/R)
            if XYZ[0] < 0.:
                phi= numpy.pi-phi
            newphi.append(phi)
            XYZ[2]+= _ZSUN
            z= XYZ[2]
            sigz= monoAbundanceMW.sigmaz(mapfehs[abindx],
                                         mapafes[abindx],
                                         r=R)
            sigr= 2.*sigz #BOVY: FOR NOW
            sigphi= sigr/numpy.sqrt(2.) #BOVY: FOR NOW
            #Estimate asymmetric drift
            va= sigr**2./2./_REFV0\
                *(-.5+R*(1./thishr+2./7.))
            #Sample from disk gaussian
            newvz.append(numpy.random.normal()*sigz)
            newvr.append(numpy.random.normal()*sigr)
            newvt.append(numpy.random.normal()*sigphi+_REFV0-va)
            newlogratio.append(numpy.log(fidDens(R,z,thishr,thishz,None))\
                                   -numpy.log(sigr)-numpy.log(sigphi)-numpy.log(sigz)-0.5*(newvr[-1]**2./sigr**2.+newvz[-1]**2./sigz**2.+(newvt[-1]-_REFV0+va)**2./sigphi**2.-qdf(R,newvr[-1],newvt[-1],z,newvz[-1],log=True)))
        newlogratio= numpy.array(newlogratio)
        thisnewlogratio= copy.copy(newlogratio)
        thisnewlogratio-= numpy.amax(thisnewlogratio)
        thisnewratio= numpy.exp(thisnewlogratio)
        print numpy.mean(thisnewratio), numpy.std(thisnewratio)
        #Rejection sample
        accept= numpy.random.uniform(size=len(thisnewratio))
        accept= (accept < thisnewratio)
        fraccomplete= float(numpy.sum(accept))/ndata
        fracsuccess= float(numpy.sum(accept))/len(thisnewratio)
    #Now collect the samples
    newrs= numpy.array(newrs)
    newls= numpy.array(newls)
    newbs= numpy.array(newbs)
    newplate= numpy.array(newplate)
    newgr= numpy.array(newgr)
    newfeh= numpy.array(newfeh)
    newvr= numpy.array(newvr)
    newvt= numpy.array(newvt)
    newvz= numpy.array(newvz)
    newphi= numpy.array(newphi)
    newds= numpy.array(newds)
    #Load into data
    oldgr= thisdata.dered_g-thisdata.dered_r
    oldr= thisdata.dered_r
    binned.data[thisdataIndx].dered_r= newrs
    binned.data[thisdataIndx].dered_g= oldgr+binned.data[thisdataIndx].dered_r
    #Also change plate and l and b
    binned.data[thisdataIndx].plate= newplate
    radec= bovy_coords.lb_to_radec(lnewls,newbs,degree=True)
    binned.data[thisdataIndx].ra= radec[:,0]
    binned.data[thisdataIndx].dec= radec[:,1]
    binned.data[thisdataIndx].l= newls
    binned.data[thisdataIndx].b= newbs
    vx, vy, vz= bovy_coords.galcencyl_to_vxvyvz(newvr,newvt,newvz,newphi,
                                                vsun=[-11.1,245.,7.25])
    vrpmllpmbb= bovy_coords.vxvyvz_to_vrpmllpmbb(vx,vy,vz,newls,newbbs,newds,
                                                 XYZ=False,degree=True)
    pmrapmdec= bovy_coords.pmllpmbb_to_pmrapmdec(vrpmllpmbb[:,1],
                                                 vrpmllpmbb[:,2],
                                                 newls,newbs,
                                                 degree=True)
    binned.data[thisdataIndx].vr= vrpmllpmbb[:,0]
    binned.data[thisdataIndx].pmra= pmrapmdec[:,0]
    binned.data[thisdataIndx].pmdec= pmrapmdec[:,1]
    return binned

