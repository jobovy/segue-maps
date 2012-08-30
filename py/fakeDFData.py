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
        newvr= []
        newvt= []
        newvz= []
        thisdata= binned(fehs[ii],afes[ii])
        ndata= len(data)
        ntot= 0
        nsamples= 0
        while nsamples < ndata:
            nneeded= numpy.amin([ndata-len(newrs),_NMIN-
            while len(thisout) < options.nmc:
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
            thisout.append([rs[jj],tgrs[cc],tfehs[ff],platelb[kk,0],platelb[kk,1],
                            sf.plates[kk],
                            _ivezic_dist(tgrs[cc],rs[jj],tfehs[ff]),
                            thisoutlier])
        #Add mock velocities
        #First calculate all R
        d= numpy.array([o[6] for o in thisout])
        l= numpy.array([o[3] for o in thisout])
        b= numpy.array([o[4] for o in thisout])
        XYZ= bovy_coords.lbd_to_XYZ(l,b,d,degree=True)
        R= ((_REFR0-XYZ[:,0])**2.+XYZ[:,1]**2.)**(0.5)
        XYZ[:,2]+= _ZSUN
        z= XYZ[:,2]
        for jj in range(options.nmc):
            sigz= monoAbundanceMW.sigmaz(mapfehs[abindx],
                                         mapafes[abindx],
                                         r=R[jj])
            sigr= 2.*sigz #BOVY: FOR NOW
            sigphi= sigr/numpy.sqrt(2.) #BOVY: FOR NOW
            #Estimate asymmetric drift
            va= sigr**2./2./_REFV0\
                *(-.5+R[jj]*(1./thishr+2./7.))
            if options.mcout and thisout[jj][7]:
                #Sample from halo gaussian
                vz= numpy.random.normal()*_SZHALO
                vr= numpy.random.normal()*_SRHALO
                vphi= numpy.random.normal()*_SPHIHALO
            else:
                #Sample from disk gaussian
                vz= numpy.random.normal()*sigz
                vr= numpy.random.normal()*sigr
                vphi= numpy.random.normal()*sigphi+_REFV0-va
            #Append to out
            if options.mcout:
                fidlogeval= numpy.log(fidDens(R[jj],z[jj],thishr,thishz,
                                              None))\
                                              -numpy.log(sigr)\
                                              -numpy.log(sigphi)\
                                              -numpy.log(sigz)\
                                              -0.5*(vr**2./sigr**2.+vz**2./sigz**2.+(vphi-_REFV0+va)**2./sigphi**2.)
                outlogeval= numpy.log(fidoutfrac)\
                    +numpy.log(outDens(R[jj],z[jj],None))\
                    -numpy.log(_SRHALO)\
                    -numpy.log(_SPHIHALO)\
                    -numpy.log(_SZHALO)\
                    -0.5*(vr**2./_SRHALO**2.+vz**2./_SZHALO**2.+vphi**2./_SPHIHALO**2.)
                thisout[jj].extend([vr,vphi,vz,#next is evaluation of f at mock
                                    logsumexp([fidlogeval,outlogeval])])
            else:
                thisout[jj].extend([vr,vphi,vz,#next is evaluation of f at mock
                                    numpy.log(fidDens(R[jj],z[jj],thishr,thishz,None))\
                                        -numpy.log(sigr)-numpy.log(sigphi)-numpy.log(sigz)-0.5*(vr**2./sigr**2.+vz**2./sigz**2.+(vphi-_REFV0+va)**2./sigphi**2.)])


