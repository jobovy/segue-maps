import cPickle as pickle
import numpy
from scipy import maxentropy, integrate
from galpy import potential
from pixelFitDF import get_options, approxFitResult, _REFV0, _REFR0, \
    setup_potential, setup_aA
from calcDFResults import setup_options
_NOTDONEYET= True
def calcDerivProps(savefilename,vo=1.,zh=400.,dlnvcdlnr=0.):
    options= setup_options(None)
    options.potential= 'dpdiskplhalofixbulgeflatwgasalt'
    options.fitdvt= False
    savefile= open(savefilename,'rb')
    try:
        if not _NOTDONEYET:
            params= pickle.load(savefile)
            mlogl= pickle.load(savefile)
        logl= pickle.load(savefile)
    except:
        return None
    finally:
        savefile.close()
    if _NOTDONEYET:
        logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
    logl[numpy.isnan(logl)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
    marglogl= numpy.zeros((logl.shape[0],logl.shape[3]))
    nderived= 8
    dmarglogl= numpy.zeros((logl.shape[0],logl.shape[3],nderived))
    rds= numpy.linspace(2.,3.4,8)
    fhs= numpy.linspace(0.,1.,16)
    ro= 1.
    for jj in range(marglogl.shape[0]):
        for kk in range(marglogl.shape[1]):
            marglogl[jj,kk]= maxentropy.logsumexp(logl[jj,0,0,kk,:,:,:,0].flatten())
            #Setup potential to calculate stuff
            potparams= numpy.array([numpy.log(rds[jj]/8.),vo,numpy.log(zh/8000.),fhs[kk],dlnvcdlnr])
            try:
                pot= setup_potential(potparams,options,0,returnrawpot=True)
            except RuntimeError:
                continue
            #First up, total surface density
            surfz= 2.*integrate.quad((lambda zz: potential.evaluateDensities(1.,zz,pot)),0.,options.height/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro
            dmarglogl[jj,kk,0]= surfz
            #Disk density
            surfzdisk= 2.*pot[0].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.*400./8000.*ro*_REFR0*1000.
            dmarglogl[jj,kk,1]= surfzdisk
            #halo density
            rhodm= pot[1].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.
            dmarglogl[jj,kk,2]= rhodm
            #total density
            rhoo= potential.evaluateDensities(1.,0.,pot)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.
            dmarglogl[jj,kk,3]= rhoo
            #mass of the disk
            rhod= pot[0].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.
            massdisk= rhod*2.*options.fixzh/8000.*numpy.exp(8./rds[jj])*rds[jj]**2./8**2.*2.*numpy.pi*(ro*_REFR0)**3./10.
            dmarglogl[jj,kk,4]= massdisk
            #pl halo
            dmarglogl[jj,kk,5]= pot[1].alpha
            #vcdvc
            vcdvc= pot[0].vcirc(2.2*rds[jj]/8.)/potential.vcirc(pot,2.2*rds[jj]/8.)
            dmarglogl[jj,kk,6]= vcdvc
            #Rd for fun
            dmarglogl[jj,kk,7]= rds[jj]
    #Calculate mean and stddv
    alogl= marglogl-maxentropy.logsumexp(marglogl.flatten())
    margp= numpy.exp(alogl)
    mean_surfz= numpy.sum(dmarglogl[:,:,0]*margp)
    std_surfz= numpy.sqrt(numpy.sum(dmarglogl[:,:,0]**2.*margp)-mean_surfz**2.)
    mean_surfzdisk= numpy.sum(dmarglogl[:,:,1]*margp)
    std_surfzdisk= numpy.sqrt(numpy.sum(dmarglogl[:,:,1]**2.*margp)-mean_surfzdisk**2.)
    mean_rhodm= numpy.sum(dmarglogl[:,:,2]*margp)
    std_rhodm= numpy.sqrt(numpy.sum(dmarglogl[:,:,2]**2.*margp)-mean_rhodm**2.)
    mean_rhoo= numpy.sum(dmarglogl[:,:,3]*margp)
    std_rhoo= numpy.sqrt(numpy.sum(dmarglogl[:,:,3]**2.*margp)-mean_rhoo**2.)
    mean_massdisk= numpy.sum(dmarglogl[:,:,4]*margp)
    std_massdisk= numpy.sqrt(numpy.sum(dmarglogl[:,:,4]**2.*margp)-mean_massdisk**2.)
    mean_plhalo= numpy.sum(dmarglogl[:,:,5]*margp)
    std_plhalo= numpy.sqrt(numpy.sum(dmarglogl[:,:,5]**2.*margp)-mean_plhalo**2.)
    mean_vcdvc= numpy.sum(dmarglogl[:,:,6]*margp)
    std_vcdvc= numpy.sqrt(numpy.sum(dmarglogl[:,:,6]**2.*margp)-mean_vcdvc**2.)
    mean_rd= numpy.sum(dmarglogl[:,:,7]*margp)
    std_rd= numpy.sqrt(numpy.sum(dmarglogl[:,:,7]**2.*margp)-mean_rd**2.)
    #load into dictionary
    out= {}
    out['surfz']= mean_surfz
    out['surfz_err']= std_surfz
    out['surfzdisk']= mean_surfzdisk
    out['surfzdisk_err']= std_surfzdisk
    out['massdisk']= mean_massdisk
    out['massdisk_err']= std_massdisk
    out['rhoo']= mean_rhoo
    out['rhoo_err']= std_rhoo
    out['rhodm']= mean_rhodm
    out['rhodm_err']= std_rhodm
    out['plhalo']= mean_plhalo
    out['plhalo_err']= std_plhalo
    out['vcdvc']= mean_vcdvc
    out['vcdvc_err']= std_vcdvc
    out['rd']= mean_rd
    out['rd_err']= std_rd
    return out

def rawDerived(marglogl,options,zh=400.,vo=1.,dlnvcdlnr=0.):
    options.fitdvt= False
    nderived= 8
    dmarglogl= numpy.zeros((marglogl.shape[0],marglogl.shape[1],nderived))
    rds= numpy.linspace(2.,3.4,8)
    fhs= numpy.linspace(0.,1.,16)
    ro= 1.
    for jj in range(marglogl.shape[0]):
        for kk in range(marglogl.shape[1]):
            #Setup potential to calculate stuff
            potparams= numpy.array([numpy.log(rds[jj]/8.),vo,numpy.log(zh/8000.),fhs[kk],dlnvcdlnr])
            try:
                pot= setup_potential(potparams,options,0,returnrawpot=True)
            except RuntimeError:
                continue
            #First up, total surface density
            surfz= 2.*integrate.quad((lambda zz: potential.evaluateDensities(1.,zz,pot)),0.,options.height/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro
            dmarglogl[jj,kk,0]= surfz
            #Disk density
            surfzdisk= 2.*pot[0].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.*400./8000.*ro*_REFR0*1000.
            dmarglogl[jj,kk,1]= surfzdisk
            #halo density
            rhodm= pot[1].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.
            dmarglogl[jj,kk,2]= rhodm
            #total density
            rhoo= potential.evaluateDensities(1.,0.,pot)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.
            dmarglogl[jj,kk,3]= rhoo
            #mass of the disk
            rhod= pot[0].dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.
            massdisk= rhod*2.*options.fixzh/8000.*numpy.exp(8./rds[jj])*rds[jj]**2./8**2.*2.*numpy.pi*(ro*_REFR0)**3./10.
            dmarglogl[jj,kk,4]= massdisk
            #pl halo
            dmarglogl[jj,kk,5]= pot[1].alpha
            #vcdvc
            vcdvc= pot[0].vcirc(2.2*rds[jj]/8.)/potential.vcirc(pot,2.2*rds[jj]/8.)
            dmarglogl[jj,kk,6]= vcdvc
            #Rd for fun
            dmarglogl[jj,kk,7]= rds[jj]
    #Calculate mean and stddv
    alogl= marglogl-maxentropy.logsumexp(marglogl.flatten())
    margp= numpy.exp(alogl)
    mean_surfz= numpy.sum(dmarglogl[:,:,0]*margp)
    std_surfz= numpy.sqrt(numpy.sum(dmarglogl[:,:,0]**2.*margp)-mean_surfz**2.)
    mean_surfzdisk= numpy.sum(dmarglogl[:,:,1]*margp)
    std_surfzdisk= numpy.sqrt(numpy.sum(dmarglogl[:,:,1]**2.*margp)-mean_surfzdisk**2.)
    mean_rhodm= numpy.sum(dmarglogl[:,:,2]*margp)
    std_rhodm= numpy.sqrt(numpy.sum(dmarglogl[:,:,2]**2.*margp)-mean_rhodm**2.)
    mean_rhoo= numpy.sum(dmarglogl[:,:,3]*margp)
    std_rhoo= numpy.sqrt(numpy.sum(dmarglogl[:,:,3]**2.*margp)-mean_rhoo**2.)
    mean_massdisk= numpy.sum(dmarglogl[:,:,4]*margp)
    std_massdisk= numpy.sqrt(numpy.sum(dmarglogl[:,:,4]**2.*margp)-mean_massdisk**2.)
    mean_plhalo= numpy.sum(dmarglogl[:,:,5]*margp)
    std_plhalo= numpy.sqrt(numpy.sum(dmarglogl[:,:,5]**2.*margp)-mean_plhalo**2.)
    mean_vcdvc= numpy.sum(dmarglogl[:,:,6]*margp)
    std_vcdvc= numpy.sqrt(numpy.sum(dmarglogl[:,:,6]**2.*margp)-mean_vcdvc**2.)
    mean_rd= numpy.sum(dmarglogl[:,:,7]*margp)
    std_rd= numpy.sqrt(numpy.sum(dmarglogl[:,:,7]**2.*margp)-mean_rd**2.)
    #load into dictionary
    out= {}
    out['surfz']= mean_surfz
    out['surfz_err']= std_surfz
    out['surfzdisk']= mean_surfzdisk
    out['surfzdisk_err']= std_surfzdisk
    out['massdisk']= mean_massdisk
    out['massdisk_err']= std_massdisk
    out['rhoo']= mean_rhoo
    out['rhoo_err']= std_rhoo
    out['rhodm']= mean_rhodm
    out['rhodm_err']= std_rhodm
    out['plhalo']= mean_plhalo
    out['plhalo_err']= std_plhalo
    out['vcdvc']= mean_vcdvc
    out['vcdvc_err']= std_vcdvc
    out['rd']= mean_rd
    out['rd_err']= std_rd
    return out

def calcSurfErr(savefilename,vo=1.,zh=400.,dlnvcdlnr=0.):
    options= setup_options(None)
    options.potential= 'dpdiskplhalofixbulgeflatwgasalt'
    options.fitdvt= False
    savefile= open(savefilename,'rb')
    try:
        if not _NOTDONEYET:
            params= pickle.load(savefile)
            mlogl= pickle.load(savefile)
        logl= pickle.load(savefile)
    except:
        return (None,None,None)
    finally:
        savefile.close()
    if _NOTDONEYET:
        logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
    logl[numpy.isnan(logl)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
    marglogl= numpy.zeros((logl.shape[0]*logl.shape[3]))
    nrs= 8
    rs= numpy.linspace(5.,13.,nrs)
    dmarglogl= numpy.zeros((logl.shape[0]*logl.shape[3],nrs))
    rds= numpy.linspace(2.,3.4,8)
    fhs= numpy.linspace(0.,1.,16)
    ro= 1.
    for jj in range(logl.shape[0]):
        for kk in range(logl.shape[3]):
            marglogl[jj*logl.shape[3]+kk]= maxentropy.logsumexp(logl[jj,0,0,kk,:,:,:,0].flatten())
            #Setup potential to calculate stuff
            potparams= numpy.array([numpy.log(rds[jj]/8.),vo,numpy.log(zh/8000.),fhs[kk],dlnvcdlnr])
            try:
                pot= setup_potential(potparams,options,0,returnrawpot=True)
            except RuntimeError:
                continue
            #First up, total surface density
            for ll in range(nrs):
                surfz= 2.*integrate.quad((lambda zz: potential.evaluateDensities(rs[ll]/_REFR0,zz,pot)),0.,options.height/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro
                dmarglogl[jj*logl.shape[3]+kk,ll]= surfz
    #Calculate mean and stddv
    alogl= marglogl-maxentropy.logsumexp(marglogl.flatten())
    margp= numpy.exp(alogl)
    margp= numpy.tile(margp,(nrs,1)).T
    mean_surfz= numpy.sum(dmarglogl*margp,axis=0)
    std_surfz= numpy.sqrt(numpy.sum(dmarglogl**2.*margp,axis=0)-mean_surfz**2.)
    return (rs,mean_surfz,std_surfz)

def calcSurfRdCorr(savefilename,vo=1.,zh=400.,dlnvcdlnr=0.):
    options= setup_options(None)
    options.potential= 'dpdiskplhalofixbulgeflatwgasalt'
    options.fitdvt= False
    savefile= open(savefilename,'rb')
    try:
        if not _NOTDONEYET:
            params= pickle.load(savefile)
            mlogl= pickle.load(savefile)
        logl= pickle.load(savefile)
    except:
        return (None,None,None)
    finally:
        savefile.close()
    if _NOTDONEYET:
        logl[(logl == 0.)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
    logl[numpy.isnan(logl)]= -numpy.finfo(numpy.dtype(numpy.float64)).max
    marglogl= numpy.zeros((logl.shape[0]*logl.shape[3]))
    nrs= 101
    rs= numpy.linspace(3.,9.,nrs)
    dmarglogl= numpy.zeros((logl.shape[0]*logl.shape[3],nrs))
    rds= numpy.linspace(2.,3.4,8)
    fhs= numpy.linspace(0.,1.,16)
    ro= 1.
    for jj in range(logl.shape[0]):
        for kk in range(logl.shape[3]):
            marglogl[jj*logl.shape[3]+kk]= maxentropy.logsumexp(logl[jj,0,0,kk,:,:,:,0].flatten())
            #Setup potential to calculate stuff
            potparams= numpy.array([numpy.log(rds[jj]/8.),vo,numpy.log(zh/8000.),fhs[kk],dlnvcdlnr])
            try:
                pot= setup_potential(potparams,options,0,returnrawpot=True)
            except RuntimeError:
                continue
            #First up, total surface density
            for ll in range(nrs):
                surfz= 2.*integrate.quad((lambda zz: potential.evaluateDensities(rs[ll]/_REFR0,zz,pot)),0.,options.height/_REFR0/ro)[0]*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*_REFR0*ro
                dmarglogl[jj*logl.shape[3]+kk,ll]= surfz
    #Calculate mean and stddv
    alogl= marglogl-maxentropy.logsumexp(marglogl.flatten())
    margp= numpy.exp(alogl)
    rds= numpy.tile(rds,(logl.shape[3],1)).T.flatten()
    mean_rd= numpy.sum(rds*margp)
    std_rd= numpy.sqrt(numpy.sum(rds**2.*margp,axis=0)-mean_rd**2.)
    margp= numpy.tile(margp,(nrs,1)).T
    mean_surfz= numpy.sum(dmarglogl*margp,axis=0)
    rds= numpy.tile(rds,(nrs,1)).T
    cov_surfz= numpy.sum(dmarglogl*rds*margp,axis=0)-mean_surfz*mean_rd
    std_surfz= numpy.sqrt(numpy.sum(dmarglogl**2.*margp,axis=0)-mean_surfz**2.)
    return (rs,mean_surfz,cov_surfz/std_rd/std_surfz,std_surfz)
