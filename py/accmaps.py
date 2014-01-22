import numpy
from scipy import special
from galpy import potential
from galpy.util import bovy_conversion
_GMBULGECUT= 43020.0 #kpc (km/s)^2 = 10^10 Msolar, but later we use 6x10^10
_RCBULGE= 1.9
_PLBULGE= 1.8
_REFV0= 220. #km/s
_REFR0= 8. #kpc
def accmaps(npix=200):
    #First setup the potential, the following are the best-fit parameters from the Sec. 5 of our paper
    params= numpy.array([-1.33663190049,0.998420232634,-3.49031638164,0.31949840593,-1.63965169376])
    try:
        pot= setup_potential(params)
    except RuntimeError: #if this set of parameters gives a nonsense potential
        raise
    #Setup grid and output
    Rs= numpy.linspace(0.01,20.,npix)/_REFR0
    Zs= numpy.linspace(0.,20.,npix)/_REFR0
    accR_dm= numpy.empty((len(Rs),len(Zs)))
    accZ_dm= numpy.empty((len(Rs),len(Zs)))
    accR_baryon= numpy.empty((len(Rs),len(Zs)))
    accZ_baryon= numpy.empty((len(Rs),len(Zs)))
    #Calculate accelerations
    for ii in range(len(Rs)):
        for jj in range(len(Zs)):
            accR_dm[ii,jj]= potential.evaluateRforces(Rs[ii],Zs[jj],[pot[0]])
            accZ_dm[ii,jj]= potential.evaluatezforces(Rs[ii],Zs[jj],[pot[0]])
            accR_baryon[ii,jj]= potential.evaluateRforces(Rs[ii],Zs[jj],pot[1:])
            accZ_baryon[ii,jj]= potential.evaluatezforces(Rs[ii],Zs[jj],pot[1:])
    accR_dm*= bovy_conversion.force_in_10m13kms2(_REFV0*params[1],_REFR0)
    accZ_dm*= bovy_conversion.force_in_10m13kms2(_REFV0*params[1],_REFR0)
    accR_baryon*= bovy_conversion.force_in_10m13kms2(_REFV0*params[1],_REFR0)
    accZ_baryon*= bovy_conversion.force_in_10m13kms2(_REFV0*params[1],_REFR0)
    return (accR_dm,accZ_dm,accR_baryon,accZ_baryon,Rs,Zs)

def write_to_fits(savefilename,npix=200):
    import fitsio
    accs= accmaps(npix=npix)
    out= numpy.recarray((npix,),
                        dtype=[('accr_dm',numpy.ndarray),
                               ('accz_dm',numpy.ndarray),
                               ('accr_baryon',numpy.ndarray),
                               ('accz_baryon',numpy.ndarray),
                               ('rs','f8'),
                               ('zs','f8')])
    for ii in range(npix):
        out[ii].accr_dm= accs[0][ii,:]
        out[ii].accz_dm= accs[1][ii,:]
        out[ii].accr_baryon= accs[2][ii,:]
        out[ii].accz_baryon= accs[3][ii,:]
        out[ii].rs= accs[4][ii]
        out[ii].zs= accs[5][ii]
    fitsio.write(savefilename,out,clobber=True)
    return None

def setup_potential(params):
    """Function for setting up the potential,
    parameters are 1) ln disk scale length / _REFR0
    2) vcirc(R=1)/_REFV0
    3) ln disk scale height / _REFR0
    4) halo fraction F_R,halo(R=1)/(F_R,halo(R=1)+F_R,disk(R=1)
    5) logarithmic derivative of the rotation curve d ln V_c(R=1) / d ln R x 30 (don't ask)"""
    ro= 1.
    vo= params[1]
    dlnvcdlnr= params[4]/30.
    ampb= _GMBULGECUT/2./numpy.pi/_RCBULGE**(3.-_PLBULGE/2.)/special.gamma(3.-_PLBULGE)*(_REFR0*ro)**(2.-_PLBULGE)/_REFV0**2./vo**2.
    #ampb= _GMBULGECUT/_REFR0/ro/_REFV0**2./vo**2. #Good enough approximation; this actually gives 10^10, but we used the above in the paper, which gives ~6x10^9
    bp= potential.PowerSphericalPotentialwCutoff(alpha=_PLBULGE,rc=_RCBULGE/_REFR0/ro,normalize=ampb)
    #Also add 13 Msol/pc^2 with a scale height of 130 pc
    gp= potential.DoubleExponentialDiskPotential(hr=2.*numpy.exp(params[0])/ro,
                                                 hz=0.130/ro/_REFR0,
                                                 normalize=1.)
    gassurfdens= 2.*gp.dens(1.,0.)*_REFV0**2.*vo**2./_REFR0**2./ro**2./4.302*10.**-3.*gp._hz*ro*_REFR0*1000.
    gp= potential.DoubleExponentialDiskPotential(hr=2.*numpy.exp(params[0])/ro,
                                                 hz=0.130/ro/_REFR0,
                                                 normalize=13./gassurfdens)
    fdfh= 1.-13./gassurfdens-ampb
    fd= (1.-params[3])*fdfh
    fh= fdfh-fd
    dp= potential.DoubleExponentialDiskPotential(hr=numpy.exp(params[0])/ro,
                                                 hz=numpy.exp(params[2])/ro,
                                                 normalize=fd)
    plhalo= plhalo_from_dlnvcdlnr(dlnvcdlnr,dp,[bp,gp],fh)
    if plhalo < 0. or plhalo > 3:
        raise RuntimeError("plhalo=%f" % plhalo)
    hp= potential.PowerSphericalPotential(alpha=plhalo,
                                          normalize=fh)
    return [hp,dp,bp,gp]

def plhalo_from_dlnvcdlnr(dlnvcdlnr,diskpot,bulgepot,fh):
    """Calculate the halo's shape corresponding to this rotation curve derivative"""
    #First calculate the derivatives dvc^2/dR of disk and bulge
    dvcdr_disk= -potential.evaluateRforces(1.,0.,diskpot)+potential.evaluateR2derivs(1.,0.,diskpot)
    dvcdr_bulge= -potential.evaluateRforces(1.,0.,bulgepot)+potential.evaluateR2derivs(1.,0.,bulgepot)
    return 2.-(2.*dlnvcdlnr-dvcdr_disk-dvcdr_bulge)/fh
