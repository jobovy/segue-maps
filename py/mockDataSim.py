import numpy
from galpy.util import bovy_coords
_DEGTORAD= numpy.pi/180.
def mockDataSim(x,v,selectionFunction=None,kDwarfs=False,ro=8.,phio=0.,
                fehrange=None,fehdist=None,
                colordist=None):
    """
    NAME:
       mockDataSim
    PURPOSE:
       generate mock SEGUE data from an N-body simulation
    INPUT:
       x - position of all simulation particles [x,y,z], [:,3] (kpc)
       v - velocity of all simulation particles [vx,vy,vz] [:,3] (km/s)
       selectionFunction - instance of a selection function
       kDwarfs= if True, generate K dwarfs
       ro= mock solar radius [kpc]
       phio= mock solar azimuth [deg]
    OUTPUT:
    HISTORY:
       2013-01-31 - Written - Bovy (IAS)
    """
    #Calculate l and b to associate data with los
    Xsun= ro*numpy.cos(phio*_DEGTORAD)
    Ysun= ro*numpy.sin(phio*_DEGTORAD)
    Zsun= 0.
    X= -(x[:,0]-Xsun)
    Y= x[:,1]-Ysun
    Z= x[:,2]-Zsun
    lbd= bovy_coords.XYZ_to_lbd(X,Y,Z,degree=True)
    #Now find all simulation particles for each los
    platelb= bovy_coords.radec_to_lb(selectionFunction.platestr.ra,
                                     selectionFunction.platestr.dec,
                                     degree=True)
    simIndices= []
    inLos= numpy.zeros(len(x[:,0]),dtype='bool')
    for ii in range(len(selectionFunction.plates)):
        thisIndx= findSimInLos(platelb[ii,0],platelb[ii,1],
                               lbd[:,0],lbd[:,1])
        simIndices.append(thisIndx)
        inLos[thisIndx]= True
        print platelb[ii,0], platelb[ii,1], numpy.sum(simIndices[-1])
    return lbd
    #Draw colors and metallicities for all stars that belong to a los
    grs= numpy.empty(len(x[:,0]))
    fehs= numpy.empty(len(x[:,0]))
    #Calculate color and metallicity distributions
    if kDwarfs:
        grmin, grmax= 0.55, 0.77
    else:
        grmin, grmax= 0.48, 0.55
    ngr, nfeh= 21, 21
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
    for ii in range(len(x[:,0])):
        if not inLos[ii]: continue
        

def findSimInLos(losl,losb,l,b,radius=1.49):
    """Find the indices of simulations particles within a los"""
    #Compute the cosine of all the differences
    cosdist= cos_sphere_dist(l*_DEGTORAD,numpy.pi/2.-b*_DEGTORAD,
                             losl*_DEGTORAD,numpy.pi/2.-losb*_DEGTORAD)
    return (cosdist >= numpy.cos(radius*_DEGTORAD))

def cos_sphere_dist(theta,phi,theta_o,phi_o):
    """
    NAME:
       cos_sphere_dist
    PURPOSE:
       computes the cosine of the spherical distance between two
       points on the sphere
    INPUT:
       theta  - polar angle [0,pi]
       phi    - azimuth [0,2pi]
       theta  - polar angle of center of the disk
       phi_0  - azimuth of the center of the disk
    OUTPUT:
       spherical distance
    HISTORY:
       2010-04-29 -Written - Bovy (NYU)
    """
    return (numpy.sin(theta)*numpy.sin(theta_o)*(numpy.cos(phi_o)*numpy.cos(phi)+
                                                 numpy.sin(phi_o)*numpy.sin(phi))+
            numpy.cos(theta_o)*numpy.cos(theta))
