#Predict the number of halo stars in our sample from the Bell et al. (2008) model
import numpy
from scipy import integrate
import isodist
from segueSelect import segueSelect, read_gdwarfs
from fitDensz import _const_colordist
import compareDataModel
def bellfunc(R,Z,params):
    return numpy.power((R*R+1./0.6/.6*Z*Z),-3.)
def haloMDF(feh):
    return numpy.exp(-0.5*(feh+1.52)**2./0.32**2.)
def predictBellHalo():
    params= None
    data= read_gdwarfs(ebv=True,sn=True)
    plates= numpy.array(list(set(list(data.plate))),dtype='int') #Only load plates that we use
    sf= segueSelect(plates=plates,sample='G',type_bright='tanhrcut',
                    type_faint='tanhrcut',sn=True)
    cdist= _const_colordist
    fehdist= haloMDF
    n,d,x= compareDataModel.comparernumberPlate(bellfunc,params,sf,cdist,fehdist,data,'all',fehmin=-1.8,fehmax=-0.8,feh=-2.,noplot=True)
    #Normalization of bell
    bell= 10**8./(integrate.dblquad(lambda r,f: numpy.sin(f)*r**2*bellfunc(r*numpy.sin(f),r*numpy.cos(f),None),
                                    0.,numpy.pi,lambda x:1., lambda x: 40.)[0]*2.*numpy.pi)
    return numpy.sum(n)*7.*(numpy.pi/180.)**2.*(20.2-14.5)/1000*bell*0.2*numpy.log(10.) #these are some factors left out of compareDataModel

def predictDiskMass(modelfunc,params,sf,cdist,fehdist,fehmin,fehmax,feh,
                    data,grmin,grmax,agemin,agemax,normalize='Z'):
    n,d,x= compareDataModel.comparernumberPlate(modelfunc,params,sf,
                                                cdist,fehdist,data,
                                                'all',
                                                fehmin=fehmin,
                                                fehmax=fehmax,
                                                feh=feh,
                                                noplot=True)
    if not normalize.lower() == 'z':
        #Normalization of model
        norm= integrate.dblquad(lambda r,f: r**2*numpy.sin(f)*modelfunc(r*numpy.sin(f),
                                                           r*numpy.cos(f),
                                                           params),
                                0.,numpy.pi,lambda x:0., lambda x: 200.)[0]\
                                *2.*numpy.pi
    else:
        norm= 2.
    pred= numpy.sum(n)*7.*(numpy.pi/180.)**2.*(20.2-14.5)/1000#\
#          *0.2*numpy.log(10.) #these are some factors left out of compareDataModel
    frac= fracMassGRRange(grmin,grmax,agemin,agemax,feh)
    avgmass= averageMassGRRange(grmin,grmax,agemin,agemax,feh)
    return len(data)/norm/pred/frac*avgmass

def averageMassGRRange(grmin,grmax,agemin,agemax,FeH):
    """
    NAME:
       averageMassGRRange
    PURPOSE:
       calculate the average mass in a range of gr using Padova isochrones
    INPUT:
       grmin, grmax -
       agemin, agemax - in Gyr
       FeH - metallicity
    OUTPUT:
       average
    HISTORY:
       2011-09-22 - Written - Bovy (IAS)
    """
    #Load Padova Isochrones
    p= isodist.PadovaIsochrone(type='sdss-ukidss',
                               Z=[0.001+ii*0.001 for ii in range(30)])
    #Find closest Z
    Zs= p.Zs()
    Z= isodist.FEH2Z(FeH)
    indx= (numpy.abs(Zs-Z)).argmin()
    logages= p.logages()


    #Loop over to marginalize
    indices= (logages >= (9.+numpy.log10(agemin)))*\
             (logages <= (9.+numpy.log10(agemax)))
    logages= logages[indices]
    out= 0.
    norm= 0.
    for logage in logages:
        #Load isochrone
        iso= p(logage,Z=Zs[indx],asrecarray=True)
        if not grmin is None:
            indices= ((iso.g-iso.r) >= grmin)*((iso.g-iso.r) <= grmax)
            if numpy.sum(indices) == 0:
                norm+= 10.**(logage-numpy.amin(logages))
                continue
            iso= iso[indices]
        indices= (iso.logg >= 3.5)
        if numpy.sum(indices) == 0:
            norm+= 10.**(logage-numpy.amin(logages))
            continue
        iso= iso[indices]
        #print logage, numpy.amin(iso.M_ini), numpy.amax(iso.M_ini), \
        #      numpy.amin(iso.int_IMF), numpy.amax(iso.int_IMF)
        out+= numpy.mean(iso.M_ini)*10.**(logage-numpy.amin(logages))
        norm+= 10.**(logage-numpy.amin(logages))
    return out/norm

def fracMassGRRange(grmin,grmax,agemin,agemax,FeH):
    """
    NAME:
       fracMassGRRange
    PURPOSE:
       calculate the fraction of mass in a range of gr using Padova isochrones
    INPUT:
       grmin, grmax -
       agemin, agemax - in Gyr
       FeH - metallicity
    OUTPUT:
       fraction
    HISTORY:
       2011-09-22 - Written - Bovy (IAS)
    """
    #Load Padova Isochrones
    p= isodist.PadovaIsochrone(type='sdss-ukidss',
                               Z=[0.001+ii*0.001 for ii in range(30)])
    return massGRRangeInt(grmin,grmax,agemin,agemax,p,FeH,dwarf=True)/\
           massGRRangeInt(None,None,agemin,agemax,p,FeH)

def massGRRangeInt(grmin,grmax,agemin,agemax,p,FeH,dwarf=False):
    #Find closest Z
    Zs= p.Zs()
    Z= isodist.FEH2Z(FeH)
    indx= (numpy.abs(Zs-Z)).argmin()
    logages= p.logages()
    #Loop over to marginalize
    indices= (logages >= (9.+numpy.log10(agemin)))*\
             (logages <= (9.+numpy.log10(agemax)))
    logages= logages[indices]
    out= 0.
    norm= 0.
    for logage in logages:
        #Load isochrone
        iso= p(logage,Z=Zs[indx],asrecarray=True)
        if not grmin is None:
            indices= ((iso.g-iso.r) >= grmin)*((iso.g-iso.r) <= grmax)
            if numpy.sum(indices) == 0:
                norm+= 10.**(logage-numpy.amin(logages))
                continue
            iso= iso[indices]
        if dwarf:
            indices= (iso.logg >= 3.5)
            if numpy.sum(indices) == 0:
                norm+= 10.**(logage-numpy.amin(logages))
                continue
            iso= iso[indices]
        #Sort on int_IMF and integrate
        iso= numpy.sort(iso,order='int_IMF')
        for jj in range(len(iso.M_ini)-1):
            out+= 0.5*(iso.M_ini[jj]+iso.M_ini[jj+1])*(iso.int_IMF[jj+1]-iso.int_IMF[jj])*10.**(logage-numpy.amin(logages))
        norm+= 10.**(logage-numpy.amin(logages))
    return out/norm
