#Predict the number of halo stars in our sample from the Bell et al. (2008) model
import numpy
from scipy import integrate
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
    n,d,x= compareDataModel.comparernumberPlate(bellfunc,params,sf,cdist,fehdist,data,'all',fehmin=-1.8,fehmax=-0.8,feh=-2.)
    #Normalization of bell
    bell= 10**8./(integrate.dblquad(lambda r,f: r**2*bellfunc(r*numpy.sin(f),r*numpy.cos(f),None),
                            0.,numpy.pi,lambda x:1., lambda x: 40.)[0]*2.*numpy.pi)
    return numpy.sum(n)*7.*(numpy.pi/180.)**2.*(20.2-14.5)/1000*bell*0.2*numpy.log(10.) #these are some factors left out of compareDataModel
