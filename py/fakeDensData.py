import sys
import os, os.path
import math
import numpy
import cPickle as pickle
from matplotlib import pyplot
from optparse import OptionParser
from galpy.util import bovy_coords
import segueSelect
from fitDensz import _HWRDensity, _FlareDensity, _TiedFlareDensity, \
    _TwoVerticalDensity, DistSpline
from fitSigz import readData
from compareDataModel import _predict_rdist_plate
def fakeDensData(parser):
    numpy.random.seed(1)
    (options,args)= parser.parse_args()
    if len(args) == 0:
        parser.print_help()
        return
    #Set up density model
    if options.model.lower() == 'hwr':
        densfunc= _HWRDensity
        if options.metal.lower() == 'rich':
            params= numpy.array([-1.34316986e+00,1.75402412e+00,5.14667706e-04])
        else:
            params= numpy.array([-0.3508148171668,0.65752,0.00206572947631])
    elif options.model.lower() == 'flare':
        densfunc= _FlareDensity
        if options.metal.lower() == 'rich':
            params= numpy.log(numpy.array([0.3,2.5,2.5]))
        else:
            params= numpy.log(numpy.array([0.3,2.5,2.5]))
    elif options.model.lower() == 'tiedflare':
        densfunc= _TiedFlareDensity
        if options.metal.lower() == 'rich':
            params= numpy.log(numpy.array([0.3,2.5]))
        else:
            params= numpy.log(numpy.array([0.3,2.5]))
    elif options.model.lower() == 'twovertical':
        densfunc= _TwoVerticalDensity
        if options.metal.lower() == 'rich':
            params= numpy.log(numpy.array([0.3,0.8,2.5,1.05]))
        else:
            params= numpy.log(numpy.array([0.3,0.8,2.5,1.05]))
    #Data
    XYZ,vxvyvz,cov_vxvyvz,rawdata= readData(metal=options.metal,
                                            sample=options.sample)
    if options.metal.lower() == 'rich':
        feh= -0.15
        fehrange= [-0.4,0.5]
    elif options.metal.lower() == 'poor':
        feh= -0.65
        fehrange= [-1.5,-0.5]
    else:
        feh= -0.5 
    #Load model distributions
    if options.sample.lower() == 'g':
        colorrange=[0.48,0.55]
    elif options.sample.lower() == 'k':
        colorrange=[0.55,0.75]
    #FeH
    fehdist= DistSpline(*numpy.histogram(rawdata.feh,bins=11,range=fehrange),
                         xrange=fehrange)
    #Color
    colordist= DistSpline(*numpy.histogram(rawdata.dered_g-rawdata.dered_r,
                                           bins=9,range=colorrange),
                           xrange=colorrange)
    #Load selection function
    sf= segueSelect.segueSelect(type_bright=options.sel_bright,
                                type_faint=options.sel_faint,
                                sample=options.sample)
    platelb= bovy_coords.radec_to_lb(sf.platestr.ra,sf.platestr.dec,
                                     degree=True)
    if options.sample.lower() == 'g':
        rmin, rmax= 14.5, 20.2
        grmin, grmax= 0.48, 0.55
    elif options.sample.lower() == 'k':
        rmin, rmax= 14.5, 19.
        grmin, grmax= 0.55, 0.75
    #Calculate the r-distribution for each plate
    nrs= 1001
    ngr, nfeh= 21, 21
    grs= numpy.linspace(grmin,grmax,ngr)
    fehs= numpy.linspace(fehrange[0],fehrange[1],nfeh)
    #Calcuate FeH and gr distriutions
    fehdists= numpy.zeros(nfeh)
    for ii in range(nfeh): fehdists[ii]= fehdist(fehs[ii])
    fehdists= numpy.cumsum(fehdists)
    fehdists/= fehdists[-1]
    colordists= numpy.zeros(ngr)
    for ii in range(ngr): colordists[ii]= colordist(grs[ii])
    colordists= numpy.cumsum(colordists)
    colordists/= colordists[-1]
    rs= numpy.linspace(rmin,rmax,nrs)
    rdists= numpy.zeros((len(sf.plates),nrs,ngr,nfeh))
    for ii in range(3):#len(sf.plates)):
        p= sf.plates[ii]
        sys.stdout.write('\r'+"Working on plate %i (%i/%i)" % (p,ii+1,len(sf.plates)))
        sys.stdout.flush()
        rdists[ii,:,:,:]= _predict_rdist_plate(rs,densfunc,params,rmin,rmax,
                                               platelb[ii,0],platelb[ii,1],
                                               grmin,grmax,
                                               fehrange[0],fehrange[1],feh,
                                               colordist,
                                               fehdist,sf,sf.plates[ii],
                                               dontmarginalizecolorfeh=True,
                                               ngr=ngr,nfeh=nfeh)
    sys.stdout.write('\r'+segueSelect._ERASESTR+'\r')
    sys.stdout.flush()
    numbers= numpy.sum(rdists,axis=3)
    numbers= numpy.sum(numbers,axis=2)
    numbers= numpy.sum(numbers,axis=1)
    numbers= numpy.cumsum(numbers)
    numbers/= numbers[-1]
    rdists= numpy.cumsum(rdists,axis=1)
    for ii in range(len(sf.plates)):
        rdists[ii,:,:,:]/= rdists[ii,-1,:,:]
    #Now sample until we're done
    out= []
    while len(out) < options.nsamples:
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
        out.append([rs[jj],grs[cc],fehs[ff],platelb[kk,0],platelb[kk,1],
                    sf.plates[kk]])
    #Save as pickle
    savefile= open(args[0],'wb')
    pickle.dump(out,savefile)
    savefile.close()

def get_options():
    usage = "usage: %prog [options] <savefilename>\n\nsavefilename= name of the file that the fake data will be saved to"
    parser = OptionParser(usage=usage)
    parser.add_option("--model",dest='model',default='HWR',
                      help="Model to fit")
    parser.add_option("--sample",dest='sample',default='g',
                      help="Use 'G' or 'K' dwarf sample")
    parser.add_option("--metal",dest='metal',default='rich',
                      help="Use metal-poor or rich sample ('poor', 'rich' or 'all')")
    parser.add_option("--sel_bright",dest='sel_bright',default='sharprcut',
                      help="Selection function to use ('constant', 'r', 'platesn_r')")
    parser.add_option("--sel_faint",dest='sel_faint',default='sharprcut',
                      help="Selection function to use ('constant', 'r', 'platesn_r')")
    parser.add_option("-n","--nsamples",dest='nsamples',type='int',
                      default=5000,
                      help="Number of fake data points to return")
    return parser

if __name__ == '__main__':
    fakeDensData(get_options())
