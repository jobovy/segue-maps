import os, os.path
import math
import numpy
import cPickle as pickle
from matplotlib import pyplot
from optparse import OptionParser
from scipy import optimize, special
from galpy.util import bovy_coords, bovy_plot
import bovy_mcmc
from segueSelect import read_gdwarfs
_VERBOSE=True
_DEBUG=False
_ARICHAFERANGE=[0.25,0.5]
_ARICHFEHRANGE=[-1.5,-0.25]
_APOORAFERANGE=[0.,0.25]
_APOORFEHRANGE=[-.5,0.25]
def fitSigz(parser):
    (options,args)= parser.parse_args()
    if len(args) == 0:
        parser.print_help()
        return
    if os.path.exists(args[0]):#Load savefile
        savefile= open(args[0],'rb')
        params= pickle.load(savefile)
        samples= pickle.load(savefile)
        savefile.close()
        if _DEBUG:
            print "Printing mean and std dev of samples ..."
            for ii in range(len(params)):
                xs= numpy.array([s[ii] for s in samples])
                print numpy.mean(xs), numpy.std(xs)
    else:
        #First read the data
        if _VERBOSE:
            print "Reading and parsing data ..."
        XYZ,vxvyvz,cov_vxvyvz,rawdata= readData(metal=options.metal,
                                                sample=options.sample)
        XYZ= XYZ.astype(numpy.float64)
        vxvyvz= vxvyvz.astype(numpy.float64)
        cov_vxvyvz= cov_vxvyvz.astype(numpy.float64)
        R= ((8.-XYZ[:,0])**2.+XYZ[:,1]**2.)**(0.5)
        d= (XYZ[:,2]-.5)
        #Optimize likelihood
        if _VERBOSE:
            print "Optimizing the likelihood ..."
        if options.model.lower() == 'hwr':
            if options.metal == 'rich':
                params= numpy.array([0.02,numpy.log(25.),0.,0.,numpy.log(6.)])
            elif options.metal == 'poor':
                params= numpy.array([0.02,numpy.log(40.),0.,0.,numpy.log(15.)])
            else:
                params= numpy.array([0.02,numpy.log(30.),0.,0.,numpy.log(15.)])
            like_func= _HWRLikeMinus
            pdf_func= _HWRLike
            #Slice sampling keywords
            step= [0.01,0.05,0.3,0.3,0.3]
            create_method=['full','step_out','step_out',
                           'step_out','step_out']
            isDomainFinite=[[True,True],[False,False],
                            [False,False],[False,False],
                            [False,False]]
            domain=[[0.,1.],[0.,0.],[0.,0.],[0.,0.],
                    [0.,4.6051701859880918]]
        elif options.model.lower() == 'isotherm':
            if options.metal == 'rich':
                params= numpy.array([0.02,numpy.log(25.),numpy.log(6.)])
            elif options.metal == 'poor':
                params= numpy.array([0.02,numpy.log(40.),numpy.log(15.)])
            else:
                params= numpy.array([0.02,numpy.log(30.),numpy.log(15.)])
            like_func= _IsothermLikeMinus
            pdf_func= _IsothermLike
            #Slice sampling keywords
            step= [0.01,0.05,0.3]
            create_method=['full','step_out','step_out']
            isDomainFinite=[[True,True],[False,False],
                            [False,True]]
            domain=[[0.,1.],[0.,0.],[0.,4.6051701859880918]]
        params= optimize.fmin_powell(like_func,params,
                                     args=(XYZ,vxvyvz,cov_vxvyvz,R,d))
        if _VERBOSE:
            print "Optimal likelihood:", params
        #Now sample
        if _VERBOSE:
            print "Sampling the likelihood ..."
        samples= bovy_mcmc.slice(params,
                                 step,
                                 pdf_func,
                                 (XYZ,vxvyvz,cov_vxvyvz,R,d),
                                 create_method=create_method,
                                 isDomainFinite=isDomainFinite,
                                 domain=domain,
                                 nsamples=options.nsamples)
        if _DEBUG:
            print "Printing mean and std dev of samples ..."
            for ii in range(len(params)):
                xs= numpy.array([s[ii] for s in samples])
                print numpy.mean(xs), numpy.std(xs)
        if _VERBOSE:
            print "Saving ..."
        savefile= open(args[0],'wb')
        pickle.dump(params,savefile)
        pickle.dump(samples,savefile)
        savefile.close()
    #Plot
    if options.plotfunc:
        #First plot the best fit
        zs= numpy.linspace(0.3,1.2,1001)
        ds= zs-0.5
        func= zfunc
        maxys= math.exp(params[1])+params[2]*ds+params[3]*ds**2.
        if options.xmin is None or options.xmax is None:
            xrange= [numpy.amin(zs)-0.2,numpy.amax(zs)+0.1]
        else:
            xrange= [options.xmin,options.xmax]
        if options.ymin is None or options.ymax is None:
            yrange= [numpy.amin(ys)-1.,numpy.amax(ys)+1.]
        else:
            yrange= [options.ymin,options.ymax]
        #Now plot the mean and std-dev from the posterior
        zmean= numpy.zeros(len(zs))
        nsigs= 3
        zsigs= numpy.zeros((len(zs),2*nsigs))
        fs= numpy.zeros((len(zs),len(samples)))
        ds= zs-0.5
        for ii in range(len(samples)):
            thisparams= samples[ii]
            fs[:,ii]= math.exp(thisparams[1])+thisparams[2]*ds+thisparams[3]*ds**2.
        #Record mean and std-devs
        zmean[:]= numpy.mean(fs,axis=1)
        bovy_plot.bovy_print()
        bovy_plot.bovy_plot(zs,zmean,'k-',xrange=xrange,yrange=yrange,
                            xlabel=options.xlabel,
                            ylabel=options.ylabel)
        for ii in range(nsigs):
            for jj in range(len(zs)):
                thisf= sorted(fs[jj,:])
                thiscut= 0.5*special.erfc((ii+1.)/math.sqrt(2.))
                zsigs[jj,2*ii]= thisf[int(math.floor(thiscut*len(samples)))]
                thiscut= 1.-thiscut
                zsigs[jj,2*ii+1]= thisf[int(math.floor(thiscut*len(samples)))]
        colord, cc= (1.-0.75)/nsigs, 1
        nsigma= nsigs
        pyplot.fill_between(zs,zsigs[:,0],zsigs[:,1],color='0.75')
        while nsigma > 1:
            pyplot.fill_between(zs,zsigs[:,cc+1],zsigs[:,cc-1],
                                color='%f' % (.75+colord*cc))
            pyplot.fill_between(zs,zsigs[:,cc],zsigs[:,cc+2],
                                color='%f' % (.75+colord*cc))
            cc+= 1.
            nsigma-= 1
        bovy_plot.bovy_plot(zs,zmean,'k-',overplot=True)
        #bovy_plot.bovy_plot(zs,maxys,'w--',overplot=True)
        bovy_plot.bovy_end_print(options.plotfile)
    else:
        xs= numpy.array([s[options.d1] for s in samples])
        ys= numpy.array([s[options.d2] for s in samples])
        if options.expd1: xs= numpy.exp(xs)
        if options.expd2: ys= numpy.exp(ys)
        if options.xmin is None or options.xmax is None:
            xrange= [numpy.amin(xs),numpy.amax(xs)]
        else:
            xrange= [options.xmin,options.xmax]
        if options.ymin is None or options.ymax is None:
            yrange= [numpy.amin(ys),numpy.amax(ys)]
        else:
            yrange= [options.ymin,options.ymax]
        bovy_plot.bovy_print()
        bovy_plot.scatterplot(xs,ys,'k,',onedhists=True,xrange=xrange,
                              yrange=yrange,xlabel=options.xlabel,
                              ylabel=options.ylabel)
        maxx, maxy= params[options.d1], params[options.d2]
        if options.expd1: maxx= math.exp(maxx)
        if options.expd2: maxy= math.exp(maxy)
        bovy_plot.bovy_plot([maxx],[maxy],'wx',
                            overplot=True,ms=10.,mew=2.)
        bovy_plot.bovy_end_print(options.plotfile)

#These are for plotting only
def zfunc(ds,params):
    return math.exp(params[1])+params[2]*ds+params[3]*ds**2.
def Rfunc(Rs,params):
    return numpy.exp(-(R-8.)/math.exp(params[4]))

def _HWRLike(params,XYZ,vxvyvz,cov_vxvyvz,R,d):
    """log likelihood for the HWR model"""
    return -_HWRLikeMinus(params,XYZ,vxvyvz,cov_vxvyvz,R,d)

def _HWRLikeMinus(params,XYZ,vxvyvz,cov_vxvyvz,R,d):
    """Minus log likelihood for the HWR model"""
    if params[0] < 0. or params[0] > 1.\
            or params[4] > 4.6051701859880918:
        return numpy.finfo(numpy.dtype(numpy.float64)).max
    #Get model sigma_z
    sigo= math.exp(params[1])
    Rs= math.exp(params[4])
    sigz= (sigo+params[2]*d+params[3]*d**2.)*numpy.exp(-(R-8.)/Rs)
    sigz2= sigz**2.+cov_vxvyvz[:,2,2]
    sigz= numpy.sqrt(sigz2)
    vz= vxvyvz[:,2]
    out= -numpy.sum(numpy.log(params[0]/numpy.sqrt(100.**2.+
                                                   cov_vxvyvz[:,2,2])\
                                  *numpy.exp(-vz**2./2./(100.**2.+
                                                         cov_vxvyvz[:,2,2]))+
                              (1.-params[0])/sigz*numpy.exp(-vz**2./2./\
                                                                 sigz2)))
    if _DEBUG:
        print "Current params, minus likelihood:", params, out
    return out

def _IsothermLike(params,XYZ,vxvyvz,cov_vxvyvz,R,d):
    """log likelihood for the HWR model"""
    return -_IsothermLikeMinus(params,XYZ,vxvyvz,cov_vxvyvz,R,d)

def _IsothermLikeMinus(params,XYZ,vxvyvz,cov_vxvyvz,R,d):
    """Minus log likelihood for the isothermal model"""
    if params[0] < 0. or params[0] > 1. or params[2] > 10.:
        return numpy.finfo(numpy.dtype(numpy.float64)).max
    #Get model sigma_z
    sigo= math.exp(params[1])
    Rs= math.exp(params[2])
    sigz= sigo*numpy.exp(-(R-8.)/Rs)
    sigz2= sigz**2.+cov_vxvyvz[:,2,2]
    sigz= numpy.sqrt(sigz2)
    vz= vxvyvz[:,2]
    out= -numpy.sum(numpy.log(params[0]/numpy.sqrt(100.**2.+
                                                   cov_vxvyvz[:,2,2])\
                                  *numpy.exp(-vz**2./2./(100.**2.+
                                                         cov_vxvyvz[:,2,2]))+
                              (1.-params[0])/sigz*numpy.exp(-vz**2./2./\
                                                                 sigz2)))
    if _DEBUG:
        print "Current params, minus likelihood:", params, out
    return out

def readData(metal='rich',sample='G'):
    if sample.lower() == 'g':
        raw= read_gdwarfs(logg=True)
        #rawdata= numpy.loadtxt(os.path.join(os.getenv('DATADIR'),'bovy',
        #                                    'segue-local','gdwarf_raw.dat'))
    elif sample.lower() == 'k':
        raw= read_kdwarfs(logg=True)
        #rawdata= numpy.loadtxt(os.path.join(os.getenv('DATADIR'),'bovy',
        #                                    'segue-local','kdwarf.dat'))
    #Select sample
    if metal == 'rich':
        indx= (raw.feh > _APOORFEHRANGE[0])*(raw.feh < _APOORFEHRANGE[1])\
            *(raw.afe > _APOORAFERANGE[0])*(raw.afe < _APOORAFERANGE[1])
    elif metal == 'poor':
        indx= (raw.feh > _ARICHFEHRANGE[0])*(raw.feh < _ARICHFEHRANGE[1])\
            *(raw.afe > _ARICHAFERANGE[0])*(raw.afe < _ARICHAFERANGE[1])
    else:
        indx= (raw.feh > -2.)*(raw.feh < 0.5)\
            *(raw.afe > -0.25)*(raw.afe < 0.5)
    raw= raw[indx]
    ndata= len(raw.ra)
    XYZ= numpy.zeros((ndata,3))
    vxvyvz= numpy.zeros((ndata,3))
    cov_vxvyvz= numpy.zeros((ndata,3,3))
    XYZ[:,0]= raw.xc
    XYZ[:,1]= raw.yc
    XYZ[:,2]= raw.zc
    vxvyvz[:,0]= raw.vxc
    vxvyvz[:,1]= raw.vyc
    vxvyvz[:,2]= raw.vzc
    cov_vxvyvz[:,0,0]= raw.vxc_err**2.
    cov_vxvyvz[:,1,1]= raw.vyc_err**2.
    cov_vxvyvz[:,2,2]= raw.vzc_err**2.
    cov_vxvyvz[:,0,1]= raw.vxvyc_rho*raw.vxc_err*raw.vyc_err
    cov_vxvyvz[:,0,2]= raw.vxvzc_rho*raw.vxc_err*raw.vzc_err
    cov_vxvyvz[:,1,2]= raw.vyvzc_rho*raw.vyc_err*raw.vzc_err
    #Load for output
    return (XYZ,vxvyvz,cov_vxvyvz,raw)
    
def get_options():
    usage = "usage: %prog [options] <savefilename>\n\nsavefilename= name of the file that the fit/samples will be saved to"
    parser = OptionParser(usage=usage)
    parser.add_option("-o",dest='plotfile',
                      help="Name of file for plot")
    parser.add_option("--model",dest='model',default='HWR',
                      help="Model to fit")
    parser.add_option("--sample",dest='sample',default='g',
                      help="Use 'G' or 'K' dwarf sample")
    parser.add_option("--metal",dest='metal',default='rich',
                      help="Use metal-poor or rich sample ('poor', 'rich' or 'all')")
    parser.add_option("-n","--nsamples",dest='nsamples',type='int',
                      default=100,
                      help="Number of MCMC samples to use")
    parser.add_option("--d1",dest='d1',type='int',default=1,
                      help="First dimension to plot")
    parser.add_option("--d2",dest='d2',type='int',default=4,
                      help="Second dimension to plot")
    parser.add_option("--expd1",action="store_true", dest="expd1",
                      default=False,
                      help="Plot exp() of d1")
    parser.add_option("--expd2",action="store_true", dest="expd2",
                      default=False,
                      help="Plot exp() of d2")
    parser.add_option("--xmin",dest='xmin',type='float',default=None,
                      help="xrange[0]")
    parser.add_option("--xmax",dest='xmax',type='float',default=None,
                      help="xrange[1]")
    parser.add_option("--ymin",dest='ymin',type='float',default=None,
                      help="yrange[0]")
    parser.add_option("--ymax",dest='ymax',type='float',default=None,
                      help="yrange[1]")
    parser.add_option("--xlabel",dest='xlabel',default=None,
                      help="xlabel")
    parser.add_option("--ylabel",dest='ylabel',default=None,
                      help="ylabel")
    parser.add_option("--plotfunc",action="store_true", dest="plotfunc",
                      default=False,
                      help="Plot samples from the inferred sigma_z(z) relation at R_0")
    parser.add_option("--plotnsamples",dest='plotnsamples',default=10,
                      type='int',
                      help="Plot this number of function samples")
    return parser

if __name__ == '__main__':
    fitSigz(get_options())
