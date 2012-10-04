import os, os.path
import numpy
import cPickle as pickle
from optparse import OptionParser
from galpy.util import bovy_plot, bovy_coords
import segueSelect
import compareDataModel
from segueSelect import read_gdwarfs, read_kdwarfs, _GDWARFFILE, _KDWARFFILE, \
    segueSelect, _ERASESTR
from fitDensz import cb, _ZSUN, DistSpline, _ivezic_dist, _NDS
from pixelFitDens import pixelAfeFeh
from pixelFitDF import *
from pixelFitDF import _SURFNRS, _SURFNZS, _PRECALCVSAMPLES, _REFR0, _REFV0
def plotDensComparisonDF(options,args):
    #Read data etc.
    print "Reading the data ..."
    raw= read_rawdata(options)
    #Bin the data
    binned= pixelAfeFeh(raw,dfeh=options.dfeh,dafe=options.dafe)
    #Map the bins with ndata > minndata in 1D
    fehs, afes= [], []
    for ii in range(len(binned.fehedges)-1):
        for jj in range(len(binned.afeedges)-1):
            data= binned(binned.feh(ii),binned.afe(jj))
            if len(data) < options.minndata:
                continue
            #print binned.feh(ii), binned.afe(jj), len(data)
            fehs.append(binned.feh(ii))
            afes.append(binned.afe(jj))
    nabundancebins= len(fehs)
    fehs= numpy.array(fehs)
    afes= numpy.array(afes)
    if not options.singlefeh is None:
        if True: #Just to keep indentation the same
            #Set up single feh
            indx= binned.callIndx(options.singlefeh,options.singleafe)
            if numpy.sum(indx) == 0:
                raise IOError("Bin corresponding to singlefeh and singleafe is empty ...")
            raw= copy.copy(binned.data[indx])
            newerrstuff= []
            for ii in range(len(binned.data)):
                if indx[ii]: newerrstuff.append(errstuff[ii])
            errstuff= newerrstuff
            print "Using %i data points ..." % (len(data))
            #Bin again
            binned= pixelAfeFeh(raw,dfeh=options.dfeh,dafe=options.dafe)
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
    if options.singles:
        run_abundance_singles_plotdens(options,args,fehs,afes)
        return None
    #Setup everything for the selection function
    print "Setting up stuff for the normalization integral ..."
    normintstuff= setup_normintstuff(options,raw,binned,fehs,afes)
    if not options.init is None:
        #Load initial parameters from file
        savefile= open(options.init,'rb')
        params= pickle.load(savefile)
        savefile.close()
    else:
        print "WARNING: Parameters not specified, using default ..."
        params= initialize(options,fehs,afes)
    #Set up density models and their parameters
    pop= 0 #assume first population
    model1= interpDens
    params1= calc_model(params,options,pop)
    model2= None
    params2= None
    model3= None
    params3= None
    """
    #Cut out bright stars on faint plates and vice versa
    indx= []
    nfaintbright, nbrightfaint= 0, 0
    for ii in range(len(data.feh)):
        if sf.platebright[str(data[ii].plate)] and data[ii].dered_r >= 17.8:
            indx.append(False)
            nbrightfaint+= 1
        elif not sf.platebright[str(data[ii].plate)] and data[ii].dered_r < 17.8:
            indx.append(False)
            nfaintbright+= 1
        else:
            indx.append(True)
    print "nbrightfaint, nfaintbright", nbrightfaint, nfaintbright
    indx= numpy.array(indx,dtype='bool')
    data= data[indx]
    XYZ= XYZ[indx,:]
    vxvyvz= vxvyvz[indx,:]
    cov_vxvyvz= cov_vxvyvz[indx,:]
    """
    #Setup everything for selection function
    thisnormintstuff= normintstuff[pop]
    if _PRECALCVSAMPLES:
        sf, plates,platel,plateb,platebright,platefaint,grmin,grmax,rmin,rmax,fehmin,fehmax,feh,colordist,fehdist,gr,rhogr,rhofeh,mr,dmin,dmax,ds, surfscale, hr, hz, surfnrs, surfnzs, surfRgrid, surfzgrid, surfvrs, surfvts, surfvzs= unpack_normintstuff(thisnormintstuff,options)
    else:
        sf, plates,platel,plateb,platebright,platefaint,grmin,grmax,rmin,rmax,fehmin,fehmax,feh,colordist,fehdist,gr,rhogr,rhofeh,mr,dmin,dmax,ds, surfscale, hr, hz= unpack_normintstuff(thisnormintstuff,options)
    data= binned(fehs[pop],afes[pop])
    #Ranges
    if options.type == 'z':
        xrange= [-0.1,5.]
    elif options.type == 'R':
        xrange= [4.8,14.2]
    elif options.type == 'r':
        xrange= [14.2,20.1]
    #We do bright/faint for 4 directions and all, all bright, all faint
    ls= [180,180,45,45]
    bs= [0,90,-23,23]
    bins= 21
    #Set up comparison
    if options.type == 'r':
        compare_func= compareDataModel.comparerdistPlate
    elif options.type == 'z':
        compare_func= compareDataModel.comparezdistPlate
    elif options.type == 'R':
        compare_func= compareDataModel.compareRdistPlate
    #all, faint, bright
    bins= [31,31,31]
    plates= ['all','bright','faint']
    for ii in range(len(plates)):
        plate= plates[ii]
        if plate == 'all':
#            thisleft_legend= left_legend
#            thisright_legend= right_legend
            thisleft_legend= None
            thisright_legend= None
        else:
            thisleft_legend= None
            thisright_legend= None
        bovy_plot.bovy_print()
        compare_func(model1,params1,sf,colordist,fehdist,
                     data,plate,color='k',
                     rmin=14.5,rmax=rmax,
                     grmin=grmin,grmax=grmax,
                     fehmin=fehmin,fehmax=fehmax,feh=feh,
                     xrange=xrange,
                     bins=bins[ii],ls='-',left_legend=thisleft_legend,
                     right_legend=thisright_legend)
        if not params2 is None:
            compare_func(model2,params2,sf,colordist,fehdist,
                         data,plate,color='k',
                         rmin=14.5,rmax=rmax,
                         grmin=grmin,grmax=grmax,
                         fehmin=fehmin,fehmax=fehmax,feh=feh,
                         xrange=xrange,
                         overplot=True,ls='--')
        if not params3 is None:
            compare_func(model3,params3,sf,colordist,fehdist,
                         data,plate,color='k',
                         rmin=14.5,rmax=rmax,
                         grmin=grmin,grmax=grmax,
                         fehmin=fehmin,fehmax=fehmax,feh=feh,
                         xrange=xrange,
                         overplot=True,ls=':')
        if options.type == 'r':
            bovy_plot.bovy_end_print(args[0]+'model_data_g_'+plate+'.'+options.ext)
        else:
            bovy_plot.bovy_end_print(args[0]+'model_data_g_'+options.type+'dist_'+plate+'.'+options.ext)
        if options.all: return None
    bins= 16
    for ii in range(len(ls)):
        #Bright
        plate= compareDataModel.similarPlatesDirection(ls[ii],bs[ii],20.,
                                                       sf,data,
                                                       faint=False)
        bovy_plot.bovy_print()
        compare_func(model1,params1,sf,colordist,fehdist,
                     data,plate,color='k',
                     rmin=14.5,rmax=rmax,
                     grmin=grmin,
                     grmax=grmax,
                     fehmin=fehmin,fehmax=fehmax,feh=feh,
                     xrange=xrange,
                     bins=bins,ls='-')
        if not params2 is None:
            compare_func(model2,params2,sf,colordist,fehdist,
                         data,plate,color='k',bins=bins,
                         rmin=14.5,rmax=rmax,
                         grmin=grmin,
                         grmax=grmax,
                         fehmin=fehmin,fehmax=fehmax,feh=feh,
                         xrange=xrange,
                         overplot=True,ls='--')
        if not params3 is None:
            compare_func(model3,params3,sf,colordist,fehdist,
                         data,plate,color='k',bins=bins,
                         rmin=14.5,rmax=rmax,
                         grmin=grmin,
                         grmax=grmax,
                         fehmin=fehmin,fehmax=fehmax,feh=feh,
                         xrange=xrange,
                         overplot=True,ls=':')
        if options.type == 'r':
            bovy_plot.bovy_end_print(args[0]+'model_data_g_'+'l%i_b%i_bright.' % (ls[ii],bs[ii])+options.ext)
        else:
            bovy_plot.bovy_end_print(args[0]+'model_data_g_'+options.type+'dist_l%i_b%i_bright.' % (ls[ii],bs[ii])+options.ext)
        #Faint
        plate= compareDataModel.similarPlatesDirection(ls[ii],bs[ii],20.,
                                                       sf,data,
                                                       bright=False)
        bovy_plot.bovy_print()
        compare_func(model1,params1,sf,colordist,fehdist,
                     data,plate,color='k',
                     rmin=14.5,rmax=rmax,
                     grmin=grmin,
                     grmax=grmax,
                     fehmin=fehmin,fehmax=fehmax,feh=feh,
                     xrange=xrange,
                     bins=bins,ls='-')
        if not params2 is None:
            compare_func(model2,params2,sf,colordist,fehdist,
                         data,plate,color='k',bins=bins,
                         rmin=14.5,rmax=rmax,grmin=grmin,
                         grmax=grmax,
                         fehmin=fehmin,fehmax=fehmax,feh=feh,
                         xrange=xrange,
                         overplot=True,ls='--')
        if not params3 is None:
            compare_func(model3,params3,sf,colordist,fehdist,
                         data,plate,color='k',bins=bins,
                         rmin=14.5,rmax=rmax,grmin=grmin,
                         grmax=grmax,
                         fehmin=fehmin,fehmax=fehmax,feh=feh,
                         xrange=xrange,
                         overplot=True,ls=':')
        if options.type == 'r':
            bovy_plot.bovy_end_print(args[0]+'model_data_g_'+'l%i_b%i_faint.' % (ls[ii],bs[ii])+options.ext)
        else:
            bovy_plot.bovy_end_print(args[0]+'model_data_g_'+options.type+'dist_l%i_b%i_faint.' % (ls[ii],bs[ii])+options.ext)
    return None

def calc_model(params,options,pop):
    nrs, nzs= _SURFNRS, _SURFNZS
    thisrmin, thisrmax= 4./_REFR0, 15./_REFR0
    thiszmin, thiszmax= 0., .8
    Rgrid= numpy.linspace(thisrmin,thisrmax,nrs)
    zgrid= numpy.linspace(thiszmin,thiszmax,nzs)
    #Model 1
    vo= get_vo(params,options,1)
    ro= get_ro(params,options)
    pot= setup_potential(params,options,1)
    aA= setup_aA(pot,options)
    dfparams= get_dfparams(params,pop,options,log=False)
    if options.dfmodel.lower() == 'qdf':
        #Normalize
        hr= dfparams[0]/ro
        sr= dfparams[1]/vo
        sz= dfparams[2]/vo
        hsr= dfparams[3]/ro
        hsz= dfparams[4]/ro
        #Setup
        qdf= quasiisothermaldf(hr,sr,sz,hsr,hsz,pot=pot,aA=aA,cutcounter=True)
    surfgrid= numpy.empty((nrs,nzs))
    for ii in range(nrs):
        for jj in range(nzs):
            surfgrid[ii,jj]= qdf.surfacemass(Rgrid[ii],zgrid[jj],
                                             nmc=options.nmcv)
        surfInterp= interpolate.RectBivariateSpline(Rgrid,zgrid,
                                                    numpy.log(surfgrid),
                                                    kx=3,ky=3,
                                                    s=10.*float(nzs*nrs))
    return surfInterp

def interpDens(R,z,surfInterp):
    """Function to give density using the interpolated representation"""
    return numpy.exp(surfInterp.ev(R/_REFR0,numpy.fabs(z)/_REFR0))

if __name__ == '__main__':
    (options,args)= get_options().parse_args()
    plotDensComparisonDF(options,args)
