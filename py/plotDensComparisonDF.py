#ython plotDensComparisonDF.py --type=z --dfeh=1. --dafe=0.6 -f ../fakeDF/fakeDF_dfeh1._dafe0.6_q0.7.fits  --savenorm=../fakeDF/fakeDFFit_dfeh1._dafe0.6_q0.7_justpot_norm.sav --nmcv=100 --minndata=10000 ../figs/fakeDFFit_dfeh1._dafe0.6_q0.7_
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
from plotDensComparisonDFMulti4gridall import calc_model
_NOTDONEYET= True
_RRANGES= False
_VARYHSZ= True
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
            allraw= copy.copy(raw)
            raw= copy.copy(binned.data[indx])
            #newerrstuff= []
            #for ii in range(len(binned.data)):
            #    if indx[ii]: newerrstuff.append(errstuff[ii])
            #errstuff= newerrstuff
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
    if options.andistances:
        data= binned(fehs[0],afes[0])
        distfac= AnDistance.AnDistance(data.dered_g-data.dered_r,
                                       data.feh)
        if options.fixdm is None:
            options.fixdm= numpy.log10(distfac)*5.
        else:
            options.fixdm= options.fixdm+numpy.log10(distfac)*5.
        options.andistances= False
        #Start over
        plotDensComparisonDF(options,args)
        return None
    #Setup everything for the selection function
    print "Setting up stuff for the normalization integral ..."
    normintstuff= setup_normintstuff(options,raw,binned,fehs,afes,allraw)
    ##########POTENTIAL PARAMETERS####################
    potparams1= numpy.array([numpy.log(2.5/8.),options.fixvc/220.,
                             numpy.log(400./8000.),0.2,0.])
    potparams2= numpy.array([numpy.log(3./8.),options.fixvc/220.,
                             numpy.log(400./8000.),0.466666666,0.])
    potparams3= numpy.array([numpy.log(2.5/8.),options.fixvc/220.,
                             numpy.log(400./8000.),0.8,0.])
    options.potential=  'dpdiskplhalofixbulgeflatwgasalt'
    #Set up density models and their parameters
    pop= 0 #assume first population
    #Load savefile
    if not options.init is None:
        #Load initial parameters from file
        savename= options.init
#        spl= savename.split('.')
#        newname= ''
#        for ll in range(len(spl)-1):
#            newname+= spl[ll]
#            if not ll == len(spl)-2: newname+= '.'
#        newname+= '_%i.' % pop
#        newname+= spl[-1]
        savefile= open(savename,'rb')
        try:
            if not _NOTDONEYET:
                params= pickle.load(savefile)
                mlogl= pickle.load(savefile)
            logl= pickle.load(savefile)
        except:
            if savetopickle:
                save_pickles(tmpfiles[jj],None)
                return None
            else:
                return None
        finally:
            savefile.close()
    else:
        raise IOError("base filename not specified ...")
    #First model is best-fit for this particular bin
    marglogl= numpy.zeros((8,16))
    for ll in range(8):
        for kk in range(16):
            marglogl[ll,kk]= logsumexp(logl[ll,0,0,kk,:,:,:,0])
    indx= numpy.unravel_index(numpy.nanargmax(marglogl),(8,16))
    print "Maximum for %i at %i,%i" % (pop,indx[0],indx[1])
    rds= numpy.linspace(2.0,3.4,options.nrds)/_REFR0
    rds= numpy.log(rds)
    fhs= numpy.linspace(0.,1.,options.nfhs)
    potparams1[0]= rds[indx[0]]
    potparams1[3]= fhs[indx[1]]
    #######DF PARAMETER RANGES###########
    hrs, srs, szs=  setup_dfgrid(fehs,afes,options)
    dfindx= numpy.unravel_index(numpy.nanargmax(logl[indx[0],0,0,indx[1],:,:,:,0]),
                                (8,8,16))
    print "Maximum for %i at %i,%i,%i" % (pop,dfindx[0],dfindx[1],dfindx[2])
    tparams= initialize(options,fehs,afes)
    startindx= 0
    if options.fitdvt: startindx+= 1
    tparams[startindx]= hrs[dfindx[0]]
    tparams[startindx+4]= srs[dfindx[1]]
    tparams[startindx+2]= szs[dfindx[2]]
    tparams[startindx+5]= 0. #outlier fraction
    tparams= set_potparams(potparams1,tparams,options,1)
    #Set up density models and their parameters
    model1= interpDens#woutlier
    print "Working on model 1 ..."
    paramsInterp= calc_model(tparams,options,0,_retsurfz=False,
                             normintstuff=normintstuff)
    params1= paramsInterp
    if False:
        indx0= numpy.argmin((potparams2[0]-rds)**2.)
        indx1= numpy.argmin((potparams2[3]-fhs)**2.)
        #indx0= indx[0]
        #indx1= indx[1]
        dfindx= numpy.unravel_index(numpy.argmax(logl[indx0,0,0,indx1,:,:,:,0]),
                                    (8,8,16))
        tparams[startindx]= hrs[dfindx[0]]
        tparams[startindx+4]= srs[dfindx[1]]
        tparams[startindx+2]= szs[dfindx[2]]
        #print "BOVY: YOU HAVE MESSED WITH MODEL 2"
        tparams= set_potparams(potparams2,tparams,toptions,1)
        model2= interpDens
        print "Working on model 2 ..."
        paramsInterp, surfz= calc_model(tparams,toptions,0,_retsurfz=True)
        params2= paramsInterp
        indx0= numpy.argmin((potparams3[0]-rds)**2.)
        indx1= numpy.argmin((potparams3[3]-fhs)**2.)
        dfindx= numpy.unravel_index(numpy.argmax(logl[indx0,0,0,indx1,:,:,:,0]),
                                    (8,8,16))
        tparams[startindx]= hrs[dfindx[0]]
        tparams[startindx+4]= srs[dfindx[1]]
        tparams[startindx+2]= szs[dfindx[2]]
        tparams= set_potparams(potparams3,tparams,toptions,1)
        model3= interpDens
        print "Working on model 3 ..."
        paramsInterp, surfz= calc_model(tparams,toptions,0,_retsurfz=True)
        params3= paramsInterp
    else:
        model2= None
        params2= None
        model3= None
        params3= None
    data= binned(fehs[pop],afes[pop])
    #Setup everything for selection function
    thisnormintstuff= normintstuff[pop]
    if _PRECALCVSAMPLES:
        sf, plates,platel,plateb,platebright,platefaint,grmin,grmax,rmin,rmax,fehmin,fehmax,feh,colordist,fehdist,gr,rhogr,rhofeh,mr,dmin,dmax,ds, surfscale, hr, hz, colorfehfac,normR, normZ,surfnrs, surfnzs, surfRgrid, surfzgrid, surfvrs, surfvts, surfvzs= unpack_normintstuff(thisnormintstuff,options)
    else:
        sf, plates,platel,plateb,platebright,platefaint,grmin,grmax,rmin,rmax,fehmin,fehmax,feh,colordist,fehdist,gr,rhogr,rhofeh,mr,dmin,dmax,ds, surfscale, hr, hz, colorfehfac, normR, normZ= unpack_normintstuff(thisnormintstuff,options)
    if True:
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
    #First do R ranges for z
    if options.type.lower() == 'z' and _RRANGES:
        bins=21
        Rmins= [None,7.,9.]
        Rmaxs= [7.,9.,None]
        nameRmins= [4,7,9]
        nameRmaxs= [7,9,13]
        for ii in range(len(Rmins)):
            plate= 'all'
            if Rmins[ii] is None:
                thisleft_legend= r'$R \leq 7\,\mathrm{kpc\ plates}$'
            elif Rmins[ii] == 7.:
                thisleft_legend= r'$7\,\mathrm{kpc} < R \leq 9\,\mathrm{kpc\ plates}$'
            elif Rmaxs[ii] is None:
                thisleft_legend= r'$R \geq 9\,\mathrm{kpc\ plates}$'
            thisright_legend= None
            bovy_plot.bovy_print()
            compare_func(model1,params1,sf,colordist,fehdist,
                         data,plate,color='k',
                         rmin=14.5,rmax=rmax,
                         grmin=grmin,grmax=grmax,
                         fehmin=fehmin,fehmax=fehmax,feh=feh,
                         xrange=xrange,
                         bins=bins,ls='-',left_legend=thisleft_legend,
                         right_legend=thisright_legend,
                         Rmin=Rmins[ii],Rmax=Rmaxs[ii])
            if not params2 is None:
                compare_func(model2,params2,sf,colordist,fehdist,
                             data,plate,color='k',bins=bins,
                             rmin=14.5,rmax=rmax,
                             grmin=grmin,grmax=grmax,
                             fehmin=fehmin,fehmax=fehmax,feh=feh,
                             xrange=xrange,
                             overplot=True,ls='--',
                             Rmin=Rmins[ii],Rmax=Rmaxs[ii])
            if not params3 is None:
                compare_func(model3,params3,sf,colordist,fehdist,
                             data,plate,color='k',bins=bins,
                             rmin=14.5,rmax=rmax,
                             grmin=grmin,grmax=grmax,
                             fehmin=fehmin,fehmax=fehmax,feh=feh,
                             xrange=xrange,
                             overplot=True,ls=':',
                             Rmin=Rmins[ii],Rmax=Rmaxs[ii])
            if options.type == 'r':
                bovy_plot.bovy_end_print(args[0]+'model_data_g_%iR%i' % (nameRmins[ii],nameRmaxs[ii])+'.'+options.ext)
            else:
                bovy_plot.bovy_end_print(args[0]+'model_data_g_'+options.type+'dist_%iR%i' % (nameRmins[ii],nameRmaxs[ii])+'.'+options.ext)
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
                         data,plate,color='k',bins=bins[ii],
                         rmin=14.5,rmax=rmax,
                         grmin=grmin,grmax=grmax,
                         fehmin=fehmin,fehmax=fehmax,feh=feh,
                         xrange=xrange,
                         overplot=True,ls='--')
        if not params3 is None:
            compare_func(model3,params3,sf,colordist,fehdist,
                         data,plate,color='k',bins=bins[ii],
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
        nodata= False
        #Bright
        plate= compareDataModel.similarPlatesDirection(ls[ii],bs[ii],20.,
                                                       sf,data,
                                                       faint=False)
        bovy_plot.bovy_print()
        try:
            compare_func(model1,params1,sf,colordist,fehdist,
                         data,plate,color='k',
                         rmin=14.5,rmax=rmax,
                         grmin=grmin,
                         grmax=grmax,
                         fehmin=fehmin,fehmax=fehmax,feh=feh,
                         xrange=xrange,
                         bins=bins,ls='-')
        except IndexError:
            #no data
            nodata= True
        if not params2 is None and not nodata:
            compare_func(model2,params2,sf,colordist,fehdist,
                         data,plate,color='k',bins=bins,
                         rmin=14.5,rmax=rmax,
                         grmin=grmin,
                         grmax=grmax,
                         fehmin=fehmin,fehmax=fehmax,feh=feh,
                         xrange=xrange,
                         overplot=True,ls='--')
        if not params3 is None and not nodata:
            compare_func(model3,params3,sf,colordist,fehdist,
                         data,plate,color='k',bins=bins,
                         rmin=14.5,rmax=rmax,
                         grmin=grmin,
                         grmax=grmax,
                         fehmin=fehmin,fehmax=fehmax,feh=feh,
                         xrange=xrange,
                         overplot=True,ls=':')
        if not nodata:
            if options.type == 'r':
                bovy_plot.bovy_end_print(args[0]+'model_data_g_'+'l%i_b%i_bright.' % (ls[ii],bs[ii])+options.ext)
            else:
                bovy_plot.bovy_end_print(args[0]+'model_data_g_'+options.type+'dist_l%i_b%i_bright.' % (ls[ii],bs[ii])+options.ext)
        #Faint
        plate= compareDataModel.similarPlatesDirection(ls[ii],bs[ii],20.,
                                                       sf,data,
                                                       bright=False)
        bovy_plot.bovy_print()
        try:
            compare_func(model1,params1,sf,colordist,fehdist,
                         data,plate,color='k',
                         rmin=14.5,rmax=rmax,
                         grmin=grmin,
                         grmax=grmax,
                         fehmin=fehmin,fehmax=fehmax,feh=feh,
                         xrange=xrange,
                         bins=bins,ls='-')
        except IndexError:
            #No data
            continue
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

def old_calc_model(params,options,pop):
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
            surfgrid[ii,jj]= qdf.density(Rgrid[ii],zgrid[jj],
                                         nmc=options.nmcv,
                                         ngl=options.ngl)
        surfInterp= interpolate.RectBivariateSpline(Rgrid,zgrid,
                                                    numpy.log(surfgrid),
                                                    kx=3,ky=3,
                                                    s=0.)
    return surfInterp

##RUNNING SINGLE BINS IN A SINGLE CALL
def run_abundance_singles_plotdens(options,args,fehs,afes):
    options.singles= False #First turn this off!
    savename= args[0]
    initname= options.init
    normname= options.savenorm
    if not options.multi is None:
        dummy= multi.parallel_map((lambda x: run_abundance_singles_plotdens_single(options,args,fehs,afes,x,
                                                                          savename,initname,normname)),
                                  range(len(fehs)),
                                  numcores=numpy.amin([len(fehs),
                                                       multiprocessing.cpu_count(),
                                                       options.multi]))
    else:
        for ii in range(len(fehs)):
            run_abundance_singles_plotdens_single(options,args,fehs,afes,ii,
                                                  savename,initname,normname)
    return None

def run_abundance_singles_plotdens_single(options,args,fehs,afes,ii,savename,
                                          initname,
                                          normname):
    if numpy.log(monoAbundanceMW.hr(fehs[ii],afes[ii],
                                    k=(options.sample.lower() == 'k'))/8.) > -0.5 and not options.conditionalr:
            #We didn't run, because we cannot model these populations with our model
        return None
    #Prepare args and options
    spl= savename.split('.')
    newname= savename+'%i_' % ii
    args[0]= newname
    if not initname is None:
        #Do the same for init
        spl= initname.split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        options.init= newname
    if not normname is None:
        #Do the same for init
        spl= normname.split('.')
        newname= ''
        for jj in range(len(spl)-1):
            newname+= spl[jj]
            if not jj == len(spl)-2: newname+= '.'
        newname+= '_%i.' % ii
        newname+= spl[-1]
        options.savenorm= newname
    options.singlefeh= fehs[ii]
    options.singleafe= afes[ii]
    #Now run
    plotDensComparisonDF(options,args)

if __name__ == '__main__':
    (options,args)= get_options().parse_args()
    plotDensComparisonDF(options,args)
