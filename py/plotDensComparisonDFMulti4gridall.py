import os, os.path
import copy
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
_NOTDONEYET= True
def getMultiComparisonBins(options):
    if options.sample.lower() == 'g':
        if options.group == 'aenhanced':
            gfehs= [-0.85,-0.75,-0.65,-0.55, #-0.95
                     -0.85,-0.75,-0.65,-0.55,-0.45, #-0.95
                     -0.85,-0.75,-0.65,-0.55] #-0.95
            gafes= [0.425,0.425,0.425,0.425, #0.425
                    0.375,0.375,0.375,0.375,0.375, #0.375
                    0.325,0.325,0.325,0.325] #0.325
            left_legend= r'$\alpha\!-\!\mathrm{old\ populations}$'
        elif options.group == 'apoor':
            gafes= [0.125,0.075,0.075,0.025,0.025,0.025,0.025]
            gfehs= [-0.05,-0.05,0.05,-0.05,0.05,0.15,0.25]
            left_legend= r'$\alpha\!-\!\mathrm{young}$'+'\n'+r'$\mathrm{populations}$'
        elif options.group == 'apoorfpoor':
            gafes= [0.025,0.075,0.075,0.075,0.075,
                    0.125,0.125,0.125,0.125,0.125,0.125,
                    0.175,0.175,0.175,0.175,
                    0.225,0.225,0.225]
            gfehs= [-0.15,-0.15,-0.25,-0.35,-0.45,
                     -0.15,-0.25,-0.35,-0.45,-0.55,-0.65,
                     -0.35,-0.45,-0.55,-0.65,
                     -0.55,-0.65,-0.75]
            left_legend= r'$\alpha\!-\!\mathrm{young},$'+'\n'+r'$\mathrm{[Fe/H]\!-\!poor}$'+'\n'+r'$\mathrm{populations}$'
        elif options.group == 'aintermediate':
            gafes= [0.275,0.275,0.275,0.275,0.275,0.275,
                    0.225,0.225,0.225,
                    0.175,0.175]
            gfehs= [-0.75,-0.65,-0.55,-0.45,-0.35,-0.25,
                     -0.45,-0.35,-0.25,
                     -0.25,-0.15]
            left_legend= r'$\alpha\!-\!\mathrm{intermediate}$'+'\n'+r'$\mathrm{populations}$'
    elif options.sample.lower() == 'k':
        if options.group == 'aenhanced':
            gfehs= [-0.85,
                     -0.95,-0.85,-0.75,
                     -0.85,-0.75,-0.65,-0.55] #-0.95
            gafes= [0.425,
                    0.375,0.375,0.375,
                    0.325,0.325,0.325,0.325] #0.325
            left_legend= r'$\alpha\!-\!\mathrm{old\ populations}$'
        elif options.group == 'apoor':
            gafes= [0.075,0.125]
            gfehs= [-0.05,-0.15]
            left_legend= r'$\alpha\!-\!\mathrm{young}$'+'\n'+r'$\mathrm{populations}$'
        elif options.group == 'apoorfpoor':
            gafes= [0.025,0.075,0.075,0.075,0.075,
                    0.125,0.125,0.125,0.125,
                    0.175,0.175,0.175,0.175,
                    0.225,0.225]
            gfehs= [-0.15,-0.15,-0.25,-0.35,-0.45,
                     -0.25,-0.35,-0.45,-0.55,
                     -0.35,-0.45,-0.55,-0.65,
                     -0.65,-0.55]
            left_legend= r'$\alpha\!-\!\mathrm{young},$'+'\n'+r'$\mathrm{[Fe/H]\!-\!poor}$'+'\n'+r'$\mathrm{populations}$'
        elif options.group == 'aintermediate':
            gafes= [0.275,0.275,0.275,0.275,
                    0.225]
            gfehs= [-0.75,-0.65,-0.55,-0.45,
                     -0.45]
            left_legend= r'$\alpha\!-\!\mathrm{intermediate}$'+'\n'+r'$\mathrm{populations}$'
    return (gafes,gfehs,left_legend)
def plotDensComparisonDFMulti(options,args):
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
    gafes, gfehs, left_legend= getMultiComparisonBins(options)
    ##########POTENTIAL PARAMETERS####################
    potparams1= numpy.array([numpy.log(2.5/8.),1.,numpy.log(400./8000.),0.2,0.])
    potparams2= numpy.array([numpy.log(2.5/8.),1.,numpy.log(400./8000.),0.466666,0.,2.])
    potparams3= numpy.array([numpy.log(2.5/8.),1.,numpy.log(400./8000.),0.466666,0.])
    #Setup everything for the selection function
    print "Setting up stuff for the normalization integral ..."
    normintstuff= setup_normintstuff(options,raw,binned,gfehs,gafes)
    M= len(gfehs)
    #Check whether fits exist, if not, pop
    removeBins= numpy.ones(M,dtype='bool')
    for jj in range(M):
        #Find pop corresponding to this bin
        pop= numpy.argmin((gfehs[jj]-fehs)**2./0.1+(gafes[jj]-afes)**2./0.0025)
        #Load savefile
        if not options.init is None:
            #Load initial parameters from file
            savename= options.init
            spl= savename.split('.')
            newname= ''
            for ll in range(len(spl)-1):
                newname+= spl[ll]
                if not ll == len(spl)-2: newname+= '.'
            newname+= '_%i.' % pop
            newname+= spl[-1]
            if not os.path.exists(newname):
                removeBins[jj]= False
        else:
            raise IOError("base filename not specified ...")
    if numpy.sum(removeBins) == 0:
        raise IOError("None of the group bins have been fit ...")
    elif numpy.sum(removeBins) < M:
        #Some bins have not been fit yet, and have to be removed
        gfehs= list((numpy.array(gfehs))[removeBins])
        gafes= list((numpy.array(gafes))[removeBins])
        print "Only using %i bins out of %i ..." % (numpy.sum(removeBins),M)
        M= len(gfehs)
    model1s= []
    model2s= []
    model3s= []
    params1= []
    params2= []
    params3= []
    data= []
    colordists= []
    fehdists= []
    fehmins= []
    fehmaxs= []
    cfehs= []
    #######DF PARAMETER RANGES###########
    hrs= numpy.log(numpy.linspace(1.5,5.,options.nhrs)/_REFR0)
    srs= numpy.log(numpy.linspace(25.,70.,options.nsrs)/_REFV0)
    szs= numpy.log(numpy.linspace(15.,60.,options.nszs)/_REFV0)
    dvts= numpy.linspace(-0.1,0.1,options.ndvts)
    pouts= numpy.linspace(10.**-5.,.3,options.npouts)
    for jj in range(M):
        print "Working on group %i / %i ..." % (jj+1,M)
        #Find pop corresponding to this bin
        pop= numpy.argmin((gfehs[jj]-fehs)**2./0.1+(gafes[jj]-afes)**2./0.0025)
        #Load savefile
        if not options.init is None:
            #Load initial parameters from file
            savename= options.init
            spl= savename.split('.')
            newname= ''
            for ll in range(len(spl)-1):
                newname+= spl[ll]
                if not ll == len(spl)-2: newname+= '.'
            newname+= '_%i.' % pop
            newname+= spl[-1]
            savefile= open(newname,'rb')
            try:
                if not _NOTDONEYET:
                    params= pickle.load(savefile)
                    mlogl= pickle.load(savefile)
                logl= pickle.load(savefile)
            except:
                continue
            finally:
                savefile.close()
        else:
            raise IOError("base filename not specified ...")
        #Set DF parameters as the maximum at R_d=2.5, f_h=0.4
        indx= numpy.unravel_index(numpy.argmax(logl[5,0,0,3,:,:,:,1:4,:,0,0]),
                                  (8,8,8,3,31))
        tparams= numpy.array([dvts[1+indx[3]],hrs[indx[0]],srs[indx[1]],
                              szs[indx[2]],numpy.log(8./_REFR0),
                              numpy.log(7./_REFR0),pouts[indx[4]],
                              0.,0.,0.,0.,0.])
        options.potential=  'dpdiskplhalofixbulgeflatwgasalt'
        tparams= set_potparams(potparams1,tparams,options,1)
        print tparams
        #Set up density models and their parameters
        model1s.append(interpDens)
        paramsInterp, surfz= calc_model(tparams,options,0,_retsurfz=True)
        params1.append(paramsInterp)
        if True:
            tparams= numpy.array([dvts[1+indx[3]],hrs[indx[0]],srs[indx[1]],
                                  szs[indx[2]],numpy.log(8./_REFR0),
                                  numpy.log(7./_REFR0),pouts[indx[4]],
                                  0.,0.,0.,0.,0.,0.])
            options.potential= 'dpdiskplhalodarkdiskfixbulgeflatwgasalt'
            tparams= set_potparams(potparams2,tparams,options,1)
            print tparams
            model2s.append(interpDens)
            paramsInterp, surfz= calc_model(tparams,options,0,_retsurfz=True)
            params2.append(paramsInterp)
            tparams= numpy.array([dvts[1+indx[3]],hrs[indx[0]],srs[indx[1]],
                                  szs[indx[2]],numpy.log(8./_REFR0),
                                  numpy.log(7./_REFR0),pouts[indx[4]],
                                  0.,0.,0.,0.,0.])
            options.potential= 'dpdiskplhalofixbulgeflatwgasalt'
            tparams= set_potparams(potparams3,tparams,options,1)
            print tparams
            model3s.append(interpDens)
            paramsInterp, surfz= calc_model(tparams,options,0,_retsurfz=True)
            params3.append(paramsInterp)
        else:
            model2s.append(None)
            params2.append(None)
            model3s.append(None)
            params3.append(None)
        data.append(binned(fehs[pop],afes[pop]))
        #Setup everything for selection function
        thisnormintstuff= normintstuff[jj]
        if _PRECALCVSAMPLES:
            sf, plates,platel,plateb,platebright,platefaint,grmin,grmax,rmin,rmax,fehmin,fehmax,feh,colordist,fehdist,gr,rhogr,rhofeh,mr,dmin,dmax,ds, surfscale, hr, hz, colorfehfac,normR, normZ,surfnrs, surfnzs, surfRgrid, surfzgrid, surfvrs, surfvts, surfvzs= unpack_normintstuff(thisnormintstuff,options)
        else:
            sf, plates,platel,plateb,platebright,platefaint,grmin,grmax,rmin,rmax,fehmin,fehmax,feh,colordist,fehdist,gr,rhogr,rhofeh,mr,dmin,dmax,ds, surfscale, hr, hz, colorfehfac, normR, normZ= unpack_normintstuff(thisnormintstuff,options)
        colordists.append(colordist)
        fehdists.append(fehdist)
        fehmins.append(fehmin)
        fehmaxs.append(fehmax)
        cfehs.append(feh)
        if True:
            #Cut out bright stars on faint plates and vice versa
            indx= []
            nfaintbright, nbrightfaint= 0, 0
            for ii in range(len(data[jj].feh)):
                if sf.platebright[str(data[jj][ii].plate)] and data[jj][ii].dered_r >= 17.8:
                    indx.append(False)
                    nbrightfaint+= 1
                elif not sf.platebright[str(data[jj][ii].plate)] and data[jj][ii].dered_r < 17.8:
                    indx.append(False)
                    nfaintbright+= 1
                else:
                    indx.append(True)
            print "nbrightfaint, nfaintbright", nbrightfaint, nfaintbright
            indx= numpy.array(indx,dtype='bool')
            if numpy.sum(indx) > 0:
                data[jj]= data[jj][indx]
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
        compare_func= compareDataModel.comparerdistPlateMulti
    elif options.type == 'z':
        compare_func= compareDataModel.comparezdistPlateMulti
    elif options.type == 'R':
        compare_func= compareDataModel.compareRdistPlateMulti
    #all, faint, bright
    bins= [31,31,31]
    plates= ['all','bright','faint']
    for ii in range(len(plates)):
        plate= plates[ii]
        if plate == 'all':
            thisleft_legend= left_legend
#            thisright_legend= right_legend
#            thisleft_legend= None
            thisright_legend= None
        else:
            thisleft_legend= None
            thisright_legend= None
        bovy_plot.bovy_print()
        compare_func(model1s,params1,sf,colordists,fehdists,
                     data,plate,color='k',
                     rmin=14.5,rmax=rmax,
                     grmin=grmin,grmax=grmax,
                     fehmin=fehmins,fehmax=fehmaxs,feh=cfehs,
                     xrange=xrange,
                     bins=bins[ii],ls='-',left_legend=thisleft_legend,
                     right_legend=thisright_legend)
        if not params2[0] is None:
            compare_func(model2s,params2,sf,colordists,fehdists,
                         data,plate,color='k',bins=bins[ii],
                         rmin=14.5,rmax=rmax,
                         grmin=grmin,grmax=grmax,
                         fehmin=fehmins,fehmax=fehmaxs,feh=cfehs,
                         xrange=xrange,
                         overplot=True,ls='--')
        if not params3[0] is None:
            compare_func(model3s,params3,sf,colordists,fehdists,
                         data,plate,color='k',bins=bins[ii],
                         rmin=14.5,rmax=rmax,
                         grmin=grmin,grmax=grmax,
                         fehmin=fehmins,fehmax=fehmaxs,feh=cfehs,
                         xrange=xrange,
                         overplot=True,ls=':')
        if options.type == 'r':
            bovy_plot.bovy_end_print(args[0]+'model_data_g_'+options.group+'_'+plate+'.'+options.ext)
        else:
            bovy_plot.bovy_end_print(args[0]+'model_data_g_'+options.group+'_'+options.type+'dist_'+plate+'.'+options.ext)
        if options.all: return None
    bins= 16
    for ii in range(len(ls)):
        #Bright
        plate= compareDataModel.similarPlatesDirection(ls[ii],bs[ii],20.,
                                                       sf,data,
                                                       faint=False)
        bovy_plot.bovy_print()
        compare_func(model1s,params1,sf,colordists,fehdists,
                     data,plate,color='k',
                     rmin=14.5,rmax=rmax,
                     grmin=grmin,
                     grmax=grmax,
                     fehmin=fehmins,fehmax=fehmaxs,feh=cfehs,
                     xrange=xrange,
                     bins=bins,ls='-')
        if not params2[0] is None:
            compare_func(model2s,params2,sf,colordists,fehdists,
                         data,plate,color='k',bins=bins,
                         rmin=14.5,rmax=rmax,
                         grmin=grmin,
                         grmax=grmax,
                         fehmin=fehmins,fehmax=fehmaxs,feh=cfehs,
                         xrange=xrange,
                         overplot=True,ls='--')
        if not params3[0] is None:
            compare_func(model3s,params3,sf,colordists,fehdists,
                         data,plate,color='k',bins=bins,
                         rmin=14.5,rmax=rmax,
                         grmin=grmin,
                         grmax=grmax,
                         fehmin=fehmins,fehmax=fehmaxs,feh=cfehs,
                         xrange=xrange,
                         overplot=True,ls=':')
        if options.type == 'r':
            bovy_plot.bovy_end_print(args[0]+'model_data_g_'+options.group+'_'+'l%i_b%i_bright.' % (ls[ii],bs[ii])+options.ext)
        else:
            bovy_plot.bovy_end_print(args[0]+'model_data_g_'+options.group+'_'+options.type+'dist_l%i_b%i_bright.' % (ls[ii],bs[ii])+options.ext)
        #Faint
        plate= compareDataModel.similarPlatesDirection(ls[ii],bs[ii],20.,
                                                       sf,data,
                                                       bright=False)
        bovy_plot.bovy_print()
        compare_func(model1s,params1,sf,colordists,fehdists,
                     data,plate,color='k',
                     rmin=14.5,rmax=rmax,
                     grmin=grmin,
                     grmax=grmax,
                     fehmin=fehmins,fehmax=fehmaxs,feh=cfehs,
                     xrange=xrange,
                     bins=bins,ls='-')
        if not params2[0] is None:
            compare_func(model2s,params2,sf,colordists,fehdists,
                         data,plate,color='k',bins=bins,
                         rmin=14.5,rmax=rmax,grmin=grmin,
                         grmax=grmax,
                         fehmin=fehmins,fehmax=fehmaxs,feh=cfehs,
                         xrange=xrange,
                         overplot=True,ls='--')
        if not params3[0] is None:
            compare_func(model3s,params3,sf,colordists,fehdists,
                         data,plate,color='k',bins=bins,
                         rmin=14.5,rmax=rmax,grmin=grmin,
                         grmax=grmax,
                         fehmin=fehmins,fehmax=fehmaxs,feh=cfehs,
                         xrange=xrange,
                         overplot=True,ls=':')
        if options.type == 'r':
            bovy_plot.bovy_end_print(args[0]+'model_data_g_'+options.group+'_'+'l%i_b%i_faint.' % (ls[ii],bs[ii])+options.ext)
        else:
            bovy_plot.bovy_end_print(args[0]+'model_data_g_'+options.group+'_'+options.type+'dist_l%i_b%i_faint.' % (ls[ii],bs[ii])+options.ext)
    return None

def calc_model(params,options,pop,_retsurfz=False):
    nrs, nzs= 21, 21
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
    if _retsurfz:
        return (surfInterp,qdf.surfacemass_z(1.,ngl=options.ngl))
    else:
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
    #Prepare args and options
    spl= savename.split('.')
    newname= ''
    for jj in range(len(spl)-1):
        newname+= spl[jj]
        if not jj == len(spl)-2: newname+= '.'
    newname+= '_%i.' % ii
    newname+= spl[-1]
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

def plotMultiBins(options,args):
    raw= read_rawdata(options)
    #Bin the data   
    binned= pixelAfeFeh(raw,dfeh=options.dfeh,dafe=options.dafe)
    if options.tighten:
        tightbinned= pixelAfeFeh(raw,dfeh=options.dfeh,dafe=options.dafe,
                                 fehmin=-1.6,fehmax=0.5,afemin=-0.05,
                                 afemax=0.55)
    else:
        tightbinned= binned
    #Load the different categories
    #aenhanced
    options.group= 'aenhanced'
    gafes_aenhanced, gfehs_aenhanced, dummy= getMultiComparisonBins(options)
    options.group= 'apoor'
    gafes_apoor, gfehs_apoor, dummy= getMultiComparisonBins(options)
    options.group= 'apoorfpoor'
    gafes_apoorfpoor, gfehs_apoorfpoor, dummy= getMultiComparisonBins(options)
    options.group= 'aintermediate'
    gafes_aintermediate, gfehs_aintermediate, dummy= getMultiComparisonBins(options)
    #Run through the pixels and gather
    plotthis= numpy.zeros((tightbinned.npixfeh(),tightbinned.npixafe()))
    plotthis[:,:]= numpy.nan
    for ii in range(tightbinned.npixfeh()):
        for jj in range(tightbinned.npixafe()):
            data= binned(tightbinned.feh(ii),tightbinned.afe(jj))
            fehindx= binned.fehindx(tightbinned.feh(ii))#Map onto regular binning
            afeindx= binned.afeindx(tightbinned.afe(jj))
            thisfeh= tightbinned.feh(ii)
            thisafe= tightbinned.afe(jj)
            if inMultiBin(thisfeh,thisafe,gfehs_aenhanced,gafes_aenhanced):
                plotthis[ii,jj]= 3
            elif inMultiBin(thisfeh,thisafe,gfehs_apoor,gafes_apoor):
                plotthis[ii,jj]= 2
            elif inMultiBin(thisfeh,thisafe,gfehs_apoorfpoor,gafes_apoorfpoor):
                plotthis[ii,jj]= 1
            elif inMultiBin(thisfeh,thisafe,gfehs_aintermediate,gafes_aintermediate):
                plotthis[ii,jj]= 0
    print "Bins accounted for: %i / 62 ..." % (numpy.sum(True-numpy.isnan(plotthis)))
    vmin, vmax= -1,3.5
    if options.tighten:
        xrange=[-1.6,0.5]
        yrange=[-0.05,0.55]
    else:
        xrange=[-2.,0.6]
        yrange=[-0.1,0.6]
    bovy_plot.bovy_print()
    bovy_plot.bovy_dens2d(plotthis.T,origin='lower',cmap='jet',
                          interpolation='nearest',
                          xlabel=r'$[\mathrm{Fe/H}]$',
                          ylabel=r'$[\alpha/\mathrm{Fe}]$',
                          xrange=xrange,yrange=yrange,
                          vmin=vmin,vmax=vmax,
                          contours=False,
                          colorbar=False)
    bovy_plot.bovy_end_print(args[0])

def inMultiBin(feh,afe,gfehs,gafes):
    return (numpy.fabs(numpy.amin((gfehs-feh)**2./0.1+(gafes-afe)**2./0.0025)) < 0.00001)

if __name__ == '__main__':
    (options,args)= get_options().parse_args()
    if options.type == 'bins':
        plotMultiBins(options,args)
    else:
        plotDensComparisonDFMulti(options,args)
