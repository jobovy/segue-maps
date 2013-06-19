import os, os.path
import tempfile
import copy
import numpy
from scipy.maxentropy import logsumexp
import cPickle as pickle
from optparse import OptionParser
import multiprocessing
from galpy.util import bovy_plot, bovy_coords, multi, save_pickles
import segueSelect
import compareDataModel
from segueSelect import read_gdwarfs, read_kdwarfs, _GDWARFFILE, _KDWARFFILE, \
    segueSelect, _ERASESTR
from fitDensz import cb, _ZSUN, DistSpline, _ivezic_dist, _NDS
import AnDistance
from pixelFitDens import pixelAfeFeh
from pixelFitDF import *
from pixelFitDF import _SURFNRS, _SURFNZS, _PRECALCVSAMPLES, _REFR0, _REFV0
_NOTDONEYET= True
_RRANGES= True
def getMultiComparisonBins(options):
    if options.sample.lower() == 'g':
        if options.group == 'aenhanced':
            gfehs= [-1.25,-1.15,-1.05,-1.05,
                     -0.95,-0.95,-0.95,-0.95,
                     -0.85,-0.85,-0.85,-0.85,
                     -0.75,-0.75,-0.75,-0.75,
                     -0.65,-0.65,-0.65,
                     -0.55,-0.55,-0.55,
                     -0.45,-0.45,-0.35]
            gafes= [0.425,0.425,0.375,0.425,
                    0.325,0.375,0.425,0.475,
                    0.325,0.375,0.425,0.475,
                    0.325,0.375,0.425,0.475,
                    0.325,0.375,0.425,
                    0.325,0.375,0.425,
                    0.325,0.375,
                    0.325]
            left_legend= r'$\alpha\!-\!\mathrm{old\ populations}$'
        elif options.group == 'apoor':
            gfehs= [-0.05,-0.05,-0.05,0.05,0.05,0.15,0.25]
            gafes= [0.025,0.075,0.125,0.025,0.075,0.025,0.025]
            gfehs= [-0.05,-0.05,0.05,0.05,0.15,0.25] #drop 57
            gafes= [0.025,0.075,0.025,0.075,0.025,0.025]
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
            gfehs= [-0.85,-0.75,-0.65,-0.55,-0.45,-0.45,
                     -0.35,-0.35,-0.25,-0.25,-0.15,-0.15]
            gafes= [0.275,0.275,0.275,0.275,0.225,0.275,
                    0.225,0.275,0.175,0.225,0.125,0.175]
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
    M= len(gfehs)
    if options.andistances:
        distancefacs= numpy.zeros_like(fehs)
        gdistancefacs= numpy.zeros_like(gfehs)
        for jj in range(M):
            #Find pop corresponding to this bin
            ii= numpy.argmin((gfehs[jj]-fehs)**2./0.1+(gafes[jj]-afes)**2./0.0025)
            print ii
            #Get the relevant data
            data= binned(fehs[ii],afes[ii])
            distancefacs[ii]= AnDistance.AnDistance(data.dered_g-data.dered_r,
                                                    data.feh)
            gdistancefacs[jj]= distancefacs[ii]
            options.fixdm= numpy.log10(distancefacs[ii])*5.
            #Apply distance factor to the data
            newraw= read_rawdata(options)
            newbinned= pixelAfeFeh(newraw,dfeh=options.dfeh,dafe=options.dafe)
            thisdataIndx= binned.callIndx(fehs[ii],afes[ii])
            binned.data.xc[thisdataIndx]= newbinned.data.xc[thisdataIndx]
            binned.data.yc[thisdataIndx]= newbinned.data.yc[thisdataIndx]
            binned.data.zc[thisdataIndx]= newbinned.data.zc[thisdataIndx]
            binned.data.plate[thisdataIndx]= newbinned.data.plate[thisdataIndx]
            binned.data.dered_r[thisdataIndx]= newbinned.data.dered_r[thisdataIndx]
    else:
        distancefacs=numpy.ones_like(fehs)
        gdistancefacs=numpy.ones_like(gfehs)
    ##########POTENTIAL PARAMETERS####################
    potparams1= numpy.array([numpy.log(2.5/8.),options.fixvc/220.,
                             numpy.log(400./8000.),0.2,0.])
    if options.group.lower() == 'aenhanced':
        potparams2= numpy.array([numpy.log(2.8/8.),options.fixvc/220.,
                                 numpy.log(400./8000.),0.266666666,0.])
        potparams3= numpy.array([numpy.log(2.8/8.),options.fixvc/220.,
                                 numpy.log(400./8000.),0.8,0.])
    elif options.group.lower() == 'aintermediate':
        potparams2= numpy.array([numpy.log(3.0/8.),options.fixvc/220.,
                                 numpy.log(400./8000.),0.3333333333333,0.])
        potparams3= numpy.array([numpy.log(3.0/8.),options.fixvc/220.,
                                 numpy.log(400./8000.),0.933333333333,0.])
    elif options.group.lower() == 'apoor':
        potparams2= numpy.array([numpy.log(2.6/8.),options.fixvc/220.,
                                 numpy.log(400./8000.),0.4,0.])
        potparams3= numpy.array([numpy.log(2.6/8.),options.fixvc/220.,
                                 numpy.log(400./8000.),1.,0.])
    options.potential=  'dpdiskplhalofixbulgeflatwgasalt'
    #Setup everything for the selection function
    print "Setting up stuff for the normalization integral ..."
    normintstuff= setup_normintstuff(options,raw,binned,gfehs,gafes,raw)
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
        gdistancefacs= gdistancefacs[removeBins]
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
    if not options.multi is None:
        #Generate list of temporary files
        tmpfiles= []
        for jj in range(M): 
            tfile, tmpfile= tempfile.mkstemp()
            os.close(tfile) #Close because it's open
            tmpfiles.append(tmpfile)
        try:
            dummy= multi.parallel_map((lambda x: run_calc_model_multi(x,M,
                                                                      gfehs,gafes,
                                                                      fehs,afes,
                                                                      options,
                                                                      potparams1,potparams2,potparams3,
                                                                      distancefacs,
                                                                      binned,normintstuff,
                                                                      True,tmpfiles)),
                                      range(M),
                                      numcores=numpy.amin([M,
                                                           multiprocessing.cpu_count(),
                                                           options.multi]))
            #Now read all of the temporary files
            for jj in range(M):
                tmpfile= open(tmpfiles[jj],'rb')
                model1= pickle.load(tmpfile)
                if model1 is None:
                    continue
                tparams1= pickle.load(tmpfile)
                model2= pickle.load(tmpfile)
                tparams2= pickle.load(tmpfile)
                model3= pickle.load(tmpfile)
                tparams3= pickle.load(tmpfile)
                tdata= pickle.load(tmpfile)
                colordist= pickle.load(tmpfile)
                fehdist= pickle.load(tmpfile)
                fehmin= pickle.load(tmpfile)
                fehmax= pickle.load(tmpfile)
                feh= pickle.load(tmpfile)
                sf= pickle.load(tmpfile)
                rmin= pickle.load(tmpfile)
                rmax= pickle.load(tmpfile)
                grmin= pickle.load(tmpfile)
                grmax= pickle.load(tmpfile)
                tmpfile.close()
                model1s.append(model1)
                model2s.append(model2)
                model3s.append(model3)
                params1.append(tparams1)
                params2.append(tparams2)
                params3.append(tparams3)
                data.append(tdata)
                colordists.append(colordist)
                fehdists.append(fehdist)
                fehmins.append(fehmin)
                fehmaxs.append(fehmax)
                cfehs.append(feh)
        finally:
            for jj in range(M):
                os.remove(tmpfiles[jj])
    else:
        for jj in range(M):
            try:
                model1,tparams1,model2,tparams2,model3,tparams3,tdata,colordist,fehdist,fehmin,fehmax,feh, sf, rmin, rmax, grmin, grmax= run_calc_model_multi(jj,M,gfehs,gafes,fehs,afes,options,
                                                                                                                                                                                                                 potparams1,potparams2,potparams3,distancefacs,
                                                                                                                                binned,normintstuff,
                                                                                                                                False,None)
            except TypeError:
                continue
            model1s.append(model1)
            model2s.append(model2)
            model3s.append(model3)
            params1.append(tparams1)
            params2.append(tparams2)
            params3.append(tparams3)
            data.append(tdata)
            colordists.append(colordist)
            fehdists.append(fehdist)
            fehmins.append(fehmin)
            fehmaxs.append(fehmax)
            cfehs.append(feh)
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
    numcores= options.multi
    #First do R ranges for z
    if options.type.lower() == 'z' and _RRANGES:
        #First determine the ranges that have nstars in them
        rranges_nstars= 1000
        data_R= []
        for jj in range(M):
            tdata_R= numpy.sqrt((8.-data[jj].xc)**2.+data[jj].yc**2.)
            data_R.extend(tdata_R)
        tdata_R= sorted(data_R)
        nbins= numpy.ceil(len(tdata_R)/float(rranges_nstars))
        rranges_nstars= int(numpy.floor(float(len(tdata_R))/nbins))
        accum= rranges_nstars
        Rmins= [None]
        Rmaxs= []
        nameRmins= [4]
        nameRmaxs= []
        while accum < len(tdata_R):
            Rmins.append(tdata_R[accum])
            Rmaxs.append(tdata_R[accum]) 
            nameRmins.append(round(tdata_R[accum]*10.)/10.)
            nameRmaxs.append(round(tdata_R[accum]*10.)/10.)
            accum+= rranges_nstars
        Rmaxs.append(None)
        nameRmaxs.append(13.0)
        print Rmins, Rmaxs
        bins=31
        for ii in range(len(Rmins)):
            plate= 'all'
            if Rmins[ii] is None:
                thisleft_legend= r'$R \leq %.1f\,\mathrm{kpc\ plates}$' % (nameRmaxs[ii])
            elif Rmaxs[ii] is None:
                thisleft_legend= r'$R \geq %.1f\,\mathrm{kpc\ plates}$' % (nameRmins[ii])
            else:
                thisleft_legend= r'$%.1f\,\mathrm{kpc} < R \leq %.1f\,\mathrm{kpc\ plates}$' % (nameRmins[ii],nameRmaxs[ii])
            thisright_legend= 'none'
            bovy_plot.bovy_print()
            compare_func(model1s,params1,sf,colordists,fehdists,
                         data,plate,color='k',
                         rmin=14.5,rmax=rmax,
                         grmin=grmin,grmax=grmax,
                         fehmin=fehmins,fehmax=fehmaxs,feh=cfehs,
                         xrange=xrange,numcores=numcores,
                         bins=bins,ls='-',left_legend=thisleft_legend,
                         right_legend=thisright_legend,
                         Rmin=Rmins[ii],Rmax=Rmaxs[ii],
                         distfac=gdistancefacs,convolve=0.1)
            if not params2[0] is None:
                compare_func(model2s,params2,sf,colordists,fehdists,
                             data,plate,color='k',bins=bins,
                             rmin=14.5,rmax=rmax,
                             grmin=grmin,grmax=grmax,
                             fehmin=fehmins,fehmax=fehmaxs,feh=cfehs,
                             xrange=xrange,numcores=numcores,
                             right_legend=thisright_legend,
                             overplot=True,ls='--',
                             Rmin=Rmins[ii],Rmax=Rmaxs[ii],
                             distfac=gdistancefacs,convolve=0.1)
            if not params3[0] is None:
                compare_func(model3s,params3,sf,colordists,fehdists,
                             data,plate,color='k',bins=bins,
                             rmin=14.5,rmax=rmax,
                             grmin=grmin,grmax=grmax,
                             fehmin=fehmins,fehmax=fehmaxs,feh=cfehs,
                             right_legend=thisright_legend,
                             xrange=xrange,numcores=numcores,
                             overplot=True,ls=':',
                             Rmin=Rmins[ii],Rmax=Rmaxs[ii],
                             distfac=gdistancefacs,convolve=0.1)
            if options.type == 'r':
                bovy_plot.bovy_end_print(args[0]+'model_data_g_%.1fR%.1f' % (nameRmins[ii],nameRmaxs[ii])+'.'+options.ext)
            else:
                bovy_plot.bovy_end_print(args[0]+'model_data_g_'+options.group+'_'+options.type+'dist_%.1fR%.1f' % (nameRmins[ii],nameRmaxs[ii])+'.'+options.ext)
    #all, faint, bright
    bins= [31,31,31]
    plates= ['all','bright','faint']
    for ii in range(len(plates)):
        plate= plates[ii]
        print "Working on %s plates" % plate
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
                     right_legend=thisright_legend,numcores=numcores,
                     distfac=gdistancefacs)
        if not params2[0] is None:
            compare_func(model2s,params2,sf,colordists,fehdists,
                         data,plate,color='k',bins=bins[ii],
                         rmin=14.5,rmax=rmax,
                         grmin=grmin,grmax=grmax,
                         fehmin=fehmins,fehmax=fehmaxs,feh=cfehs,
                         xrange=xrange,
                         overplot=True,ls='--',numcores=numcores,
                     distfac=gdistancefacs)
        if not params3[0] is None:
            compare_func(model3s,params3,sf,colordists,fehdists,
                         data,plate,color='k',bins=bins[ii],
                         rmin=14.5,rmax=rmax,
                         grmin=grmin,grmax=grmax,
                         fehmin=fehmins,fehmax=fehmaxs,feh=cfehs,
                         xrange=xrange,
                         overplot=True,ls=':',numcores=numcores,
                         distfac=gdistancefacs)
        if options.type == 'r':
            bovy_plot.bovy_end_print(args[0]+'model_data_g_'+options.group+'_'+plate+'.'+options.ext)
        else:
            bovy_plot.bovy_end_print(args[0]+'model_data_g_'+options.group+'_'+options.type+'dist_'+plate+'.'+options.ext)
        if options.all: return None
    bins= 16
    for ii in range(len(ls)):
        print "Working on plates in the direction of l = %.0f, b=%.0f" % (ls[ii],bs[ii])
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
                     bins=bins,ls='-',numcores=numcores,
                     distfac=gdistancefacs)
        if not params2[0] is None:
            compare_func(model2s,params2,sf,colordists,fehdists,
                         data,plate,color='k',bins=bins,
                         rmin=14.5,rmax=rmax,
                         grmin=grmin,
                         grmax=grmax,
                         fehmin=fehmins,fehmax=fehmaxs,feh=cfehs,
                         xrange=xrange,
                         overplot=True,ls='--',numcores=numcores,
                         distfac=gdistancefacs)
        if not params3[0] is None:
            compare_func(model3s,params3,sf,colordists,fehdists,
                         data,plate,color='k',bins=bins,
                         rmin=14.5,rmax=rmax,
                         grmin=grmin,
                         grmax=grmax,
                         fehmin=fehmins,fehmax=fehmaxs,feh=cfehs,
                         xrange=xrange,
                         overplot=True,ls=':',numcores=numcores,
                         distfac=gdistancefacs)
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
                     bins=bins,ls='-',numcores=numcores,
                     distfac=gdistancefacs)
        if not params2[0] is None:
            compare_func(model2s,params2,sf,colordists,fehdists,
                         data,plate,color='k',bins=bins,
                         rmin=14.5,rmax=rmax,grmin=grmin,
                         grmax=grmax,
                         fehmin=fehmins,fehmax=fehmaxs,feh=cfehs,
                         xrange=xrange,
                         overplot=True,ls='--',numcores=numcores,
                         distfac=gdistancefacs)
        if not params3[0] is None:
            compare_func(model3s,params3,sf,colordists,fehdists,
                         data,plate,color='k',bins=bins,
                         rmin=14.5,rmax=rmax,grmin=grmin,
                         grmax=grmax,
                         fehmin=fehmins,fehmax=fehmaxs,feh=cfehs,
                         xrange=xrange,
                         overplot=True,ls=':',
                         numcores=numcores,
                         distfac=gdistancefacs)
        if options.type == 'r':
            bovy_plot.bovy_end_print(args[0]+'model_data_g_'+options.group+'_'+'l%i_b%i_faint.' % (ls[ii],bs[ii])+options.ext)
        else:
            bovy_plot.bovy_end_print(args[0]+'model_data_g_'+options.group+'_'+options.type+'dist_l%i_b%i_faint.' % (ls[ii],bs[ii])+options.ext)
    return None

def run_calc_model_multi(jj,M,gfehs,gafes,fehs,afes,options,
                         potparams1,potparams2,potparams3,
                         distancefacs,
                         binned,normintstuff,
                         savetopickle,tmpfiles):
    print "Working on group %i / %i ..." % (jj+1,M)
    #Find pop corresponding to this bin
    pop= numpy.argmin((gfehs[jj]-fehs)**2./0.1+(gafes[jj]-afes)**2./0.0025)
    #Apply distance factor
    toptions= copy.copy(options)
    if options.andistances:
        toptions.fixdm= numpy.log10(distancefacs[pop])*5.
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
    hrs, srs, szs=  setup_dfgrid([gfehs[jj]],[gafes[jj]],toptions)
    dfindx= numpy.unravel_index(numpy.nanargmax(logl[indx[0],0,0,indx[1],:,:,:,0]),
                                (8,8,16))
    print "Maximum for %i at %i,%i,%i" % (pop,dfindx[0],dfindx[1],dfindx[2])
    tparams= initialize(toptions,[gfehs[jj]],[gafes[jj]])
    startindx= 0
    if options.fitdvt: startindx+= 1
    tparams[startindx]= hrs[dfindx[0]]
    tparams[startindx+4]= srs[dfindx[1]]
    tparams[startindx+2]= szs[dfindx[2]]
    tparams[startindx+5]= 0. #outlier fraction
    tparams= set_potparams(potparams1,tparams,toptions,1)
    #Set up density models and their parameters
    model1= interpDens
    print "Working on model 1 ..."
    paramsInterp, surfz= calc_model(tparams,toptions,0,_retsurfz=True)
    params1= paramsInterp
    if True:
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
    thisnormintstuff= normintstuff[jj]
    if _PRECALCVSAMPLES:
        sf, plates,platel,plateb,platebright,platefaint,grmin,grmax,rmin,rmax,fehmin,fehmax,feh,colordist,fehdist,gr,rhogr,rhofeh,mr,dmin,dmax,ds, surfscale, hr, hz, colorfehfac,normR, normZ,surfnrs, surfnzs, surfRgrid, surfzgrid, surfvrs, surfvts, surfvzs= unpack_normintstuff(thisnormintstuff,toptions)
    else:
        sf, plates,platel,plateb,platebright,platefaint,grmin,grmax,rmin,rmax,fehmin,fehmax,feh,colordist,fehdist,gr,rhogr,rhofeh,mr,dmin,dmax,ds, surfscale, hr, hz, colorfehfac, normR, normZ= unpack_normintstuff(thisnormintstuff,toptions)
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
        if numpy.sum(indx) > 0:
            data= data[indx]
    if savetopickle:
        save_pickles(tmpfiles[jj],model1,params1,model2,params2,model3,params3,
                     data,
                     colordist,fehdist,fehmin,fehmax,feh,sf,rmin,rmax,grmin,grmax)
        return None
    else:
        return (model1,params1,model2,params2,model3,params3,
                data,
                colordist,fehdist,fehmin,fehmax,feh,sf,rmin,rmax,grmin,grmax)

def calc_model(params,options,pop,_retsurfz=False,
               normintstuff=None):
    nrs, nzs= 16, 16 #5,5
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
    if not params[6] == 0.:
        print "Calculating normalization of qdf ..."
        normalization_qdf= calc_normint(qdf,pop,normintstuff,params,1,options,
                                        -numpy.finfo(numpy.dtype(numpy.float64)).max)
        print "Calculating normalization of outliers ..."
        normalization_out= calc_normint(qdf,pop,normintstuff,params,1,options,
                                        0.,fqdf=0.)
        return (surfInterp,params[6]*normalization_qdf/normalization_out/12.)
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
