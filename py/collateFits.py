#Grab all of the fits, output them as fits
#Suggested use:
#python collateFits.py -o ~/Repos/monoAbundanceMW/monoAbundanceMW/data/monoAbundanceResults.fits ../fits/pixelFitG_DblExp_BigPix0.1.sav ../fits/pixelFitG_DblExp_BigPix0.1_1000samples.sav ../fits/pixelFitG_Mass_DblExp_BigPix0.1_simpleage.sav ../fits/pixelFitG_Mass_DblExp_BigPix0.1_simpleage_100samples.sav ../fits/pixelFitG_Vel_HWR_BigPix0.1.sav ../fits/pixelFitG_Vel_HWR_BigPix0.1_10ksamples.sav ../fits/pixelFitG_VelR_HWR_BigPix0.1.sav ../fits/pixelFitG_VelR_HWR_BigPix0.1_10ksamples.sav
#For K
#python collateFits.py -o ~/Repos/monoAbundanceMW/monoAbundanceMW/data/monoAbundanceResults_k.fits ../fits/pixelFitK_DblExp_BigPix0.1_minndata50.sav ../fits/pixelFitK_DblExp_BigPix0.1_minndata50_1000samples.sav ../fits/pixelFitG_Mass_DblExp_BigPix0.1_simpleage.sav ../fits/pixelFitG_Mass_DblExp_BigPix0.1_simpleage_100samples.sav ../fits/pixelFitK_Vel_HWR_BigPix0.1_minndata50.sav ../fits/pixelFitK_Vel_HWR_BigPix0.1_minndata50_1000samples.sav ../fits/pixelFitK_VelR_HWR_BigPix0.1_minndata50.sav ../fits/pixelFitK_VelR_HWR_BigPix0.1_minndata50_1000samples.sav --sample=k --select=program --minndata=50
import os, os.path
import cPickle as pickle
from optparse import OptionParser
import numpy
import fitsio
from pixelFitDens import pixelAfeFeh
from segueSelect import read_gdwarfs, read_kdwarfs, _gi_gr, _mr_gi, \
    segueSelect, _GDWARFFILE, _KDWARFFILE
from fitSigz import _ZSUN
def collateFits(options,args):
    if options.sample.lower() == 'g':
        if options.select.lower() == 'program':
            raw= read_gdwarfs(_GDWARFFILE,logg=True,ebv=True,sn=True)
        else:
            raw= read_gdwarfs(logg=True,ebv=True,sn=True)
    elif options.sample.lower() == 'k':
        if options.select.lower() == 'program':
            raw= read_kdwarfs(_KDWARFFILE,logg=True,ebv=True,sn=True)
        else:
            raw= read_kdwarfs(logg=True,ebv=True,sn=True)
    #Bin the data
    binned= pixelAfeFeh(raw,dfeh=options.dfeh,dafe=options.dafe)
    #Savefiles
    if os.path.exists(args[0]):#Load density fits
        savefile= open(args[0],'rb')
        densfits= pickle.load(savefile)
        savefile.close()
    else:
        raise IOError("density fits file not included")
    if os.path.exists(args[1]):#Load density fits
        savefile= open(args[1],'rb')
        denssamples= pickle.load(savefile)
        savefile.close()
    else:
        raise IOError("density samples file not included")
    if os.path.exists(args[2]):#Load density fits
        savefile= open(args[2],'rb')
        mass= pickle.load(savefile)
        savefile.close()
    else:
        raise IOError("masses file not included")
    #if os.path.exists(args[3]):#Load density fits
    #    savefile= open(args[3],'rb')
    #    masssamples= pickle.load(savefile)
    #    savefile.close()
    #else:
    #    raise IOError("mass samples file not included")
    if os.path.exists(args[4]):#Load density fits
        savefile= open(args[4],'rb')
        velfits= pickle.load(savefile)
        savefile.close()
    else:
        raise IOError("vertical velocity file not included")
    if os.path.exists(args[5]):#Load density fits
        savefile= open(args[5],'rb')
        velsamples= pickle.load(savefile)
        savefile.close()
    else:
        raise IOError("vertical velocity samples  file not included")
    if os.path.exists(args[6]):#Load density fits
        savefile= open(args[6],'rb')
        velrfits= pickle.load(savefile)
        savefile.close()
    else:
        raise IOError("radial velocity file not included")
    if os.path.exists(args[7]):#Load density fits
        savefile= open(args[7],'rb')
        velrsamples= pickle.load(savefile)
        savefile.close()
    else:
        raise IOError("radial velocity samples  file not included")
    nrecs= len([r for r in densfits if not r is None])
    out= numpy.recarray(nrecs,dtype=[('feh',float),
                                     ('afe',float),
                                     ('hz',float),
                                     ('hr',float),
                                     ('bc',float),
                                     ('mass',float),
                                     ('sz',float),
                                     ('hsz',float),
                                     ('p1',float),
                                     ('p2',float),
                                     ('sr',float),
                                     ('hsr',float),
                                     ('zmin',float),
                                     ('zmax',float),
                                     ('zmedian',float),
                                     ('hz_err',float),
                                     ('hr_err',float),
                                     ('mass_err',float),
                                     ('sz_err',float),
                                     ('hsz_err',float),
                                     ('p1_err',float),
                                     ('p2_err',float),
                                     ('szp1_corr',float),
                                     ('szp2_corr',float),
                                     ('szhsz_corr',float),
                                     ('p1hsz_corr',float),
                                     ('p2hsz_corr',float),
                                     ('p1p2_corr',float),
                                     ('sr_err',float)])
                                     
    nout= 0
    #Start filling it up
    for ii in range(binned.npixfeh()):
        for jj in range(binned.npixafe()):
            data= binned(binned.feh(ii),binned.afe(jj))
            fehindx= binned.fehindx(binned.feh(ii))#Map onto regular binning
            afeindx= binned.afeindx(binned.afe(jj))#Unnecessary here
            if afeindx+fehindx*binned.npixafe() >= len(densfits):
                continue
            thisdensfit= densfits[afeindx+fehindx*binned.npixafe()]
            thisdenssamples= denssamples[afeindx+fehindx*binned.npixafe()]
            thismass= mass[afeindx+fehindx*binned.npixafe()]
            #thismasssamples= masssamples[afeindx+fehindx*binned.npixafe()]
            thisvelfit= velfits[afeindx+fehindx*binned.npixafe()]
            thesevelsamples= velsamples[afeindx+fehindx*binned.npixafe()]
            thisvelrfit= velrfits[afeindx+fehindx*binned.npixafe()]
            thesevelrsamples= velrsamples[afeindx+fehindx*binned.npixafe()]
            if thisdensfit is None:
                continue
            if len(data) < options.minndata:
                continue
            out['feh'][nout]= binned.feh(ii)
            out[nout]['afe']= binned.afe(jj)
            if options.densmodel.lower() == 'hwr' \
                    or options.densmodel.lower() == 'dblexp':
                out[nout]['hz']= numpy.exp(thisdensfit[0])*1000.
                if options.densmodel.lower() == 'dblexp':
                    out[nout]['hr']= numpy.exp(-thisdensfit[1])
                else:
                    out[nout]['hr']= numpy.exp(thisdensfit[1])
                out[nout]['bc']= thisdensfit[2]
                theseerrors= []
                xs= numpy.array([s[0] for s in thisdenssamples])
                theseerrors.append(0.5*(-numpy.exp(numpy.mean(xs)-numpy.std(xs))+numpy.exp(numpy.mean(xs)+numpy.std(xs))))
                out[nout]['hz_err']= theseerrors[0]*1000.
                if options.densmodel.lower() == 'dblexp':
                    xs= numpy.array([-s[1] for s in thisdenssamples])
                else:
                    xs= numpy.array([s[1] for s in thisdenssamples])
                theseerrors.append(0.5*(-numpy.exp(numpy.mean(xs)-numpy.std(xs))+numpy.exp(numpy.mean(xs)+numpy.std(xs))))
                out[nout]['hr_err']= theseerrors[1]
            #mass
            if options.sample.lower() == 'k':
                out[nout]['mass']= numpy.nan
            else:
                out[nout]['mass']= thismass/10.**6.
            #out[nout]['mass_err']= numpy.std(numpy.array(thismasssamples)/10.**6.)
            #Velocities
            if options.velmodel.lower() == 'hwr':
                out[nout]['sz']= numpy.exp(thisvelfit[1])
                out[nout]['hsz']= numpy.exp(thisvelfit[4])
                out[nout]['sr']= numpy.exp(thisvelrfit[1])
                out[nout]['hsr']= numpy.exp(thisvelrfit[4])
                out[nout]['p1']= thisvelfit[2]
                out[nout]['p2']= thisvelfit[3]
                zsorted= sorted(numpy.fabs(data.zc+_ZSUN))
                out[nout]['zmin']= zsorted[int(numpy.ceil(0.16*len(zsorted)))]*1000.
                out[nout]['zmax']= zsorted[int(numpy.floor(0.84*len(zsorted)))]*1000.
                out[nout]['zmedian']= numpy.median(numpy.fabs(data.zc)-_ZSUN)*1000.
                #Errors
                xs= numpy.array([s[1] for s in thesevelsamples])
                out[nout]['sz_err']= 0.5*(-numpy.exp(numpy.mean(xs)-numpy.std(xs))+numpy.exp(numpy.mean(xs)+numpy.std(xs)))
                xs= numpy.array([s[4] for s in thesevelsamples])
                out[nout]['hsz_err']= 0.5*(-numpy.exp(numpy.mean(xs)-numpy.std(xs))+numpy.exp(numpy.mean(xs)+numpy.std(xs)))
                xs= numpy.array([s[2] for s in thesevelsamples])
                out[nout]['p1_err']= numpy.std(xs)
                xs= numpy.array([s[3] for s in thesevelsamples])
                out[nout]['p2_err']= numpy.std(xs)
                xs= numpy.exp(numpy.array([s[1] for s in thesevelsamples]))
                ys= numpy.array([s[2] for s in thesevelsamples])
                out[nout]['szp1_corr']= numpy.corrcoef(xs,ys)[0,1]
                ys= numpy.array([s[3] for s in thesevelsamples])
                out[nout]['szp2_corr']= numpy.corrcoef(xs,ys)[0,1]
                xs= numpy.array([s[2] for s in thesevelsamples])
                out[nout]['p1p2_corr']= numpy.corrcoef(xs,ys)[0,1]
                xs= numpy.exp(numpy.array([s[4] for s in thesevelsamples]))
                ys= numpy.exp(numpy.array([s[1] for s in thesevelsamples]))
                out[nout]['szhsz_corr']= numpy.corrcoef(xs,ys)[0,1]
                ys= numpy.array([s[2] for s in thesevelsamples])
                out[nout]['p1hsz_corr']= numpy.corrcoef(xs,ys)[0,1]
                ys= numpy.array([s[3] for s in thesevelsamples])
                out[nout]['p2hsz_corr']= numpy.corrcoef(xs,ys)[0,1]
                xs= numpy.array([s[1] for s in thesevelrsamples])
                out[nout]['sr_err']= 0.5*(-numpy.exp(numpy.mean(xs)-numpy.std(xs))+numpy.exp(numpy.mean(xs)+numpy.std(xs)))
            nout+= 1
    #Write output
    fitsio.write(options.outfile,out,clobber=True)
    
def get_options():
    usage = "usage: %prog [options] <savefiles> <savefile>\n\nsavefiles= (in this order) name of the file with the density fits\nname of the file with density samples\nname of the file with masses\nname of the file with mass samples\nname of the file with vertical velocity fits\nname of the file with vertical velocity samples"
    parser = OptionParser(usage=usage)
    parser.add_option("-o",dest='outfile',default=None,
                      help="Name for output file")
    parser.add_option("--sample",dest='sample',default='g',
                      help="Use 'G' or 'K' dwarf sample")
    parser.add_option("--select",dest='select',default='all',
                      help="Select 'all' or 'program' stars")
    parser.add_option("--dfeh",dest='dfeh',default=0.1,type='float',
                      help="FeH bin size")   
    parser.add_option("--dafe",dest='dafe',default=0.05,type='float',
                      help="[a/Fe] bin size")   
    parser.add_option("--minndata",dest='minndata',default=100,type='int',
                      help="Minimum number of objects in a bin to perform a fit")   
    parser.add_option("--densmodel",dest='densmodel',default='hwr',
                      help="Density Model of fit")
    parser.add_option("--velmodel",dest='velmodel',default='hwr',
                      help="Density Model of fit")
    return parser
  
if __name__ == '__main__':
    parser= get_options()
    options,args= parser.parse_args()
    collateFits(options,args)

