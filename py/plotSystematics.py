import os, os.path
import sys
import numpy
from scipy import optimize, interpolate
import pickle
from galpy.util import bovy_plot
from matplotlib import pyplot
from matplotlib.ticker import NullFormatter
import monoAbundanceMW
from pixelFitDF import read_rawdata
from pixelFitDens import pixelAfeFeh
import AnDistance
from calcDFResults import setup_options
def plotSystematics(plotfilename):
    #hard-code these paths 
    savefilename= '../pdfs_margvrvt_gl_hsz/realDFFit_dfeh0.1_dafe0.05_dpdiskplhalofixbulgeflatwgasalt_gridall_fixvc230_an_margvrvt_staeckel_singles_bestr_rvssurf.sav'
    savefilename_vo250= '../pdfs_margvrvt_gl_hsz/realDFFit_dfeh0.1_dafe0.05_dpdiskplhalofixbulgeflatwgasalt_gridall_fixvc250_an_margvrvt_staeckel_singles_bestr_rvssurf.sav'
    savefilename_vo210= '../pdfs_margvrvt_gl_hsz/realDFFit_dfeh0.1_dafe0.05_dpdiskplhalofixbulgeflatwgasalt_gridall_fixvc250_an_margvrvt_staeckel_singles_bestr_rvssurf.sav'
    savefilename_zh200= '../pdfs_margvrvt_gl_hsz/realDFFit_dfeh0.1_dafe0.05_dpdiskplhalofixbulgeflatwgasalt_gridall_fixvc230_an_zh200_margvrvt_staeckel_singles_bestr_rvssurf.sav'
    savefilename_dvcm3= '../pdfs_margvrvt_gl_hsz/realDFFit_dfeh0.1_dafe0.05_dpdiskplhalofixbulgeflatwgasalt_gridall_fixvc230_an_dlnvcdlnr-3_margvrvt_staeckel_singles_bestr_rvssurf.sav'
    savefilename_dvcp3= '../pdfs_margvrvt_gl_hsz/realDFFit_dfeh0.1_dafe0.05_dpdiskplhalofixbulgeflatwgasalt_gridall_fixvc230_an_dlnvcdlnr3_margvrvt_staeckel_singles_bestr_rvssurf.sav'
    #Read surface densities
    if os.path.exists(savefilename):
        surffile= open(savefilename,'rb')
        surfrs= pickle.load(surffile)
        surfs= pickle.load(surffile)
        surferrs= pickle.load(surffile)
        kzs= pickle.load(surffile)
        kzerrs= pickle.load(surffile)
        surffile.close()
    else:
        raise IOError("savefilename with surface-densities has to exist")
    #Read surface densities
    if os.path.exists(savefilename_vo250):
        surffile= open(savefilename_vo250,'rb')
        surfrs_vo250= pickle.load(surffile)
        surfs_vo250= pickle.load(surffile)
        surferrs_vo250= pickle.load(surffile)
        kzs_vo250= pickle.load(surffile)
        kzerrs_vo250= pickle.load(surffile)
        altsurfrs_vo250= pickle.load(surffile)
        altsurfs_vo250= pickle.load(surffile)
        altsurferrs_vo250= pickle.load(surffile)
        altkzs_vo250= pickle.load(surffile)
        altkzerrs_vo250= pickle.load(surffile)
        surffile.close()
    else:
        raise IOError("savefilename with surface-densities has to exist")
    #Read surface densities
    if os.path.exists(savefilename_vo210):
        surffile= open(savefilename_vo210,'rb')
        surfrs_vo210= pickle.load(surffile)
        surfs_vo210= pickle.load(surffile)
        surferrs_vo210= pickle.load(surffile)
        kzs_vo210= pickle.load(surffile)
        kzerrs_vo210= pickle.load(surffile)
        altsurfrs_vo210= pickle.load(surffile)
        altsurfs_vo210= pickle.load(surffile)
        altsurferrs_vo210= pickle.load(surffile)
        altkzs_vo210= pickle.load(surffile)
        altkzerrs_vo210= pickle.load(surffile)
        surffile.close()
    else:
        raise IOError("savefilename with surface-densities has to exist")
    #Read surface densities
    if os.path.exists(savefilename_zh200):
        surffile= open(savefilename_zh200,'rb')
        surfrs_zh200= pickle.load(surffile)
        surfs_zh200= pickle.load(surffile)
        surferrs_zh200= pickle.load(surffile)
        kzs_zh200= pickle.load(surffile)
        kzerrs_zh200= pickle.load(surffile)
        altsurfrs_zh200= pickle.load(surffile)
        altsurfs_zh200= pickle.load(surffile)
        altsurferrs_zh200= pickle.load(surffile)
        altkzs_zh200= pickle.load(surffile)
        altkzerrs_zh200= pickle.load(surffile)
        surffile.close()
    else:
        raise IOError("savefilename with surface-densities has to exist")
    #Read surface densities
    if os.path.exists(savefilename_dvcm3):
        surffile= open(savefilename_dvcm3,'rb')
        surfrs_dvcm3= pickle.load(surffile)
        surfs_dvcm3= pickle.load(surffile)
        surferrs_dvcm3= pickle.load(surffile)
        kzs_dvcm3= pickle.load(surffile)
        kzerrs_dvcm3= pickle.load(surffile)
        altsurfrs_dvcm3= pickle.load(surffile)
        altsurfs_dvcm3= pickle.load(surffile)
        altsurferrs_dvcm3= pickle.load(surffile)
        altkzs_dvcm3= pickle.load(surffile)
        altkzerrs_dvcm3= pickle.load(surffile)
        surffile.close()
    else:
        raise IOError("savefilename with surface-densities has to exist")
    #Read surface densities
    if os.path.exists(savefilename_dvcp3):
        surffile= open(savefilename_dvcp3,'rb')
        surfrs_dvcp3= pickle.load(surffile)
        surfs_dvcp3= pickle.load(surffile)
        surferrs_dvcp3= pickle.load(surffile)
        kzs_dvcp3= pickle.load(surffile)
        kzerrs_dvcp3= pickle.load(surffile)
        altsurfrs_dvcp3= pickle.load(surffile)
        altsurfs_dvcp3= pickle.load(surffile)
        altsurferrs_dvcp3= pickle.load(surffile)
        altkzs_dvcp3= pickle.load(surffile)
        altkzerrs_dvcp3= pickle.load(surffile)
        surffile.close()
    else:
        raise IOError("savefilename with surface-densities has to exist")
    if True:#options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif False:#options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    indx= numpy.isnan(surfrs)
    indx[50]= True
    indx[57]= True
    indx= True - indx
    surfrs= surfrs[indx]
    surfs= surfs[indx]
    surferrs= surferrs[indx]
    kzs= kzs[indx]
    kzerrs= kzerrs[indx]
    #vo250
    surfrs_vo250= surfrs_vo250[indx]
    surfs_vo250= surfs_vo250[indx]
    surferrs_vo250= surferrs_vo250[indx]
    kzs_vo250= kzs_vo250[indx]
    kzerrs_vo250= kzerrs_vo250[indx]
    altsurfrs_vo250= altsurfrs_vo250[indx]
    altsurfs_vo250= altsurfs_vo250[indx]
    altsurferrs_vo250= altsurferrs_vo250[indx]
    altkzs_vo250= altkzs_vo250[indx]
    altkzerrs_vo250= altkzerrs_vo250[indx]
    #vo210
    surfrs_vo210= surfrs_vo210[indx]
    surfs_vo210= surfs_vo210[indx]
    surferrs_vo210= surferrs_vo210[indx]
    kzs_vo210= kzs_vo210[indx]
    kzerrs_vo210= kzerrs_vo210[indx]
    altsurfrs_vo210= altsurfrs_vo210[indx]
    altsurfs_vo210= altsurfs_vo210[indx]
    altsurferrs_vo210= altsurferrs_vo210[indx]
    altkzs_vo210= altkzs_vo210[indx]
    altkzerrs_vo210= altkzerrs_vo210[indx]
    #zh200
    surfrs_zh200= surfrs_zh200[indx]
    surfs_zh200= surfs_zh200[indx]
    surferrs_zh200= surferrs_zh200[indx]
    kzs_zh200= kzs_zh200[indx]
    kzerrs_zh200= kzerrs_zh200[indx]
    altsurfrs_zh200= altsurfrs_zh200[indx]
    altsurfs_zh200= altsurfs_zh200[indx]
    altsurferrs_zh200= altsurferrs_zh200[indx]
    altkzs_zh200= altkzs_zh200[indx]
    altkzerrs_zh200= altkzerrs_zh200[indx]
    #dvc-3
    surfrs_dvcm3= surfrs_dvcm3[indx]
    surfs_dvcm3= surfs_dvcm3[indx]
    surferrs_dvcm3= surferrs_dvcm3[indx]
    kzs_dvcm3= kzs_dvcm3[indx]
    kzerrs_dvcm3= kzerrs_dvcm3[indx]
    altsurfrs_dvcm3= altsurfrs_dvcm3[indx]
    altsurfs_dvcm3= altsurfs_dvcm3[indx]
    altsurferrs_dvcm3= altsurferrs_dvcm3[indx]
    altkzs_dvcm3= altkzs_dvcm3[indx]
    altkzerrs_dvcm3= altkzerrs_dvcm3[indx]
    #dvc+3
    surfrs_dvcp3= surfrs_dvcp3[indx]
    surfs_dvcp3= surfs_dvcp3[indx]
    surferrs_dvcp3= surferrs_dvcp3[indx]
    kzs_dvcp3= kzs_dvcp3[indx]
    kzerrs_dvcp3= kzerrs_dvcp3[indx]
    altsurfrs_dvcp3= altsurfrs_dvcp3[indx]
    altsurfs_dvcp3= altsurfs_dvcp3[indx]
    altsurferrs_dvcp3= altsurferrs_dvcp3[indx]
    altkzs_dvcp3= altkzs_dvcp3[indx]
    altkzerrs_dvcp3= altkzerrs_dvcp3[indx]
    fehs= fehs[indx]
    afes= afes[indx]
    #Plot
    bovy_plot.bovy_print(fig_height=8.,fig_width=7.)
    dx= 0.8/5.
    #vo250
    left, bottom, width, height= 0.1, 0.9-dx, 0.8, dx
    axTop= pyplot.axes([left,bottom,width,height])
    allaxes= [axTop]
    fig= pyplot.gcf()
    fig.sca(axTop)
    bovy_plot.bovy_plot(surfrs,numpy.log(altsurfs_vo250/surfs),'ko',
                        overplot=True,
                        zorder=10)
    pyplot.errorbar(surfrs,numpy.log(altsurfs_vo250/surfs),
                    yerr=surferrs/surfs,
                    elinewidth=1.,capsize=3,
                    linestyle='none',zorder=5,
                    color='k')
    pyplot.plot([4.,10.],
                [numpy.log((250./230.)**2.),numpy.log((250./230.)**2.)],'--',
                color='0.5',lw=2.)
    pyplot.plot([4.,10.],
                [0.,0.],'-',
                color='0.5',lw=2.)
    bovy_plot.bovy_text(8.5,0.18,r'$V_c = 250\,\mathrm{km\,s}^{-1}$',
                        size=14.)
    #bovy_plot.bovy_plot(surfrs,numpy.log(altkzs_vo250/kzs),'kd',
    #                    overplot=True,color='0.35',
    #                    zorder=10)
    thisax= pyplot.gca()
    nullfmt   = NullFormatter()         # no labels
    thisax.xaxis.set_major_formatter(nullfmt)
    thisax.set_ylim(-0.29,0.29)
    thisax.set_xlim(4.,10.)
    bovy_plot._add_ticks()
    #vo210
    left, bottom, width, height= 0.1, 0.9-2.*dx, 0.8, dx
    axTop= pyplot.axes([left,bottom,width,height])
    allaxes= [axTop]
    fig= pyplot.gcf()
    fig.sca(axTop)
    bovy_plot.bovy_plot(surfrs,numpy.log(altsurfs_vo210/surfs),'ko',
                        overplot=True,
                        zorder=10)
    pyplot.errorbar(surfrs,numpy.log(altsurfs_vo210/surfs),
                    yerr=surferrs/surfs,
                    elinewidth=1.,capsize=3,
                    linestyle='none',zorder=5,
                    color='k')
    pyplot.plot([4.,10.],
                [numpy.log((210./230.)**2.),numpy.log((210./230.)**2.)],'--',
                color='0.5',lw=2.)
    pyplot.plot([4.,10.],
                [0.,0.],'-',
                color='0.5',lw=2.)
    bovy_plot.bovy_text(8.5,0.18,r'$V_c = 210\,\mathrm{km\,s}^{-1}$',
                        size=14.)
    #bovy_plot.bovy_plot(surfrs,numpy.log(altkzs_vo210/kzs),'kd',
    #                    overplot=True,color='0.35',
    #                    zorder=10)
    thisax= pyplot.gca()
    nullfmt   = NullFormatter()         # no labels
    thisax.xaxis.set_major_formatter(nullfmt)
    thisax.set_ylim(-0.29,0.29)
    thisax.set_xlim(4.,10.)
    bovy_plot._add_ticks()
    #zh200
    left, bottom, width, height= 0.1, 0.9-3.*dx, 0.8, dx
    axTop= pyplot.axes([left,bottom,width,height])
    allaxes= [axTop]
    fig= pyplot.gcf()
    fig.sca(axTop)
    bovy_plot.bovy_plot(surfrs,numpy.log(altsurfs_zh200/surfs),'ko',
                        overplot=True,
                        zorder=10)
    pyplot.errorbar(surfrs,numpy.log(altsurfs_zh200/surfs),
                    yerr=surferrs/surfs,
                    elinewidth=1.,capsize=3,
                    linestyle='none',zorder=5,
                    color='k')
    pyplot.plot([4.,10.],
                [0.,0.],'-',
                color='0.5',lw=2.)
    bovy_plot.bovy_text(8.5,0.18,r'$z_h = 200\,\mathrm{pc}$',
                        size=14.)
    #bovy_plot.bovy_plot(surfrs,numpy.log(altkzs_zh200/kzs),'kd',
    #                    overplot=True,color='0.35',
    #                    zorder=10)
    thisax= pyplot.gca()
    nullfmt   = NullFormatter()         # no labels
    thisax.xaxis.set_major_formatter(nullfmt)
    thisax.set_ylim(-0.29,0.29)
    thisax.set_xlim(4.,10.)
    pyplot.ylabel(r'$\ln \Sigma_{1.1}^{\mathrm{alt}} / \Sigma_{1.1}^{\mathrm{fid}}$')
    bovy_plot._add_ticks()
    #dvcm3
    left, bottom, width, height= 0.1, 0.9-4.*dx, 0.8, dx
    thisax= pyplot.axes([left,bottom,width,height])
    allaxes.append(thisax)
    fig.sca(thisax)    
    bovy_plot.bovy_plot(surfrs,numpy.log(altsurfs_dvcm3/surfs),'ko',
                        overplot=True,
                        zorder=10)
    pyplot.errorbar(surfrs,numpy.log(altsurfs_dvcm3/surfs),
                    yerr=surferrs/surfs,
                    elinewidth=1.,capsize=3,
                    linestyle='none',zorder=5,
                    color='k')
    xs= numpy.linspace(4.,10.,1001)
    ys= numpy.log((69.*numpy.exp(-(xs-8.)/2.5)+67*(8./xs)**2.*(-0.1))/
                  (69.*numpy.exp(-(xs-8.)/2.5)))
    pyplot.plot(xs,ys,'--',color='0.5',lw=2.)
    pyplot.plot([4.,10.],
                [0.,0.],'-',
                color='0.5',lw=2.)
    bovy_plot.bovy_plot(surfrs,numpy.log(altkzs_dvcm3/kzs),'kd',
                        overplot=True,color='0.35',
                        xrange=[4.,10.],
                        yrange=[-0.41,0.41],zorder=10)
    bovy_plot.bovy_text(7.65,0.18,
                        r'$\mathrm{d} \ln V_c(R_0) / \mathrm{d} \ln R = -0.1$',
                        size=14.)
    thisax= pyplot.gca()
    thisax.set_ylim(-0.29,0.29)
    pyplot.xlim(4.,10.)
    bovy_plot._add_ticks()
    nullfmt   = NullFormatter()         # no labels
    thisax.xaxis.set_major_formatter(nullfmt)
    #dvcp3
    left, bottom, width, height= 0.1, 0.9-5.*dx, 0.8, dx
    thisax= pyplot.axes([left,bottom,width,height])
    allaxes.append(thisax)
    fig.sca(thisax)    
    bovy_plot.bovy_plot(surfrs,numpy.log(altsurfs_dvcp3/surfs),'ko',
                        overplot=True,
                        zorder=10)
    pyplot.errorbar(surfrs,numpy.log(altsurfs_dvcp3/surfs),
                    yerr=surferrs/surfs,
                    elinewidth=1.,capsize=3,
                    linestyle='none',zorder=5,
                    color='k')
    xs= numpy.linspace(4.,10.,1001)
    ys= numpy.log((69.*numpy.exp(-(xs-8.)/2.5)+67*(8./xs)**2.*(+0.1))/
                  (69.*numpy.exp(-(xs-8.)/2.5)))
    pyplot.plot(xs,ys,'--',color='0.5',lw=2.)
    pyplot.plot([4.,10.],
                [0.,0.],'-',
                color='0.5',lw=2.)
    bovy_plot.bovy_plot(surfrs,numpy.log(altkzs_dvcp3/kzs),'kd',
                        overplot=True,color='0.35',
                        xrange=[4.,10.],
                        yrange=[-0.41,0.41],zorder=10)
    bovy_plot.bovy_text(7.65,0.18,
                        r'$\mathrm{d} \ln V_c(R_0) / \mathrm{d} \ln R = 0.1$',
                        size=14.)
    thisax= pyplot.gca()
    thisax.set_ylim(-0.29,0.29)
    pyplot.xlim(4.,10.)
    bovy_plot._add_ticks()
    pyplot.xlabel(r'$R\ (\mathrm{kpc})$')
    bovy_plot.bovy_end_print(plotfilename)

def plotDistanceSystematics(plotfilename):
    #hard-code these paths 
    savefilename= '../pdfs_margvrvt_gl_hsz/realDFFit_dfeh0.1_dafe0.05_dpdiskplhalofixbulgeflatwgasalt_gridall_fixvc230_an_margvrvt_staeckel_singles_bestr_rvssurf.sav'
    savefilename_iv= '../pdfs_margvrvt_gl_hsz/realDFFit_dfeh0.1_dafe0.05_dpdiskplhalofixbulgeflatwgasalt_gridall_fixvc230_margvrvt_staeckel_singles_bestr_rvssurf.sav'
    #Read surface densities
    if os.path.exists(savefilename):
        surffile= open(savefilename,'rb')
        surfrs= pickle.load(surffile)
        surfs= pickle.load(surffile)
        surferrs= pickle.load(surffile)
        surffile.close()
    else:
        raise IOError("savefilename with surface-densities has to exist")
    #Read surface densities
    if os.path.exists(savefilename_iv):
        surffile= open(savefilename_iv,'rb')
        surfrs_iv= pickle.load(surffile)
        surfs_iv= pickle.load(surffile)
        surferrs_iv= pickle.load(surffile)
        altsurfrs_iv= pickle.load(surffile)
        altsurfs_iv= pickle.load(surffile)
        altsurferrs_iv= pickle.load(surffile)
        surffile.close()
    else:
        raise IOError("savefilename with surface-densities has to exist")
    if True:#options.sample.lower() == 'g':
        savefile= open('binmapping_g.sav','rb')
    elif False:#options.sample.lower() == 'k':
        savefile= open('binmapping_k.sav','rb')
    fehs= pickle.load(savefile)
    afes= pickle.load(savefile)
    indx= numpy.isnan(surfrs)
    indx[50]= True
    indx[57]= True
    indx= True - indx
    surfrs= surfrs[indx]
    surfs= surfs[indx]
    surferrs= surferrs[indx]
    #vo250
    surfrs_iv= surfrs_iv[indx]
    surfs_iv= surfs_iv[indx]
    surferrs_iv= surferrs_iv[indx]
    altsurfrs_iv= altsurfrs_iv[indx]
    altsurfs_iv= altsurfs_iv[indx]
    altsurferrs_iv= altsurferrs_iv[indx]
    fehs= fehs[indx]
    afes= afes[indx]
    #Plot
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(surfrs,numpy.log(altsurfs_iv/surfs),'ko',
                        xlabel=r'$R\ (\mathrm{kpc})$',
                        ylabel=r'$\ln \Sigma_{1.1}^{\mathrm{Ivezic}} / \Sigma_{1.1}^{\mathrm{An}}$',
                        xrange=[4.,10.],
                        yrange=[-0.41,0.41],zorder=10)
    pyplot.errorbar(surfrs,numpy.log(altsurfs_iv/surfs),
                    yerr=surferrs/surfs,
                    elinewidth=1.,capsize=3,
                    linestyle='none',zorder=5,
                    color='k')
    #Get distance factors
    options= setup_options(None)
    raw= read_rawdata(options)
    binned= pixelAfeFeh(raw,dfeh=options.dfeh,dafe=options.dafe)
    #Get fehs from monoAb
    tfehs= monoAbundanceMW.fehs(k=(options.sample.lower() == 'k'))
    tafes= monoAbundanceMW.afes(k=(options.sample.lower() == 'k'))
    plotthis= numpy.zeros_like(tfehs)
    for ii in range(len(tfehs)):
        #Get the relevant data
        data= binned(tfehs[ii],tafes[ii])
        plotthis[ii]= AnDistance.AnDistance(data.dered_g-data.dered_r,
                                            data.feh)
    plotthis= plotthis[indx]
    #Spline interpolate
    distfunc= interpolate.UnivariateSpline(surfrs,numpy.log(plotthis))
    plotrs= numpy.linspace(4.5,9.,1001)
    pyplot.plot(plotrs,distfunc(plotrs),'--',color='0.5',lw=2.)
    pyplot.plot(plotrs,-distfunc(plotrs),'--',color='0.5',lw=2.)
    bovy_plot.bovy_end_print(plotfilename)

if __name__ == '__main__':
    if sys.argv[1].lower() == 'distance':
        plotDistanceSystematics(sys.argv[2])
    else:
        plotSystematics(sys.argv[2])
        
