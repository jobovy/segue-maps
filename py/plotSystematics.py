import os, os.path
import sys
import numpy
from scipy import optimize
import pickle
from galpy.util import bovy_plot
from matplotlib import pyplot, cm
from selectFigs import _squeeze
from plotOverview import expcurve
def plotSurf(plotfilename):
    #hard-code these paths 
    savefilename= '../pdfs_margvrvt_gl_hsz/realDFFit_dfeh0.1_dafe0.05_dpdiskplhalofixbulgeflatwgasalt_gridall_fixvc230_an_margvrvt_staeckel_singles_bestr_rvssurf.sav'
    savefilename_vo250= '../pdfs_margvrvt_gl_hsz/realDFFit_dfeh0.1_dafe0.05_dpdiskplhalofixbulgeflatwgasalt_gridall_fixvc250_an_margvrvt_staeckel_singles_bestr_rvssurf.sav'
    savefilename_noan= '../pdfs_margvrvt_gl_hsz/realDFFit_dfeh0.1_dafe0.05_dpdiskplhalofixbulgeflatwgasalt_gridall_fixvc230_margvrvt_staeckel_singles_bestr_rvssurf.sav'
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
    if os.path.exists(savefilename):
        surffile= open(savefilename_vo250,'rb')
        surfrs_vo250= pickle.load(surffile)
        surfs_vo250= pickle.load(surffile)
        surferrs_vo250= pickle.load(surffile)
        altsurfrs_vo250= pickle.load(surffile)
        altsurfs_vo250= pickle.load(surffile)
        altsurferrs_vo250= pickle.load(surffile)
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
    surfrs_vo250= surfrs_vo250[indx]
    surfs_vo250= surfs_vo250[indx]
    surferrs_vo250= surferrs_vo250[indx]
    altsurfrs_vo250= altsurfrs_vo250[indx]
    altsurfs_vo250= altsurfs_vo250[indx]
    altsurferrs_vo250= altsurferrs_vo250[indx]
    fehs= fehs[indx]
    afes= afes[indx]
    #Plot
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(surfrs,numpy.log(altsurfs_vo250/surfs),'ko',
                        xlabel=r'$R\ (\mathrm{kpc})$',
                        ylabel=r'$\ln \Sigma_{1.1}^{\mathrm{alt}} / \Sigma_{1.1}^{\mathrm{fid}}$',
                        xrange=[4.,10.],
                        yrange=[-0.41,0.41],zorder=10)
    pyplot.errorbar(surfrs,numpy.log(altsurfs_vo250/surfs),
                    yerr=surferrs/surfs,
                    elinewidth=1.,capsize=3,
                    linestyle='none',zorder=5,
                    color='k')
    pyplot.plot([4.,10.],
                [numpy.log((250./230.)**2.),numpy.log((250./230.)**2.)],'--',
                color='0.5',lw=2.)
    bovy_plot.bovy_text(7.5,0.3,r'$V_c = 250\,\mathrm{km\,s}^{-1}$',
                        size=14.)
    bovy_plot.bovy_end_print(plotfilename)

if __name__ == '__main__':
    plotSurf(sys.argv[1])

