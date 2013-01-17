import sys
import os, os.path
import math
import numpy
import cPickle as pickle
from optparse import OptionParser
from pixelFitDF import _REFV0, _REFR0
from calcDFResults import calcDFResults, setup_options
_GFIT= '../realDF/realDFFit_dfeh0.1_dafe0.05_mpdiskplhalofixbulgeflat_staeckelg_singles.sav'
_KFIT= '../realDF/realDFFitK_dfeh0.1_dafe0.05_mpdiskplhalofixbulgeflat_staeckelg_singles.sav'
def dfResultsTable(args):
    cmdline= '%python dfResultsTable.py '+args
    #Load g, k, and combined fits
    options= setup_options(None)
    options.sample= 'g'
    options.select= 'all'
    gfits= calcDFResults(options,[_GFIT])
    options.sample= 'k'
    options.select= 'program'   
    kfits= calcDFResults(options,[_KFIT])
    options.sample= 'gk'
    gkfits= calcDFResults(options,[_GFIT,_KFIT])
    #Set up sections
    names= ['$R_d\ (\mathrm{kpc})$',
            '$z_h\ (\mathrm{pc})$',
            '$V_{c,\mathrm{disk}}/V_c\,(R_0)$',
            '$V_{c,\mathrm{disk}}/V_c\,(2.2\,R_d)$',
            '$\\Sigma_{\mathrm{disk}}\ (M_{\odot}\,\mathrm{pc}^{-2})$',
            '$M_{\mathrm{disk}}\ (10^{10}\,M_{\odot})$',
            '$\\alpha_{h}$',
            '$\\rho_{\mathrm{DM}}\,(R_0,0)\ (M_{\odot}\,\mathrm{pc}^{-3})$',
            '$V_c(R_0)\ [\mathrm{km\ s}^{-1}]$',
            '$\\frac{\mathrm{d}\ln V_c}{\mathrm{d}\ln R}\,(R_0)$',
            '$\\rho_{\mathrm{total}}\,(R_0,0)\ (M_{\odot}\,\mathrm{pc}^{-3})$',
            '$\\Sigma(R_0,|Z|\leq 0.8\,\mathrm{kpc})\ (M_{\odot}\,\mathrm{pc}^{-2})$',
            '$\\Sigma(R_0,|Z|\leq 1.1\,\mathrm{kpc})\ (M_{\odot}\,\mathrm{pc}^{-2})$']
    skip= [0,0,0,0,0,1,0,1,0,0,0,0,0] #1 if line skip after this parameter
    scale= [_REFR0,_REFR0*1000.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]
    key= ['rdexp','zhexp','vcdvcro','vcdvc','surfzdisk','massdisk',
          'plhalo','rhodm',
          'vc','dlnvcdlnr','rhooalt','surfz800','surfz']
    #Make table
    outfile= open(args,'w')
    for ii in range(len(names)):
        #Set up line
        printline= names[ii]
        #G
        printline+= ' & '
        bestfit= gfits[key[ii]+'_m']*scale[ii]
        err= gfits[key[ii]+'_merr']*scale[ii]
        #Prepare
        if math.log10(err) >= 0.:
            value= '$%.0f$' % bestfit
            err= '$\pm$%.0f' % err
        elif math.log10(err) >= -1.:
            value= '$%.1f$' % bestfit
            err= '$\pm$%.1f' % err
        elif math.log10(err) >= -2.:
            value= '$%.2f$' % bestfit
            err= '$\pm$%.2f' % err        
        elif math.log10(err) >= -3.:
            value= '$%.3f$' % bestfit
            err= '$\pm$%.3f' % err 
        else:
            value= '$%.4f$' % bestfit
            err= '$\pm$%.4f' % err
        printline+= value+'&'+err
        #K
        printline+= ' & '
        bestfit= kfits[key[ii]+'_m']*scale[ii]
        err= kfits[key[ii]+'_merr']*scale[ii]
        #Prepare
        if math.log10(err) >= 0.:
            value= '$%.0f$' % bestfit
            err= '$\pm$%.0f' % err
        elif math.log10(err) >= -1.:
            value= '$%.1f$' % bestfit
            err= '$\pm$%.1f' % err
        elif math.log10(err) >= -2.:
            value= '$%.2f$' % bestfit
            err= '$\pm$%.2f' % err        
        elif math.log10(err) >= -3.:
            value= '$%.3f$' % bestfit
            err= '$\pm$%.3f' % err 
        else:
            value= '$%.4f$' % bestfit
            err= '$\pm$%.4f' % err
        printline+= value+'&'+err
        #combined
        printline+= ' & '
        bestfit= gkfits[key[ii]+'_m']*scale[ii]
        err= gkfits[key[ii]+'_merr']*scale[ii]
        #Prepare
        if math.log10(err) >= 0.:
            value= '$%.0f$' % bestfit
            err= '$\pm$%.0f' % err
        elif math.log10(err) >= -1.:
            value= '$%.1f$' % bestfit
            err= '$\pm$%.1f' % err
        elif math.log10(err) >= -2.:
            value= '$%.2f$' % bestfit
            err= '$\pm$%.2f' % err        
        elif math.log10(err) >= -3.:
            value= '$%.3f$' % bestfit
            err= '$\pm$%.3f' % err 
        else:
            value= '$%.4f$' % bestfit
            err= '$\pm$%.4f' % err
        printline+= value+'&'+err
        if not ii == (len(names)-1):
            printline+= '\\\\'
        if skip[ii]:
            printline+= '\\\\'
            #Write the line
        outfile.write(printline+'\n')
    outfile.write('\\enddata\n')
    outfile.write(cmdline+'\n')
    outfile.close()

if __name__ == '__main__':
    if len(sys.argv) < 2: raise IOError("must specify filename for table")
    dfResultsTable(sys.argv[1])
