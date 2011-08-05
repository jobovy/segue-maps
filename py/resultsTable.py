import sys
import os, os.path
import math
import numpy
import cPickle as pickle
from optparse import OptionParser
_RICHDBLEXPNAMES= ['HWR_dens_rich_g.sav',
                   'HWR_dens_rich_gbright.sav',
                   'HWR_dens_rich_gfaint.sav',
                   'HWR_dens_rich_g_south.sav',
                   'HWR_dens_rich_g_north.sav',
                   'HWR_dens_rich_g_bmin45.sav',
                   'HWR_dens_rich_g_bmax45.sav']
_RICHFLARENAMES= ['HWR_dens_rich_g_flare.sav',
                  'HWR_dens_rich_gbright_flare.sav',
                  'HWR_dens_rich_gfaint_flare.sav',
                  'HWR_dens_rich_gsouth_flare.sav',
                  'HWR_dens_rich_gnorth_flare.sav',
                  'HWR_dens_rich_gbmin45_flare.sav',
                  'HWR_dens_rich_gbmax45_flare.sav']
_RICHTWODBLEXPNAMES= ['HWR_dens_rich_g_twodblexp.sav',
                      'HWR_dens_rich_gbright_twodblexp.sav',
                      'HWR_dens_rich_gfaint_twodblexp.sav',
                      'HWR_dens_rich_gsouth_twodblexp.sav',
                      'HWR_dens_rich_gnorth_twodblexp.sav',
                      'HWR_dens_rich_gbmin45_twodblexp.sav',
                      'HWR_dens_rich_gbmax45_twodblexp.sav']
_POORDBLEXPNAMES= ['HWR_dens_poor_g.sav',
                   'HWR_dens_poor_gbright.sav',
                   'HWR_dens_poor_gfaint.sav',
                   'HWR_dens_poor_g_south.sav',
                   'HWR_dens_poor_g_north.sav',
                   'HWR_dens_poor_g_bmin45.sav',
                   'HWR_dens_poor_g_bmax45.sav']
_POORFLARENAMES= ['HWR_dens_poor_g_flare.sav',
                  'HWR_dens_poor_gbright_flare.sav',
                  'HWR_dens_poor_gfaint_flare.sav',
                  'HWR_dens_poor_gsouth_flare.sav',
                  'HWR_dens_poor_gnorth_flare.sav',
                  'HWR_dens_poor_gbmin45_flare.sav',
                  'HWR_dens_poor_gbmax45_flare.sav']
_POORTWODBLEXPNAMES= ['HWR_dens_poor_g_twodblexp.sav',
                      'HWR_dens_poor_gbright_twodblexp.sav',
                      'HWR_dens_poor_gfaint_twodblexp.sav',
                      'HWR_dens_poor_gsouth_twodblexp.sav',
                      'HWR_dens_poor_gnorth_twodblexp.sav',
                      'HWR_dens_poor_gbmin45_twodblexp.sav',
                      'HWR_dens_poor_gbmax45_twodblexp.sav']
_RICHFEHNAMES= ['HWR_dens_richpoorest_g_twodblexp.sav',
                'HWR_dens_richpoor_g_twodblexp.sav',
                'HWR_dens_richrich_g_twodblexp.sav']
_POORFEHNAMES= ['HWR_dens_poorpoor_g_twodblexp.sav',
                'HWR_dens_poorrich_g_twodblexp.sav']
_AFENAMES= ['HWR_dens_apoorpoor_g_twodblexp.sav',
            'HWR_dens_apoorrich_g_twodblexp.sav',
            'HWR_dens_arichpoor_g_twodblexp.sav',
            'HWR_dens_arichrich_g_twodblexp.sav']
def resultsTable(parser):
    (options,args)= parser.parse_args()
    cmdline= '%python resultsTable '+args[0]+' --table='+options.table
    if len(args) == 0:
        parser.print_help()
        return
    #Set up sections
    if options.table.lower() == 'richresults':
        sections= [_RICHDBLEXPNAMES,_RICHFLARENAMES,_RICHTWODBLEXPNAMES,
                   _RICHFEHNAMES]
        format= ['hz','hR','hz1','hR1','a2','hf','ac']
    elif options.table.lower() == 'poorresults':
        sections= [_POORDBLEXPNAMES,_POORFLARENAMES,_POORTWODBLEXPNAMES,
                   _POORFEHNAMES]
        format= ['hz','hR','hz1','hR1','a2','hf','ac']
    elif options.table.lower() == 'afe':
        sections= [_AFENAMES]
        format= ['hz','hR','hz1','hR1','a2'] #just list the two-dblexp parameters
    #Make table
    outfile= open(args[0],'w')
    for section in sections:
        for name in section:
            #Open savefile
            savefile= open(os.path.join('..','fits',name),'rb')
            params= pickle.load(savefile)
            samples= pickle.load(savefile)
            savefile.close()
            if 'twodblexp' in name.lower():
                paramnames= ['hz','hz1','hR','hR1','a2']
            elif 'flare' in name.lower():
                paramnames= ['hz','hf','hR']
            else:
                paramnames= ['hz','hR','ac']
            thisline= {}
            for ii in range(len(paramnames)):
                xs= numpy.array([s[ii] for s in samples])
                if paramnames[ii] == 'a2' or paramnames[ii] == 'ac':
                    thisline[paramnames[ii]]= numpy.mean(xs)
                    err= numpy.std(xs)
                    thisline[paramnames[ii]+'_err']= err                  
                else:
                    thisline[paramnames[ii]]= numpy.exp(numpy.mean(xs))
                    err_low= numpy.exp(numpy.mean(xs))-numpy.exp(numpy.mean(xs)-numpy.std(xs))
                    err_high= -numpy.exp(numpy.mean(xs))+numpy.exp(numpy.mean(xs)+numpy.std(xs))
                    if err_low/err_high > 1.4 or err_low/err_high < 0.6:
                        thisline[paramnames[ii]+'_low']= err_low
                        thisline[paramnames[ii]+'_high']= err_high
                        thisline[paramnames[ii]+'_err']= 0.5*(err_low+err_high)
                    else:
                        thisline[paramnames[ii]+'_err']= 0.5*(err_low+err_high)
                    if thisline[paramnames[ii]] > 4.5:
                        #Also list lower limit
                        xs= sorted(numpy.array([s[ii] for s in samples]))
                        indx2= int(numpy.floor(0.01*len(samples)))
                        thisline[paramnames[ii]+'_ll']= numpy.exp(xs[indx2])
            #Set up line
            if 'bmin' in name:
                printline= '$|b| > 45^\circ$ '
            elif 'bmax' in name:
                printline= '$|b| < 45^\circ$  '
            elif 'north' in name:
                printline= '$b > 0^\circ$  '
            elif 'south' in name:
                printline= '$b < 0^\circ$  '               
            elif 'faint' in name:
                printline= 'faint plates  '
            elif 'bright' in name:
                printline= 'bright plates  '
            elif 'apoorpoor' in name:
                printline= '0.00 $<$ [$\\alpha$/Fe] $<$ 0.15  '
            elif 'apoorrich' in name:
                printline= '0.15 $\leq$ [$\\alpha$/Fe] $<$ 0.25  '
            elif 'arichpoor' in name:
                printline= '0.25 $\leq$ [$\\alpha$/Fe] $<$ 0.35  '
            elif 'arichrich' in name:
                printline= '0.35 $\leq$ [$\\alpha$/Fe] $<$ 0.5\phantom{0}  '
            elif 'poorpoor' in name:
                printline= '\protect{[}Fe/H] $<$ -0.7  '
            elif 'poorrich' in name:
                printline= '\protect{[}Fe/H] $>$ -0.7  '
            elif 'richpoorest' in name:
                printline= '\phantom{-1.00 $<$} \protect{[}Fe/H] $<$ -0.5\phantom{0}  '
            elif 'richpoor' in name:
                printline= '-0.5\phantom{0} $<$ [Fe/H] $<$ -0.25  '
            elif 'richrich' in name:
                printline= '-0.25 $<$ \protect{[}Fe/H] \phantom{$<$ -0.25}  '
            else:
                printline= 'all plates '
            for paramname in format:
                if not thisline.has_key(paramname):
                    printline+= '& \ldots & '
                    continue
                if thisline[paramname] > 4.5 \
                        and thisline[paramname] < 6.:
                    ll, valerr= True, True
                elif thisline[paramname] >= 6.:
                    ll, valerr= True, False
                else:
                    ll, valerr= False, True
                #hz, hz1 are in pc
                for key in thisline.keys():
                    if 'hz' in key \
                            and (key == paramname \
                                     or key == (paramname+'_err') \
                                     or key == (paramname+'_low') \
                                     or key == (paramname+'_high') \
                                     or key == (paramname+'_ll')):
                        thisline[key]*= 1000.
                #Prepare
                if math.log10(thisline[paramname+'_err']) >= 0.:
                    value= '%.0f' % thisline[paramname]
                    if thisline.has_key(paramname+'_low'):
                        err= '$^{+%.0f}_{-%.0f}$' % (thisline[paramname+'_low'],thisline[paramname+'_high'])
                    else:
                        err= '$\pm$%.0f' % thisline[paramname+'_err']
                elif math.log10(thisline[paramname+'_err']) >= -1.:
                    value= '%.1f' % thisline[paramname]
                    if thisline.has_key(paramname+'_low'):
                        err= '$^{+%.1f}_{-%.1f}$' % (thisline[paramname+'_low'],thisline[paramname+'_high'])
                    else:
                        err= '$\pm$%.1f' % thisline[paramname+'_err']
                elif math.log10(thisline[paramname+'_err']) >= -2.:
                    value= '%.2f' % thisline[paramname]
                    if thisline.has_key(paramname+'_low'):
                        err= '$^{+%.2f}_{-%.2f}$' % (thisline[paramname+'_low'],thisline[paramname+'_high'])
                    else:
                        err= '$\pm$%.2f' % thisline[paramname+'_err']
                elif math.log10(thisline[paramname+'_err']) >= -3.:
                    value= '%.3f' % thisline[paramname]
                    if thisline.has_key(paramname+'_low'):
                        err= '$^{+%.3f}_{-%.3f}$' % (thisline[paramname+'_low'],thisline[paramname+'_high'])
                    else:
                        err= '$\pm$%.3f' % thisline[paramname+'_err']
                elif math.log10(thisline[paramname+'_err']) >= -4.:
                    value= '%.4f' % thisline[paramname]
                    if thisline.has_key(paramname+'_low'):
                        err= '$^{+%.4f}_{-%.4f}$' % (thisline[paramname+'_low'],thisline[paramname+'_high'])
                    else:
                        err= '$\pm$%.4f' % thisline[paramname+'_err']
                elif math.log10(thisline[paramname+'_err']) >= -5.:
                    value= '%.5f' % thisline[paramname]
                    if thisline.has_key(paramname+'_low'):
                        err= '$^{+%.5f}_{-%.5f}$' % (thisline[paramname+'_low'],thisline[paramname+'_high'])
                    else:
                        err= '$\pm$%.5f' % thisline[paramname+'_err']
                if ll and valerr:
                    #Both error and lower limit
                    lower_lim= '%.1f' % thisline[paramname+'_ll']
                    printline+= '& $>$'+lower_lim+' ('+value+'&'+err+')'
                elif ll:
                    #Just lower limit
                    lower_lim= '%.0f' % thisline[paramname+'_ll']
                    printline+= '& $>$'+lower_lim+' & '
                else:
                    #Print value+err
                    printline+= '& '+value+'&'+err
            printline+= '\\\\'
            #Write the line
            outfile.write(printline+'\n')
        outfile.write('\\\\\n')
    outfile.write(cmdline+'\n')
    outfile.close()

def get_options():
    usage = "usage: %prog [options] <outputfilename>\n\noutputfilename= name of the file that the table will be saved to"
    parser = OptionParser(usage=usage)
    parser.add_option("--table",dest='table',default='richresults',
                      help="Table to prepare ('richresults', 'poorresults','afe')")
    return parser

if __name__ == '__main__':
    resultsTable(get_options())
