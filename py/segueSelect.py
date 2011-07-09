import os, os.path
import sys
import copy
import numpy
from scipy import special
import pyfits
from galpy.util import bovy_plot
import matplotlib
_SEGUESELECTDIR=os.getenv('SEGUESELECTDIR')
_GDWARFFILE= os.path.join(_SEGUESELECTDIR,'gdwarf_raw_nodups.fit')
class segueSelect:
    """Class that contains selection function for SEGUE targets"""
    def __init__(self,sample='G',remove_dups=True,plates=None,
                 logg=True,ug=False,ri=False,sn=True,
                 ebv=False):
        """
        NAME:
           __init__
        PURPOSE:
           load the selection function for this sample
        INPUT:
           sample= sample to load ('G' or 'K', 'GK' loads all)
           plates= if set, only consider this plate or list of plates
           
           SPECTROSCOPIC SAMPLE SELECTION:
              logg= if False, don't cut on logg
              ug= if True, cut on u-g
              ri= if True, cut on r-i
              sn= if False, don't cut on SN
              ebv= if True, cut on E(B-V)
        OUTPUT:
           object
        HISTORY:
           2011-07-08 - Written - Bovy (NYU)
        """
        self.sample=sample.lower()
        #Load plates
        self.platestr= _load_fits(os.path.join(_SEGUESELECTDIR,'segueplates.fits'))
        if plates is None:
            self.plates= list(self.platestr.field('plate'))
        else:
            if isinstance(plates,str):
                self.plates= self.platestr.field('plate')
                if plates[0] == '>':
                    self.plates= self.plates[(self.plates > int(plates[1:len(plates)]))]
                elif plates[0] == '<':
                    self.plates= self.plates[(self.plates < int(plates[1:len(plates)]))]
                else:
                    print "'plates=' format not understood, check documentation"
                    return
                self.plates= list(self.plates)
            elif not isinstance(plates,(list,numpy.ndarray)):
                self.plates= [plates]
            elif isinstance(plates,numpy.ndarray):
                self.plates= list(plates)
            else:
                self.plates= plates
        #Remove 2820 for now BOVY DEAL WITH PLATE 2820, 2560, 2799, 2550
        if 2820 in self.plates:
            self.plates.remove(2820)
        if 2560 in self.plates:
            self.plates.remove(2560)
        if 2799 in self.plates:
            self.plates.remove(2799)           
        if 2550 in self.plates:
            self.plates.remove(2550)           
        if remove_dups:
            #Remove duplicate plates
            self.plates= numpy.array(sorted(list(set(self.plates))))
            #Match platestr to plates again
            allIndx= numpy.arange(len(self.platestr),dtype='int')
            reIndx= numpy.zeros(len(self.plates),dtype='int')-1
            for ii in range(len(self.plates)):
                indx= (self.platestr.field('plate') == self.plates[ii])
                reIndx[ii]= (allIndx[indx][0])
            self.platestr= self.platestr[reIndx]
        #load the photometry for the SEGUE plates
        self.platephot= {}
        for ii in range(len(self.plates)):
            plate= self.plates[ii]
            sys.stdout.write('\r'+"Loading photometry for plate %i" % plate)
            sys.stdout.flush()
            platefile= os.path.join(_SEGUESELECTDIR,'segueplates',
                                    '%i.fit' % plate)
            self.platephot[str(plate)]= _load_fits(platefile)
            #Split into bright and faint
            if 'faint' in self.platestr[ii].field('programname'):
                indx= (self.platephot[str(plate)].field('r') > 17.8)
                self.platephot[str(plate)]= self.platephot[str(plate)][indx]
            else:
                indx= (self.platephot[str(plate)].field('r') < 17.8)
                self.platephot[str(plate)]= self.platephot[str(plate)][indx]
        sys.stdout.write('\r'+"                                              \r")
        sys.stdout.flush()
        #Flesh out samples
        for plate in self.plates:
            if self.sample == 'g':
                indx= ((self.platephot[str(plate)].field('g')\
                           -self.platephot[str(plate)].field('r')) < 0.55)\
                           *((self.platephot[str(plate)].field('g')\
                           -self.platephot[str(plate)].field('r')) > 0.48)\
                           *(self.platephot[str(plate)].field('r') < 20.2)\
                           *(self.platephot[str(plate)].field('r') > 14.5)
            elif self.sample == 'k':
                indx= ((self.platephot[str(plate)].field('g')\
                            -self.platephot[str(plate)].field('r')) > 0.55)\
                           *((self.platephot[str(plate)].field('g')\
                           -self.platephot[str(plate)].field('r')) < 0.75)\
                           *(self.platephot[str(plate)].field('r') < 19.)\
                           *(self.platephot[str(plate)].field('r') > 14.5)
            if self.sample == 'gk':
                indx= ((self.platephot[str(plate)].field('g')\
                           -self.platephot[str(plate)].field('r')) < 0.75)\
                           *((self.platephot[str(plate)].field('g')\
                           -self.platephot[str(plate)].field('r')) > 0.48)\
                           *(self.platephot[str(plate)].field('r') < 20.2)\
                           *(self.platephot[str(plate)].field('r') > 14.5)
            self.platephot[str(plate)]= self.platephot[str(plate)][indx]
        #Now load the spectroscopic data
        if sample.lower() == 'g':
            self.spec= read_gdwarfs(logg=logg,ug=ug,ri=ri,sn=ri,
                                        ebv=ebv)
        elif sample.lower() == 'k':
            specfile= os.path.join(_SEGUESELECTDIR,'kdwarf.dat') 
        self.platespec= {}
        for plate in self.plates:
            #Find spectra for each plate
            indx= (self.spec.field('plate') == plate)
            self.platespec[str(plate)]= self.spec[indx]
        return None

    def plot(self,x='gr',y='r',plate='all',spec=False,scatterplot=True,
             bins=None,specbins=None,type=None):
        if isinstance(plate,str) and plate == 'all':
            plate= self.plates
        elif isinstance(plate,(list,numpy.ndarray)):
            plate=plate
        else:
            plate= [plate]
        xs, ys= [], []
        specxs, specys= [], []
        for ii in range(len(plate)):
            p=plate[ii]
            thisplatephot= self.platephot[str(p)]
            thisplatespec= self.platespec[str(p)]
            if not type is None:
                pindx= (self.platestr.field('plate') == p)
            if not type is None and type.lower() == 'bright':
                if 'faint' in self.platestr[pindx].field('programname')[0]:
                    continue
            elif not type is None and type.lower() == 'faint':
                if not 'faint' in self.platestr[pindx].field('programname')[0]:
                    continue
            if len(x) > 1: #Color
                xs.extend(thisplatephot.field(x[0])\
                              -thisplatephot.field(x[1])) #dereddened
                specxs.extend(thisplatespec.field('dered_'+x[0])\
                                  -thisplatespec.field('dered_'+x[1]))
            else:
                xs.extend(thisplatephot.field(x[0]))
                specxs.extend(thisplatespec.field('dered_'+x[0]))
            if len(y) > 1: #Color
                ys.extend(thisplatephot.field(y[0])\
                              -thisplatephot.field(y[1])) #dereddened
                specys.extend(thisplatespec.field('dered_'+y[0])\
                                  -thisplatespec.field('dered_'+y[1]))
            else:
                ys.extend(thisplatephot.field(y[0]))
                specys.extend(thisplatespec.field('dered_'+y[0]))
        xs= numpy.array(xs)
        xs= numpy.reshape(xs,numpy.prod(xs.shape))
        ys= numpy.array(ys)
        ys= numpy.reshape(ys,numpy.prod(ys.shape))
        specxs= numpy.array(specxs)
        specxs= numpy.reshape(specxs,numpy.prod(specxs.shape))
        specys= numpy.array(specys)
        specys= numpy.reshape(specys,numpy.prod(specys.shape))
        if len(x) > 1:
            xlabel= '('+x[0]+'-'+x[1]+')_0'
        else:
            xlabel= x[0]+'_0'
        xlabel= r'$'+xlabel+r'$'
        if len(y) > 1:
            ylabel= '('+y[0]+'-'+y[1]+')_0'
        else:
            ylabel= y[0]+'_0'
        ylabel= r'$'+ylabel+r'$'
        if len(xs) > 1: #color
            xrange= [numpy.amin(xs)-0.1,numpy.amax(xs)+0.1]
        else:
            xrange= [numpy.amin(xs)-0.7,numpy.amax(xs)+0.7]
        if len(ys) > 1: #color
            yrange= [numpy.amin(ys)-0.1,numpy.amax(ys)+0.1]
        else:
            yrange= [numpy.amin(ys)-0.7,numpy.amax(ys)+0.7]
        if bins is None:
            bins= int(numpy.ceil(0.3*numpy.sqrt(len(xs))))
        if specbins is None: specbins= bins
        if scatterplot:
            if len(xs) > 100000: symb= 'w,'
            else: symb= 'k,'
            if spec:
                #First plot spectroscopic sample
                cdict = {'red': ((.0, 1.0, 1.0),
                                 (1.0, 1.0, 1.0)),
                         'green': ((.0, 1.0, 1.0),
                                   (1.0, 1.0, 1.0)),
                         'blue': ((.0, 1.0, 1.0),
                                  (1.0, 1.0, 1.0))}
                allwhite = matplotlib.colors.LinearSegmentedColormap('allwhite',cdict,256)
                speclevels= list(special.erf(0.5*numpy.arange(1,4)))
                speclevels.append(1.01)#HACK TO REMOVE OUTLIERS
                bovy_plot.scatterplot(specxs,specys,symb,onedhists=True,
                                      levels=speclevels,
                                      onedhistec='r',
                                      cntrcolors='r',
                                      cmap=allwhite,
                                      xlabel=xlabel,ylabel=ylabel,
                                      xrange=xrange,yrange=yrange,
                                      bins=specbins)
                
            bovy_plot.scatterplot(xs,ys,symb,onedhists=True,
                                  xlabel=xlabel,ylabel=ylabel,
                                  xrange=xrange,yrange=yrange,bins=bins,
                                  overplot=spec)
        else:
            bovy_plot.bovy_plot(xs,ys,'k,',onedhists=True,
                                xlabel=xlabel,ylabel=ylabel,
                                xrange=xrange,yrange=yrange)
        return None                

def _load_fits(file,ext=1):
    hdulist= pyfits.open(file)
    out= hdulist[1].data
    hdulist.close()
    return out

def read_gdwarfs(file=_GDWARFFILE,logg=True,ug=False,ri=False,sn=True,
                 ebv=False):
    """
    NAME:
       read_gdwarfs
    PURPOSE:
       read the spectroscopic G dwarf sample
    INPUT:
       logg= if False, don't cut on logg
       ug= if True, cut on u-g
       ri= if True, cut on r-i
       sn= if False, don't cut on SN
       ebv= if True, cut on E(B-V)
    OUTPUT:
       cut data, still pyfits format
    HISTORY:
       2011-07-08 - Written - Bovy (NYU)
    """
    raw= _load_fits(file)
    #First cut on r
    indx= (raw.field('dered_r') < 20.2)*(raw.field('dered_r') > 14.5)
    raw= raw[indx]
    #Then cut on g-r
    indx= ((raw.field('dered_g')-raw.field('dered_r')) < 0.55)\
        *((raw.field('dered_g')-raw.field('dered_r')) > .48)
    raw= raw[indx]
    #Cut on velocity errs
    indx= (raw.field('pmra_err') > 0.)*(raw.field('pmdec_err') > 0.)\
        *(raw.field('vr_err') > 0.)
    raw= raw[indx]
    #Cut on logg?
    if logg:
        indx= (raw.field('logga') > 3.75)
        raw= raw[indx]
    if ug:
        indx= ((raw.field('dered_u')-raw.field('dered_g')) < 2.)\
            *((raw.field('dered_u')-raw.field('dered_g')) > .6)
        raw= raw[indx]
    if ri:
        indx= ((raw.field('dered_r')-raw.field('dered_i')) < .4)\
            *((raw.field('dered_r')-raw.field('dered_i')) > -.1)
        raw= raw[indx]
    if sn:
        indx= (raw.field('sna') > 15.)
        raw= raw[indx]
    if ebv:
        indx= (raw.field('ebv') < .3)
        raw= raw[indx]
    #BOVY: distances
    return raw
    
