import os, os.path
import sys
import copy
import numpy
import pyfits
from galpy.util import bovy_plot
_SEGUESELECTDIR=os.getenv('SEGUESELECTDIR')
class segueSelect:
    """Class that contains selection function for SEGUE targets"""
    def __init__(self,sample='G',remove_dups=True,plates=None):
        """
        NAME:
           __init__
        PURPOSE:
           load the selection function for this sample
        INPUT:
           sample= sample to load ('G' or 'K')
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
            if not isinstance(plates,(list,numpy.ndarray)):
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
            self.plates= numpy.array(list(set(self.plates)))
            #Match platestr to plates again
            allIndx= numpy.arange(len(self.platestr),dtype='int')
            reIndx= numpy.zeros(len(self.plates),dtype='int')-1
            for ii in range(len(self.plates)):
                indx= (self.platestr.field('plate') == self.plates[ii])
                reIndx[ii]= (allIndx[indx][0])
            self.platestr= self.platestr[reIndx]
        #load the photometry for the SEGUE plates
        self.platephot= {}
        for plate in sorted(self.plates):
            sys.stdout.write('\r'+"Loading photometry for plate %i" % plate)
            sys.stdout.flush()
            platefile= os.path.join(_SEGUESELECTDIR,'segueplates',
                                    '%i.fit' % plate)
            self.platephot[str(plate)]= _load_fits(platefile)
        sys.stdout.write('\r'+"                                              \r")
        sys.stdout.flush()
        #Now load the spectroscopic data
        if sample.lower() == 'g':
            specfile= os.path.join(_SEGUESELECTDIR,'gdwarf_raw.dat')
        elif sample.lower() == 'k':
            specfile= os.path.join(_SEGUESELECTDIR,'kdwarf.dat')
        numpy.loadtxt(specfile)
        if sample.lower() == 'g':
            self.platespec= {}
            for plate in self.plates:
                pass
        return None

    def plot(self,x='gr',y='r',plate='all',spec=False):
        if isinstance(plate,str) and plate == 'all':
            plate= self.plates
        elif isinstance(plate,(list,numpy.ndarray)):
            plate=plate
        else:
            plate= [plate]
        xs, ys= [], []
        for p in plate:
            thisplatephot= self.platephot[str(p)]
            if len(x) > 1: #Color
                xs.extend(thisplatephot.field(x[0])\
                              -thisplatephot.field(x[1])) #dereddened
            else:
                xs.extend(thisplatephot.field(x[0]))
            if len(y) > 1: #Color
                ys.extend(thisplatephot.field(y[0])\
                              -thisplatephot.field(y[1])) #dereddened
            else:
                ys.extend(thisplatephot.field(y[0]))
        xs= numpy.array(xs)
        xs= numpy.reshape(xs,numpy.prod(xs.shape))
        ys= numpy.array(ys)
        ys= numpy.reshape(ys,numpy.prod(ys.shape))
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
            xrange= [numpy.amin(xs)-0.5,numpy.amax(xs)+0.5]
        if len(ys) > 1: #color
            yrange= [numpy.amin(ys)-0.1,numpy.amax(ys)+0.1]
        else:
            yrange= [numpy.amin(ys)-0.5,numpy.amax(ys)+0.5]
        bovy_plot.scatterplot(xs,ys,'k,',onedhists=True,
                              xlabel=xlabel,ylabel=ylabel,
                              xrange=xrange,yrange=yrange)
        return None
                

def _load_fits(file,ext=1):
    hdulist= pyfits.open(file)
    out= hdulist[1].data
    hdulist.close()
    return out
                        
