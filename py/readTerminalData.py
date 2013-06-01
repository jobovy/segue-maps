import numpy
def readClemens():
    data= numpy.loadtxt('clemens1985_table2.dat',delimiter='|',
                        comments='#')
    glon= data[:,0]
    vterm= data[:,1]
    #Correct for new LSR motion
    return (glon,vterm)
