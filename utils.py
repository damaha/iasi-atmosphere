import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import scipy.sparse.linalg as ssl

from mpl_toolkits.basemap import Basemap, cm

def pccs(x, n_comps = "all"):
    if n_comps == "all":
        n_comps = x.shape[0]-1
    shp = x.shape
    S = np.cov(x.reshape((shp[0],np.prod(shp[1:]))))
    val, vec = ssl.eigs(S, k = n_comps)
    sort = np.argsort(-val)
    val = np.real(val[sort])
    vec = np.real(vec[:,sort])    
    return(vec,val)

def mnfs(x, n_comps = "all", method = "acorr"):
    if n_comps == "all":
        n_comps = x.shape[0]-1
        
    x = x - x.mean(axis = (1,2))[0]
    if method == "acorr":
        x_delta = x[:,:-1,:-1] - (x[:,1:,:-1] + x[:,:-1,1:])/2
        x = x[:,:-1,:-1]
    elif method == "quadres":
        filt = np.array([[-2,1,-2],[1,4,1],[-2,1,-2]])*1/9
        x_delta = np.zeros((x.shape[0],x.shape[1]-(filt.shape[0]-1),x.shape[2]-(filt.shape[0]-1)))
        for i in range(x.shape[0]):
            x_delta[i,:,:] = convolve2d(x[i,:,:], filt, mode = 'valid')
        x = x[:,int((filt.shape[0]-1)/2):-int((filt.shape[0]-1)/2),
                 int((filt.shape[1]-1)/2):-int((filt.shape[1]-1)/2)]
    shp = x.shape
    S = np.cov(x.reshape((shp[0],np.prod(shp[1:]))))
    S_d = np.cov(x_delta.reshape((shp[0],np.prod(shp[1:]))))
    
    val, vec = ssl.eigs(S, k = n_comps, M = S_d)
    sort = np.argsort(-val)
    val = np.real(val[sort])
    vec = np.real(vec[:,sort])
    return(vec,val)

def geoplot(lons, lats, data, proj='Sphere', plotting='continous', 
            title='Geoplot', lab='', intv=(None,None), viewp = (0,0), 
            great_circle=False):
    fig = plt.figure(figsize=(10,10))
    tricolor=False
    
    if proj=='Sphere':
        m = Basemap(projection='ortho',lat_0=viewp[1],lon_0=viewp[0],resolution='l')
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
    
    if proj=='extSphere':
        m = Basemap(projection='eck4',lon_0=viewp[0], lat_0=viewp[1],resolution='l')
        m.drawcoastlines()
        m.drawparallels(np.arange(-90.,99.,30.))
        m.drawmeridians(np.arange(-180.,180.,60.))
        m.drawmapboundary(fill_color='0.3')
        tricolor=True
        lons, lats, data = (lons.flatten(), lats.flatten(), data.flatten())
        
    if proj=='square':
        offs = 37.
#        m = Basemap(llcrnrlon=-145.5,llcrnrlat=1.,urcrnrlon=-2.566,urcrnrlat=46.352,
#                    rsphere=(6378137.00,6356752.3142),
#                    resolution='l',area_thresh=1000.,projection='lcc',
#                    lat_1=50.,lon_0=-107.)
        m = Basemap(width=1200000,height=800000, resolution='l',
                    projection='stere', lat_ts=0,lat_0=viewp[1],lon_0=viewp[0])
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
        
    if great_circle:
        # great_circle =  (start lon, start lat, end lon, end lat)
        m.drawgreatcircle(great_circle[0],great_circle[1],great_circle[2],
                          great_circle[3],linewidth=2,color='b')
        
    x, y = m(lons, lats) # compute map proj coordinates.
    
    if plotting=='continous':
        cs = m.pcolor(x,y,data,cmap=plt.cm.jet, snap=True, tri=tricolor)
        cbar = m.colorbar(cs,"bottom", size="5%", pad="2%")
        
    ### try tricolor!!! ###
    if plotting=='contour':
        cs = m.contourf(x,y,data,cmap='jet')
        cbar = m.colorbar(cs,location='bottom',pad="5%")
    
    cbar.set_label(lab+' [K]', color=(0.33,0.33,0.33))
    cs.set_clim(vmin=intv[0],vmax=intv[1])
    plt.title(title, y=1.08, color=(0.33,0.33,0.33), size=ts+2)