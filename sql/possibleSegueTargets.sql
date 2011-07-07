SELECT p.ra, p.dec, p.dered_u as u,p.dered_g as g,p.dered_r as r,
p.dered_i as i,p.dered_z as z,
into mydb.DBNAME from fGetNearbyObjEq(PLATERA,PLATEDEC,89.4) n, PhotoPrimary p
WHERE n.objID=p.objID
p.dered_r between 14.5 and 20.e
and (p.dered_g-p.dered_r between 0.48 and 0.75
and (p.flags & (dbo.fPhotoFlags('BRIGHT')+dbo.fPhotoFlags('EDGE')+dbo.fPhotoFlags('BLENDED')+dbo.fPhotoFlags('SATURATED')+dbo.fPhotoFlags('NOTCHECKED')+dbo.fPhotoFlags('NODEBLEND')+dbo.fPhotoFlags('INTERP_CENTER')+dbo.fPhotoFlags('DEBLEND_NOPEAK')+dbo.fPhotoFlags('PEAKCENTER'))) = 0