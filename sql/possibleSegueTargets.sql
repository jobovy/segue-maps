SELECT p.ra, p.dec, p.dered_u as u,p.dered_g as g,p.dered_r as r,
p.dered_i as i,p.dered_z as z
into mydb.DBNAME from dbo.fGetNearbyObjEq(PLATERA,PLATEDEC,89.4) n, Star p
WHERE n.objID=p.objID
and ((p.dered_r between 14.5 and 20.2) or (p.dered_g between 14.5 and 20.2))
and (p.dered_g-p.dered_r) between 0.2 and 0.75
and (p.flags & (dbo.fPhotoFlags('SATURATED')+dbo.fPhotoFlags('EDGE')+dbo.fPhotoFlags('PSF_FLUX_INTERP')+dbo.fPhotoFlags('BAD_COUNTS_ERROR'))) = 0
and (((p.flags & dbo.fPhotoFlags('INTERP_CENTER')) = 0) or
((p.flags & dbo.fPhotoFlags('INTERP_CENTER')) > 0 
and (p.flags & dbo.fPhotoFlags('COSMIC_RAY')) = 0))