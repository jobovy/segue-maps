OUTDIR=../figs
EXT=png

#Metal rich
$(OUTDIR)/HWR_rich_g_sR.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py HWR_rich_g.sav -o $@ --expd1 --expd2 -n 10000 \
	--xmin=15. --xmax=25. --ymin=0. --ymax=30. \
	--xlabel='\sigma_z(z=500\ \mathrm{pc})\ [\mathrm{km\ s}^{-1}]' \
	--ylabel='R_\sigma\ [\mathrm{kpc}]'

$(OUTDIR)/HWR_rich_g_spbad.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py HWR_rich_g.sav -o $@ --expd1 -n 10000 \
	--xmin=15. --xmax=25. --ymin=0. --ymax=.05 --d2=0 \
	--xlabel='\sigma_z(z=500\ \mathrm{pc})\ [\mathrm{km\ s}^{-1}]' \
	--ylabel='P_{\mathrm{bad}}'

$(OUTDIR)/HWR_rich_g_func.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py HWR_rich_g.sav -o $@ -n 10000 \
	--plotfunc --plotnsamples=1000 \
	--xlabel='|z|\ [\mathrm{kpc}]' \
	--ylabel='\sigma_z\ [\mathrm{km\ s}^{-1}]' \
	--ymin=0. --ymax=50.

#Metal poor
$(OUTDIR)/HWR_poor_g_sR.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py HWR_poor_g.sav -o $@ --expd1 --expd2 -n 10000 \
	--xmin=35. --xmax=45. --ymin=0. --ymax=30. \
	--xlabel='\sigma_z(z=500\ \mathrm{pc})\ [\mathrm{km\ s}^{-1}]' \
	--ylabel='R_\sigma\ [\mathrm{kpc}]' --metal=poor

$(OUTDIR)/HWR_poor_g_spbad.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py HWR_poor_g.sav -o $@ --expd1 -n 10000 \
	--xmin=35. --xmax=45. --ymin=0. --ymax=.05 --d2=0 \
	--xlabel='\sigma_z(z=500\ \mathrm{pc})\ [\mathrm{km\ s}^{-1}]' \
	--ylabel='P_{\mathrm{bad}}' --metal=poor

$(OUTDIR)/HWR_poor_g_func.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py HWR_poor_g.sav -o $@ -n 10000 \
	--plotfunc --plotnsamples=1000 \
	--xlabel='|z|\ [\mathrm{kpc}]' \
	--ylabel='\sigma_z\ [\mathrm{km\ s}^{-1}]' \
	--ymin=0. --ymax=50. --metal=poor


#Metal rich, isothermal model
$(OUTDIR)/isotherm_rich_g_sR.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py isotherm_rich_g.sav -o $@ --expd1 --expd2 \
	-n 10000 --d2=2 --model=isotherm \
	--xmin=15. --xmax=25. --ymin=0. --ymax=30. \
	--xlabel='\sigma_z(z=500\ \mathrm{pc})\ [\mathrm{km\ s}^{-1}]' \
	--ylabel='R_\sigma\ [\mathrm{kpc}]'

$(OUTDIR)/isotherm_rich_g_spbad.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py isotherm_rich_g.sav -o $@ --expd1 -n 10000 \
	--xmin=15. --xmax=25. --ymin=0. --ymax=.05 --d2=0 \
	--xlabel='\sigma_z(z=500\ \mathrm{pc})\ [\mathrm{km\ s}^{-1}]' \
	--ylabel='P_{\mathrm{bad}}' --model=isotherm

$(OUTDIR)/isotherm_rich_g_func.$(EXT): fitSigz.py Makefile
	$(PYTHON) fitSigz.py isotherm_rich_g.sav -o $@ -n 10000 \
	--plotfunc --plotnsamples=1000 --model=isotherm \
	--xlabel='|z|\ [\mathrm{kpc}]' \
	--ylabel='\sigma_z\ [\mathrm{km\ s}^{-1}]' \
	--ymin=0. --ymax=50.
