# ANAIS112_analysis
An independent search for annual modulation and its significance in ANAIS-112 data

We first do an annual modulation fit by using the same background parameters as those of the ANAIS collaboration. 
The code for this analysis is const_bg_analysis.py
The data files for this (fig2_16.dat and fig2_26.dat) contain data after subtracting the exponential background from it
(provided to us by the ANAIS collaboration)


We then allow the background values to float and do  a  combined  fit  to  both  the  signal  and  background.
For doing a fit to the signal hypothesis, we also vary the phase and period in addition to the amplitude. 
The code for this analysis is exp_bg_analysis.py
The data files for this are data16.dat and data26.dat (provided to us by the ANAIS collaboration)
