# Stage 1: Flux

The purpose of this stage is to get the expected flux produced in the atmosphere for different particles at different energies and angles. It is therefore possible to reweight a dataset using the effective area to investigate uncertainties from using different flux models. The neutrino files come from several different measurements made at different positions on the earth and currently can only use azimuth-averaged data. 

## Main Module

The main module for this stage is `Flux.py`, which gets a flux service and then produces flux maps for all angles and energies. It implements a function

```get_flux_maps(flux_service, ebins, czbins)```

### Parameters

*  `flux_service`: A flux service that calculates flux as a function of energy and cos(zenith), which is queried for a flux for each of the primaries (`nue`, `nue_bar`, `numu`, `numu_bar`).
*  `ebins`: Edges of the energy bins in units of GeV, default is 40 edges (39 bins) from 1.0 to 80 GeV in logarithmic spacing. 
*  `czbins`: Edges of the cos(zenith) bins, default is 21 edges (20 bins) from -1. (upward) to 0. horizontal in linear spacing.

  
#### Output

This function returns 2D _maps_ of flux (`map`) in bins of energy (`ebins`) and cos(zenith) (`czbins`).
```
map := {'ebins': [...],
        'czbins' : [...],
        'map': [[...]]}
```
One map is returned for each flavour
```
{
  "nue" : map
  "nue_bar": : map   
  "numu": map
  "numu_bar": map
}
```

## Services

###HondaFluxService

```
class HondaFluxService():
    def __init__(self, flux_file=None, smooth=0.05, **params):
        logging.info("Loading atmospheric flux table %s" %flux_file)
```

This class loads a differential neutrino flux from Honda-styles flux tables in units of [GeV^-1 m^-2 s^-1 sr^-1] and creates a 2D spline interpolated function for each flavour. For now this Service only supports azimuth-averaged input files. The default flux is the `spl-solmax-aa.d` model from [Honda's web-page.](http://www.icrr.u-tokyo.ac.jp/~mhonda/). Due to the steepness of the spectrum, the spline interpolation is carried out in `log10(flux)` space.

* `get_flux(self, ebins, czbins, prim)`: Get the flux in units [m^-2 s^-1] for the given bin edges in energy(`ebins`) and cos(zenith)(`czbins`) and the primary(`prim`).

The flux _F_ in the bin _i_ is obtained from the differential flux by multiplying the flux at the center of each bin with the bin size in energy and cos(zenith)

![Flux](Flux.png)

where

![Flux bins](flux-bins.png)


## References

"Atmospheric neutrino flux at INO, South Pole and Pyhäsalmi" , Physics Letters B Volume 718, Issues 4–5, 29 January 2013, Pages 1375–1380   
"Calculation of atmospheric neutrino flux using the interaction model calibrated with atmospheric muon data", Phys. Rev. D 75, 043006 
