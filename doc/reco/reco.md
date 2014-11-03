# Stage 4: Reconstruction

The purpose of this stage is to apply the detector resolution to 
the detected events by convertubg the events' _true_ energies and directions 
into the _reconstructed_ ones.

## Main module

The main module for this stage is `Reco.py`, which provides a function that takes a set of triggered event rate truth maps (output of `Aeff.py`) as input and produces a 
set of reconstructed templates of nue CC, numu CC, nutau CC, and NC 
events. Note that in this stage, the counts of neutrinos and corresponding  anti-neutrinos are summed.
```
def get_reco_maps(true_event_maps,reco_service=None,
                  e_reco_scale=None, cz_reco_scale=None, **kwargs)
```
### Parameters

* `event_rate_maps`: Event rate input dictionary with one map for each flavour and interaction type:

    ```
    {"nue": {'cc':{'czbins':[], 'ebins':[], 'map':[]},'nc':...},
     "numu": {...},
     "nutau": {...},
     "nue_bar": {...},
     "numu_bar": {...}, 
     "nutau_bar": {...} }
    ```
* `reco_service`: A _reconstruction service_ that provides resolution kernels for energy and cos(zenith) as a function of the true energy and cos(zenith). See below for a choice of services.
* `e_reco_scale`, `cz_reco_scale`: Scales the width of the energy  or cos(zenith) reco by a given factor. Currently only supported by the parametrized reconstruction service.

### Output

This stage returns maps of reconstructed event counts for each _flavour_ and _interaction type_:

```
{"nue_cc": {'czbins':[], 'ebins':[], 'map':[]},
 "numu_cc": {...},
 "nutau_cc": {...},
 "nuall_nc": {...}}
```

## Services

### RecoServiceBase
The base service class `RecoServiceBase` contains the functions that actually apply the reconstruction kernels to the data, as 
well as sanity checks of the kernels. This method defined here are:

* `get_reco_kernels`: This method is called to construct the reco kernels, 
 i.e. a 4D histogram of true vs. reconstructed energy and cos(zenith). The order of the axis is
     1. true energy
     2. true cos(zenith)
     3. reconstructed energy
     4. reconstructed cos(zenith) 

    This method is individually implemented in the derived classes, since the way the reco kernels are generated depends on the reconstruction service. 
* `check_kernels`: Test whether the reco kernels have the correct shape
 to match the given input maps.
* `normalize_kernels`: Ensure that all kernels are normalized.
* `get_reco_maps(true_event_maps, recalculate=False, **kwargs)`: Apply the 
 reconstruction kernels to the "true" event histograms and return the 
 "reconstructed" histograms. If `recalculate` is `True`, call `recalculate_kernels` 
 before and pass `kwargs` to it (might be needed when scanning a systematic 
 parameter related to reconstruction).
* `recalculate_kernels(**kwargs)`: Call `get_reco_kernels` again, passing `kwargs` 
 and do all necessary checks. If new kernels are corrupted, stick with the old ones.
* `store_kernels(filename)`: Store reconstruction kernels in `JSON` format 
 to re-use them later.

The different derived reconstruction services (i.e. different implementations 
of `get_reco_kernels`) are:

### MCRecoService

Argument(s):
* `simfile`: MC file in `HDF5` format containing data from all flavours for a 
 particular instumental geometry

Creates the 4D reco kernel (see above) from histogramming `simfile`.
Expects the file format to be:
```
{'nue': {
   'cc': {
     'true_energy': np.array,
     'true_coszen': np.array,
     'reco_energy': np.array,
     'reco_coszen': np.array
     },
   'nc': {...
     }
   },
 'nue_bar' {...},
 ...
}
```

### ParamRecoService

Argument(s):
* `paramfile`: `JSON` file containing the parametrizations as strings defining 
 python lambda functions. `numpy` can be used as `np`, also `scipy.norm`.
* `e_reco_scale`, `cz_reco_scale`: `double`, scales the width of the energy (zenith) 
 reco by a given factor (default: 1.0).

Assumes all resolutions can be described by double gaussians whose parameters depend 
on the true neutrino energy. The input file should have the format
```
{'nue': {
   'cc': {
     'energy': {
       'loc1': 'lambda E: some_func(E)',
       'loc2': '...',
       'width1': '...',
       'width2': '...',
       'fraction': '...'
     },
     'coszen': {...}
   }
   'nc': {...
   }
 },
 'nue_bar' {...},
 ...
}
```
such that the `'lambda E: ...'` definitions can be read via `eval()`. The double 
gaussians then take the form
```
   F(E_true) = (1-fraction)*exp((E_true-loc1)^2/width1^2)
                 + fraction*exp((E_true-loc2)^2/width2^2
```
with `fraction`, `loc1`, `loc2`, `width1`, and `width2` all being functions of 
the true energy as defined in the parametrization file.

### KernelFileRecoService

Argument(s):
* `kernelfile`: `JSON` file containing a previously calculated reconstruction 
 kernel as produced by the `store_kernels` method.

Loads the kernels from disk and uses them for reconstruction.
