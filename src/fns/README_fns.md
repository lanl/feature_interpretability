[//]: <> (THIS IS A MARKDOWN FILE, VIEW IN A MARKDOWN VIEWER OR CONVERT)
<a name="top"></a>

# Functions

This folder contains all the custom-made functions that support the work in the mian directory. This README will disucss the organization and purpose of code in the `fns` directory specifically. There is a seperate README for the whole suite [elsewhere](../README.md).

These tools were developed by **Skylar Callis**, Z#365331. They currently still work at LANL! Some contact info is below:  
>sjcallis@lanl.gov
>
>(505) 66 5-6457
 
---

---
## Folders & Functions Oh MY!

[Back to Top](#top)

The scripts in the fns directory are sorted based on their use case, with the most general scripts outside of any subdirectories. All of the functions in each script have documentation on their inputs, outputs, and use cases.

All scripts in the main module take as input the *package* (`-P`) used to create the neural networks and the *experiment* (`-E`) data that the models were trainined on.

The scipts in the outermost directory and the scripts in the `setup` directory can be used with any combination of package and experiment. The other subdirectories correspond to a specific selection of package or experiment.

There is an `__init__.py` file at every level in the fns directory to maintain smooth operations. Some of them contain automatic imports for submodules. Broadly, the modules import all submodules they can without creating import loops.

- **Outmost files**: scripts with the most general use cases
     + [**clear_test_cache.py**](./clear_test_cache): function that finds the test_cache directory and deletes all the files inside of it; runs by default everytime the fns module is imported
     + [**mat.py**](./mat.py): functions related to matrix operations, such as normalizing or resizing
     + [**plots.py**](./plots.py): functions to make `matplotlib.pyplot` plots
     + [**save.py**](./save.py): functions to save common outputs
     + [**misc.py**](./misc.py): functions that do not fit into any other categories

 - [**Setup**](./setup): scripts that are used to setup more complicated scripts in the outer couponmlactivation suite
     + [**copy_code.txt**](./setup/copy_code.txt): a text file with code to copy into the setup of a new couponmlactivation script; includes argparse argument creation, print statements, and check statements
     + [**args.py**](./setup/args.py): functions to set up the `argparse` arguments standard across the couponmlactivation suite
     + [**data_prints.py**](./setup/data_prints.py): functions to print out lists of options for the data-based input arguments
     + [**data_checks.py**](./setup/data_checks.py): functions to check that the data-based input arguments passed are valid

 - [**Coupon Data**](./coupondata): scripts to process data from coupon experiments
     + [**process.py**](./coupondata/process.py): functions to process data from coupon experiments
     + [**prints.py**](./coupondata/prints.py): functions that print out lists of options for the data-based input arguments specific to the coupon experiment

 - [**Nested Cylinder Data**](./nestedcylinder): scripts to process data from nested cylinder experiments
     + [**process.py**](./nestedcylinderdata/process.py): functions to process data from nested cylinder experiments
     + [**prints.py**](./nestedcylinderdata/prints.py): functions that print out lists of options for the data-based input arguments specific to the nested cylinder experiment 

 - [**TFCustom**](./tfcustom): scripts with custom functions specific to `tensorflow` neural networks
     + [**fts.py**](./tfcustom/fts.py): functions to extract features from a `tensorflow` neural network
     + [**prints.py**](./tfcustom/prints.py): functions to print out lists of options for the model-based input arguments
     + [**checks.py**](./tfcustom/checks.py): functions to check that the model-based input arguments passed are valid
     + [**calico_fns.py**](./tfcustom/calico_fns.py): functions to create a calico network and do prints/checks on the calcio network inputs; for more information see the [calico network README](../CALICO_README.md)
     + [**calico_seq.py**](./tfcustom/calico_seq.py): `keras` sequence object definition for the calico networks; for more information see the [calico network README](../CALICO_README.md)

 - [**PytorchCustom**](./tfcustom): scripts with custom functions specific to `pytorch` neural networks
     + [**fts.py**](./pytorchcustom/fts.py): functions to extract features from a `pytorch` neural network
     + [**prints.py**](./pytorchcustom/prints.py): functions to print out lists of options for the model-based input arguments
     + [**checks.py**](./pytorchcustom/checks.py): functions to check that the model-based input arguments passed are valid
     
 - **Depricated**: contains scripts that are no longer used but that Skylar can't get rid of due to unending fear

---

---
