# CO2 INJECTION OHAAKI MODEL

This project calibrates a model of the Ohaaki geothermal reservoir from txt files "cs_q.txt", "cs_p.txt", "cs_c.txt" and "cs_cc.txt" 
and predicts the result of scenarios 30 years forward.

Produces plots of data:
Injection_VS_concentration.png
MassRate_VS_Pressure.png
Pressure_Benchmarking.png
Pressure_Convergence.png
Concentration_Benchmarking.png
Concentration_Convergence.png
Pressure_Calibration.png
Pressure_Misfit.png
Concentration_Calibration.png
Concentration_Misfit.png
Pressure_Predictions.png
Concentration_Predictions.png
Pressure_Predictions_Uncertainty.png
Concentration_Predictions_Uncertainty.png

## Getting Started

Install the following files from the depository in the same directory:
cs_q.txt
cs_p.txt
cs_c.txt
cs_cc.txt
main.py
functions.py
curve_fit_ode_functions.py
unit_tests.py

Open directory in Visual Studio Code and select '233env' interpreter.

Note: the 'literature' folder contains the project brief and relevant literature and the 'unused'
folder contains code fragments that were used to build our model, but are not used in the running of
the final model. 

### Prerequisites

Requires Visual Studio Code installed and 233env interpreter

## Running the script

Open Visual Studio Code and run main.py under the '233env' interpreter. All plots will be saved automatically.
Notes: 
It will take approximately 3-4 minutes to run the full script. 

Currently for Uncertainty, 1000 samples are tested. If you need to run more samples, change 'N=1000' in line 450 and 518
to 'N=x' with x is the number of samples. 

The parameters are not printed on the screen. Uncomment lines 215 and 241 to see the parameters. 
For 215, the parameters will be in order of [a,b,c] and [M0,d] for 241

To print the [5%, 95%] interval, uncomment lines 469-470 for Pressure and 538-539 for Concentration
2 lists will be printed. The first list is upperbound and is 2nd list is lowerbound.
The predictions are in order of [quadruple, triple, double, unchanged, 0]


### Unit tests

Runs code against hand solved answer to ensure functionality
Open Anaconda Prompt or any other cmd and run pytest -v to run the unit test.

## Versioning

We use [Bitbucket](https://bitbucket.org/acli141/engsci-263-group-16/src/master/) for versioning.

## Authors

* **Adam Clifford** - acli141@aucklanduni.ac.nz
* **Hung Ngu** - hugn556@aucklanduni.ac.nz
* **Shalin Shah** - shsa437@aucklanduni.ac.nz
* **George Timings** - gtim999@aucklanduni.ac.nz
