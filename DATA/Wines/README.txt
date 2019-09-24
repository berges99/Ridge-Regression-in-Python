Additional dataset for the Ridge Regression problem.

Dataset from the UCI data archive (red wine file)
link: https://archive.ics.uci.edu/ml/datasets/Wine+Quality

The file downloaded was a .csv file, so we needed to convert it into something compatible with python and ampl input data.
In order to do so, we wrote an R script. It outputs two .dat files that are ready to be used as python and ampl input.
(Note: we needed to add manually the parameter specifications in the ampl.dat file of the R script output) (the file inlcluded here is ready to work!)