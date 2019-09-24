# Set current working directory
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

# Read wines file (.csv format)
raw.data <- read.csv('winequality-red.csv', sep = ';', header = TRUE)
# Dimensions
n <- nrow(raw.data); p <- ncol(raw.data)
# Check variable types (must be all numeric for ridge regression and support vector machines)
str(raw.data)


# Write file in PYTHON format (matrix with A and y in the last column)
write.table(raw.data, file = 'wines_python.dat', row.names = FALSE, col.names = FALSE)


# Write file in AMPL format
# Response value: y
y <- raw.data[,p]
y <- data.frame(seq(1,n),y)
# Predictors: A
A <- raw.data[,1:p-1]
A <- data.frame(seq(1,n),A)
write.table(y, file = 'wines_ampl.dat', row.names = FALSE, col.names = FALSE)
write.table(A, file = 'wines_ampl.dat', row.names = FALSE, col.names = FALSE, append = TRUE)

