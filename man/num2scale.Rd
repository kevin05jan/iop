% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/tmp.R
\name{num2scale}
\alias{num2scale}
\title{Feature scaling of numeric variables}
\usage{
num2scale(x)
}
\arguments{
\item{x}{: data.frame with independent variables of type continous}
}
\value{
num2norm object
}
\description{
Feature scaling of numeric variables
}
\examples{
  m = num2norm(iris[,1:2])
  predict(m, iris[,1:4])

}
