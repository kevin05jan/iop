% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/tmp.R
\name{num2gasvm}
\alias{num2gasvm}
\title{Feature selection using the wrapper method (GA + SVM)}
\usage{
num2gasvm(x, y, run = 10)
}
\arguments{
\item{x}{: data.frame with continuous variables}

\item{y}{= vector with binary class labels: {0, 1}}
}
\value{
num2gasvm object
}
\description{
Feature selection using the wrapper method (GA + SVM)
}
\examples{
  x = iris[1:100,1:4]
  y = ifelse(iris[1:100, 5] == "setosa", 1, 0)
  m = num2gasvm(x, y)
  newx = predict(m, x)

}
