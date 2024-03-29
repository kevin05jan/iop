% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/tmp.R
\name{predict.num2bcp}
\alias{predict.num2bcp}
\title{Box/Cox Power (BCP) Transform}
\usage{
\method{predict}{num2bcp}(model, newx, ...)
}
\arguments{
\item{model}{: num2bcp object}

\item{newx}{: data.frame}
}
\value{
data.frame;  dim is preserved
}
\description{
Box/Cox Power (BCP) Transform
}
\examples{
  m = num2bcp(iris[,1:2])
  predict(m, iris[,1:4])

}
