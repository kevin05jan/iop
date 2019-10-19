




#' Anomaly Detection:  isofor: iForest
#'
#' @examples
#'   m = data2isofor(iris[,1:4])
#'   p = predict(p, iris)
#' 
#' @export
#'
data2isof <- function(x, ...){
    isofor::iForest(X=x, 10, 32, ...)
}

#' Anomaly Detection:   e1071: svm with “one-classification” type
#'
#' @examples
#'   m = data2svm1(Species ~ ., iris)
#'   predict(m, iris)
#' 
#' @export
#'
data2svm1 <- function(f, x, ...){
    m = e1071::svm(x=train, type="one-classification", kernel="linear", nu=.1, ...)
}

## Clustering:  amap::hclust
## Clustering:  neighbr::neighbr
## Clustering:  stats::kmeans


## K Nearest Neighbors:  neighbr::knn


## Linear Models:  glmnet::cv.glmnet with gaussian and poisson family types
## Linear Models:  nnet::multinom
## Linear Models:  stats::glmn without interaction terms


#' Naive Bayes:   e1071: naiveBayes
#'
#' @examples
#'   iris[["Species"]] = as.factor(iris[["Species"]])
#'   m = data2nb(Species ~ ., iris)
#'   predict(m, newdata=iris, type="class")
#' 
#' @export
#' 
data2nb <- function(f, x, ...){
    e1071::naiveBayes(f, x, ...)
}


#' Neural Networks:  nnet::nnet
#'
#' @examples
#'   m = data2nnet(Species ~ ., iris)
#'   predict(m, iris)
#' 
#' @export
#'
data2nnet <- function(f, x, ...){
    nnet::nnet(f, data=x,size=10)
}



#' Support Vector Machines:  e1071::svm
#'
#' @param
#'   f : formula
#'
#' @param
#'   x : data.frame
#'
#' @examples
#'   X = 1:2
#'   Y = 5
#'   iris[["Species"]] = as.factor(iris[["Species"]])
#'   svm = e1071::svm(Species ~ ., data=iris[,c(Y, X)])
#'   p = predict(svm, newdata=iris[,X])##--important
#'   table(p, a=iris[["Species"]])
#' 
#' @export
#' 
data2svm <- function(f, x){
    m = e1071::svm(f, data=x)
}


#' Support Vector Machines: kernlab::ksvm with rbfdot, polydot, ..., kernels
#'
#' @param
#'   f : formula
#'
#' @param
#'   x : data.frame
#'
#' @examples
#'   ksvm = kernlab::ksvm(Species ~ ., data=iris)
#'   p = predict(ksvm, newx=iris)
#'   table(p, a=iris[["Species"]])
#' 
#' @export
#' 
data2kernlab <- function(f, x){
    m = kernlab::ksvm(formula, data=train, kernel="rbfdot")
}


## Time Series
##   forecast: Arima


## Tree-based Models and Ensembles:  ada::ada
## Tree-based Models and Ensembles:  gbm::gbm with bernoulli, poisson, multinomial” distribution types


#' Tree-based models and ensembles:  randomForest::randomForest
#'
#' @param
#'   f : formula
#'
#' @param
#'   x : data.frame
#' 
#' @examples
#'   iris[["Species"]] = as.factor(iris[["Species"]])
#'   dt = randomForest::randomForest(Species ~ ., data=iris)
#'   p = predict(dt, newdata=iris)
#'   table(p, a=iris[["Species"]])
#' 
#' @export
#'
data2rf <- function(f, x){
    m = randomForest::randomForest(f, x, ntree=10)
}


#' Tree-based models and ensembles:  rpart::rpart
#'
#' methods to deal with imbalanced data
#' (a) prior:  proprotion of 1/0 labels:
#'   rpart(..., params=list(prior=c(.5, .5))
#' (b) cost/loss: cost sensitive training/loss learning:  employs cost in misclassification learning
#'
#'                   TP  FP
#'                   FN  TN
#' 
#' 
#'   loss = matrix(c( 0,  1,
#'                   20,  0), ncol=2)
#'   rpart(..., params=list(loss=loss))
#'  (c) case weight: increase case weight for samples of minority class: applies to observation
#' 
#' 
#'
#' 
#' @param
#'   f : formula
#'
#' @param
#'   x : data.frame
#' 
#' @examples
#'   dt = rpart::rpart(f, data=x)
#'   p = predict(dt, newdata=iris, type="class")
#'   table(p, a=iris[["Species"]])
#' 
#' @export
#'
data2dt <- function(f, x, ...){
    rpart::rpart(f, data=x, ...)
}


#' Tree-based models and ensembles:  xgboost::xgb.Booster  with multi:softprob, multi:softmax, and binary:logistic objectives
#'
#' @examples
#'   iris[["Species"]] = as.character(iris[["Species"]])
#'   xgb = data2xgb(Species ~ ., iris, iris[["Species"]]
#'   p = predict(xgb, iris)
#'   table(p, a=iris[["Species"]]
#' 
#' @export
#' 
data2xbg <- function(f, x, y){
    label = y
    matrix = Matrix::sparse.model.matrix(f, data=x)
    xgboost::xgboost(data = matrix, label = label, max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
}


##   arules: rules and itemsets
##   survival: coxph



## [1]
## Data cleansing
##   Removing or correcting records with corrputed
##   or invalid values from raw data, as well as
##   removing records that are missing a large
##   number of columns


## [2]
## Instances selection and partioning
##   Selecting data points from the input dataset
##   to create training, evaluation (validation),
##   and test sets. This process includes techniques
##   for repeatable random sampling, minority
##   classes oversampling, and stratified partitioning


#' Create subsets for learning, evaluating or testing
#' 
#' @param
#'   y :character vector; a label is an element of
#'   a set of length two, e.g. {0, 1}
#'
#' @return
#'   list : with
#'     $id_train := vector/row index
#'     $id_eval := vector/row index
#'     $id_learn := vector/row index
#'     $reps := 1:nrep
#'     $nrep := INTEGER
#' 
#' @examples
#'   iris$y = ifelse(iris$Species == "", 1, 0)
#'   sets = data2sets(iris$y, sample="down")
#'   train = iris[sets$id_train,]
#'   model = svm(y ~ ., data=train)
#'   eval = iris[sets$id_eval,]
#'   predict(model, newx=eval)
#'
#'   ## learn multiple models
#'   models = lapply(sets$reps, function(r){
#'     learn = iris[sets$id_learn_[[r]],]
#'     svm(y ~ ., data=learn)
#'   })
#'   ## average models
#' 
#' @export
#'
data2sets <- function(y, t=NULL, sample="down", split=.5, scale=1, nquant=5, nrep=11){
    sets = switch(sample,
                  "down" = { downsample(y, t, split, scale, nquant, nrep) },
                  "up"   = {   upsample(y, t, split, scale, nquant, nrep) },
                  "sub"  = {  subsample(y, t, split, scale, nquant, nrep) }
                  )
    return(sets)
}

downsample <- function(y, t=NULL, split=.5, scale=1, nquant=5, nrep=11){
    if(is.null(t)){
        t = 1:length(y)
    }
    sep = as.integer(length(y) * split)
    id_train = 1:sep
    id_eval = (sep+1):length(y)
    bin = seq(0, 1, 1/nquant)
    q = as.integer(quantile(t[id_train], bin))
    quants = 1:(length(q)-1)
    quant = list(nquant)
    quant[quants] = lapply(quants, function(i){
        id = q[i] <= t & t < q[i+1]+1
        y0 = which(y[id] == 0)
        y1 = which(y[id] == 1)
        n1 = length(y1)
        n0 = as.integer(scale * n1)
        rep = list(nrep)
        reps = 1:nrep
        rep[reps] = lapply(reps, function(j){
            c(sample(y0, n0, T),
              sample(y1, n1))
        })
        names(rep) = sprintf("r%03d", reps)
        rep
    })
    id_target
    names(quant) = sprintf("q%03d", quants)
    lut = list(id_train=id_train,
               id_eval=id_eval,
               id_learn=quant,
               t = t,
               sep=sep,
               scale=scale,
               nquant=nquant,
               bin = bin,
               q = q,
               nrep=nrep)
    return(lut)
}

upsample <- function(y, t=NULL, split=.5, scale=1, nquant=5, nrep=11){
    if(is.null(t)){
        t = 1:length(y)
    }
    sep = as.integer(length(y) * split)
    id_train = 1:sep
    id_eval = (sep+1):length(y)
    bin = seq(0, 1, 1/nquant)
    q = as.integer(quantile(t[id_train], bin))
    quants = 1:(length(q)-1)
    quant = list(nquant)
    quant[quants] = lapply(quants, function(i){
        id = q[i] <= t & t < q[i+1]+1
        y0 = which(y[id] == 0)
        y1 = which(y[id] == 1)
        n0 = length(y0)
        n1 = as.integer(scale * n0)
        rep = list(nrep)
        reps = 1:nrep
        rep[reps] = lapply(reps, function(j){
            c(sample(y0, n0),
              sample(y1, n1, T))
        })
        names(rep) = sprintf("r%03d", reps)
        rep
    })
    names(quant) = sprintf("q%03d", quants)
    lut = list(id_train=id_train,
               id_eval=id_eval,
               id_learn=quant,
               t = t,
               sep=sep,
               scale=scale,
               nquant=nquant,
               bin = bin,
               q = q,
               nrep=nrep)
    return(lut)
}


subsample <- function(y, t=NULL, split=.5, scale=1, nquant=2, nrep=11){
    if(is.null(t)){
        t = 1:length(y)
    }
    sep = as.integer(length(y) * split)
    id_train = 1:sep
    id_eval = (sep+1):length(y)
    bin = seq(0, 1, 1/nquant)
    q = as.integer(quantile(t[id_train], bin))
    quants = 1:(length(q)-1)
    quant = list(nquant)
    quant[quants] = lapply(quants, function(i){
        id = q[i] <= t & t < q[i+1]+1
        n = as.integer(length(id)/2)
        reps = 1:nrep
        rep[reps] = lapply(reps, function(j){
            sample(id, n)
        })
        names(rep) = sprintf("r%03d", reps)
        rep
    })
    names(quant) = sprintf("q%03d", quants)
    lut = list(id_train=id_train,
               id_eval=id_eval,
               id_learn=quant,
               t = t,
               sep=sep,
               scale=scale,
               nquant=nquant,
               bin = bin,
               q = q,
               nrep=nrep)
    return(lut)
}


## [3]
## Feature tuning
##   Improving the quality of a feature for ML, which
##   includes scaling and normalising numeric values,
##   imputing missing values, clipping outliers, and
##   adjusting values with skewed distributions

#' Box/Cox Power (BCP) Transform
#'
#' @param
#'   x : data.frame
#'
#' @return
#'   num2bcp object
#' 
#' @examples
#'   m = num2bcp(iris[,1:2])
#'   predict(m, iris[,1:4])
#' 
#' @export
#' 
num2bcp <- function(x, ...){
    labs = colnames(x)
    lut = lapply(labs, function(lab){ car::powerTransform(x[[lab]]) })
    names(lut) = labs
    model = list(lut=lut)
    class(model) = c("num2bcp")
    return(model)
}

#' Box/Cox Power (BCP) Transform
#'
#' @param
#'   model : num2bcp object
#'
#' @param
#'   newx : data.frame
#' 
#' @return
#'   data.frame;  dim is preserved
#'
#' @examples
#'   m = num2bcp(iris[,1:2])
#'   predict(m, iris[,1:4])
#'
#' @export
#' 
predict.num2bcp <- function(model, newx, ...){
    cols = intersect(colnames(newx), names(model$lut))
    newx[,cols] = lapply(cols, function(col){
        car::bcPower(newx[[col]], model$lut[[col]]$lambda)
    })
    return(newx)
}


#' Feature scaling of numeric variables
#'
#' @param
#'   x : data.frame with independent variables of type continous
#'
#' @return
#'   num2norm object
#'
#' @examples
#'   m = num2norm(iris[,1:2])
#'   predict(m, iris[,1:4])
#' 
#' @export
#' 
num2scale <- function(x){
    labs = colnames(x)
    lut = lapply(labs, function(lab){
        v = na.omit(x[[lab]])
        list(min=min(v), max=max(v), mean=mean(v), sd=sd(v))
    })
    names(lut) = labs
    model = list(lut=lut)
    class(model) = c("num2scale")
    return(model)
}                 

#' Feature scaling of numeric variables
#'
#' @param
#'   model : num2scale object
#'
#' @param
#'   method : [scaling (default) | normalising | standardising ]
#'   scaling = (x-xmin) / (xmax-xmin)
#'   normalising = (x-xmean) / (xmax-xmin)
#'   standardising = (x-xmean) / xsd
#'
#' @param
#'   newx : data.frame
#'
#' @return
#'   data.frame;  dim is preserved
#'
#' @examples
#'   m = num2scale(iris[,1:2])
#'   predict(m, iris[,1:4], )
#' 
#' @export
#' 
predict.num2scale <- function(model, newx, method="scaling"){
    cols = intersect(colnames(newx), names(model$lut))
    methods = c("scaling", "normalising", "standardising")
    choice = methods[pmatch(method, methods)]
    if(is.na(choice)){
        return(NULL)
    }else{
        newx[,cols] = lapply(cols, function(col){
            vals = model$lut[[col]]
            switch(choice,
                   scaling = { (newx[[col]]-vals$min)/(vals$max-vals$min) },
                   normalising = { (newx[[col]]-vals$mean)/(vals$max-vals$min) },
                   standardising = { (newx[[col]]-vals$mean)/(vals$sd) }
                   )
        })
        return(newx)
    }
}



        
## [4]
## Representation transformation
##   Converting a numeric feature to a categorical
##   feature (through bucketisation), and converting
##   categorical features to a numeric representation
##   (through one-hot encoding, learning with counts,
##   sparse feature embeddings). Some models work only
##   with numeric or categorical features, while others
##   can handle both types, they can benefit from
##   different representation (numeric and categorical)
##   of the same feature

#' Data dictionary definition based on input data
#' 
#' @param
#'   x : data.frame with categorical or numerical variables
#' 
#' @param
#'   file : STRING, e.g. "dict.csv"
#'
#' @param
#'   sep : CHAR, default ","
#'
#' @return
#'   data2dict object
#'
#'   if file is not null, then a file is created with the columns:-
#'       var := [x|y|-]
#'     class := [cat|num]
#'      name := STRING
#'
#' @examples
#'   dict = data2dict(iris, file="dict.csv")
#'   ## emacs -nw dict.csv ## edit var=y
#'   dict = data2dict(file="dict.csv")
#'   data = predict(dict, iris)
#'
#' @export
#' 
data2dict <- function(x=NULL, file=NULL, sep=","){
    if(!is.null(x)){
        name = colnames(x)
        class = sapply(name, function(i){ class(x[[i]]) })
        class = gsub("integer", "num", class)
        class = gsub("numeric", "num", class)
        class = gsub("logical", "num", class)
        class = gsub("character", "cat", class)
        class = gsub("factor", "cat", class)        
        var = "x"
        model = data.frame(var=var, class=class, name=name, stringsAsFactors=FALSE)
        if(!is.null(file)){
            write.table(model, file=file, quote=FALSE, sep=sep, row.names=FALSE)
        }
        class(model) = c("data2dict")        
        return(model)
    }
    if(is.null(x) && !is.null(file) && file.exists(file)){
        model = read.table(file, sep=sep, stringsAsFactors=FALSE, header=TRUE)
        class(model) = c("data2dict")
        return(model)
    }
}

#' Data dictionary applied on input data
#'
#' Operations include:-
#'   - casting x numeric variables
#'   - casting x categorical variables
#'   - casting y variable [TODO]
#'
#' @param
#'   model : data2dict object
#'
#' @param
#'   newx : data.frame
#'
#' @return
#'   data.frame;  dim is preserved
#' 
#' @examples
#'   dict = data2dict(iris, file="dict.csv")
#'   ## emacs -nw dict.csv ## edit var=y
#'   dict = data2dict(file="dict.csv")
#'   data = predict(dict, iris)
#'
#' @export
#' 
predict.data2dict <- function(model, newx, ...){
    X = dict2x(model, newx)
    Y = dict2y(model, newx)
    Xnum = dict2xnum(model, newx)
    Xcat = dict2xcat(model, newx)
    newx[,Xnum] = lapply(Xnum, function(i){ as.numeric(newx[[i]]) })
    newx[,Xcat] = lapply(Xnum, function(i){ as.character(newx[[i]]) })
    ##--TODO:  cast Y
    return(newx)
}




#' Extract numerical variables on the basis of the current data.frame
#'
#' @param
#'   x : data.frame
#'
#' @param
#'   model : dictionary
#'
#' @return
#'   vector with numerical variable names
#'
#' @export
#' 
data2xnum <- function(x, model){
    labs = colnames(x)
    s = sapply(labs, function(i)is.numeric(x[[i]]))
    id_xnum = labs[s]
    id_y = dict2y(model, x)
    id_xnum = id_xnum[!id_xnum %in% id_y]
}


#' Extract categorical variables on the basis of the current data.frame
#'
#' @param
#'   x : data.frame
#'
#' @param
#'   model : dictionary
#'
#' @return
#'   vector with categorical variable names
#' 
#' @export
#' 
data2xcat <- function(x, model){
    labs = colnames(x)
    s = sapply(labs, function(i)is.character(x[[i]]))
    id_xcat = labs[s]
    id_y = dict2y(model, x)
    id_xcat = id_xcat[!id_xcat %in% id_y]
}



#' Extract x variables
#'
#' @param
#'   model : data2dict object
#'
#' @param
#'   newx : data.frame
#'
#' @return
#'   character vector
#' 
#' @export
#' 
dict2x <- function(model, newx, ...){
    intersect(model$name[model$var == "x"], colnames(newx))
}

#' Extract y variables
#'
#' @param
#'   model : data2dict object
#'
#' @param
#'   newx : data.frame
#'
#' @return
#'   character vector
#' 
#' @export
#' 
dict2y <- function(model, newx, ...){
    intersect(model$name[model$var == "y"], colnames(newx))
}

#' Extract numeric x variables
#'
#' @param
#'   model : data2dict object
#'
#' @param
#'   newx : data.frame
#'
#' @return
#'   character vector
#' 
#' @export
#' 
dict2xnum <- function(model, newx, ...){
    intersect(model$name[model$var == "x" & model$class == "num"], colnames(newx))
}

#' Extract categorical x variables
#'
#' @param
#'   model : data2dict object
#'
#' @param
#'   newx : data.frame
#'
#' @return
#'   character vector
#' 
#' @export
#' 
dict2xcat <- function(model, newx, ...){
    intersect(model$name[model$var == "x" & model$class == "cat"], colnames(newx))
}


#' Create formula from dict
#'
#' @param
#'   model : data2dict object
#'
#' @param
#'   newx : data.frame
#'
#' @return
#'   formula object
#'
#' @export
#' 
data2formula <- function(model, newx, varx=NULL, vary=NULL){
    if(is.null(vary)){
        id_y = dict2y(model, newx)
    }else{
        id_y = vary
    }
    if(is.null(varx)){
        id_x = dict2x(model, newx)
    }else{
        switch(varx,
               "x" = {id_x = dict2x(model, newx)},
               "xnum" = {id_x = dict2xnum(model, newx)},
               "xcat" = {id_x = dict2xcat(model, newx)},
               {id_x = varx}
               )
    }
    s = paste0(id_y, " ~ ")
    if(length(id_x) > 1){
        s = paste0(s, paste0(id_x, collapse=" + "))
    }else if(length(x) == 0){
        s = paste0(s, id_x)
    }
    return(as.formula(s))
}





#' Converts categorical variables to continuous variables
#'
#' Based on the mutual information MI(xi, y) each categorical
#' variable value is represented by its score
#' 
#' @param
#'   x :  data.frame with independent variables of type category
#' @param
#'   y : character vector
#'
#' @return
#'   cat2num object
#'
#' @examples
#'   m = cat2num()
#'   predict(m, )
#' 
#' @export
#' 
cat2num <- function(x, y, ...){
    labs = colnames(x)
    lut = lapply(labs, function(lab){ mi(x[[lab]], y, ...) })
    names(lut) = labs
    model = list(lut=lut)
    class(model) = c("cat2num")
    return(model)
}       

## assumes the matrix is like this
##     y0 y1
##  x0
##  x1
mi <- function(X, Y, smooth=.000001){
    M = as.matrix(table(X, Y))
    . = apply(M, 2, sum)
    .M = t(apply(M, 1, function(x). - x))
    N11 =  M[,2] + smooth
    N10 =  M[,1] + smooth
    N00 = .M[,1] + smooth
    N01 = .M[,2] + smooth
    N   = N11 + N10 + N00 + N01
    N1. = N11 + N10 + smooth
    N.1 = N11 + N01 + smooth
    N0. = N01 + N00 + smooth
    N.0 = N10 + N00 + smooth
    log2N   = log2(N)
    log2N11 = log2(N11)
    log2N1. = log2(N1.)
    log2N.1 = log2(N.1)   
    log2N00 = log2(N00)
    log2N0. = log2(N0.)
    log2N.0 = log2(N.0) 
    log2N10 = log2(N10)
    log2N01 = log2(N01)
    I = (N11 / N) * (log2N + log2N11 - log2N1. - log2N.1) +
        (N01 / N) * (log2N + log2N01 - log2N0. - log2N.1) +
        (N10 / N) * (log2N + log2N10 - log2N1. - log2N.0) +
        (N00 / N) * (log2N + log2N00 - log2N0. - log2N.0)
    I[is.na(I)] = 0
    return(I)
}

#' Converts categorical variables to continuous variables
#' 
#' @param
#'   model : cat2num object
#'
#' @param
#'   newx : data.frame
#'
#' @return
#'   data.frame;  dim is preserved
#'
#' @examples
#'   m = cat2num()
#'   predict(m, )
#' 
#' @export
predict.cat2num <- function(model, newx, ...){
    cols = intersect(colnames(newx),  names(model$lut))
    newx[,cols] = lapply(cols, function(col){
        vals = model$lut[[col]]
        vals[match(newx[[col]], names(vals))]
    })
    return(newx)
}




#' Converts a continuous variable to bins
#'
#' @param
#'   x : ata.frame with independent variables of type continous
#'
#' @return
#'   num2bin object
#'
#' @examples
#'   n = 100
#'   rows = sample(1:nrow(iris), n)
#'   train = iris[n,]
#'   sel = 1:30
#'   calib = train[sel,]
#'   learn = train[-sel,]
#'   m = num2bin(calib[,1:4])
#'   learn = predict(m, learn)
#' 
#' @export
#' 
num2bin <- function(x, ...){
    labs = colnames(x)
    lut = lapply(labs, function(lab){ tdigest(x[[lab]], ...) })
    names(lut) = labs
    model = list(lut=lut)
    class(model) = c("num2bin")
    return(model)
}

tdigest <- function(X, bins = 100, delta = 10){
    dens = density(X, na.rm=TRUE)
    csum = cumsum(dens$y)
    cuts = suppressWarnings(Hmisc::cut2(csum, g=bins, onlycuts=TRUE))
    step = length(cuts) - 1
    labs = seq(1/step, 1, 1/step)
    q = as.numeric(as.character(cut(csum, breaks=cuts, labels=labs)))
    k = round(delta*((asin(2*q-1)/pi) + .5))
    df = data.frame(x=dens$x, y=dens$y, csum=csum,q,k)
    df = aggregate(df, by=list(k), FUN=mean)
    k = df$k
    k = c(0, k, delta)
    names(k) = c(-Inf, df$x, Inf)
    return(k)
}      

#' Converts a continuous variable to bins
#'
#' @param
#'   model : num2bin object
#'
#' @param
#'   newx : data.frame
#'
#' @return
#'   data.frame;  dim is preserved
#' 
#' @examples
#'   n = 100
#'   rows = sample(1:nrow(iris), n)
#'   train = iris[n,]
#'   sel = 1:30
#'   calib = train[sel,]
#'   learn = train[-sel,]
#'   m = num2bin(calib[,1:4])
#'   learn = predict(m, learn)
#' 
#' @export
#' 
predict.num2bin <- function(model, newx, ...){
    cols = intersect(colnames(newx),  names(model$lut))
    newx[,cols] = lapply(cols, function(col){
        vals = model$lut[[col]]
        cuts = names(vals)
        labs = vals[-length(vals)]
        cut(newx[[col]], breaks=cuts, labels=labs)
    })
    return(newx)
}




## [5]
## Feature extraction
##   Reducing the number of features by creating lower-
##   dimension, more powerful data representations using
##   techniques such as PCA, embedding extraction, and
##   hashing

#' PCA based transformation of numerical data
#'
#' @param
#'   x : data.frame with numerical variables
#'
#' @return
#'   num2pca object
#'
#' @examples
#'   m = num2pca(iris[,1:4])
#'   predict(m, iris)  ##--TODO
#' 
#' @export
#' 
num2pca <- function(x, ...){
    pca = prcomp(x, center=TRUE, scale=TRUE)
    importance = summary(pca)$importance[3,]
    model = list(pca=pca, importance=importance)
    class(model) = c("num2pca")
    return(model)
}

predict.num2pca <- function(model, newx, thresh=.95, ...){
    sel = model$importance <= thresh
    newx = predict(model$pca, newx)[,sel]
    return(newx)
}



## [6]
## Feature selection
##   Selecting a subset of the input features for training
##   the model, and ignoring the irrelevant or redundant
##   ones, using filter or wrapper methods. This can also
##   involve simply dropping features if the features are
##   missing a large number of values

#' Remove NA variables
#'
#' @param
#'   x : data.frame with categorical or numerical variables
#'
#' @param
#'   thresh : DOUBLE
#'   IF num_NA / len_var > thresh
#'   THEN remove
#'
#' @examples
#'   m = na2rm(iris[,1:4])b
#'   predict(m, iris)
#' 
#' @return
#'   na2rm object
#'
#' @export
#' 
na2rm <- function(x, thresh=.05){
    cols = colnames(x)
    na = sapply(cols, function(xx){        
        sum(is.na(x[[xx]]))
    }) / nrow(x)
    lut = cols[na > thresh]
    model = list(lut=lut)
    class(model) = c("na2rm")
    return(model)
}

#' Remove NA variables
#'
#' @param
#'   model : na2rm object
#'
#' @param
#'   newx : data.frame
#'
#' @examples
#'   m = na2rm(iris[,1:4])b
#'   predict(m, iris)
#'
#' @return
#'   data.frame;  number of columns is reduced
#'
#' @export
#' 
predict.na2rm <- function(model, newx){
    newx[, model$lut] = NULL
    return(newx)
}


#' Remove constant variables
#'
#' @param
#'   x : data.frame with categorical or numerical variables
#'
#' @return
#'   const2rm object
#'
#' @examples
#'   m = const2rm(iris[,1:4])
#'   predict(m, iris)
#'
#' @export
#' 
const2rm <- function(x){
    cols = colnames(x)
    const = sapply(cols, function(xx){
        length(unique(x[[xx]]))
    })
    lut = cols[const == 1]
    model = list(lut=lut)
    class(model) = c("const2rm")
    return(model)
}

#' Remove constant variables
#'
#' @param
#'   model : const2rm object
#'
#' @param
#'   newx : data.frame
#'
#' @return
#'   data.frame;  number of columns is reduced
#'
#' @examples
#'   m = const2rm(iris[,1:4])
#'   predict(m, iris)
#'
#' @export
#' 
predict.const2rm <- function(model, newx){
    newx[, model$lut] = NULL
    return(newx)
}

#' Remove near zero variance (NZV) variables
#'
#' @param
#'   x : data.frame with numerical variables
#' 
#' @param
#'   ratio : DOUBLE;
#'   IF num_NA / len_var >= RATIO
#'   THEN  remove NA
#' 
#' @param
#'   thresh : DOUBLE;
#'   IF nzv < thresh
#'   THEN  exclude NZV variables
#' 
#' @return
#'   nzv2rm object
#'
#' @examples
#'   sel = rm_nzv(iris[,1:4])
#'   learn = iris[,sel]
#' 
#' @export
#' 
nzv2rm <- function(x, ratio=.5, thresh=.05){
    cols = colnames(x)
    nzv = sapply(cols, function(xx){
        rel = sum(is.na(x[[xx]])) / length(x[[xx]])
        NA.RM = ifelse(rel >= ratio, TRUE, FALSE)
        nzv = ifelse(is.numeric(x[[xx]]),
                     var(x[[xx]], na.rm=NA.RM),
                     ifelse(length(table(x[[xx]]) > 1, 1, 0))
                     )
    })
    lut = na.omit(cols[nzv < thresh])
    model = list(lut=lut, nzv=nzv)
    class(model) = c("nzv2rm")
    return(model)
}

#' Remove near zero variance (NZV) variables
#'
#' @param
#'   model : nzv2rm object
#'
#' @param
#'   newx : data.frame
#'
#' @return
#'   data.frame : number of columns is reduced
#' 
#' @examples
#'   sel = rm_nzv(iris[,1:4])
#'   learn = iris[,sel]
#' 
#' @export
#' 
predict.nzv2rm <- function(model, newx){
    newx[, model$lut] = NULL
    return(newx)
}


#' Remove correlating variables
#'
#' @param
#'   x : data.frame with numeric variables
#'
#' @param
#'   thresh : DOUBLE;
#'   IF cor > thresh
#'   THEN exclude variable
#'
#' @return
#'   cor2rm object
#'
#' @examples
#'   m = cor2rm(iris[,1:4])
#'   predict(m, iris)
#'
#' @export
#' 
cor2rm <- function(x, thresh=.95){
    cols = colnames(x)
    tmp = cor(x[,cols], use="complete.obs")
    tmp[upper.tri(tmp)] = 0
    diag(tmp) = 0
    lut = cols[apply(tmp, 2, function(xx){ any(abs(xx) > thresh) })]
    model = list(lut=lut, thresh=thresh, cormat=tmp, cols=cols)
    class(model) = c("cor2rm")
    return(model)
}

#' Remove correlating variables
#'
#' @param
#'   model : cor2rm object
#'
#' @param
#'   newx : data.frame
#'
#' @return
#'   data.frame : number of columns is reduced
#'
#' @examples
#'   m = cor2rm(iris[,1:4])
#'   predict(m, iris)
#'
#' @export
#' 
predict.cor2rm <- function(model, newx){
    newx[, model$lut] = NULL
    return(newx)
}

#' Identify which numericcal variables needs to be log
#' The decision is based on the measurement of their skewness
#'
#' @param
#'   x : data.frame with numerical variables
#'
#' @return
#'   num2log object
#' 
#' @export
#' 
num2log <- function(x, thresh=1){
    labs = colnames(x)
    s = sapply(labs, function(i){
        e1071::skewness(x[[i]], na.rm=T)
    })
    sel = labs[s >= 1]
    model = list(sel=sel, skewness=s, name=labs)
    class(model) = c("num2log")
    return(model)
}

#' Identify which numericcal variables needs to be log
#' The decision is based on the measurement of their skewness
#'
#' @param
#'   model : num2log object
#'
#' @param
#'   newx : data.frame with numerical variables
#'
#' @param
#'   smooth : double;  value is added to numerical variable before log:  log(newx[[i]] + smooth)
#' 
#' @return
#'   data.frame; number of dimensions may be reduced
#' 
#' @export
#' 
predict.num2log <- function(model, newx, smooth=.001){
    labs = intersect(colnames(newx), model$sel)
    newx[,!colnames(newx) %in% labs] = NULL
    newx[labs] = lapply(labs, function(i) log(newx[[i]] + smooth) )
    newx
}



#' Feature selection using the wrapper method (GA + SVM)
#'
#' @param
#'   x : data.frame with continuous variables
#'
#' @param
#'   y = vector with binary class labels: {0, 1}
#'
#' @return
#'   num2gasvm object
#'
#' @examples
#'   x = iris[1:100,1:4]
#'   y = ifelse(iris[1:100, 5] == "setosa", 1, 0)
#'   m = num2gasvm(x, y)
#'   newx = predict(m, x)
#' 
#' @export
#' 
num2gasvm <- function(x, y, run=10){
    i0 = which(y == 0)
    i1 = which(y == 1)
    r0 = sample(i0, as.integer(length(i0)/2))
    r1 = sample(i1, as.integer(length(i1)/2))
    id_learn = c(r0, r1)
    id_eval = -c(r0, r1)
    ## fitness function
    fit <- function(vars, x, y, id_learn, id_eval){
        id_xnum = colnames(x)[vars]
        svm = LiblineaR::LiblineaR(data=x[id_learn, id_xnum],target=y[id_learn])
        p = predict(svm, newx=x[id_eval, id_xnum])$predictions
        a = y[id_eval]
        conf = as.matrix(table(p,a))
        TP = conf[2,2]
        FP = conf[2,1]
        FN = conf[1,2]
        PPV = TP / (TP+FP)
        REC = TP / (TP+FN)
        F1 = PPV*REC*2/(REC+PPV)
        F1
        F1 / sum(vars)
    }
    ## genetics algorithm to search optimal subset of features
    ga = GA::ga(fitness=function(vars){ fit(vars, x, y, id_learn, id_eval) },
                type="binary",
                elitism=3,
                pmutation=.5,
                popSize=10,
                nBits=ncol(x),
                names=colnames(x),
                run=run,
                maxiter=10,
                monitor=plot,
                keepBest=TRUE,
                parallel=FALSE,
                seed=84211)
    sel = which(as.integer(ga@bestSol[[run]]) == 1)
    cols = colnames(x)[sel]
    model = list(cols=cols)
    class(model) = c("num2gasvm")
    return(model)
}

#' Feature selection using the wrapper method
#' 
#' @param
#'   model : num2gasvm object
#'
#' @param
#'   newx : data.frame
#'
#' @examples
#'   x = iris[1:100,1:4]
#'   y = ifelse(iris[1:100, 5] == "setosa", 1, 0)
#'   m = num2gasvm(x, y)
#'   newx = predict(m, x)#'
#' 
#' @return
#'   data.frame; the number of columns is reduced
#' 
#' @export
#' 
predict.num2gasvm <- function(model, newx, ...){
    cols = intersect(colnames(newx), names(model$cols))
    return(newx[,cols])
}



#' Feature selection using randomForest
#'
#' Features are selected on the basis of variable$importance
#'
#' @param
#'   x : data.frame with numerical variables and class label
#'
#' @param
#'   f : formula
#'
#' @param
#'   thresh : numerical;  selects variables with importance >= thresh
#'
#' @param
#'   ntree : integer;  number of trees
#'
#' @return
#'   num2imp object
#'
#' @examples
#'   m = num2imp(Species ~ ., iris)
#'   dat = predict(m, iris)
#' 
#' @export
#' 
num2imp <- function(x, y, thresh=.5, ntree=10){
    rf = randomForest::randomForest(x, y, ntree=10)
    labs = rownames(rf$importance)
    imp = rf$importance
    imp = (imp-min(imp))/diff(range(imp))
    name = row.names(imp)
    sel = labs[imp >= thresh]
    model = list(sel=sel, imp=imp, name=name, thresh=thresh, rf=rf)
    class(model) = c("num2imp")
    return(model)
}

#' Feature selection using randomForest
#'
#' Features are selected on the basis of variable$importance
#'
#' @param
#'   model : num2imp object
#'
#' @param
#'   newx : data.frame with numerical variables
#'
#' @return
#'   data.frame dimension may be reduced
#'
#' @examples
#'   m = num2imp(Species ~ ., iris)
#'   dat = predict(m, iris)
#' 
#' @export
#' 
predict.num2imp <- function(model, newx){
    newx[,model$sel]
}



num2sel <- function(x, y, method="randomforest", ...){
    methods = c("RF", "GASVM")
    choice = methods[pmatch(method, methods)]
    switch(choice,
           RF = { return(num2imp(x, y, ...)) },
           GASVM = { return(num2gasvm(x, ...)) }
           )
    return(NULL)
}



## [7]
## Feature construction
##   Creating new features either by using typical techniques,
##   such as polynomial expansion (by using univariate
##   mathematical functions), or feature corssing (to
##   capture feature interactions). Features can also be
##   constructed by using business logic from the domain
##   of the ML use case



#' Interaction of numerical variables
#'
#' @param
#'   x : data.frame with numerical variables
#'   if t=NULL (default), then all variable combinations are used and four
#'   different interaction operations used to compute new variables:
#'   a*b (new variable infix = name(a) _dot_ name(b)
#'   a+b (new variable infix = name(a) _sum_ name(b)
#'   a-b (new variable infix = name(a) _dif_ name(b)
#'   a/b (new variable infix = name(a) _div_ name(b)
#'
#' @param
#'   t : numeric vector;  default NULL
#'   if available then each numeric variable is multiplied by t: t*xi
#' 
#' @return
#'   data.frame
#'
#' @examples
#'   num2int(iris[,1:3])
#' 
#' @export
#' 
num2int <- function(x, t=NULL){
    labs = colnames(x)
    if(is.null(t)){
        comb = combn(labs,m=2)
        combs = 1:ncol(comb)
        ## dot = a*b
        xdot = list()
        xdot[combs] = lapply(combs, function(i){
            a = comb[1,i]
            b = comb[2,i]
            x[[a]] * x[[b]]
        })
        xdot.name = sapply(combs, function(i){
            a = comb[1,i]
            b = comb[2,i]
            paste0(a,"_dot_",b)
        })
        ## sum = a+b
        xsum = list()
        xsum[combs] = lapply(combs, function(i){
            a = comb[1,i]
            b = comb[2,i]
            x[[a]] + x[[b]]
        })
        xsum.name = sapply(combs, function(i){
            a = comb[1,i]
            b = comb[2,i]
            paste0(a,"_sum_",b)
        })
        ## dif = a-b
        xdif = list()
        xdif[combs] = lapply(combs, function(i){
            a = comb[1,i]
            b = comb[2,i]
            x[[a]] - x[[b]]
        })
        xdif.name = sapply(combs, function(i){
            a = comb[1,i]
            b = comb[2,i]
            paste0(a,"_dif_",b)
        })
        ## div = a/b
        xdiv = list()
        xdiv[combs] = lapply(combs, function(i){
            a = comb[1,i]
            b = comb[2,i]
            x[[a]] - x[[b]]
        })
        xdiv.name = sapply(combs, function(i){
            a = comb[1,i]
            b = comb[2,i]
            paste0(a,"_div_",b)
        })
        newx = cbind(as.data.frame(xdot, col.names=xdot.name),
                     as.data.frame(xsum, col.names=xsum.name),
                     as.data.frame(xdif, col.names=xdif.name),
                     as.data.frame(xdiv, col.names=xdiv.name)
                     )
        return(newx)
    }else{
        newx = list()
        newx[labs] = lapply(labs, function(i){
            t * x[[i]]
        })
        name = paste0("t_dot_", labs)
        return(as.data.frame(newx, col.names=name))
    }
    return(NULL)
}

#' Feature interaction:  f1 + f2
#' 
#' @param
#'   x : data.frame with numerical variables
#'
#' @param
#'   y : vector with class label
#'
#' @param
#'   thresh : numerical;  selects variables with importance >= thresh
#' 
#' @param
#'   ntree : integer;  number of trees
#'
#' @return
#'   num2sum object
#'
#' @examples
#'   df = iris[1:100,]
#'   df$Species = as.factor(ifelse(df$Species == "setosa", 1, 0))
#'   m = num2sum(df[,1:4], df[,5])
#'   xsum = predict(m, df[,1:4]
#'   data = cbind(df, xsum)
#' 
#' @export
#' 
num2sum <- function(x, y=NULL, thresh=.5, ntree=10){
    labs = colnames(x)
    comb = expand.grid(labs, labs)
    combs = 1:nrow(comb)
    int = list()
    int[combs] = lapply(combs, function(i){
        a = comb[i,1]
        b = comb[i,2]
        x[[a]] + x[[b]]
    })
    infix = "_add_"
    name = paste0(comb[,1], infix, comb[,2])
    df = as.data.frame(int, col.names=name)
    if(is.null(y)){
        model = list(sel=name, imp=NULL, infix=infix, name=name, thresh=NULL, ntree=NULL, out=df)
    }else{
        m = num2imp(df, y, thresh, ntree)
        model = list(sel=m$sel, imp=m$imp, infix=infix, name=m$name, thresh=thresh, ntree=ntree, out=NULL)
    }
    class(model) = c("num2sum")
    return(model)
}


#' Feature interaction:  f1 + f2
#' 
#' @param
#'   model : num2sum object
#'
#' @param
#'   newx : data.frame with numerical variables
#'
#' @return
#'   data.frame with most important interacting features
#'
#' @examples
#'   df = iris[1:100,]
#'   df$Species = as.factor(ifelse(df$Species == "setosa", 1, 0))
#'   m = num2sum(df[,1:4], df[,5])
#'   xsum = predict(m, df[,1:4]
#'   data = cbind(df, xsum)
#' 
#' @export
#' 
predict.num2sum <- function(model, newx){
    int = list()
    int[model$sel] = lapply(model$sel, function(i){
        a = strsplit(i, model$infix)[[1]][1]
        b = strsplit(i, model$infix)[[1]][2]
        newx[[a]] + newx[[b]]
    })
    df = as.data.frame(int, col.names=model$sel)
    return(df)
}

#' Feature interaction:  f1 * f2
#' 
#' @param
#'   x : data.frame with numerical variables
#'
#' @param
#'   y : vector with class label
#'
#' @param
#'   thresh : numerical;  selects variables with importance >= thresh
#' 
#' @param
#'   ntree : integer;  number of trees
#'
#' @return
#'   num2dot object
#'
#' @examples
#'   df = iris[1:100,]
#'   df$Species = as.factor(ifelse(df$Species == "setosa", 1, 0))
#'   m = num2dot(df[,1:4], df[,5])
#'   xdot = predict(m, df[,1:4]
#'   data = cbind(df, xdot)
#' 
#' @export
#' 
num2dot <- function(x, y=NULL, thresh=.5, ntree=10){
    labs = colnames(x)
    comb = expand.grid(labs, labs)
    combs = 1:nrow(comb)
    int = list()
    int[combs] = lapply(combs, function(i){
        a = comb[i,1]
        b = comb[i,2]
        x[[a]] * x[[b]]
    })
    infix = "_dot_"
    name = paste0(comb[,1], infix, comb[,2])
    df = as.data.frame(int, col.names=name)
    if(is.null(y)){
        model = list(sel=name, imp=NULL, infix=infix, name=name, thresh=NULL, ntree=NULL, out=df)
        
    }else{
        m = num2imp(df, y, thresh, ntree)
        model = list(sel=m$sel, imp=m$imp, infix=infix, name=m$name, thresh=thresh, ntree=ntree)
    }
    class(model) = c("num2dot")
    return(model)
}


#' Feature interaction:  f1 * f2
#' 
#' @param
#'   model : num2dot object
#'
#' @param
#'   newx : data.frame with numerical variables
#'
#' @return
#'   data.frame with most important interacting features
#'
#' @examples
#'   df = iris[1:100,]
#'   df$Species = as.factor(ifelse(df$Species == "setosa", 1, 0))
#'   m = num2dot(df[,1:4], df[,5])
#'   xdot = predict(m, df[,1:4]
#'   data = cbind(df, xdot)
#' 
#' @export
#' 
predict.num2dot <- function(model, newx){
    int = list()
    int[model$sel] = lapply(model$sel, function(i){
        a = strsplit(i, model$infix)[[1]][1]
        b = strsplit(i, model$infix)[[1]][2]
        newx[[a]] * newx[[b]]
    })
    df = as.data.frame(int, col.names=model$sel)
    return(df)
}


#' Feature interaction:  f1 - f2
#' 
#' @param
#'   x : data.frame with numerical variables
#'
#' @param
#'   y : vector with class label
#'
#' @param
#'   thresh : numerical;  selects variables with importance >= thresh
#' 
#' @param
#'   ntree : integer;  number of trees
#'
#' @return
#'   num2dif object
#'
#' @examples
#'   df = iris[1:100,]
#'   df$Species = as.factor(ifelse(df$Species == "setosa", 1, 0))
#'   m = num2dif(df[,1:4], df[,5])
#'   xdif = predict(m, df[,1:4]
#'   data = cbind(df, xdif)
#' 
#' @export
#' 
num2dif <- function(x, y=NULL, thresh=.5, ntree=10){
    labs = colnames(x)
    comb = combn(labs, m=2)
    combs = 1:ncol(comb)
    int = list()
    int[combs] = lapply(combs, function(i){
        a = comb[1,i]
        b = comb[2,i]
        x[[a]] - x[[b]]
    })
    infix = "_dif_"
    name = sapply(combs, function(i){
        paste0(comb[1,i], infix, comb[2,i])
    })
    df = as.data.frame(int, col.names=name)
    if(is.null(y)){
        model = list(sel=name, imp=NULL, infix=infix, name=name, thresh=NULL, ntree=NULL, out=df)        
    }else{
        m = num2imp(df, y, thresh, ntree)
        model = list(sel=m$sel, imp=m$imp, infix=infix, name=m$name, thresh=thresh, ntree=ntree)
    }
    class(model) = c("num2dif")
    return(model)
}

#' Feature interaction:  f1 - f2
#' 
#' @param
#'   model : num2dif object
#'
#' @param
#'   newx : data.frame with numerical variables
#'
#' @return
#'   data.frame with most important interacting features
#'
#' @examples
#'   df = iris[1:100,]
#'   df$Species = as.factor(ifelse(df$Species == "setosa", 1, 0))
#'   m = num2dif(df[,1:4], df[,5])
#'   xdif = predict(m, df[,1:4]
#'   data = cbind(df, xdif)
#' 
#' @export
#' 
predict.num2dif <- function(model, newx){
    int = list()
    int[model$sel] = lapply(model$sel, function(i){
        a = strsplit(i, model$infix)[[1]][1]
        b = strsplit(i, model$infix)[[1]][2]
        newx[[a]] - newx[[b]]
    })
    df = as.data.frame(int, col.names=model$sel)
    return(df)
}




#' Feature interaction:  f1 / f2
#' 
#' @param
#'   x : data.frame with numerical variables
#'
#' @param
#'   y : vector with class label
#'
#' @param
#'   thresh : numerical;  selects variables with importance >= thresh
#' 
#' @param
#'   ntree : integer;  number of trees
#'
#' @return
#'   num2div object
#'
#' @examples
#'   df = iris[1:100,]
#'   df$Species = as.factor(ifelse(df$Species == "setosa", 1, 0))
#'   m = num2div(df[,1:4], df[,5])
#'   xdiv = predict(m, df[,1:4]
#'   data = cbind(df, xdiv)
#' 
#' @export
#' 
num2div <- function(x, y=NULL, thresh=.5, ntree=10, smooth=.001){
    labs = colnames(x)
    comb = combn(labs, m=2)
    combs = 1:ncol(comb)
    int = list()
    int[combs] = lapply(combs, function(i){
        a = comb[1,i]
        b = comb[2,i]
        (x[[a]] + smooth) / (x[[b]] + smooth)
    })
    infix = "_div_"
    name = sapply(combs, function(i){
        paste0(comb[1,i], infix, comb[2,i])
    })
    df = as.data.frame(int, col.names=name)
    if(is.null(y)){
        model = list(sel=name, imp=NULL, infix=infix, name=name, thresh=NULL, ntree=NULL, out=df)
    }else{
        m = num2imp(df, y, thresh, ntree)
        model = list(sel=m$sel, imp=m$imp, infix=infix, name=m$name, thresh=thresh, ntree=ntree)
    }
    class(model) = c("num2div")
    return(model)
}

#' Feature interaction:  f1 / f2
#' 
#' @param
#'   model : num2div object
#'
#' @param
#'   newx : data.frame with numerical variables
#'
#' @return
#'   data.frame with most important interacting features
#'
#' @examples
#'   df = iris[1:100,]
#'   df$Species = as.factor(ifelse(df$Species == "setosa", 1, 0))
#'   m = num2div(df[,1:4], df[,5])
#'   xdiv = predict(m, df[,1:4]
#'   data = cbind(df, xdiv)
#' 
#' @export
#' 
predict.num2div <- function(model, newx, smooth=.001){
    int = list()
    int[model$sel] = lapply(model$sel, function(i){
        a = strsplit(i, model$infix)[[1]][1]
        b = strsplit(i, model$infix)[[1]][2]
        (newx[[a]] + smooth) / (newx[[b]] + smooth)
    })
    df = as.data.frame(int, col.names=model$sel)
    return(df)
}


#' Decision tree feature
#'
#' @param
#'   f : formula
#'
#' @param
#'   x : data.frame with numeric variables and class label
#'
#' @param
#'   loss : matrix; describing the loss:
#'                 c(TP, FP,
#'                   FN, TN)
#' @return
#'   num2dtf object
#'
#' @examples
#'   m = num2dtf(Species ~ ., iris)
#'   dtf = predict(m, iris)
#'   cbind(iris, dtf)
#' 
#' @export
#' 
num2dtf <- function(f, x, loss=matrix(c(0, 1, 5, 0), ncol=2, byrow=TRUE)){
    library(partykit)
    dt = rpart::rpart(f, data=x, parms=list(loss=loss, split="information"))
    fit = fitted(as.party(dt))
    id_node = fit[,1]
    nodes = unique(id_node)
    model = list(dt=dt, nodes=nodes)
    class(model) = c("num2dtf")
    return(model)
}


#' Decision tree feature
#'
#' @param
#'   model : num2dtf object
#'
#' @param
#'   newx : data.frame with numeric variables
#'
#' @return
#'   class label
#'
#' @examples
#'   m = num2dtf(Species ~ ., iris)
#'   dtf = predict(m, iris)
#'   cbind(iris, dtf)
#' 
#' @export
#' 
predict.num2dtf <- function(model, newx){
    predict(partykit::as.party(model$dt), newdata=newx, type="node")
}


#' Extract time related features
#'
#' @param
#'   x :  POSIXct containing YYYY-mm-dd
#' 
#' @param
#'   width : integer; window for rolling statistics
#' 
#' @return
#'   data.frame
#'
#' @examples
#'   
#' 
#' @export
#' 
time2feat <- function(x, width){
    WOD = as.numeric(format(x, "%w")) + 1
    DOM = as.numeric(format(x, "%d"))
    HOD = as.numeric(format(x, "%H"))
    WOD. = zoo::rollapply(c(rep(NA, (width-1)), WOD), width, function(xx){
                    min = min(xx)
                    max = max(xx)
                    diff = diff(xx, lag=1)
                    mean = mean(diff)
                    median = median(diff)
                    var = var(diff)
                    minmax = diff(range(xx))
                    delta = tail(xx, 1) - head(xx, 1)
                })
    colnames(WOD.) = paste0("WOD.", width, ".", colnames(WOD.))
    DOM. = zoo::rollapply(c(rep(NA, (width-1)), DOM), width, function(xx){
                    min = min(xx)
                    max = max(xx)
                    diff = diff(xx, lag=1)
                    mean = mean(diff)
                    median = median(diff)
                    var = var(diff)
                    minmax = diff(range(xx))
                    delta = tail(xx, 1) - head(xx, 1)
                })
    colnames(DOM.) = paste0("DOM.", width, ".", colnames(DOM.))    
    HOD. = zoo::rollapply(c(rep(NA, (width-1)), HOD), width, function(xx){
                    min = min(xx)
                    max = max(xx)
                    diff = diff(xx, lag=1)
                    mean = mean(diff)
                    median = median(diff)
                    var = var(diff)
                    minmax = diff(range(xx))
                    delta = tail(xx, 1) - head(xx, 1)
                    c(min, max, mean, median, var, minmax, delta)
                })
    colnames(HOD.) = paste0("HOD.", width, ".", colnames(HOD.))
    x. = zoo::rollapply(c(rep(NA, (width-1)), x), width, function(xx){
                  diffhours = difftime(xx, lag(xx), units="hours")
                  diffdays = difftime(xx, lag(xx), units="days")
                  c(diffhours, diffdays)
              })
    colnames(x.) = paste0("DTS.", width, ".", colnames(x.))
    data.frame(WOD, DOM, HOD, WOD., DOM., HOD., x.)
}





###############################################################





#' Clustering by committee
#'
#' @param x A data.frame with independent variables of type continous
#' @param top A subset of highly similar entries from which the initial cluster is built
#' @param theta1 Similarity threshold between centroids of clusters
#' @param theta2 Similarity threshold between an observation and a committee
#' @param method Similarity metric available from the "proxy" package:
#'   "Jaccard"
#'   "Kulczynski1" "Kulczynski2"
#'   "Mountford"
#'   "Fager"
#'   "Russel"
#'   "simple matching"
#'   "Hamman"
#'   "Faith"
#'   "Tanimoto"
#'   "Dice"
#'   "Phi"
#'   "Stiles"
#'   "Michael"
#'   "Mozley"
#'   "Yule" "Yule2"
#'   "Ochiai"
#'   "Simpson"
#'   "Braun-Blanquet"
#'   "cosine"
#'   "eJaccard" "fJaccard"
#'   "correlation"
#'   "Chi-squared"
#'   "Phi-squared"
#'   "Tschuprow"
#'   "Cramer"
#'   "Pearson"
#'   "Gower"
#'   "Euclidean"
#'   "Mahalanobis"
#'   "Bhjattacharyya"
#'   "Manhattan"
#'   "supremum"
#'   "Minkowski"
#'   "Canberra"
#'   "Wave"
#'   "divergence"
#'   "Kullback"
#'   "Bray"
#'   "Soergel"
#'   "Levenshtein"
#'   "Podani"
#'   "Chord"
#'   "Geodesic"
#'   "Whittaker"
#'   "Hellinger"   
#' 
#' @return A model to predict cluster
#'
#' @examples
#'   n = 100
#'   rows = sample(1:nrow(iris), n)
#'   train = iris[n,]
#'   test = iris[-rows,]
#'   m = cbc(train[,1:4])
#'   p = predict(m, test)
#' 
#' @export                 
cbc <- function(x, top=50, theta1=.35, theta2=.25, method="Euclidean", lim=nrow(x)/2){
    ## Input:  A list of elements E to be clustered, a similarity database S
    ##   from Phase 1, threhsolds theta1, theta2
    E = x
    C = list()
    iter = 0
    nrow.E.last = 0
    while(nrow(E) > 0 && iter < lim){
                                        #    while(iter < 1){
        ## step 1:  For each element e elem E
        ##   Cluster the top similar elements of e from S using average-link
        ##   clustering
        ##   For each discovered cluster c, compute the following score:
        ##   |c| x avgsim(c), where |c| is the number of elements in c and
        ##   avgsim(c) is the average pairwise similarity between elements in c.
        ##   Store the highest scoring cluster in a list L.
        
        
        data.table::setorderv(E)     
        S = head(E, top)
        ##        S = E[sample(1:nrow(E), ifelse(top>=nrow(E),nrow(E), top)), ]
        
        hc = cutree(hclust(proxy::as.dist(proxy::simil(S)), method="ave"), k=1:nrow(S))
        
        print(hc)
        
        hc.scores = apply(hc, 2, function(k){
            uniq = unique(k)
            score = sapply(uniq, function(k.){
                c = S[k == k.,]
                len.c = nrow(c)      
                d = as.matrix(proxy::simil(c, method=method))
                avgsim.c = mean(d[lower.tri(d)])                         
                len.c * avgsim.c
            })
            score[match(k, uniq)]
        })
        
        print(hc.scores)
        
        ## step 2:  Sort the clusters in L in descending order of their scores
        ##---  TODO
        best = which(hc.scores == max(hc.scores, na.rm = T), arr.ind=TRUE)[1,]
        row = best[1]
        col = best[2]
        sel = (hc[,col] == hc[row,col])
        
        L = list()
        L[[(length(L)+1)]] = S[sel,]
        
        ## step 3:  Let C be a list of committees, initially empty.
        ##   For each cluster c elem L in sorted order
        ##   Compute the centroid of c by averaging the feature vectors of its
        ##   elements and computing the mutual information scores in the same
        ##   way as we did for individual elements.
        ##   If c's similarity to the centroid of each committee previously added
        ##   to C is below a threshold theta1, add c to C.
        tmp = lapply(1:length(L), function(i){
            l = L[[i]]
            cen.l = apply(l, 2, mean)
            if(length(C) > 0){
                censim = max(sapply(1:length(C), function(j){
                    c = C[[j]]
                    cen.c = apply(c, 2, mean)
                    v = proxy::simil(x=cen.l, y=cen.c, method=method)
                    ifelse(is.nan(v), 0, v)
                }))
                print(paste0("  censim: ", censim))
                
                if(censim < theta1){
                    C[[(length(C)+1)]] <<- l
                }            
            }else{
                C[[(length(C)+1)]] <<- l
            }
            NULL
        })
        ## step 4:  If C is empty, we are done and return C.
        model = list(C=C, method=method)
        class(model) = "cbc"   

        ## step 5:  For each element e elem E
        ##   If e's similarity to every committee in C is below threshold theta2, add
        ##   e to a list of residues R
        p = predict.cbc(model, x, theta2)
        
        ee = x
        ee$p = as.character(p)
        pp = ggplot(ee, aes(Petal.Length, Petal.Width, color=p)) + geom_point(data=C[[length(C)]], aes(Petal.Length, Petal.Width), color="black", size=5, alpha=.2)+ geom_point() 
        print(pp)
        ## step 6:  If R is empty, we are done and return C.
        ##   Otherwise, return the union of C and the output of a recursive call
        ##   to Phase II using the same input except replacing E with R
        R = x[is.na(p),]
        E = R        
        iter = iter + 1
    }
    
    model = list(C=C, R=R, iter=iter, L=L, p=p, method=method, E=E, p=p)
    class(model) = "cbc"  
    return(model)
}


#' @export
predict.cbc <- function(model, newx, theta2=.50){
    if(class(newx) == "data.frame"){
        sapply(1:nrow(newx), function(ii){
            e = newx[ii,]
            sim = sapply(1:length(model$C), function(i){
                c = model$C[[i]]
                df = rbind(e, c)
                mat = as.matrix(proxy::simil(df, method=model$method))
                max(mat[1,-1])
            })
            hit = as.character(which(sim > theta2, arr.ind = T))
            label = ifelse(length(hit)>0, hit[1], NA)
            label
        })
    }
}







## find cluster/cliques in graph



#' Extract profile information from numerical variables
#' 
#' @param
#'   x : data.frame with numerical variables
#'
#' @param
#'   group : a vector with group variable
#'
#' @param
#'   width : INTEGER,  window for rollapply operation
#'
#' @param
#'   lag : INTEGER,  used to compute the normalised difference index
#'   of n and r = lag(n, k)
#'
#' @return
#'   a data.frame with
#'   z
#'   min
#'   max
#'   median
#'   ndi
#' 
#' @examples
#'   df = num2profile(iris[,1:4], iris[["Species"]], width=10)
#'   learn = cbind(iris[["Species"]], df)
#'   svm = e1071::svm(Species ~ ., data=learn)
#'
#' @export
#' 
num2profile <- function(x, group, width=100, lag=as.integer(width/2)){
    cols = colnames(x)
    tmp = by(x,
       INDICES=list(group),
       FUN=function(g){
           if(nrow(g) == 1){
               g[,cols] = 0
           }else{
               as.data.frame(lapply(g[,cols], function(x){
                   m = zoo::rollapplyr(x, width, mean, fill=NA)
                   s = zoo::rollapplyr(x, width, sd, fill=NA)
                   mi = zoo::rollapplyr(x, width, min, fill=NA)
                   ma = zoo::rollapplyr(x, width, max, fill=NA)
                   n = cumsum(x)
                   r = lag(n, lag)
                   ndi = (n-r) / (n+r)
                   mima = (x-mi)/(ma-mi)
                   z = (x-m)/s
                   data.frame(z=z, mima=mima, ndi=ndi)
               }))
           }
       })
    do.call(rbind, tmp)
}






########

## model

###


#' NBTree
#'
#' Scaling up the accuracy of NB classifiers: a decision tree hybrid -- Ron Kohavi
#'
#' @param
#'   x : data.frame with numeric variables
#'
#' @param
#'   y : character; class label
#'
#' @return
#'   nbtree object
#' 
#' @examples
#'   tmp = iris[1:100,]
#'   tmp[["Species"]] = ifelse(tmp[["Species"]] == "setosa", 1, 0)
#'   m = nbtree(tmp[,1:4], tmp[["Species"]])
#'   p = predict(m, iris)
#' 
#' @export
#' 
nbtree <- function(x, y, loss = matrix(c(0, 1, 5, 0), ncol=2, byrow=TRUE)){
    data = x
    data[["y"]] = y
    library(partykit)
    dt = rpart::rpart(y ~ ., data=data, parms=list(loss=loss, split="information"))
    fit = fitted(as.party(dt))
    id_node = fit[,1]
    nodes = unique(id_node)
    nbtree = list()
    priors = as.numeric(table(y)/length(y))
    tmp = lapply(nodes, function(node){
        sel = id_node == node
        xx = x[sel,]
        xx[is.na(xx)] = 0
        yy = y[sel]
        nbtree[[node]] <<- fastNaiveBayes::fastNaiveBayes(x=xx, y=yy, priors=priors)
    })
    model = list(dt=dt, nbtree=nbtree, nodes=nodes)
    class(model) = c("nbtree")
    return(model)
}

#' NBTree
#'
#' Scaling up the accuracy of NB classifiers: a decision tree hybrid -- Ron Kohavi
#'
#' @param
#'   x : data.frame with numeric variables
#'
#' @param
#'   y : character; class label
#'
#' @return
#'   nbtree object
#' 
#' @examples
#'   tmp = iris[1:100,]
#'   tmp[["Species"]] = ifelse(tmp[["Species"]] == "setosa", 1, 0)
#'   m = nbtree(tmp[,1:4], tmp[["Species"]])
#'   p = predict(m, iris)
#' 
#' @export
#' 
predict.nbtree <- function(model, newx, na.action=0, ...){
    p_dt = as.numeric(as.character(predict(model$dt, newdata=newx, type="class")))
    library(fastNaiveBayes)
    p_node = predict(as.party(model$dt), newdata=newx, type="node")
    p_nb = list()
    p_nb[model$nodes] = lapply(model$nodes, function(node){
        predict(model$nbtree[[node]], newdata=newx, type="class")
    })
    p_nbtree = as.numeric(as.character(sapply(1:length(p_node), function(i){
        p_nb[[p_node[i]]][i]
    })))
    p_nbtree[is.na(p_nbtree)] = na.action
    p = ifelse(p_dt == 0, p_dt, ifelse(p_nbtree == 1, p_nbtree, 0))
    return(p)
}




### incremental svm
## store support vectors
## https://hal.archives-ouvertes.fr/hal-00988202/file/NeuroComputing-HumanRecognition_IncrementalM.pdf



## gasvm / rRecord / rVariable


#' Building an ensemble of SVM classifiers using random sampling
#'
#' @param
#'   x : data.frame with numeric variables
#'   nx = sqrt(ncol(x)) is used to randomly select variables per learning
#'
#' @param
#'   y : vector with label [0,1]
#'   n0 = sqrt(length(i0)) is used to randomly select negative class records
#'
#' @param
#'   nsvm : integer; number of SVMs to be trained
#'
#' @return
#'   rsvm object
#'
#' @examples
#'   dt = iris[1:100,]
#'   dt$Species = ifelse(dt$Species == "setosa", 1, 0)
#'   m = rsvm(dt[,1:4], dt$Species)
#'   predict(m, dt)
#' 
#' @export
#' 
rssvm <- function(x, y, iter=100, nsvm=10){
    i0 = which(y == 0)
    i1 = which(y == 1)
    id_xnum = colnames(x)
    n0 = as.integer(length(i0)/2)
    n1 = length(i1)
    nx = as.integer(sqrt(ncol(x)))
    iters = 1:iter
    svms = list()
    svms[iters] = lapply(iters, function(i){
        r0 = sample(i0, n0)
        r1 = sample(i1, n1)
        rx = sample(id_xnum, nx)
        learn = c(r0, r1)
        LiblineaR::LiblineaR(data=x[learn, rx],target=y[learn])
    })
    ## eval
    perf = rep(0, iter)
    perf[iters] = sapply(iters, function(i){
        p = predict(svms[[i]], newx=x)$predictions
        cm(p,y)$PPV
    })
    o = order(perf, decreasing=T)
    sel = o[1:nsvm]
    out = list()
    out[1:length(sel)] = lapply(sel, function(i){
        rsvm[[i]]
    })    
    model = list(rsvm=out, nsvm=nsvm, iter=iter, nx=nx)
    class(model) = c("rsvm")
    return(model)
}

#' @export
#' 
cm <- function(p,a){
    pp = as.numeric(as.character(p))
    aa = as.numeric(as.character(a))
    TP = sum(pp == aa & aa == 1)
    TN = sum(pp == aa & aa == 0)
    FP = sum(pp != aa & pp == 1)
    FN = sum(pp != aa & pp == 0)
    PPV = TP/(TP+FP)
    REC = TP/(TP+FN)
    F1 = PPV*REC*2/(REC+PPV)
    CM = matrix(c(TN, FN, FP, TP), ncol=2, byrow=TRUE, dimnames=list(P=c("N", "P"), A=c("N", "P")))
    list(TP=TP, TN=TN, FP=FP, FN=FN, PPV=PPV, REC=REC, F1=F1, CM=CM)
}

#' Building an ensemble of SVM classifiers using random sampling
#'
#' @param
#'   x : data.frame with numeric variables
#'   nx = sqrt(ncol(x)) is used to randomly select variables per learning
#'
#' @param
#'   y : vector with label [0,1]
#'   n0 = sqrt(length(i0)) is used to randomly select negative class records
#'
#' @param
#'   nsvm : integer; number of SVMs to be trained
#'
#' @return
#'   rsvm object
#'
#' @examples
#'   dt = iris[1:100,]
#'   dt$Species = ifelse(dt$Species == "setosa", 1, 0)
#'   m = rsvm(dt[,1:4], dt$Species)
#'   predict(m, dt)
#' 
#' @export
#' 
predict.rsvm <- function(model, newx, thresh=.5){
    ids = 1:model$nsvm
    mat = as.data.frame(matrix(0, ncol=model$nsvm, nrow=nrow(newx)))
    mat[,ids] = lapply(ids, function(i){
        as.numeric(as.character(predict(model$rsvm[[i]]$svm, newx=newx)$predictions))
    })
    ifelse(apply(mat, 1, sum)/model$nsvm >= thresh, 1, 0)
}


#' K means clustering based SVM
#' 
#' @export
#' 
kmsvm <- function(x, y, centers){
    i0 = which(y == 0)
    i1 = which(y == 1)
    n0 = length(i1)
    id_learn = c(i0, i1)
    tsne = Rtsne::Rtsne(x[id_learn,], dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)
    df = data.frame(x=tsne$Y[,1], y=tsne$Y[,2])
    ## centers = --centers
    ## centers =   centers
    ## centers = ++centers    
    km0 = kmeans(df, centers)
    km1 = kmeans(df, centers-1)
    km2 = kmeans(df, centers+1)    
    p0 = predict(km0, df)
    p1 = predict(km1, df)
    p2 = predict(km2, df)
    c0 = unique(p0)
    c1 = unique(p1)
    c2 = unique(p2)
    kmsvm0 = list()
    kmsvm0[c0] = lapply(c0, function(cc){
        id_learn = c(i0, i1[p0 == cc])
        LiblineaR::LiblineaR(x[id_learn,], y[id_learn])
    })
    kmsvm1 = list()
    kmsvm1[c1] = lapply(c1, function(cc){
        id_learn = c(i0, i1[p1 == cc])
        LiblineaR::LiblineaR(x[id_learn,], y[id_learn])
    })
    kmsvm2 = list()
    kmsvm2[c2] = lapply(c2, function(cc){
        id_learn = c(i0, i1[p2 == cc])
        LiblineaR::LiblineaR(x[id_learn,], y[id_learn])
    })
    model = list(kmsvm0=kmsvm0, kmsvm1=kmsvm1, kmsvm2=kmsvm2,
                 c0=c0, c1=c1, c2=c2)
    class(model) = c("kmsvm")
    return(model)
}

#' @export
predict.kmsvm <- function(model, newx, opt){
    p0 = list()
    p0[model$c0] = lapply(model$c0, function(i){
        predict(model$kmsvm0[[i]], newx=newx)$predictions
    })
    return(p0)
}

## 
