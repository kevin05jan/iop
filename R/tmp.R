
### TODO
### dict2xnum : extract based on current variable classes
### dict2xcat : extract based on current variable classes





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

#' Box-Cox transformation
#'
#' @param
#'   x : data.frame
#'
#' @return
#'   num2trafo object
#' 
#' @examples
#'   m = num2trafo(iris[,1:2])
#'   predict(m, iris[,1:4])
#' 
#' @export
#' 
num2trafo <- function(x, ...){
    labs = colnames(x)
    lut = lapply(labs, function(lab){ car::powerTransform(x[[lab]]) })
    names(lut) = labs
    model = list(lut=lut)
    class(model) = c("num2trafo")
    return(model)
}

#' Box-Cox transformation
#'
#' @param
#'   model : num2trafo object
#'
#' @param
#'   newx : data.frame
#' 
#' @return
#'   data.frame;  dim is preserved
#'
#' @examples
#'   m = num2trafo(iris[,1:2])
#'   predict(m, iris[,1:4])
#'
#' @export
#' 
predict.num2trafo <- function(model, newx, ...){
    cols = intersect(colnames(newx), names(model$lut))
    newx[,cols] = lapply(cols, function(col){
        car::bcPower(newx[[col]], model$lut[[col]]$lambda)
    })
    return(newx)
}


#' Continuous variable scaling
#'
#' @param
#'   x : data.frame with independent variables of type continous
#'
#' @param
#'   method : string = ["minmax", "znorm"]
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
num2norm <- function(x, method="minmax", ...){
    labs = colnames(x)
    lut = lapply(labs, function(lab){ norm(x[[lab]], method, ...) })
    names(lut) = labs
    model = list(lut=lut)
    class(model) = c("num2norm")
    return(model)
}                 

norm <- function(X, method="minmax"){
    v = na.omit(X)
    list(method=method, min=min(v), max=max(v), mean=mean(v), sd=sd(v))
}

#' Continuous variable scaling
#'
#' @param
#'   model : num2norm object
#'
#' @param
#'   newx : data.frame
#'
#' @return
#'   data.frame;  dim is preserved
#'
#' @examples
#'   m = num2norm(iris[,1:2])
#'   predict(m, iris[,1:4])
#' 
#' @export
#' 
predict.num2norm <- function(model, newx, method=NULL, ...){
    cols = intersect(colnames(newx), names(model$lut))
    newx[,cols] = lapply(cols, function(col){
        vals = model$lut[[col]]
        switch(ifelse(is.null(method), vals$method, method),
               minmax = { (newx[[col]]-vals$min)/(vals$max-vals$min) },
               zscore = { (newx[[col]]-vals$mean)/(vals$sd) }
               )
    })
    return(newx)
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
        model = data.frame(var=var, class=class, name=name, stringsAsFactors=F)
        if(!is.null(file)){
            write.table(model, file=file, quote=F, sep=sep, row.names=F)
        }
        class(model) = c("data2dict")        
        return(model)
    }
    if(is.null(x) && !is.null(file) && file.exists(file)){
        model = read.table(file, sep=sep, stringsAsFactors=F, header=T)
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



num2interaction <- function(x, ...){
    labs = colnames(x)
    comb = combn(labs,m=2)
    combs = 1:ncol(comb)
    newx = list()
    newx[combs] = lapply(combs, function(i){
        a = comb[1,i]
        b = comb[2,i]
        x[[a]] * x[[b]]
    })
    name = sapply(combs, function(i){
        a = comb[1,i]
        b = comb[2,i]
        paste0(a,"_x_",b)
    })
    return(as.data.frame(newx, col.names=name))
}




num2interactionT <- function(x, t, ...){
    labs = colnames(x)
    newx = list()
    newx[labs] = lapply(labs, function(i){
        t * x[[i]]
    })
    name = paste0("t_x_",labs)
    return(as.data.frame(newx, col.names=name))
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
    dens = density(X, na.rm=T)
    csum = cumsum(dens$y)
    cuts = suppressWarnings(Hmisc::cut2(csum, g=bins, onlycuts=T))
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
    pca = prcomp(x, center=T, scale=T)
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
        NA.RM = ifelse(rel >= ratio, T, F)
        nzv = ifelse(is.numeric(x[[xx]]),
                     var(x[[xx]], na.rm=NA.RM),
                     ifelse(length(table(x[[xx]]) > 1, 1, 0))
                     )
    })
    lut = na.omit(cols[nzv < thresh])
    model = list(lut=lut)
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
    lut = cols[apply(tmp, 2, function(xx){ any(xx > thresh) })]
    model = list(lut=lut)
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




## [7]
## Feature construction
##   Creating new features either by using typical techniques,
##   such as polynomial expansion (by using univariate
##   mathematical functions), or feature corssing (to
##   capture feature interactions). Features can also be
##   constructed by using business logic from the domain
##   of the ML use case

#' @export
data2prep <- function(x){
    model = x
    class(model) = c("data2prep")
    return(model)
}

#' @export
predict.data2prep <- function(model, newx){
    nmod = length(model)
    x = newx
    for(i in 1:length(model)){
        x <<- predict(model[[i]], x)
    }
    return(x)
}


#' Calibration of a data.frame
#'
#' @param models A list of models
#' @param append=true Boolean indicating if the output of each model should be appended to the right of the data.frame
#'   If append=false the operation of each model is on the output of the previous one.
#' @return A calibrated data.frame
#'
#' @examples
#'   n = 100
#'   rows = sample(1:nrow(iris), n)
#'   train = iris[n,]
#'   sel = 1:30
#'   calib = train[sel,]
#'   learn = train[-sel,]
#'   m1 = conti2bins(calib[,1:4])
#'   m2 = conti2norm(calib[,1:4])
#'   M = list()
#'   M[1] = m1
#'   M[2] = m2
#'   learn = calib(learn, M, append=T)
#'

calib <- function(x, models=list(0), append=T){
    if(append){
        out = list()
        iter = 1:length(models)
        out[[iter]] = sapply(iter, function(i){
            model = models[[i]]
            predict(x, model)
        })
    }else{
    }
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
        best = which(hc.scores == max(hc.scores, na.rm = T), arr.ind=T)[1,]
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


#' Extract time related features
#'
#' @param x A vector with timestamps of type POSIXct
#' @param y A vector with group label
#'
#' @return A data.frame with features
#'
#' @export
tfeat <- function(x, width){
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



##
    ## NBTree
    ##
    ## - Decision tree-based paritioning of the data (unbalanced)
    ## - No interaction data (needed)
    ## - Data subsets at each leaf node is a Naive Bayes Classifer
    ## - Prediction:
    ##     if DT == 0 -> 0
    ##     if NB == 1 -> 1
    ##     else       -> 0
    ##
    ## i = 5
    ## .ym = ym[i]
    ## y = data[[id_y]]
    ## i0 = which(ymx == ym[i] & y == 0)
    ## i1 = which(ymx <= ym[i] & y == 1)
    ## id_learn = c(i0, i1)
    ## LEARN = data[id_learn,]

data2nbtree <- function(x, y, ...){
    f = iop::data2formula(dict, CALIB, varx="xnum")
    loss = matrix(c(0, 1, 5, 0), ncol=2, byrow=T)
    library(partykit)
    dt = rpart::rpart(f, data=LEARN, parms=list(loss=loss, split="information"))
    fit = fitted(as.party(dt))
    id_node = fit[,1]
    nodes = unique(id_node)
    id_xnum = iop::dict2xnum(dict, CALIB)
    NBTree = list()
    tt = table(LEARN[[id_y]])
    priors = as.numeric(tt/sum(tt))
    tmp = lapply(nodes, function(node){
        sel = id_node == node
        x = data[sel,id_xnum]
        x[is.na(x)] = 0
        y = data[[id_y]][sel]
        NBTree[[node]] <<- fastNaiveBayes::fastNaiveBayes(x=x, y=y, priors=priors)
    })
    model = list(nbtree=NBTree)
    class(model) = c("data2nbtree")
    return(model)
}


    ## y = data[[id_y]]
    ## i0 = which(ymx == ym[i+1] & y == 0)
    ## i1 = which(ymx <= ym[i+1] & y == 1)
    ## n0 = length(i1)
    ## id_eval = c(sample(i0, n0), i1)
    ## EVAL = data[id_eval,]
    ## dim(EVAL)

predict.data2nbtree <- function(model, newx, ...){
    library(fastNaiveBayes)
    p_node = predict(as.party(dt), newdata=EVAL[,id_xnum], type="node")
    p_nb = rep(NA,length(p_node))
    p_nb = list()
    p_nb[nodes] = lapply(nodes, function(node){
        predict(NBTree[[node]], newdata=EVAL[,id_xnum], type="class")
    })
    p_dt = as.character(predict(dt, newdata=EVAL[,id_xnum], type="class"))
    p_nbtree = as.character(sapply(1:length(p_node), function(i){
        p_nb[[p_node[i]]][i]
    }))
    p = ifelse(p_dt == 0, p_dt, ifelse(p_nbtree == 1, p_nbtree, 0))
    return(p)
}


#' Feature selection using the wrapper method
#'
#' @param
#'   x : data.frame with continuous variables
#'
#' @param
#'   y = vector with binary class labels: {0, 1}
#'
#' @return
#'   num2sel object
#'
#' @examples
#'   x = iris[1:100,1:4]
#'   y = ifelse(iris[1:100, 5] == "setosa", 1, 0)
#'   m = num2sel(x, y)
#'   newx = predict(m, x)
#' 
#' @export
#' 
num2sel <- function(x, y, run=10){
    ## fitness function
    fit <- function(vars, x, y){
        i0 = which(y == 0)
        i1 = which(y == 1)
        r0 = sample(i0, as.integer(length(i0)/2))
        r1 = sample(i1, as.integer(length(i1)/2))
        id_learn = c(r0, r1)
        id_eval = -c(r0, r1)
        svm = LiblineaR::LiblineaR(data=x[id_learn,vars],target=y[id_learn])
        p = predict(svm, newx=x[id_eval,vars])$predictions
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
    ga = GA::ga(fitness=function(vars){ fit(vars=vars, x=x, y=y) },
                type="binary",
                elitism=3,
                pmutation=.5,
                popSize=10,
                nBits=ncol(x),
                names=colnames(x),
                run=run,
                maxiter=10,
                monitor=plot,
                keepBest=T,
                parallel=F,
                seed=84211)
    sel = which(as.integer(ga@bestSol[[run]]) == 1)
    cols = colnames(x)[sel]
    model = list(cols=cols)
    class(model) = c("num2sel")
    return(model)
}

#' Feature selection using the wrapper method
#' 
#' @param
#'   model : num2sel object
#'
#' @param
#'   newx : data.frame
#'
#' @examples
#'   x = iris[1:100,1:4]
#'   y = ifelse(iris[1:100, 5] == "setosa", 1, 0)
#'   m = num2sel(x, y)
#'   newx = predict(m, x)#'
#' 
#' @return
#'   data.frame; the number of columns is reduced
#' 
#' @export
#' 
predict.num2sel <- function(model, newx, ...){
    cols = intersect(colnames(newx), names(model$cols))
    return(newx[,cols])
}
