link.model <-
function(formula, weights, spatial = FALSE, tran = "F", data, prox = 0){
     if (tran != "F" && tran != "log" && tran != "logit")
          stop("Only log or logit transformations are allowed")
     if (spatial && tran != "F")
          stop("Unmatched spatial model is not currently supported")
     if (!missing(data)){
          formuladata <- model.frame(formula, na.action = na.omit, data)
          X <- model.matrix(formula, data)
     }
     else{
          formuladata <- model.frame(formula, na.action = na.omit)
          X <- model.matrix(formula)
     }
     if (missing(weights))
          weights <- rep(1, nrow(X))
     if (nrow(X) <= ncol(X) || det(crossprod(X, X)) == 0)
          stop("the design matrix is required to be column full rank")
     if (nrow(X) != length(weights))
          stop("number of rows of X and length of weights are not the same")
     if (spatial){
          if (ncol(prox) != 2)
               stop("prox must have 2 columns")
          l <- nrow(prox)
          li1 <- as.integer(prox)[1:l]
          li2 <- as.integer(prox)[(1:l)+l]
          if (any(prox - cbind(li1, li2) != 0))
               stop("elements in prox shoul be integers")
          if (any((li1 - li2) == 0))
               stop("elements in the same row of prox are required not to be the same")
          prox[li1 > li2, 1] = li2[li1 > li2]
          prox[li1 > li2, 2] = li1[li1 > li2]
          prox = unique(prox) 
          m <- nrow(X)
          if (max(c(li1, li2)) > m || min(c(li1, li2)) < 1)
               stop("elements in prox must be bounded by 1 and number of areas")
     }
     else
          prox = as.matrix(0)
     link <- list(X = X, b = weights, spatial = spatial, tran = tran, prox = prox)
}
