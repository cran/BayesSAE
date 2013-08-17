samp.model <-
function(y, vardir, innov = "normal", df = NULL, data){
     if (innov != "normal" && innov != "t")
          stop("Sampling model must be with normal or t innovations") 
     if(!missing(data)){
          y <- deparse(substitute(y))
          vardir <- deparse(substitute(vardir))
          y <- data[, y]
          vardir <- data[, vardir]
     }
     m <- length(y)
     if (length(vardir) != m)
          stop("y and varidr must be of the same length")
     if (innov == "t" && any(df <= 0))
          stop("degree of freedom must be positive")
     if (innov == "t" && length(df) != m)
          stop("y and df must be of the same length")
     if (innov == "normal")
          df = 0
     samp <- list(y = y, vardir = vardir, innov = innov, df = df)
}
