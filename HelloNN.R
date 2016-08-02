##################################################################
# This is a toy data set here. We have 50 (x,y) data points.     #
# At first, the data is perfectly linear                         #
##################################################################
num_examples <- 50
x <- seq(-2, 4, length = num_examples)
y <- seq(-6, 6, length = num_examples)
plot(x,y)

# Then we perturb it with noise
x <- x + rnorm(num_examples)
y <- y + rnorm(num_examples)
plot(x,y)

# What we're trying to do is calculate the green line below
polyfit <- lm(y ~ x)
plot(x,y)
lines(x, predict(polyfit, data.frame(x=x)), col = 'green')

loss_polyfit <- sum(polyfit$residuals^2) / 2
weights_polyfit <- polyfit$coefficients

##################
#### Let's go ####
##################
lm_gradiental <- function(x, y, training_steps = 100, learning_rate = 0.002){
  # Set up all the tensors.
  # Add the bias node which always has a value of 1
  x_with_bias <- data.frame(x=x, bias = 1)
  input <- x_with_bias
  
  # Keep track of the loss at each iteration so we can chart it later
  losses <- vector(mode="numeric", length=0)
  
  # Our target is the y values. They need to be massaged to the right shape.
  target = y
  
  # Weights are initialized to random values (gaussian, mean 0, stdev 0.1)
  weights <- data.frame(x=rnorm(1, mean = 0, sd = 0.1), 
                        bias = rnorm(1, mean = 0, sd = 0.1))
  
  for (i in 1:training_steps) {
    # For all x values, generate our estimate on all y given our current
    # weights. So, this is computing y = w2 * x + w1 * bias
    yhat <- data.matrix(input) %*% t(data.matrix(weights)) 
    
    # Compute the error, which is just the difference between our 
    # estimate of y and what y actually is
    yerror <- yhat - target
    
    # We are going to minimize the L2 loss. The L2 loss is the sum of the
    # squared error for all our estimates of y. This penalizes large errors
    # a lot, but small errors only a little.
    loss <- 0.5*sum(yerror^2)
    losses <-c(losses, loss)
    
    # Change the weights by subtracting derivative with respect to that weight
    gradient <- apply(input*yerror, 2, sum) 
    weights <- weights - learning_rate*gradient 
  }
  
  returnList <- list("weights" = weights, "losses" = losses, "yhat"=yhat, "yerror" = yerror)
  
  return(returnList)
  
}
#################################################
#End of function                                #
#################################################

# Give it a try
fitNN <- lm_gradiental(x, y,
                       training_steps = 500, 
                       learning_rate = 0.001)
plot(x,y)
lines(x, fitNN$yhat, col = "blue")
weights_polyfit[c(2,1)]
fitNN$weights 
loss_polyfit
fitNN$losses[length(fitNN$losses)]

#################################################



