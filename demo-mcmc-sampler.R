

num_mcmc_steps <- 1000000 # the length of the MCMC chain
num_samples <- 10000      # the length after thinning.


## TODO This is where we compute R0. But the ratio is missing so it
## doesn't really work...
r0 <- function(tau, gamma) {
  (tau / gamma)
}

## This is where we define how good a particular (tau, gamma) pair is
## by measuring how close the resulting R0 of the parameters is to the
## threshold value of 1.0
target_log_density <- function(p, a) {
  param_tau <- p[1]
  param_gamma <- p[2]
  if (param_tau < 0 || param_tau > 1 || param_gamma < 0 || param_gamma > 1) {
    return(-Inf)
  }
  - a * (r0(param_tau, param_gamma) - 1) ^ 2
}

library(mcmc) # popular MCMC library from CRAN.


## Here we use MCMC to sample from the target distribution.
mcmc_sample_from_target <- mcmc::metrop(
  function(theta) target_log_density(theta, a = 10),
  initial = c(0.5, 0.5),
  nbatch = num_mcmc_steps,
  scale = 0.1
)

## Thin out the same because they are auto-correlated so this
## represents a sample from the target distribution.
thinned_samples <- mcmc_sample_from_target$batch[round(seq(1, num_mcmc_steps, length = num_samples)), ]
thinned_samples <- as.data.frame(thinned_samples)
colnames(thinned_samples) <- c("tau", "gamma")

## Write that out to a CSV file so we can use it in the next step.
write.csv(thinned_samples, "adaptive-parameter-samples.csv", row.names = FALSE)


## ---------------------------------------------------------


uniform_sample <- data.frame(
  tau = runif(num_samples, min = 0, max = 1),
  gamma = runif(num_samples, min = 0, max = 1)
)

write.csv(uniform_sample, "uniform-parameter-samples.csv", row.names = FALSE)



## ---------------------------------------------------------
