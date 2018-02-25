data {
  int<lower=1> n; // number of reviews
  int<lower=1> p; // number of users
  vector[n] y; // all observed reviews
  int user_idx[n]; // subject observation indicator
}

parameters {
  vector[p] mu_i; // mean for user p
  // vector<lower=0>[p] sigma_i; // variance for user p
  real mu;  // global mean
  real<lower=0> tau2;  // variance between user means
  real<lower=0> sigma2;  // variance within one user's reviews
}

transformed parameters {
  real<lower=0> tau;
  real<lower=0> sigma;

  tau = sqrt(tau2);
  sigma = sqrt(sigma2);
}

model {
  // Prior on global mean: normally distributed, centered around 3 with stddev of 1
  // (so 95% probability in range 1-5)
  mu ~ normal(3, 1);
  // Prior on variance between user means is inverse-gamma:
  tau2 ~ inv_gamma(1, 0.5);

  // Prior on global within-user review variance
  // TODO: Make this hierarchical too
  // TODO: This should probably be slightly more uninformative.
  sigma2 ~ inv_gamma(1, 0.5);

  mu_i ~ normal(mu, tau);

  for (i in 1:n) {
    y[i] ~ normal(mu_i[user_idx[i]], sigma);
  }
}
