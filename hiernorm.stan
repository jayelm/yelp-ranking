data {
  int<lower=1> n; // number of reviews
  int<lower=1> p; // number of users
  vector[n] y; // all observed reviews
  int user_idx[n]; // subject observation indicator
}

parameters {
  real mu;  // global mean
  real<lower=0> tau2;  // variance between user means
  vector[p] mu_i; // mean for user p

  real<lower=0> sigma2;  // global review variance
  // variance between user variances
  // (we approximate this with the non-conjugate
  // normal prior because it's more interpretable)
  real<lower=0> eta2;  // variance between user variances

  vector<lower=0>[p] sigma_i; // review variance for user p
}

transformed parameters {
  real<lower=0> tau;
  real<lower=0> sigma;
  real<lower=0> eta;

  tau = sqrt(tau2);
  sigma = sqrt(sigma2);
  eta = sqrt(eta2);
}

model {
  // Prior on global mean: normally distributed, centered around 3 with stddev of 1
  // (so 95% probability in range 1-5)
  mu ~ normal(3, 1);
  // Prior on variance between user means is inverse-gamma:
  tau2 ~ inv_gamma(1, 0.5);

  // Global review variance mean
  sigma2 ~ inv_gamma(1, 0.5);
  eta2 ~ inv_gamma(1, 0.5);

  mu_i ~ normal(mu, tau);
  sigma_i ~ normal(sigma2, eta);

  for (i in 1:n) {
    y[i] ~ normal(mu_i[user_idx[i]], sigma_i[user_idx[i]]);
  }
}
