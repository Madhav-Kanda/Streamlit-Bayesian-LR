import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from sklearn.datasets import make_regression
from jax import random
import streamlit as st


# Define the model
def linear_regression(X, y, alpha_prior, beta_prior, sigma_prior):
    alpha = numpyro.sample('alpha', alpha_prior)
    beta = numpyro.sample('beta', beta_prior)
    sigma = numpyro.sample('sigma', sigma_prior)
    mean = alpha + beta * X
    numpyro.sample('obs', dist.Normal(mean, sigma), obs=y)


def run_linear_regression(X, y, alpha_prior, beta_prior, sigma_prior):
    # Run MCMC
    rng_key = random.PRNGKey(0)
    nuts_kernel = NUTS(linear_regression)
    mcmc = MCMC(nuts_kernel, num_warmup=50, num_samples=1000)
    mcmc.run(rng_key, jnp.array(X), jnp.array(y), alpha_prior=alpha_prior, beta_prior=beta_prior, sigma_prior=sigma_prior)

    mcmc.print_summary()

    # Get posterior samples
    samples = mcmc.get_samples()

    # Plot the results
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].hist(samples['alpha'], bins=20, density=True)
    ax[0].set_title('alpha')
    ax[1].hist(samples['beta'], bins=20, density=True)
    ax[1].set_title('beta')
    ax[2].hist(samples['sigma'], bins=20, density=True)
    ax[2].set_title('sigma')
    st.write("The plot of posterior samples is shown below:")
    st.pyplot(fig)
    st.write("The mean of alpha is", np.mean(samples['alpha']))
    st.write("The mean of beta is", np.mean(samples['beta']))
    st.write("The mean of sigma is", np.mean(samples['sigma']))

    st.write("The plot of predicted line is shown below using mean values of alpha and beta (Xβ+α):")
    

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X, y, color='blue', alpha=0.5, label='data')
    light_color = (1.0, 0.5, 0.5, 0.7)
    for i in range(499):
        alpha_i = samples['alpha'][i]
        beta_i = samples['beta'][i]
        ax.plot(X, alpha_i + beta_i * X, color=light_color)
    alpha_i = samples['alpha'][498]
    beta_i = samples['beta'][498]
    ax.plot(X, alpha_i + beta_i * X, color=light_color,label='MCMC samples')
    ax.plot(X, np.mean(samples['alpha']) + np.mean(samples['beta']) * X, color='red', label='mean')
    ax.legend(loc='upper left')
    st.pyplot(fig)

# User Input
st.write("# Bayesian Linear Regression")


""" Prior: p(α) = N(μ0,Σ0), p(β) = N(μ1,Σ1), p(σ) = N(Σ2)"""
""" Likelihood: p(y|X,β,α,σ) = N(Xβ+α,σ)"""
""" Posterior: p(α|X,y) ∝ p(y|X,β,σ)p(α), p(β|X,y) ∝ p(y|X,α,σ)p(β), p(σ|X,y) ∝ p(y|X,β,α)p(σ)"""



alpha_prior_option = st.selectbox("Choose an option for alpha prior:", ["Normal", "Laplace", "Cauchy"])
if alpha_prior_option == "Normal":
    alpha_loc = st.slider("Select a mean value for prior of alpha(α) (μ0)", -10.0, 10.0, 0.0, 0.1)
    alpha_scale = st.slider("Select a standard deviation value for prior of alpha(α) (Σ0)", 0.01, 10.0, 1.0, 0.1)
    alpha_prior = dist.Normal(alpha_loc, alpha_scale)
elif alpha_prior_option == "Laplace":
    alpha_loc = st.slider("Select a mean value for prior of alpha(α) (μ0)", -10.0, 10.0, 0.0, 0.1)
    alpha_scale = st.slider("Select a standard deviation value for prior of alpha(α) (Σ0)", 0.01, 10.0, 1.0, 0.1)
    alpha_prior = dist.Laplace(alpha_loc, alpha_scale)
elif alpha_prior_option == "Cauchy":
    alpha_loc = st.slider("Select a mean value for prior of alpha(α) (μ0)", -10.0, 10.0, 0.0, 0.1)
    alpha_scale = st.slider("Select a standard deviation value for prior of alpha(α) (Σ0)", 0.01, 10.0, 1.0, 0.1)
    alpha_prior = dist.Cauchy(alpha_loc, alpha_scale)

beta_prior_option = st.selectbox("Choose an option for beta prior:", ["Normal", "Laplace", "Cauchy"])
if beta_prior_option == "Normal":
    beta_loc = st.slider("Select a mean value for prior of beta(β) (μ1)", -10.0, 10.0, 0.0, 0.1)
    beta_scale = st.slider("Select a standard deviation value for prior of beta(β) (Σ1)", 0.01, 10.0, 1.0, 0.1)
    beta_prior = dist.Normal(beta_loc, beta_scale)
elif beta_prior_option == "Laplace":
    beta_loc = st.slider("Select a mean value for prior of beta(β) (μ1)", -10.0, 10.0, 0.0, 0.1)
    beta_scale = st.slider("Select a standard deviation value for prior of beta(β) (Σ1)", 0.01, 10.0, 1.0, 0.1)
    beta_prior = dist.Laplace(beta_loc, beta_scale)
elif beta_prior_option == "Cauchy":
    beta_loc = st.slider("Select a mean value for prior of beta(β) (μ1)", -10.0, 10.0, 0.0, 0.1)
    beta_scale = st.slider("Select a standard deviation value for prior of beta(β) (Σ1)", 0.01, 10.0, 1.0, 0.1)
    beta_prior = dist.Cauchy(beta_loc, beta_scale)

sigma_prior_option = st.selectbox("Choose an option for sigma prior:", ["HalfNormal", "HalfCauchy"])
if sigma_prior_option == "HalfNormal":
    sigma_scale = st.slider("Select a scale value for prior of sigma(σ) (Σ2)", 0.01, 10.0, 1.0, 0.1)
    sigma_prior = dist.HalfNormal(sigma_scale)
elif sigma_prior_option == "HalfCauchy":
    sigma_scale = st.slider("Select a scale value for prior of sigma(σ) (Σ2)", 0.01, 10.0, 1.0, 0.1)
    sigma_prior = dist.HalfCauchy(sigma_scale)

rng_key = random.PRNGKey(0)
X, y = make_regression(n_samples=50, n_features=1, noise=10.0, random_state=0)
X = X.reshape(50)

if alpha_prior and beta_prior and sigma_prior:
    run_linear_regression(X, y, alpha_prior, beta_prior, sigma_prior)