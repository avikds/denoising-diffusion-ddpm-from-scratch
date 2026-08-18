"""
Denoising Diffusion (DDPM) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - linear_beta_schedule
import torch
import torch.nn.functional as F

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    # Return a linear beta schedule of length T
    return torch.linspace(
        beta_start,
        beta_end,
        T,
        dtype=torch.float32
    )

# Step 2 - alphas_from_betas
def alphas_from_betas(betas):
    # Return alpha_t = 1 - beta_t
    return 1.0 - betas

# Step 3 - cumprod_alphas
def cumprod_alphas(alphas):
    # Compute cumulative product of alphas
    return torch.cumprod(alphas, dim=0)

# Step 4 - extract_into_batch
def extract_into_batch(a, t, x):
    # Gather schedule values at the given timesteps
    # and reshape for broadcasting with x
    return a[t].reshape(t.shape[0], 1, 1, 1)

# Step 5 - q_sample
def q_sample(x0, t, noise, alphas_cumprod):
    # Get cumulative alpha values for each timestep
    alpha_bar_t = alphas_cumprod[t].reshape(t.shape[0], 1, 1, 1)

    # Closed-form forward diffusion:
    # x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
    return (
        torch.sqrt(alpha_bar_t) * x0
        + torch.sqrt(1.0 - alpha_bar_t) * noise
    )

# Step 6 - build_diffusion_schedule
def build_diffusion_schedule(
    T: int = 100,
    beta_start: float = 1e-4,
    beta_end: float = 0.02
) -> dict:
    # Linear beta schedule
    betas = torch.linspace(
        beta_start,
        beta_end,
        T,
        dtype=torch.float32
    )

    # Alpha schedule: alpha_t = 1 - beta_t
    alphas = 1.0 - betas

    # Cumulative product: alpha_bar_t = prod(alpha_1, ..., alpha_t)
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # Useful terms for the closed-form forward diffusion
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "T": int(T),
    }

# Step 7 - noise_prediction_loss
def noise_prediction_loss(noise_pred, noise):
    # Mean squared error between predicted and true noise
    return torch.mean((noise - noise_pred) ** 2)

# Step 8 - diffusion_training_loss
def diffusion_training_loss(model, x0, t, noise, alphas_cumprod):
    # Sample the noisy image x_t using the closed-form forward process
    x_t = q_sample(x0, t, noise, alphas_cumprod)

    # Predict the noise added to x0
    noise_pred = model(x_t, t)

    # Compute the simplified DDPM noise-prediction loss
    return noise_prediction_loss(noise_pred, noise)

# Step 9 - timestep_embedding
def timestep_embedding(t, dim: int):
    # Compute half of the embedding dimension for sin/cos pairs
    half = dim // 2

    # Handle the half == 1 case by using exponent 0
    if half == 1:
        exponent = torch.zeros(1, device=t.device, dtype=torch.float32)
    else:
        exponent = torch.arange(
            half,
            device=t.device,
            dtype=torch.float32
        ) / (half - 1)

    # Compute the frequencies
    frequencies = 10000.0 ** exponent

    # Scale each timestep by the corresponding frequency
    angles = t.float().unsqueeze(1) / frequencies.unsqueeze(0)

    # Concatenate sine and cosine components
    return torch.cat([
        torch.sin(angles),
        torch.cos(angles)
    ], dim=1)

# Step 10 - init_tiny_unet
def init_tiny_unet(
    in_ch: int = 1,
    hidden: int = 16,
    time_dim: int = 16,
    seed: int = 0
) -> dict:
    # Set the random seed for reproducible initialization
    torch.manual_seed(seed)

    # Helper to initialize weights from N(0, 0.02^2)
    def init_weight(*shape):
        return torch.randn(*shape, dtype=torch.float32) * 0.02

    # Initialize parameters
    params = {
        "conv_in_w": init_weight(hidden, in_ch, 3, 3),
        "conv_in_b": torch.zeros(hidden, dtype=torch.float32),

        "time_mlp_w": init_weight(hidden, time_dim),
        "time_mlp_b": torch.zeros(hidden, dtype=torch.float32),

        "conv_mid_w": init_weight(hidden, hidden, 3, 3),
        "conv_mid_b": torch.zeros(hidden, dtype=torch.float32),

        "conv_out_w": init_weight(in_ch, hidden, 3, 3),
        "conv_out_b": torch.zeros(in_ch, dtype=torch.float32),
    }

    # Make all parameters trainable
    for name in params:
        params[name].requires_grad_(True)

    return params

# Step 11 - tiny_unet_forward
def tiny_unet_forward(x, t, params: dict):
    # Input convolution
    h = F.conv2d(
        x,
        params["conv_in_w"],
        params["conv_in_b"],
        padding=1
    )

    # Sinusoidal timestep embedding
    temb = timestep_embedding(
        t,
        params["time_mlp_w"].shape[1]
    )

    # Time embedding MLP
    temb = F.linear(
        temb,
        params["time_mlp_w"],
        params["time_mlp_b"]
    )
    temb = F.relu(temb)

    # Add time conditioning to the convolutional features
    h = h + temb[:, :, None, None]

    # Residual denoising layers
    h = F.relu(h)

    h = F.relu(
        F.conv2d(
            h,
            params["conv_mid_w"],
            params["conv_mid_b"],
            padding=1
        )
    )

    # Predict the noise
    return F.conv2d(
        h,
        params["conv_out_w"],
        params["conv_out_b"],
        padding=1
    )

# Step 12 - make_blob_dataset
def make_blob_dataset(n: int = 128, size: int = 8, seed: int = 0):
    # Set seed for reproducibility
    torch.manual_seed(seed)

    radius = size // 4

    # Create an empty dataset
    x = torch.zeros(n, 1, size, size, dtype=torch.float32)

    # Coordinate grids
    yy, xx = torch.meshgrid(
        torch.arange(size),
        torch.arange(size),
        indexing="ij"
    )

    # Place one filled disk in each image
    for i in range(n):
        center = torch.randint(radius, size - radius, (2,))
        cy, cx = center[0], center[1]

        # Filled disk mask
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2

        x[i, 0][mask] = 1.0

    return x

# Step 13 - ddpm_train_step
def ddpm_train_step(
    params: dict,
    x0,
    schedule: dict,
    lr: float = 1e-2,
    seed: int = 0
) -> tuple[dict, float]:
    # Seed RNG for reproducible timestep and noise sampling
    torch.manual_seed(seed)

    B = x0.shape[0]
    T = schedule["T"]

    # Sample random timesteps t ~ Uniform{0, ..., T-1}
    t = torch.randint(
        0,
        T,
        (B,),
        device=x0.device
    )

    # Sample Gaussian noise with the same shape as x0
    noise = torch.randn_like(x0)

    # Compute the DDPM noise-prediction loss
    loss = diffusion_training_loss(
        lambda x, t: tiny_unet_forward(x, t, params),
        x0,
        t,
        noise,
        schedule["alphas_cumprod"]
    )

    # Backpropagate
    loss.backward()

    # Perform one SGD update and detach the updated parameters
    new_params = {}

    for name, p in params.items():
        if p.grad is not None:
            p_new = (p - lr * p.grad).detach().requires_grad_(True)
        else:
            p_new = p.clone()

        new_params[name] = p_new

    return new_params, float(loss)

# Step 14 - train_ddpm
def train_ddpm(
    dataset,
    params: dict,
    schedule: dict,
    num_steps: int = 50,
    batch_size: int = 16,
    lr: float = 1e-2,
    seed: int = 0
) -> tuple[dict, list]:
    # Store the loss from each training step
    history = []

    n = dataset.shape[0]

    for step in range(num_steps):
        # Seed the RNG for reproducible minibatch sampling
        torch.manual_seed(seed + step)

        # Sample minibatch indices
        indices = torch.randint(
            0,
            n,
            (batch_size,),
            device=dataset.device
        )

        x0 = dataset[indices]

        # Perform one DDPM SGD update
        params, loss = ddpm_train_step(
            params,
            x0,
            schedule,
            lr=lr,
            seed=seed + step
        )

        history.append(float(loss))

    return params, history

# Step 15 - predict_x0_from_eps
def predict_x0_from_eps(x_t, t, eps, alphas_cumprod):
    # Extract alpha_bar_t for each sample in the batch
    alpha_bar_t = extract_into_batch(alphas_cumprod, t, x_t)

    # Recover the clean image estimate:
    # x0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
    return (
        x_t - torch.sqrt(1.0 - alpha_bar_t) * eps
    ) / torch.sqrt(alpha_bar_t)

# Step 16 - ddpm_p_mean_variance
def ddpm_p_mean_variance(x_t, t, eps, schedule: dict):
    # Recover and clamp the predicted clean image
    x0_hat = predict_x0_from_eps(
        x_t,
        t,
        eps,
        schedule["alphas_cumprod"]
    )
    x0_hat = x0_hat.clamp(-1.0, 1.0)

    # Extract schedule values for the current timestep
    alphas_t = extract_into_batch(
        schedule["alphas"],
        t,
        x_t
    )
    betas_t = extract_into_batch(
        schedule["betas"],
        t,
        x_t
    )
    alpha_bar_t = extract_into_batch(
        schedule["alphas_cumprod"],
        t,
        x_t
    )

    # alpha_bar_{t-1}, with alpha_bar_{-1} := 1 for t == 0
    alpha_bar_prev = torch.ones_like(alpha_bar_t)

    mask = (t > 0).float().reshape(-1, 1, 1, 1)
    t_prev = torch.clamp(t - 1, min=0)

    alpha_bar_prev_values = schedule["alphas_cumprod"][t_prev]
    alpha_bar_prev = (
        mask * alpha_bar_prev_values.reshape(-1, 1, 1, 1)
        + (1.0 - mask)
    )

    # DDPM posterior mean:
    # mu = [sqrt(alpha_bar_{t-1}) * beta_t / (1-alpha_bar_t)] * x0_hat
    #    + [sqrt(alpha_t) * (1-alpha_bar_{t-1}) / (1-alpha_bar_t)] * x_t
    denom = 1.0 - alpha_bar_t

    coef_x0 = (
        torch.sqrt(alpha_bar_prev)
        * betas_t
        / denom
    )

    coef_xt = (
        torch.sqrt(alphas_t)
        * (1.0 - alpha_bar_prev)
        / denom
    )

    mean = coef_x0 * x0_hat + coef_xt * x_t

    # Simple fixed-variance DDPM choice
    variance = betas_t

    return mean, variance, x0_hat

# Step 17 - ddpm_p_sample
def ddpm_p_sample(x_t, t, params: dict, schedule: dict, noise=None):
    # Predict the noise at the current timestep
    eps = tiny_unet_forward(x_t, t, params)

    # Compute the reverse-process Gaussian parameters
    mean, var, _ = ddpm_p_mean_variance(
        x_t,
        t,
        eps,
        schedule
    )

    # Sample Gaussian noise if none was provided
    if noise is None:
        noise = torch.randn_like(x_t)

    # No noise is added at t == 0 because the final step is deterministic
    nonzero_mask = (t != 0).float().reshape(-1, 1, 1, 1)

    x_prev = mean + torch.sqrt(var) * noise * nonzero_mask

    return x_prev

# Step 18 - ddpm_sample_loop
def ddpm_sample_loop(params: dict, schedule: dict, shape: tuple, seed: int = 0):
    # Seed RNG for reproducible sampling
    torch.manual_seed(seed)

    # Start from pure Gaussian noise
    x = torch.randn(shape)

    T = schedule["T"]
    B = shape[0]

    # Run the reverse diffusion process from T-1 down to 0
    for t in range(T - 1, -1, -1):
        t_batch = torch.full(
            (B,),
            t,
            dtype=torch.long,
            device=x.device
        )

        x = ddpm_p_sample(
            x,
            t_batch,
            params,
            schedule
        )

    return x

# Step 19 - sample_quality_mse (not yet solved)
# TODO: implement

# Step 20 - ddpm_experiment (not yet solved)
# TODO: implement

