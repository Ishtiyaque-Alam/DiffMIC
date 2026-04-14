import math
import torch
import numpy as np

def make_beta_schedule(schedule="linear", num_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == "linear":
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == "const":
        betas = end * torch.ones(num_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, num_timesteps) ** 2
    elif schedule == "jsd":
        betas = 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine" or schedule == "cosine_reverse":
        max_beta = 0.999
        cosine_s = 0.008
        betas = torch.tensor(
            [min(1 - (math.cos(((i + 1) / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2) / (
                    math.cos((i / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2), max_beta) for i in
             range(num_timesteps)])
    elif schedule == "cosine_anneal":
        betas = torch.tensor(
            [start + 0.5 * (end - start) * (1 - math.cos(t / (num_timesteps - 1) * math.pi)) for t in
             range(num_timesteps)])
    return betas


def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


# Forward functions
def q_sample(y, y_0_hat, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, t, noise=None):
    """
    y_0_hat: prediction of pre-trained guidance classifier; can be extended to represent 
        any prior mean setting at timestep T.
    """
    if noise is None:
        noise = torch.randn_like(y).to(y.device)
    sqrt_alpha_bar_t = extract(alphas_bar_sqrt, t, y)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    # q(y_t | y_0, x)
    y_t = sqrt_alpha_bar_t * y + (1 - sqrt_alpha_bar_t) * y_0_hat + sqrt_one_minus_alpha_bar_t * noise

    return y_t


# Reverse function -- sample y_{t-1} given y_t
def p_sample(
    model,
    x,
    y,
    y_0_hat,
    y_T_mean,
    t,
    alphas,
    one_minus_alphas_bar_sqrt,
    y_0_global=None,
    y_0_local=None,
):
    """
    Reverse diffusion process sampling -- one time step.

    y: sampled y at time step t, y_t.
    y_0_hat: prediction of pre-trained guidance model.
    y_T_mean: mean of prior distribution at timestep T.
    We replace y_0_hat with y_T_mean in the forward process posterior mean computation, emphasizing that 
        guidance model prediction y_0_hat = f_phi(x) is part of the input to eps_theta network, while 
        in paper we also choose to set the prior mean at timestep T y_T_mean = f_phi(x).
    """
    device = next(model.parameters()).device
    z = torch.randn_like(y)
    t = torch.tensor([t]).to(device)
    alpha_t = extract(alphas, t, y)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    sqrt_one_minus_alpha_bar_t_m_1 = extract(one_minus_alphas_bar_sqrt, t - 1, y)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    sqrt_alpha_bar_t_m_1 = (1 - sqrt_one_minus_alpha_bar_t_m_1.square()).sqrt()
    # y_t_m_1 posterior mean component coefficients
    gamma_0 = (1 - alpha_t) * sqrt_alpha_bar_t_m_1 / (sqrt_one_minus_alpha_bar_t.square())
    gamma_1 = (sqrt_one_minus_alpha_bar_t_m_1.square()) * (alpha_t.sqrt()) / (sqrt_one_minus_alpha_bar_t.square())
    gamma_2 = 1 + (sqrt_alpha_bar_t - 1) * (alpha_t.sqrt() + sqrt_alpha_bar_t_m_1) / (
        sqrt_one_minus_alpha_bar_t.square())
    eps_theta = model(
        x,
        y,
        t,
        y_0_hat,
        yhat_global=y_0_global,
        yhat_local=y_0_local,
        use_inference_fusion=True,
    ).to(device).detach()
    # y_0 reparameterization
    y_0_reparam = 1 / sqrt_alpha_bar_t * (
            y - (1 - sqrt_alpha_bar_t) * y_T_mean - eps_theta * sqrt_one_minus_alpha_bar_t)
    # posterior mean
    y_t_m_1_hat = gamma_0 * y_0_reparam + gamma_1 * y + gamma_2 * y_T_mean
    # posterior variance
    beta_t_hat = (sqrt_one_minus_alpha_bar_t_m_1.square()) / (sqrt_one_minus_alpha_bar_t.square()) * (1 - alpha_t)
    y_t_m_1 = y_t_m_1_hat.to(device) + beta_t_hat.sqrt().to(device) * z.to(device)
    return y_t_m_1




# Reverse function -- sample y_0 given y_1
def p_sample_t_1to0(
    model,
    x,
    y,
    y_0_hat,
    y_T_mean,
    one_minus_alphas_bar_sqrt,
    y_0_global=None,
    y_0_local=None,
):
    device = next(model.parameters()).device
    t = torch.tensor([0]).to(device)  # corresponding to timestep 1 (i.e., t=1 in diffusion models)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    eps_theta = model(
        x,
        y,
        t,
        y_0_hat,
        yhat_global=y_0_global,
        yhat_local=y_0_local,
        use_inference_fusion=True,
    ).to(device).detach()
    # y_0 reparameterization
    y_0_reparam = 1 / sqrt_alpha_bar_t * (
            y - (1 - sqrt_alpha_bar_t) * y_T_mean - eps_theta * sqrt_one_minus_alpha_bar_t)
    y_t_m_1 = y_0_reparam.to(device)
    return y_t_m_1


def y_0_reparam(model, x, y, y_0_hat, y_T_mean, t, one_minus_alphas_bar_sqrt):
    """
    Obtain y_0 reparameterization from q(y_t | y_0), in which noise term is the eps_theta prediction.
    Algorithm 2 Line 4 in paper.
    """
    device = next(model.parameters()).device
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    eps_theta = model(x, y, t, y_0_hat).to(device).detach()
    # y_0 reparameterization
    y_0_reparam = 1 / sqrt_alpha_bar_t * (
            y - (1 - sqrt_alpha_bar_t) * y_T_mean - eps_theta * sqrt_one_minus_alpha_bar_t).to(device)
    return y_0_reparam


def p_sample_loop(model, x, y_0_hat, y_T_mean, n_steps, alphas, one_minus_alphas_bar_sqrt,
                  only_last_sample=False, y_0_global=None, y_0_local=None):
    num_t, y_p_seq = None, None
    device = next(model.parameters()).device
    z = torch.randn_like(y_T_mean).to(device)
    cur_y = z + y_T_mean  # sampled y_T
    if only_last_sample:
        num_t = 1
    else:
        y_p_seq = [cur_y]
    for t in reversed(range(1, n_steps)):
        y_t = cur_y
        cur_y = p_sample(
            model,
            x,
            y_t,
            y_0_hat,
            y_T_mean,
            t,
            alphas,
            one_minus_alphas_bar_sqrt,
            y_0_global=y_0_global,
            y_0_local=y_0_local,
        )  # y_{t-1}
        if only_last_sample:
            num_t += 1
        else:
            y_p_seq.append(cur_y)
    if only_last_sample:
        assert num_t == n_steps
        y_0 = p_sample_t_1to0(
            model,
            x,
            cur_y,
            y_0_hat,
            y_T_mean,
            one_minus_alphas_bar_sqrt,
            y_0_global=y_0_global,
            y_0_local=y_0_local,
        )
        return y_0
    else:
        assert len(y_p_seq) == n_steps
        y_0 = p_sample_t_1to0(
            model,
            x,
            y_p_seq[-1],
            y_0_hat,
            y_T_mean,
            one_minus_alphas_bar_sqrt,
            y_0_global=y_0_global,
            y_0_local=y_0_local,
        )
        y_p_seq.append(y_0)
        return y_p_seq


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd


# ---------------------------------------------------------------------------
# DPM++ 2M solver using HuggingFace Diffusers
# ---------------------------------------------------------------------------
def dpm_pp_sample_loop(
    model,
    x_img,
    y_0_hat,
    y_0_global,
    y_0_local,
    y_T_mean,
    n_steps,
    beta_start,
    beta_end,
    beta_schedule,
    num_train_timesteps,
    device,
):
    """
    DPM++ 2nd-order multistep reverse sampling (HuggingFace Diffusers).

    Mathematical justification for the heterologous shift:
      DiffMIC forward process:
        y_t = sqrt_abar_t * y_0 + (1 - sqrt_abar_t) * y_T_mean + sigma_t * eps
      Define centred variable:
        x_t = y_t - y_T_mean
             = sqrt_abar_t * (y_0 - y_T_mean) + sigma_t * eps
      => x_t follows *standard* DDPM around x_0 = y_0 - y_T_mean.
      Initial sample: x_T = z ~ N(0,I)  =>  y_T = z + y_T_mean  (same as DDPM loop).
      Model call:  model(x_img, y_t=x_t + y_T_mean, t, y_0_hat) -> eps_theta.
      Recovery:    y_0 = x_0 + y_T_mean.
    """
    from diffusers import DPMSolverMultistepScheduler

    # Map DiffMIC beta schedule names to diffusers convention
    hf_schedule = {
        "linear":          "linear",
        "cosine":          "squaredcos_cap_v2",
        "cosine_reverse":  "squaredcos_cap_v2",
        "cosine_anneal":   "squaredcos_cap_v2",
        "quad":            "linear",
        "jsd":             "linear",
        "sigmoid":         "linear",
        "const":           "linear",
    }.get(beta_schedule, "linear")

    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=hf_schedule,
        solver_order=2,
        prediction_type="epsilon",
        thresholding=False,
    )
    scheduler.set_timesteps(n_steps, device=device)

    # Initialise centred sample: x_T = z ~ N(0, I)
    z = torch.randn_like(y_T_mean).to(device)
    x_centered = z  # shape: (B, n_classes)

    for t in scheduler.timesteps:
        # Expand scalar timestep to batch dimension for the embedding lookup
        t_batch = t.long().unsqueeze(0).expand(x_centered.shape[0]).to(device)
        # Reconstruct heterologous y_t
        y_t = x_centered + y_T_mean
        with torch.no_grad():
            eps_theta = model(
                x_img,
                y_t,
                t_batch,
                y_0_hat,
                yhat_global=y_0_global,
                yhat_local=y_0_local,
                use_inference_fusion=True,
            )
        # DPM++ step on centred variable
        x_centered = scheduler.step(eps_theta, t, x_centered).prev_sample

    # Shift back to label space
    y_0 = x_centered + y_T_mean
    return y_0
