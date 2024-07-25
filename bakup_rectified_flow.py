import torch
from torch.distributions import LogisticNormal
from dataclasses import dataclass
import numpy as np
from typing import List, Optional, Tuple, Union

from diffusers.utils import BaseOutput
from diffusers.schedulers import SchedulerMixin, KarrasDiffusionSchedulers
from diffusers.configuration_utils import ConfigMixin, register_to_config
from tqdm import tqdm
from einops import rearrange


@dataclass
class DDPMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None



def timestep_transform(
    t,
    model_kwargs,
    base_resolution=512 * 512,
    base_num_frames=1,
    scale=1.0,
    num_timesteps=1,
):
    # Force fp16 input to fp32 to avoid nan output
    for key in ["height", "width", "num_frames"]:
        if model_kwargs[key].dtype == torch.float16:
            model_kwargs[key] = model_kwargs[key].float()

    t = t / num_timesteps
    resolution = model_kwargs["height"] * model_kwargs["width"]
    ratio_space = (resolution / base_resolution).sqrt()
    # NOTE: currently, we do not take fps into account
    # NOTE: temporal_reduction is hardcoded, this should be equal to the temporal reduction factor of the vae
    if model_kwargs["num_frames"][0] == 1:
        num_frames = torch.ones_like(model_kwargs["num_frames"])
    else:
        num_frames = model_kwargs["num_frames"] // 17 * 5
    ratio_time = (num_frames / base_num_frames).sqrt()

    ratio = ratio_space * ratio_time * scale
    new_t = ratio * t / (1 + (ratio - 1) * t)

    new_t = new_t * num_timesteps
    return new_t


def mean_flat(tensor: torch.Tensor, mask=None):
    """
    Take the mean over all non-batch dimensions.
    """
    assert mask is None
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def _extract_into_tensor(arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: List[int]):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = arr.to(timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)




# From https://github.dev/hpcaitech/Open-Sora
class RFlowScheduler:
    def __init__(
        self,
        num_timesteps=1000,
        num_sampling_steps=10,
        use_discrete_timesteps=False,
        sample_method="uniform",
        loc=0.0,
        scale=1.0,
        use_timestep_transform=False,
        transform_scale=1.0,
    ):
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.use_discrete_timesteps = use_discrete_timesteps
        self.order = 1

        # sample method
        assert sample_method in ["uniform", "logit-normal"]
        assert (
            sample_method == "uniform" or not use_discrete_timesteps
        ), "Only uniform sampling is supported for discrete timesteps"
        self.sample_method = sample_method
        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(x.device)

        # timestep transform
        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale


    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        return sample


    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
        """

        self.num_inference_steps = num_inference_steps

        timesteps = (
            np.linspace(0, self.num_timesteps - 1, num_inference_steps)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        
        timesteps = torch.from_numpy(timesteps).to(device=device)

        self.timesteps = timesteps
            
        
    def training_losses(self, controlnet, unet, x_start, weight_dtype=torch.float32, model_kwargs=None, noise=None, weights=None, t=None):
        """
        Compute training losses for a single timestep.
        Arguments format copied from opensora/schedulers/iddpm/gaussian_diffusion.py/training_losses
        Note: t is int tensor and should be rescaled from [0, num_timesteps-1] to [1,0]
        """
        if t is None:
            if self.use_discrete_timesteps:
                t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
            elif self.sample_method == "uniform":
                t = torch.rand((x_start.shape[0],), device=x_start.device) * self.num_timesteps
            elif self.sample_method == "logit-normal":
                t = self.sample_t(x_start) * self.num_timesteps

            if self.use_timestep_transform:
                t = timestep_transform(t, model_kwargs, scale=self.transform_scale, num_timesteps=self.num_timesteps)
        
        t = t.to(dtype=weight_dtype)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start, device=x_start.device, dtype=weight_dtype)
        assert noise.shape == x_start.shape

        x_t = self.add_noise(x_start, noise, t).to(dtype=weight_dtype)


        down_block_res_samples, mid_block_res_sample = controlnet(
            x_t,
            t,
            **model_kwargs,
            return_dict=False,
        )

        model_pred = unet(
            x_t,
            t,
            encoder_hidden_states=model_kwargs['encoder_hidden_states'],
            down_block_additional_residuals=[
                sample.to(dtype=weight_dtype) for sample in down_block_res_samples
            ],
            mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
        ).sample

        if weights is None:
            loss = mean_flat((model_pred - (x_start - noise)).pow(2))
        else:
            weight = _extract_into_tensor(weights, t, x_start.shape)
            loss = mean_flat(weight * (model_pred - (x_start - noise)).pow(2))

        return loss

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints  # [1,1/1000]

        # timepoint  (bsz) noise: (bsz, 4, w ,h)
        # expand timepoint to noise shape
        timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        timepoints = timepoints.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3])

        return timepoints * original_samples + (1 - timepoints) * noise


    def previous_timestep(self, timestep):

        num_inference_steps = (
            self.num_inference_steps if self.num_inference_steps else self.num_timesteps
        )
        prev_t = timestep - self.num_timesteps // num_inference_steps

        return prev_t    

    def step(
        self,
        model_output: torch.Tensor,  # predicted flow
        timestep: int,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        
        t = timestep
        prev_t = self.previous_timestep(t)

        # please allow me to ignore CFG here!
        # TODO: add consideration for CFGs
        
        all_t = t.float() / self.num_timesteps
        delta_t = (t - prev_t) / self.num_timesteps
        
        v_pred = model_output
        new_sample = sample + delta_t * v_pred
        est_x0 = sample + all_t * v_pred

        return DDPMSchedulerOutput(prev_sample=new_sample, pred_original_sample=est_x0)
