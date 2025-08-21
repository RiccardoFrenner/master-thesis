from collections.abc import Sequence

import keras

from bayesflow.distributions import Distribution
from bayesflow.types import Shape, Tensor
from bayesflow.utils import (
    expand_right_as,
    find_network,
    integrate,
    jacobian_trace,
    layer_kwargs,
    optimal_transport,
    weighted_mean,
    tensor_utils,
)
from bayesflow.utils.serialization import serialize, deserialize, serializable
from ..inference_network import InferenceNetwork


@serializable("bayesflow.networks")
class FlowMatching(InferenceNetwork):
    """(IN) Implements Optimal Transport Flow Matching, originally introduced as Rectified Flow, with ideas incorporated
    from [1-3].

    [1] Rectified Flow: arXiv:2209.03003
    [2] Flow Matching: arXiv:2210.02747
    [3] Optimal Transport Flow Matching: arXiv:2302.00482
    """

    MLP_DEFAULT_CONFIG = {
        "widths": (256, 256, 256, 256, 256),
        "activation": "mish",
        "kernel_initializer": "he_normal",
        "residual": True,
        "dropout": 0.05,
        "spectral_normalization": False,
    }

    OPTIMAL_TRANSPORT_DEFAULT_CONFIG = {
        "method": "log_sinkhorn",
        "regularization": 0.1,
        "max_steps": 100,
        "atol": 1e-5,
        "rtol": 1e-4,
    }

    INTEGRATE_DEFAULT_CONFIG = {
        "method": "euler",
        "steps": 100,
    }

    def __init__(
        self,
        subnet: str | type | keras.Layer = "mlp",
        base_distribution: str | Distribution = "normal",
        use_optimal_transport: bool = False,
        loss_fn: str | keras.Loss = "mse",
        integrate_kwargs: dict[str, any] = None,
        optimal_transport_kwargs: dict[str, any] = None,
        subnet_kwargs: dict[str, any] = None,
        high_fidelity_data=None,
        **kwargs,
    ):
        """
        Initializes a flow-based model with configurable subnet architecture, loss function, and optional optimal
        transport integration.

        This model learns a transformation from a base distribution to a target distribution using a specified subnet
        type, which can be an MLP or a custom network. It supports flow matching with optional optimal transport for
        improved sample efficiency.

        The integration and transport steps can be customized with additional parameters available in the respective
        configuration dictionaries.

        Parameters
        ----------
        subnet : str or keras.Layer, optional
            Architecture for the transformation network. Can be "mlp", a custom network class, or
            a Layer object, e.g., `bayesflow.networks.MLP(widths=[32, 32])`. Default is "mlp".
        base_distribution : str, optional
            The base probability distribution from which samples are drawn, such as "normal".
            Default is "normal".
        use_optimal_transport : bool, optional
            Whether to apply optimal transport for improved training stability. Default is False.
            Note: this will increase training time by approximately ~2.5 times, but may lead to faster inference.
        loss_fn : str, optional
            The loss function used for training, such as "mse". Default is "mse".
        integrate_kwargs : dict[str, any], optional
            Additional keyword arguments for the integration process. Default is None.
        optimal_transport_kwargs : dict[str, any], optional
            Additional keyword arguments for configuring optimal transport. Default is None.
        subnet_kwargs: dict[str, any], optional, deprecated
            Keyword arguments passed to the subnet constructor or used to update the default MLP settings.
        concatenate_subnet_input: bool, optional
            Flag for advanced users to control whether all inputs to the subnet should be concatenated
            into a single vector or passed as separate arguments. If set to False, the subnet
            must accept three separate inputs: 'x' (noisy parameters), 't' (time),
            and optional 'conditions'. Default is True.
        **kwargs
            Additional keyword arguments passed to the subnet and other components.
        """
        super().__init__(base_distribution, **kwargs)

        self.high_fidelity_data = high_fidelity_data
        self.use_optimal_transport = use_optimal_transport

        self.integrate_kwargs = FlowMatching.INTEGRATE_DEFAULT_CONFIG | (integrate_kwargs or {})
        self.optimal_transport_kwargs = FlowMatching.OPTIMAL_TRANSPORT_DEFAULT_CONFIG | (optimal_transport_kwargs or {})

        self.loss_fn = keras.losses.get(loss_fn)

        self.seed_generator = keras.random.SeedGenerator()

        subnet_kwargs = subnet_kwargs or {}
        if subnet == "mlp":
            subnet_kwargs = FlowMatching.MLP_DEFAULT_CONFIG | subnet_kwargs
        self._concatenate_subnet_input = kwargs.get("concatenate_subnet_input", True)

        self.subnet = find_network(subnet, **subnet_kwargs)
        self.output_projector = keras.layers.Dense(units=None, bias_initializer="zeros", name="output_projector")

    def build(self, xz_shape: Shape, conditions_shape: Shape = None) -> None:
        if self.built:
            # building when the network is already built can cause issues with serialization
            # see https://github.com/keras-team/keras/issues/21147
            return

        self.base_distribution.build(xz_shape)

        self.output_projector.units = xz_shape[-1]

        input_shape = list(xz_shape)
        if self._concatenate_subnet_input:
            # construct time vector
            input_shape[-1] += 1
            if conditions_shape is not None:
                input_shape[-1] += conditions_shape[-1]
            input_shape = tuple(input_shape)

            self.subnet.build(input_shape)
            out_shape = self.subnet.compute_output_shape(input_shape)
        else:
            # Multiple separate inputs
            time_shape = tuple(xz_shape[:-1]) + (1,)  # same batch/sequence dims, 1 feature
            self.subnet.build(x_shape=xz_shape, t_shape=time_shape, conditions_shape=conditions_shape)
            out_shape = self.subnet.compute_output_shape(
                x_shape=xz_shape, t_shape=time_shape, conditions_shape=conditions_shape
            )

        self.output_projector.build(out_shape)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
            "subnet": self.subnet,
            "base_distribution": self.base_distribution,
            "use_optimal_transport": self.use_optimal_transport,
            "loss_fn": self.loss_fn,
            "integrate_kwargs": self.integrate_kwargs,
            "optimal_transport_kwargs": self.optimal_transport_kwargs,
            "concatenate_subnet_input": self._concatenate_subnet_input,
            # we do not need to store subnet_kwargs
        }

        return base_config | serialize(config)

    def _apply_subnet(
        self, x: Tensor, t: Tensor, conditions: Tensor = None, training: bool = False
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """
        Prepares and passes the input to the subnet either by concatenating the latent variable `x`,
        the time `t`, and optional conditions or by returning them separately.

        Parameters
        ----------
        x : Tensor
            The parameter tensor, typically of shape (..., D), but can vary.
        t : Tensor
            The time tensor, typically of shape (..., 1).
        conditions : Tensor, optional
            The optional conditioning tensor (e.g. parameters).
        training : bool, optional
            The training mode flag, which can be used to control behavior during training.

        Returns
        -------
        Tensor
            The output tensor from the subnet.
        """
        if self._concatenate_subnet_input:
            t = keras.ops.broadcast_to(t, keras.ops.shape(x)[:-1] + (1,))
            xtc = tensor_utils.concatenate_valid([x, t, conditions], axis=-1)
            return self.subnet(xtc, training=training)
        else:
            if training is False:
                t = keras.ops.broadcast_to(t, keras.ops.shape(x)[:-1] + (1,))
            return self.subnet(x=x, t=t, conditions=conditions, training=training)

    def velocity(self, xz: Tensor, time: float | Tensor, conditions: Tensor = None, training: bool = False) -> Tensor:
        time = keras.ops.convert_to_tensor(time, dtype=keras.ops.dtype(xz))
        time = expand_right_as(time, xz)

        subnet_out = self._apply_subnet(xz, time, conditions, training=training)
        return self.output_projector(subnet_out, training=training)

    def _velocity_trace(
        self, xz: Tensor, time: Tensor, conditions: Tensor = None, max_steps: int = None, training: bool = False
    ) -> (Tensor, Tensor):
        def f(x):
            return self.velocity(x, time=time, conditions=conditions, training=training)

        v, trace = jacobian_trace(f, xz, max_steps=max_steps, seed=self.seed_generator, return_output=True)

        return v, keras.ops.expand_dims(trace, axis=-1)

    def _forward(
        self, x: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        if density:

            def deltas(time, xz):
                v, trace = self._velocity_trace(xz, time=time, conditions=conditions, training=training)
                return {"xz": v, "trace": trace}

            state = {"xz": x, "trace": keras.ops.zeros(keras.ops.shape(x)[:-1] + (1,), dtype=keras.ops.dtype(x))}
            state = integrate(deltas, state, start_time=1.0, stop_time=0.0, **(self.integrate_kwargs | kwargs))

            z = state["xz"]
            log_density = self.base_distribution.log_prob(z) + keras.ops.squeeze(state["trace"], axis=-1)

            return z, log_density

        def deltas(time, xz):
            return {"xz": self.velocity(xz, time=time, conditions=conditions, training=training)}

        state = {"xz": x}
        state = integrate(deltas, state, start_time=1.0, stop_time=0.0, **(self.integrate_kwargs | kwargs))

        z = state["xz"]

        return z

    def _inverse(
        self, z: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        if density:

            def deltas(time, xz):
                v, trace = self._velocity_trace(xz, time=time, conditions=conditions, training=training)
                return {"xz": v, "trace": trace}

            state = {"xz": z, "trace": keras.ops.zeros(keras.ops.shape(z)[:-1] + (1,), dtype=keras.ops.dtype(z))}
            state = integrate(deltas, state, start_time=0.0, stop_time=1.0, **(self.integrate_kwargs | kwargs))

            x = state["xz"]
            log_density = self.base_distribution.log_prob(z) - keras.ops.squeeze(state["trace"], axis=-1)

            return x, log_density

        def deltas(time, xz):
            return {"xz": self.velocity(xz, time=time, conditions=conditions, training=training)}

        state = {"xz": z}
        state = integrate(deltas, state, start_time=0.0, stop_time=1.0, **(self.integrate_kwargs | kwargs))

        x = state["xz"]

        return x

    def compute_metrics(
        self,
        x: Tensor | Sequence[Tensor, ...],
        conditions: Tensor = None,
        sample_weight: Tensor = None,
        stage: str = "training",
    ) -> dict[str, Tensor]:
        if isinstance(x, Sequence):
            # already pre-configured
            x0, x1, t, x, target_velocity = x
        else:
            # not pre-configured, resample
            x1 = x
            if not self.built:
                xz_shape = keras.ops.shape(x1)
                conditions_shape = None if conditions is None else keras.ops.shape(conditions)
                self.build(xz_shape, conditions_shape)
            x0 = self.base_distribution.sample(keras.ops.shape(x1)[:-1])

            if self.use_optimal_transport:
                # we must choose between resampling x0 or x1
                # since the data is possibly noisy and may contain outliers, it is better
                # to possibly drop some samples from x1 than from x0
                # in the marginal over multiple batches, this is not a problem
                x0, x1, assignments = optimal_transport(
                    x0,
                    x1,
                    seed=self.seed_generator,
                    **self.optimal_transport_kwargs,
                    return_assignments=True,
                )
                if conditions is not None:
                    # conditions must be resampled along with x1
                    conditions = keras.ops.take(conditions, assignments, axis=0)

            t = keras.random.uniform((keras.ops.shape(x0)[0],), seed=self.seed_generator)
            t = expand_right_as(t, x0)

            x = t * x1 + (1 - t) * x0
            target_velocity = x1 - x0

        base_metrics = super().compute_metrics(x1, conditions=conditions, stage=stage)

        # print("<>"*40)
        # print(f"{x.shape=}")
        # print(f"{t.shape=}")
        # print(f"{conditions.shape=}")
        # print("<>"*40)
        predicted_velocity = self.velocity(x, time=t, conditions=conditions, training=stage == "training")

        loss = self.loss_fn(target_velocity, predicted_velocity)
        loss = weighted_mean(loss, sample_weight)

        if self.high_fidelity_data is not None:
            loss_taylor = _compute_taylor_loss(self.velocity, self.summary_net, self.high_fidelity_data, self.loss_fn, self.base_distribution.sample, self.seed_generator, stage)
            loss += keras.ops.mean(loss_taylor)

        return base_metrics | {"loss": loss}


def _vel_fields(model, x_1, y, x_0_sampler, seed_generator, stage):
    x_0 = x_0_sampler(keras.ops.shape(x_1)[:-1])
    t = keras.random.uniform((keras.ops.shape(x_0)[0],), seed=seed_generator)
    t = expand_right_as(t, x_0)
    x_t = (1 - t) * x_0 + t * x_1
    dx_t = x_1 - x_0
    # print(x_0.shape)
    # print(x_1.shape)
    # print(t.shape)
    # print(x_t.shape)
    # print(dx_t.shape)
    # print(y.shape)
    v_pred = model(x_t, t, y, training=stage == "training")
    # print(v_pred.shape)
    return (v_pred, dx_t), (t, x_t, x_0)


def _compute_taylor_loss(vel_net, summary_net, high_fidelity_data, loss_fn, x_0_sampler, seed_generator, stage):
    def actual_model(x, t, y, **kwargs):
        return vel_net(x, time=t, conditions=summary_net(y), **kwargs)

    x_true, y_true, delta_y = high_fidelity_data
    (v_pred, dx_t), (t, x_t, _) = _vel_fields(actual_model, x_true, y_true, x_0_sampler,seed_generator, stage)


    # Batched JVP: J_{v,y}(xt, y_s_true) * delta_y
    func_for_jvp = lambda y_arg_batch: actual_model(x_t, t, y_arg_batch, training=stage == "training")
    # print(f"{y_true.shape=}")
    # print(f"{delta_y.shape=}")
    _, jvp_val = jvp_keras(func_for_jvp, (y_true,), (delta_y,))
    # print(f"{jvp_val.shape=}")
    # print("z"*500)
    v_pred_corrected = v_pred + jvp_val  # v_t + JVP term

    return loss_fn(dx_t, v_pred_corrected)


import keras
import torch
import tensorflow as tf
import jax
import jax.numpy as jnp
import numpy as np

def jvp_keras(func, primals, tangents):
    """
    Computes the Jacobian-vector product in a backend-agnostic way.

    Args:
        func: The function to be differentiated.
        primals: The point at which to compute the JVP.
        tangents: The vector with which to multiply the Jacobian.

    Returns:
        A tuple containing the output of the function and the JVP.
    """
    backend = keras.backend.backend()

    if backend == "torch":
        # Ensure primals and tangents are PyTorch tensors
        primals_torch = tuple(torch.tensor(p, requires_grad=True) for p in primals)
        tangents_torch = tuple(torch.tensor(t) for t in tangents)

        # The user's original torch.autograd.functional.jvp can be used here
        return torch.autograd.functional.jvp(func, primals_torch, tangents_torch)

    elif backend == "tensorflow":
        primals_tf = [tf.convert_to_tensor(p) for p in primals]
        tangents_tf = [tf.convert_to_tensor(t) for t in tangents]
        # print("oioi"*100)

        with tf.autodiff.ForwardAccumulator(primals_tf, tangents_tf) as acc:
            outputs = func(*primals_tf)
        # print("huhuhuhu"*100)
        jvp_val = acc.jvp(outputs)
        # print("asdasdasd"*100)
        return outputs, jvp_val

    elif backend == "jax":
        # Ensure primals and tangents are JAX arrays
        primals_jax = tuple(jnp.array(p) for p in primals)
        tangents_jax = tuple(jnp.array(t) for t in tangents)

        return jax.jvp(func, primals_jax, tangents_jax)

    else:
        raise ValueError(f"Backend '{backend}' not supported for JVP computation.")
