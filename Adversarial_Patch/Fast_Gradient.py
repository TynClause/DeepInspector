from __future__ import annotations

import logging
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import (
    compute_success,
    get_labels_np_array,
    random_sphere,
    projection,
    check_and_transform_label_format,
)

if TYPE_CHECKING:  # only for typing
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)


class FastGradientMethod(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        "norm",
        "eps",
        "eps_step",
        "targeted",
        "num_random_init",
        "batch_size",
        "minimal",
    ]

    _estimator_requirements = (BaseEstimator, LossGradientsMixin)

    # ------------------------------------------------------------------ #
    #                              INIT                                  #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        estimator: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        norm: Union[int, float, str] = np.inf,
        eps: Union[int, float, np.ndarray] = 0.3,
        eps_step: Union[int, float, np.ndarray] = 0.1,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        minimal: bool = False,
    ) -> None:
        super().__init__(estimator=estimator)

        self.norm = norm
        self.eps = eps
        self.eps_step = eps_step

        self.targeted = targeted        # <‑‑ publik (digunakan di banyak tempat)
        self._targeted = targeted       # tetap simpan versi private jika dibutuhkan

        self.num_random_init = num_random_init
        self.batch_size = batch_size
        self.minimal = minimal

        self._project = True
        FastGradientMethod._check_params(self)

    # ------------------------------------------------------------------ #
    #                       HELPER / VALIDATION                          #
    # ------------------------------------------------------------------ #
    def _check_compatibility_input_and_eps(self, x: np.ndarray):
        """
        Pastikan `eps` dan `eps_step` broadcastable ke input.
        """
        for arr, name in ((self.eps, "eps"), (self.eps_step, "eps_step")):
            if isinstance(arr, np.ndarray):
                if arr.ndim > x.ndim:
                    raise ValueError(f"The `{name}` shape must be broadcastable to input shape.")

    def _check_params(self) -> None:
        if self.norm not in [1, 2, np.inf, "inf"]:
            raise ValueError("`norm` must be 1, 2, np.inf, or 'inf'.")

        same_type = (
            isinstance(self.eps, (int, float)) and isinstance(self.eps_step, (int, float))
            or isinstance(self.eps, np.ndarray) and isinstance(self.eps_step, np.ndarray)
        )
        if not same_type:
            raise TypeError("`eps` and `eps_step` must share type int/float or both np.ndarray.")

        if isinstance(self.eps, (int, float)) and self.eps < 0:
            raise ValueError("`eps` must be non‑negative.")
        if isinstance(self.eps, np.ndarray) and (self.eps < 0).any():
            raise ValueError("`eps` must be non‑negative.")

        if isinstance(self.eps_step, (int, float)) and self.eps_step <= 0:
            raise ValueError("`eps_step` must be positive.")
        if isinstance(self.eps_step, np.ndarray) and (self.eps_step <= 0).any():
            raise ValueError("`eps_step` must be positive.")

        if isinstance(self.eps, np.ndarray) and isinstance(self.eps_step, np.ndarray):
            if self.eps.shape != self.eps_step.shape:
                raise ValueError("`eps` and `eps_step` must have the same shape.")

        if not isinstance(self.targeted, bool):
            raise TypeError("`targeted` must be bool.")

        if not isinstance(self.num_random_init, (int, np.integer)):
            raise TypeError("`num_random_init` must be int.")
        if self.num_random_init < 0:
            raise ValueError("`num_random_init` must be >= 0.")

        if self.batch_size <= 0:
            raise ValueError("`batch_size` must be positive.")

        if not isinstance(self.minimal, bool):
            raise TypeError("`minimal` must be bool.")

    # ------------------------------------------------------------------ #
    #                    CORE FGM FUNCTIONS                              #
    # ------------------------------------------------------------------ #
    def _compute_perturbation(
        self,
        batch: np.ndarray,
        batch_labels: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """Return normalized gradient (sign/L1/L2)."""
        tol = 1e-8
        grad = self.estimator.loss_gradient(batch, batch_labels) * (1 - 2 * int(self.targeted))

        if np.isnan(grad).any():
            logger.warning("NaNs in gradient replaced with 0.")
            grad = np.nan_to_num(grad, copy=False)

        if mask is not None:
            grad = np.where(mask == 0.0, 0.0, grad)

        def _apply_norm(g: np.ndarray, object_type: bool = False):
            if np.isinf(g).any():
                logger.info("Gradient contains inf.")

            if self.norm in [np.inf, "inf"]:
                return np.sign(g)

            if self.norm in [1, 2]:
                axis = None if object_type else tuple(range(1, len(batch.shape)))
                if self.norm == 1:
                    denom = np.sum(np.abs(g), axis=axis, keepdims=True) + tol
                else:  # L2
                    denom = np.sqrt(np.sum(np.square(g), axis=axis, keepdims=True)) + tol
                return g / denom

            return g

        if batch.dtype == object:
            for i in range(batch.shape[0]):
                grad[i] = _apply_norm(grad[i], object_type=True)
        else:
            grad = _apply_norm(grad)

        return grad

    def _apply_perturbation(
        self,
        batch: np.ndarray,
        perturbation: np.ndarray,
        eps_step: Union[int, float, np.ndarray],
    ) -> np.ndarray:
        step = eps_step * perturbation
        step = np.nan_to_num(step, copy=False)
        batch = batch + step

        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
            batch = np.clip(batch, clip_min, clip_max)

        return batch

    # ------------------------------------------------------------------ #
    #                  MAIN COMPUTE (outer loop)                         #
    # ------------------------------------------------------------------ #
    def _compute(
        self,
        x: np.ndarray,
        x_init: np.ndarray,
        y: np.ndarray,
        mask: Optional[np.ndarray],
        eps: Union[int, float, np.ndarray],
        eps_step: Union[int, float, np.ndarray],
        project: bool,
        random_init: bool,
    ) -> np.ndarray:
        if random_init:
            n = x.shape[0]
            m = np.prod(x.shape[1:]).item()
            rand = random_sphere(n, m, eps, self.norm).reshape(x.shape).astype(ART_NUMPY_DTYPE)
            if mask is not None:
                rand *= mask.astype(ART_NUMPY_DTYPE)
            x_adv = x.astype(ART_NUMPY_DTYPE) + rand
            if self.estimator.clip_values:
                clip_min, clip_max = self.estimator.clip_values
                x_adv = np.clip(x_adv, clip_min, clip_max)
        else:
            x_adv = x.astype(ART_NUMPY_DTYPE) if x.dtype != object else x.copy()

        # iterate over batches
        for b0 in range(0, x.shape[0], self.batch_size):
            b1 = min(b0 + self.batch_size, x.shape[0])
            batch = x_adv[b0:b1]
            batch_labels = y[b0:b1]          # <-- FIXED

            mask_batch = mask
            if mask is not None and len(mask.shape) == len(x.shape):
                mask_batch = mask[b0:b1]

            perturb = self._compute_perturbation(batch, batch_labels, mask_batch)

            # broadcast eps / eps_step if arrays
            batch_eps = eps[b0:b1] if isinstance(eps, np.ndarray) and eps.shape[:1] == x.shape[:1] else eps
            batch_eps_step = (
                eps_step[b0:b1]
                if isinstance(eps_step, np.ndarray) and eps_step.shape[:1] == x.shape[:1]
                else eps_step
            )

            x_adv[b0:b1] = self._apply_perturbation(batch, perturb, batch_eps_step)

            if project:
                if x_adv.dtype == object:
                    for i in range(b0, b1):
                        proj = projection(x_adv[i] - x_init[i], batch_eps[i - b0], self.norm)
                        x_adv[i] = x_init[i] + proj
                else:
                    proj = projection(x_adv[b0:b1] - x_init[b0:b1], batch_eps, self.norm)
                    x_adv[b0:b1] = x_init[b0:b1] + proj

        return x_adv

    # ------------------------------------------------------------------ #
    #                       PUBLIC API                                   #
    # ------------------------------------------------------------------ #
    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        mask = self._get_mask(x, **kwargs)
        self._check_compatibility_input_and_eps(x)

        if isinstance(self.estimator, ClassifierMixin):
            y = check_and_transform_label_format(y, self.estimator.nb_classes)

            if y is None:
                if self.targeted:
                    raise ValueError("Target labels `y` required for targeted attack.")
                logger.info("Using model predictions as labels for untargeted FGM.")
                y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
            y = y / np.sum(y, axis=1, keepdims=True)

            if self.minimal:
                logger.info("Minimal‑perturbation FGM.")
                adv_best = self._minimal_perturbation(x, y, mask)
            else:
                adv_best = None
                best_rate = -1.0
                for _ in range(max(1, self.num_random_init)):
                    adv = self._compute(x, x, y, mask, self.eps, self.eps_step, self._project, True)
                    if self.num_random_init > 1:
                        rate = 100 * compute_success(self.estimator, x, y, adv, self.targeted, self.batch_size)
                        if rate > best_rate:
                            best_rate, adv_best = rate, adv
                    else:
                        adv_best = adv
            success = 100 * compute_success(self.estimator, x, y, adv_best, self.targeted, self.batch_size)
            logger.info("FGM success rate: %.2f%%", success)
            return adv_best

        # non‑classifier estimator branch
        if self.minimal:
            raise ValueError("`minimal=True` only supported for classifiers.")
        if y is None:
            if self.targeted:
                raise ValueError("Target labels `y` required for targeted attack.")
            logger.info("Using model predictions as labels for untargeted FGM.")
            y = self.estimator.predict(x, batch_size=self.batch_size)

        return self._compute(x, x, y, mask, self.eps, self.eps_step, self._project, self.num_random_init > 0)

    # ------------------------------------------------------------------ #
    #                       MASK HELPER                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_mask(x: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        mask = kwargs.get("mask")
        if mask is not None:
            if mask.ndim > x.ndim:
                raise ValueError("Mask shape must be broadcastable to input.")
            if not (np.issubdtype(mask.dtype, np.floating) or mask.dtype == bool):
                raise ValueError("`mask` must be float32/float64 or bool.")
            if np.issubdtype(mask.dtype, np.floating) and np.amin(mask) < 0.0:
                raise ValueError("Float mask requires non‑negative values.")
        return mask