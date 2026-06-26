"""Small numpy/scipy QRNN used by all Chapter 5 simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SQRT_2PI = np.sqrt(2.0 * np.pi)


def parameter_count(p: int, J: int) -> int:
    return p * J + J + J + 1


def pack_params(W: np.ndarray, b: np.ndarray, v: np.ndarray, c: float) -> np.ndarray:
    return np.concatenate([W.ravel(), b.ravel(), v.ravel(), np.array([c], dtype=float)])


def unpack_params(params: np.ndarray, p: int, J: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    w_end = p * J
    b_end = w_end + J
    v_end = b_end + J
    W = params[:w_end].reshape(p, J)
    b = params[w_end:b_end]
    v = params[b_end:v_end]
    c = float(params[v_end])
    return W, b, v, c


def normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x**2) / SQRT_2PI


def normal_cdf(x: np.ndarray) -> np.ndarray:
    """Fast normal CDF approximation, avoiding a scipy dependency."""

    x = np.asarray(x, dtype=float)
    b1, b2, b3, b4, b5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    p = 0.2316419
    ax = np.abs(x)
    t = 1.0 / (1.0 + p * ax)
    poly = (((((b5 * t + b4) * t) + b3) * t + b2) * t + b1) * t
    cdf_pos = 1.0 - normal_pdf(ax) * poly
    return np.where(x >= 0.0, cdf_pos, 1.0 - cdf_pos)


def cs_loss_values(e: np.ndarray, tau: float, h: float) -> np.ndarray:
    h = max(float(h), 1e-8)
    a = e / h
    return e * (tau - normal_cdf(-a)) + h * normal_pdf(a)


def cs_score_values(e: np.ndarray, tau: float, h: float) -> np.ndarray:
    h = max(float(h), 1e-8)
    return tau - normal_cdf(-e / h)


def check_loss_values(e: np.ndarray, tau: float) -> np.ndarray:
    return np.where(e >= 0.0, tau * e, (tau - 1.0) * e)


def huber_loss_values(e: np.ndarray, tau: float, epsilon: float) -> np.ndarray:
    eps = max(float(epsilon), 1e-8)
    abs_e = np.abs(e)
    huber = np.where(abs_e <= eps, e**2 / (2.0 * eps), abs_e - 0.5 * eps)
    return np.where(e >= 0.0, tau * huber, (1.0 - tau) * huber)


def huber_score_values(e: np.ndarray, tau: float, epsilon: float) -> np.ndarray:
    eps = max(float(epsilon), 1e-8)
    slope = np.where(np.abs(e) <= eps, e / eps, np.sign(e))
    return np.where(e >= 0.0, tau * slope, (1.0 - tau) * slope)


def loss_values(e: np.ndarray, tau: float, h: float, loss_kind: str, epsilon: float) -> np.ndarray:
    if loss_kind == "cs":
        return cs_loss_values(e, tau, h)
    if loss_kind == "huber":
        return huber_loss_values(e, tau, epsilon)
    if loss_kind == "check":
        return check_loss_values(e, tau)
    raise ValueError(f"unknown loss kind: {loss_kind}")


def score_values(e: np.ndarray, tau: float, h: float, loss_kind: str, epsilon: float) -> np.ndarray:
    if loss_kind == "cs":
        return cs_score_values(e, tau, h)
    if loss_kind == "huber":
        return huber_score_values(e, tau, epsilon)
    if loss_kind == "check":
        return np.where(e >= 0.0, tau, tau - 1.0)
    raise ValueError(f"unknown loss kind: {loss_kind}")


def forward(params: np.ndarray, Xs: np.ndarray, p: int, J: int) -> tuple[np.ndarray, np.ndarray]:
    W, b, v, c = unpack_params(params, p, J)
    H = np.tanh(Xs @ W + b)
    pred = H @ v + c
    return pred, H


def loss_and_grad(
    params: np.ndarray,
    Xs: np.ndarray,
    y: np.ndarray,
    p: int,
    J: int,
    tau: float,
    lam: float,
    h: float,
    loss_kind: str = "cs",
    epsilon: float = 0.1,
    correction: np.ndarray | None = None,
    include_penalty: bool = True,
) -> tuple[float, np.ndarray]:
    """Return objective and analytic gradient for the single-hidden-layer QRNN."""

    n = float(len(y))
    W, b, v, c = unpack_params(params, p, J)
    z = Xs @ W + b
    H = np.tanh(z)
    pred = H @ v + c
    e = y - pred
    values = loss_values(e, tau, h, loss_kind, epsilon)
    psi = score_values(e, tau, h, loss_kind, epsilon)

    common = -(psi[:, None] * (1.0 - H**2) * v[None, :])
    gW = Xs.T @ common / n
    gb = common.mean(axis=0)
    gv = -(psi[:, None] * H).mean(axis=0)
    gc = -float(psi.mean())
    grad = pack_params(gW, gb, gv, gc)
    obj = float(values.mean())

    if include_penalty and lam > 0.0:
        scale = lam / max(p * J, 1)
        obj += float(scale * np.sum(W**2))
        gW_pen = 2.0 * scale * W
        grad += pack_params(gW_pen, np.zeros(J), np.zeros(J), 0.0)

    if correction is not None:
        obj -= float(np.dot(params, correction))
        grad -= correction

    return obj, grad


@dataclass
class QRNNModel:
    p: int
    J: int
    tau: float
    lam: float
    h: float
    params: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    loss_kind: str = "cs"
    epsilon: float = 0.1
    backend: str = "numpy"
    device: str = "cpu"
    torch_dtype: str = "float32"

    def standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean) / self.x_scale

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.backend == "torch":
            try:
                from .torch_backend import predict_torch

                return predict_torch(self, X)
            except RuntimeError:
                pass
        Xs = self.standardize(X)
        pred, _ = forward(self.params, Xs, self.p, self.J)
        return pred

    def empirical_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(X)
        return float(loss_values(y - pred, self.tau, self.h, self.loss_kind, self.epsilon).mean())

    def empirical_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.backend == "torch":
            try:
                from .torch_backend import empirical_gradient_torch

                return empirical_gradient_torch(self, X, y)
            except RuntimeError:
                pass
        Xs = self.standardize(X)
        _, grad = loss_and_grad(
            self.params,
            Xs,
            y,
            self.p,
            self.J,
            self.tau,
            0.0,
            self.h,
            self.loss_kind,
            self.epsilon,
            include_penalty=False,
        )
        return grad

    def with_params(self, params: np.ndarray) -> "QRNNModel":
        return QRNNModel(
            p=self.p,
            J=self.J,
            tau=self.tau,
            lam=self.lam,
            h=self.h,
            params=params.copy(),
            x_mean=self.x_mean.copy(),
            x_scale=self.x_scale.copy(),
            loss_kind=self.loss_kind,
            epsilon=self.epsilon,
            backend=self.backend,
            device=self.device,
            torch_dtype=self.torch_dtype,
        )
