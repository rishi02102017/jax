import numpy as np, jax, jax.numpy as jnp
print("jax", jax.__version__, "numpy", np.__version__)
def show(label, f):
    try:
        r = f(); print(f"{label}:\n{np.asarray(r)!r}  shape={np.shape(r)}")
    except Exception as e:
        print(f"{label}: {type(e).__name__}: {e}")
cases = {
 "A scalar start/stop, base=[2,10]": dict(start=0., stop=2., num=3, base=np.array([2., 10.])),
 "B start/stop (2,), base (2,), axis=1, num=2 (silent?)": dict(start=np.array([0., 5.]), stop=np.array([5., 0.]), num=2, base=np.array([2., 3.]), axis=1),
 "C start/stop (2,), base (2,), axis=-1, num=5": dict(start=np.array([0., 5.]), stop=np.array([5., 0.]), num=5, base=np.array([2., 3.]), axis=-1),
 "D docstring example axis=0": dict(start=np.array([0., 5.]), stop=np.array([5., 0.]), num=5, base=np.array([2., 3.])),
 "E base (2,1), axis=1": dict(start=0., stop=2., num=3, base=np.array([[2.], [10.]]), axis=1),
 "F base (3,), num=3 scalar start/stop": dict(start=0., stop=2., num=3, base=np.array([2., 3., 10.])),
}
for k, kw in cases.items():
    print("=====", k)
    show("  numpy", lambda: np.logspace(**kw))
    show("  jax  ", lambda: jnp.logspace(**kw))
    show("  jax jit", lambda: jax.jit(lambda s, t, b: jnp.logspace(s, t, kw['num'], base=b, axis=kw.get('axis', 0)))(kw['start'], kw['stop'], kw['base']))
print("===== matrix_rank rtol")
M = np.array([[4., 1, 0], [1, 3, 1], [0, 1, 2]])
print("svals", np.linalg.svd(M, compute_uv=False))
for kw in [dict(rtol=0.5), dict(tol=0.5), dict(rtol=0.3), dict(tol=2.5)]:
    print(kw, "numpy", np.linalg.matrix_rank(M, **kw), "jax", jnp.linalg.matrix_rank(M, **kw), "pinv-consistent?")
print("===== linspace int / pad linear_ramp")
print("np ", np.linspace(-4, 8, 3, endpoint=False, dtype=int), "jax", jnp.linspace(-4, 8, 3, endpoint=False, dtype=int))
print("np ", np.pad(np.array([1, 2, 4, 8]), (0, 3), mode='linear_ramp', end_values=(0, -4)), "jax", jnp.pad(np.array([1, 2, 4, 8]), (0, 3), mode='linear_ramp', end_values=(0, -4)))
