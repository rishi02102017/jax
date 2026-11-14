def cmp(name, jf, nf, rtol=1e-6, atol=1e-10):
    try:
        r = nf(); e1 = None
    except Exception as e:
        r = None; e1 = e
    try:
        j = jf(); e2 = None
    except Exception as e:
        j = None; e2 = e
    if e1 or e2:
        if bool(e1) != bool(e2):
            print(f"MISMATCH-ERR {name}: np={type(e1).__name__ if e1 else r!r} jax={type(e2).__name__+': '+str(e2)[:150] if e2 else j!r}")
        return
    def norm(x):
        if isinstance(x, (tuple, list)): return [norm(y) for y in x]
        return np.asarray(x)
    r, j = norm(r), norm(j)
    def eq(a, b):
        if isinstance(a, list):
            return isinstance(b, list) and len(a)==len(b) and all(eq(x,y) for x,y in zip(a,b))
        if isinstance(b, list): return False
        if a.shape != b.shape: return False
        try:
            return np.allclose(a.astype(complex) if a.dtype.kind=='c' or b.dtype.kind=='c' else a.astype(float), b.astype(complex) if a.dtype.kind=='c' or b.dtype.kind=='c' else b.astype(float), rtol=rtol, atol=atol, equal_nan=True)
        except Exception:
            return np.array_equal(a, b)
    if not eq(r, j):
        print(f"MISMATCH {name}:\n   np ={r!r}\n   jax={j!r}")

