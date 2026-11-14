import warnings
warnings.filterwarnings("ignore")
import numpy as np, scipy, scipy.special as ss, scipy.stats as st, scipy.signal as sig, scipy.linalg as sl, scipy.ndimage as nd, scipy.fft as sfft, scipy.integrate as si
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, jax.scipy.special as jss, jax.scipy.stats as jst, jax.scipy.signal as jsig, jax.scipy.linalg as jsl
import jax.scipy.ndimage as jnd, jax.scipy.fft as jfft, jax.scipy.integrate as jsi
exec(open("cmp.py").read())
C = []
def case(name, jf, nf, **kw): C.append((name, jf, nf, kw))
# special functions on edge values
X = np.array([-np.inf, -2., -1., -0.5, 0., 1e-300, 0.5, 1., 2., 30., np.inf, np.nan])
P = np.array([0., 1e-300, 0.3, 0.5, 1., 1.5, 2., 10.])
unary = ["gammaln","digamma","erf","erfc","erfinv","erfcinv","erfcx","logit","expit","entr","ndtr","ndtri","log_ndtr","i0e","i0","i1e","i1","expi","exp1","spence","dawsn","gamma","loggamma","gammasgn","factorial","wofz"]
for f in unary:
    if hasattr(jss, f) and hasattr(ss, f):
        case(f"{f}(X)", lambda f=f: getattr(jss,f)(X), lambda f=f: getattr(ss,f)(X), rtol=1e-5)
binary = ["xlogy","xlog1py","gammainc","gammaincc","betaln","beta","kl_div","rel_entr","zeta","poch","boxcox","boxcox1p","polygamma","multigammaln","expn","gammaincinv","gammainccinv","hyp1f1","comb", "owens_t"]
A = np.array([-1., 0., 0.5, 1., 2., 3., np.inf, np.nan])
for f in binary:
    if hasattr(jss, f) and hasattr(ss, f):
        aa, xx = np.meshgrid(A, X, indexing='ij')
        if f in ("polygamma","expn"):
            aa = np.floor(np.abs(np.nan_to_num(aa, posinf=3)))
        if f == "multigammaln":
            for d in [1,2,3]:
                case(f"multigammaln d={d}", lambda d=d: jss.multigammaln(np.array([0.5,1.,1.5,2,5]), d), lambda d=d: np.array([ss.multigammaln(v, d) if v>(d-1)/2 else np.nan for v in [0.5,1.,1.5,2,5]]), rtol=1e-6)
            continue
        if f == "gammaincinv" or f == "gammainccinv":
            pp = np.array([0, 0.1, 0.5, 0.9, 1., 1.5, -0.1])
            aa, xx = np.meshgrid(np.array([0., 0.5, 1, 3, -1]), pp, indexing='ij')
        if f == "comb":
            case("comb exact?", lambda: jss.comb(np.array([5., 5, 5, 3]), np.array([2., 0, 7, -1])), lambda: ss.comb(np.array([5., 5, 5, 3]), np.array([2., 0, 7, -1])))
            continue
        case(f"{f}", lambda f=f,aa=aa,xx=xx: getattr(jss,f)(aa, xx), lambda f=f,aa=aa,xx=xx: getattr(ss,f)(aa, xx), rtol=1e-5)
for f in ["betainc", "hyp2f1"]:
    if hasattr(jss, f):
        g = np.array(np.meshgrid(np.array([0., 0.5, 1., 2.]), np.array([0., 0.5, 2.]), np.array([0., 0.3, 1., 1.5, -0.1]), indexing='ij'))
        case(f, lambda f=f,g=g: getattr(jss,f)(*g), lambda f=f,g=g: getattr(ss,f)(*g), rtol=1e-5)
case("logsumexp b sign", lambda: jss.logsumexp(np.array([1., 2, 3]), b=np.array([1., -1, -0.5]), return_sign=True), lambda: ss.logsumexp(np.array([1., 2, 3]), b=np.array([1., -1, -0.5]), return_sign=True))
case("logsumexp b zero all", lambda: jss.logsumexp(np.array([1., 2]), b=np.array([0., 0])), lambda: ss.logsumexp(np.array([1., 2]), b=np.array([0., 0])))
case("logsumexp neg nosign", lambda: jss.logsumexp(np.array([1., 2]), b=np.array([1., -1])), lambda: ss.logsumexp(np.array([1., 2]), b=np.array([1., -1])))
case("logsumexp inf", lambda: jss.logsumexp(np.array([np.inf, 1])), lambda: ss.logsumexp(np.array([np.inf, 1])))
case("logsumexp -inf all", lambda: jss.logsumexp(np.array([-np.inf, -np.inf]), return_sign=True), lambda: ss.logsumexp(np.array([-np.inf, -np.inf]), return_sign=True))
case("logsumexp axis kd", lambda: jss.logsumexp(np.arange(12.).reshape(3,4), axis=(0,1), keepdims=True), lambda: ss.logsumexp(np.arange(12.).reshape(3,4), axis=(0,1), keepdims=True))
case("logsumexp complex", lambda: jss.logsumexp(np.array([1+1j, 2-1j])), lambda: ss.logsumexp(np.array([1+1j, 2-1j])))
case("logsumexp empty", lambda: jss.logsumexp(np.array([])), lambda: ss.logsumexp(np.array([])))
case("softmax axis tuple", lambda: jss.softmax(np.arange(6.).reshape(2,3), axis=(0,1)), lambda: ss.softmax(np.arange(6.).reshape(2,3), axis=(0,1)))
case("log_softmax", lambda: jss.log_softmax(np.array([1., np.inf]), axis=0), lambda: ss.log_softmax(np.array([1., np.inf]), axis=0))
case("sph_harm_y", lambda: jss.sph_harm_y(np.array([2]), np.array([1]), np.array([0.3]), np.array([0.7]), n_max=3), lambda: ss.sph_harm_y(2, 1, 0.3, 0.7))
case("bernoulli", lambda: jss.bernoulli(5), lambda: ss.bernoulli(5))
case("bessel_jn", lambda: jss.bessel_jn(np.array([0.5, 2., 10.]), v=3), lambda: np.stack([ss.jv(n, np.array([0.5, 2., 10.])) for n in range(4)]), rtol=1e-5)
case("sici", lambda: jss.sici(np.array([-1., 0., 1., 50.])), lambda: ss.sici(np.array([-1., 0., 1., 50.])))
case("fresnel?", lambda: jss.fresnel(np.array([0., 1.])), lambda: ss.fresnel(np.array([0., 1.])))
case("ndtri_exp", lambda: jss.ndtri_exp(np.array([-np.inf, -1., -1e-10, 0.])), lambda: ss.ndtri_exp(np.array([-np.inf, -1., -1e-10, 0.])))
# stats
xs = np.array([-1., 0., 0.5, 1., 2., np.inf, -np.inf, np.nan])
ks = np.array([-1., 0., 1., 2., 2.5, 10.])
dists = [
 ("norm", (0.5, 2.)), ("cauchy", (0.5, 2.)), ("laplace", (0.5, 2.)), ("logistic", (0.5, 2.)), ("expon", (0.5, 2.)), ("uniform", (0., 1.)),
 ("gamma", (2.,)), ("gamma", (1.,)), ("gamma", (0.5,)), ("beta", (2., 3.)), ("beta", (1., 1.)), ("beta", (0.5, 1.)), ("beta",(1.,0.5)), ("chi2", (2.,)), ("chi2", (1.,)), ("t", (3.,)), ("pareto", (2.,)),
 ("gennorm", (1.5,)), ("gumbel_l", ()), ("gumbel_r", ()), ("vonmises", (2.,)), ("truncnorm", (-1., 2.)), ("wrapcauchy", (0.5,)),
]
for dn, args in dists:
    jd = getattr(jst, dn); sd = getattr(st, dn)
    for meth in ["logpdf","pdf","cdf","logcdf","sf","logsf","ppf","isf"]:
        if hasattr(jd, meth) and hasattr(sd, meth):
            inp = np.array([-0.1, 0., 0.25, 0.5, 1., 1.1, np.nan]) if meth in ("ppf","isf") else xs
            if dn in ("vonmises","wrapcauchy") and meth not in ("logpdf","pdf"): continue
            case(f"{dn}{args}.{meth}", lambda jd=jd,meth=meth,inp=inp,args=args: getattr(jd,meth)(inp,*args), lambda sd=sd,meth=meth,inp=inp,args=args: getattr(sd,meth)(inp,*args), rtol=1e-5, atol=1e-8)
    for meth in ["logpdf","cdf"]:
        if hasattr(jd, meth) and dn not in ("vonmises","wrapcauchy","truncnorm"):
            try:
                case(f"{dn}{args}.{meth} loc/scale", lambda jd=jd,meth=meth,args=args: getattr(jd,meth)(np.array([0.,1.5,3.]),*args, 1., 2.) if dn not in("norm","cauchy","laplace","logistic","expon","uniform") else getattr(jd,meth)(np.array([0.,1.5,3.]),1., 2.), lambda sd=sd,meth=meth,args=args: getattr(sd,meth)(np.array([0.,1.5,3.]),*args, 1., 2.) if dn not in("norm","cauchy","laplace","logistic","expon","uniform") else getattr(sd,meth)(np.array([0.,1.5,3.]),1.,2.), rtol=1e-5)
            except Exception: pass
discrete = [("poisson", (2.,)), ("poisson", (0.,)), ("geom", (0.3,)), ("geom", (1.,)), ("binom", (5, 0.3)), ("binom", (5, 0.)), ("binom", (5, 1.)), ("nbinom", (3, 0.4)), ("bernoulli", (0.3,)), ("bernoulli", (0.,)), ("betabinom", (5, 2., 3.))]
for dn, args in discrete:
    jd = getattr(jst, dn); sd = getattr(st, dn)
    for meth in ["logpmf","pmf","cdf","logcdf","sf","logsf","ppf"]:
        if hasattr(jd, meth) and hasattr(sd, meth):
            inp = np.array([-0.1, 0., 0.25, 0.5, 1., 1.1, np.nan]) if meth == "ppf" else ks
            case(f"{dn}{args}.{meth}", lambda jd=jd,meth=meth,inp=inp,args=args: getattr(jd,meth)(inp,*args), lambda sd=sd,meth=meth,inp=inp,args=args: getattr(sd,meth)(inp,*args), rtol=1e-5, atol=1e-8)
    if hasattr(jd, "logpmf"):
        case(f"{dn}{args}.logpmf loc", lambda jd=jd,args=args: jd.logpmf(ks, *args, 1), lambda sd=sd,args=args: sd.logpmf(ks, *args, 1), rtol=1e-5)
case("multinomial", lambda: jst.multinomial.logpmf(np.array([1., 2, 0]), 3, np.array([0.2, 0.8, 0.])), lambda: st.multinomial.logpmf(np.array([1., 2, 0]), 3, np.array([0.2, 0.8, 0.])))
case("multinomial bad n", lambda: jst.multinomial.logpmf(np.array([1., 2, 0]), 4, np.array([0.2, 0.3, 0.5])), lambda: st.multinomial.logpmf(np.array([1., 2, 0]), 4, np.array([0.2, 0.3, 0.5])))
case("dirichlet", lambda: jst.dirichlet.logpdf(np.array([0.2, 0.8]), np.array([1., 1.])), lambda: st.dirichlet.logpdf(np.array([0.2, 0.8]), np.array([1., 1.])))
case("dirichlet K-1", lambda: jst.dirichlet.logpdf(np.array([0.2]), np.array([2., 3.])), lambda: st.dirichlet.logpdf(np.array([0.2]), np.array([2., 3.])))
case("mvn", lambda: jst.multivariate_normal.logpdf(np.array([1., 2]), np.array([0., 1]), np.array([[2., .5],[.5, 1]])), lambda: st.multivariate_normal.logpdf(np.array([1., 2]), np.array([0., 1]), np.array([[2., .5],[.5, 1]])))
case("mvn scalar cov", lambda: jst.multivariate_normal.logpdf(np.array([1., 2]), np.array([0., 1]), 2.), lambda: st.multivariate_normal.logpdf(np.array([1., 2]), np.array([0., 1]), 2.))
case("mvn 1d cov", lambda: jst.multivariate_normal.logpdf(np.array([1., 2]), np.array([0., 1]), np.array([2., 3.])), lambda: st.multivariate_normal.logpdf(np.array([1., 2]), np.array([0., 1]), np.array([2., 3.])))
case("mvn cdf?", lambda: jst.multivariate_normal.pdf(np.array([[1., 2],[0,0]]), np.array([0., 1])), lambda: st.multivariate_normal.pdf(np.array([[1., 2],[0,0]]), np.array([0., 1])))
case("mode", lambda: tuple(jst.mode(np.array([[1, 2, 2], [3, 3, 1]]), axis=1, keepdims=True)), lambda: tuple(st.mode(np.array([[1, 2, 2], [3, 3, 1]]), axis=1, keepdims=True)))
case("mode None", lambda: tuple(jst.mode(np.array([[1, 2, 2], [3, 3, 1]]), axis=None)), lambda: tuple(st.mode(np.array([[1, 2, 2], [3, 3, 1]]), axis=None)))
for m in ["average","min","max","dense","ordinal"]:
    case(f"rankdata {m}", lambda m=m: jst.rankdata(np.array([[3, 1, 3], [2, 2, 1]]), method=m, axis=1), lambda m=m: st.rankdata(np.array([[3, 1, 3], [2, 2, 1]]), method=m, axis=1))
case("rankdata nan", lambda: jst.rankdata(np.array([3., np.nan, 1]), nan_policy='omit'), lambda: st.rankdata(np.array([3., np.nan, 1]), nan_policy='omit'))
case("rankdata None", lambda: jst.rankdata(np.array([[3, 1], [2, 2]])), lambda: st.rankdata(np.array([[3, 1], [2, 2]])))
case("sem", lambda: jst.sem(np.array([[1., 2, 4], [3, 3, 9]]), axis=None, ddof=0), lambda: st.sem(np.array([[1., 2, 4], [3, 3, 9]]), axis=None, ddof=0))
case("sem omit", lambda: jst.sem(np.array([1., np.nan, 4, 7]), nan_policy='omit'), lambda: st.sem(np.array([1., np.nan, 4, 7]), nan_policy='omit'))
case("kde", lambda: jst.gaussian_kde(np.array([0., 1, 3, 7])).evaluate(np.array([0.5, 2.])), lambda: st.gaussian_kde(np.array([0., 1, 3, 7])).evaluate(np.array([0.5, 2.])))
case("kde silverman w", lambda: jst.gaussian_kde(np.array([0., 1, 3, 7]), bw_method='silverman', weights=np.array([1., 2, 1, 1])).evaluate(np.array([0.5, 2.])), lambda: st.gaussian_kde(np.array([0., 1, 3, 7]), bw_method='silverman', weights=np.array([1., 2, 1, 1])).evaluate(np.array([0.5, 2.])))
case("kde integrate_box_1d", lambda: jst.gaussian_kde(np.array([0., 1, 3, 7])).integrate_box_1d(0., 2.), lambda: st.gaussian_kde(np.array([0., 1, 3, 7])).integrate_box_1d(0., 2.))
case("kde integrate_gaussian", lambda: jst.gaussian_kde(np.array([0., 1, 3, 7])).integrate_gaussian(np.array([1.]), np.array([[2.]])), lambda: st.gaussian_kde(np.array([0., 1, 3, 7])).integrate_gaussian(np.array([1.]), np.array([[2.]])))
case("kde integrate_kde", lambda: jst.gaussian_kde(np.array([0., 1, 3, 7])).integrate_kde(jst.gaussian_kde(np.array([2., 5]))), lambda: st.gaussian_kde(np.array([0., 1, 3, 7])).integrate_kde(st.gaussian_kde(np.array([2., 5]))))
case("kde logpdf", lambda: jst.gaussian_kde(np.array([[0., 1, 3, 7],[1,0,2,2]])).logpdf(np.array([[0.5],[1.]])), lambda: st.gaussian_kde(np.array([[0., 1, 3, 7],[1,0,2,2]])).logpdf(np.array([[0.5],[1.]])))
# signal
s1 = np.array([1., 2, 3, 0, -1]); s2 = np.array([0.5, 1j, 2])
for mode in ["full","same","valid"]:
    case(f"convolve {mode}", lambda mode=mode: jsig.convolve(s1, s2, mode=mode), lambda mode=mode: sig.convolve(s1, s2, mode=mode))
    case(f"correlate {mode}", lambda mode=mode: jsig.correlate(s1, s2, mode=mode), lambda mode=mode: sig.correlate(s1, s2, mode=mode))
    case(f"correlate swap {mode}", lambda mode=mode: jsig.correlate(s2, s1, mode=mode), lambda mode=mode: sig.correlate(s2, s1, mode=mode))
    case(f"fftconvolve {mode}", lambda mode=mode: jsig.fftconvolve(s1, s2, mode=mode), lambda mode=mode: sig.fftconvolve(s1, s2, mode=mode))
    case(f"convolve2d {mode}", lambda mode=mode: jsig.convolve2d(np.arange(12.).reshape(3,4), np.array([[1., 2],[0, -1]]), mode=mode), lambda mode=mode: sig.convolve2d(np.arange(12.).reshape(3,4), np.array([[1., 2],[0, -1]]), mode=mode))
    case(f"correlate2d {mode}", lambda mode=mode: jsig.correlate2d(np.arange(12.).reshape(3,4), np.array([[1., 2j],[0, -1]]), mode=mode), lambda mode=mode: sig.correlate2d(np.arange(12.).reshape(3,4), np.array([[1., 2j],[0, -1]]), mode=mode))
case("fftconvolve axes", lambda: jsig.fftconvolve(np.ones((3,4)), np.ones((1,2)), axes=1), lambda: sig.fftconvolve(np.ones((3,4)), np.ones((1,2)), axes=1))
case("detrend const", lambda: jsig.detrend(np.arange(12.).reshape(3,4)**2, axis=0, type='constant'), lambda: sig.detrend(np.arange(12.).reshape(3,4)**2, axis=0, type='constant'))
case("detrend bp", lambda: jsig.detrend(np.arange(12.)**2, bp=[5]), lambda: sig.detrend(np.arange(12.)**2, bp=[5]))
rng = np.random.RandomState(0); sx = rng.randn(64); sy = rng.randn(64)
case("welch", lambda: jsig.welch(sx, nperseg=16), lambda: sig.welch(sx, nperseg=16))
case("welch median", lambda: jsig.welch(sx, nperseg=16, average='median'), lambda: sig.welch(sx, nperseg=16, average='median'))
case("welch spectrum dt", lambda: jsig.welch(sx, fs=3., nperseg=16, scaling='spectrum', detrend='linear', return_onesided=False), lambda: sig.welch(sx, fs=3., nperseg=16, scaling='spectrum', detrend='linear', return_onesided=False))
case("csd", lambda: jsig.csd(sx, sy, nperseg=16, noverlap=4), lambda: sig.csd(sx, sy, nperseg=16, noverlap=4))
case("stft", lambda: jsig.stft(sx, nperseg=16), lambda: sig.stft(sx, nperseg=16))
case("stft nopad", lambda: jsig.stft(sx[:60], nperseg=16, boundary=None, padded=False), lambda: sig.stft(sx[:60], nperseg=16, boundary=None, padded=False))
case("istft", lambda: jsig.istft(sig.stft(sx, nperseg=16)[2], nperseg=16), lambda: sig.istft(sig.stft(sx, nperseg=16)[2], nperseg=16))
case("welch odd nperseg", lambda: jsig.welch(sx, nperseg=15), lambda: sig.welch(sx, nperseg=15))
case("welch hann tuple win", lambda: jsig.welch(sx, window=('tukey', 0.3), nperseg=16), lambda: sig.welch(sx, window=('tukey', 0.3), nperseg=16))
case("welch nperseg>len", lambda: jsig.welch(sx[:10], nperseg=16), lambda: sig.welch(sx[:10], nperseg=16))
# linalg
M = np.array([[4., 1, 0], [1, 3, 1], [0, 1, 2]])
case("block_diag empty", lambda: jsl.block_diag(), lambda: sl.block_diag())
case("block_diag 1d/0d", lambda: jsl.block_diag(np.array([1, 2]), 3, np.zeros((0, 2))), lambda: sl.block_diag(np.array([1, 2]), 3, np.zeros((0, 2))))
case("toeplitz", lambda: jsl.toeplitz(np.array([1, 2j, 3]), np.array([9, 5])), lambda: sl.toeplitz(np.array([1, 2j, 3]), np.array([9, 5])))
case("toeplitz c only complex", lambda: jsl.toeplitz(np.array([1, 2j, 3])), lambda: sl.toeplitz(np.array([1, 2j, 3])))
case("hilbert", lambda: jsl.hilbert(3), lambda: sl.hilbert(3))
case("expm", lambda: jsl.expm(M), lambda: sl.expm(M))
case("sqrtm", lambda: jsl.sqrtm(M), lambda: sl.sqrtm(M))
case("polar", lambda: jsl.polar(M), lambda: sl.polar(M))
case("polar left", lambda: jsl.polar(M[:2], side='left'), lambda: sl.polar(M[:2], side='left'))
case("cho_solve", lambda: jsl.cho_solve(sl.cho_factor(M), np.array([1., 2, 3])), lambda: sl.cho_solve(sl.cho_factor(M), np.array([1., 2, 3])))
case("lu_factor/solve trans", lambda: jsl.lu_solve(sl.lu_factor(M+np.triu(M)), np.array([1., 2, 3]), trans=1), lambda: sl.lu_solve(sl.lu_factor(M+np.triu(M)), np.array([1., 2, 3]), trans=1))
case("solve_triangular", lambda: jsl.solve_triangular(np.triu(M), np.array([1., 2, 3]), trans='T', unit_diagonal=True), lambda: sl.solve_triangular(np.triu(M), np.array([1., 2, 3]), trans='T', unit_diagonal=True))
case("solve assume pos", lambda: jsl.solve(M, np.array([1., 2, 3]), assume_a='pos', lower=True), lambda: sl.solve(M, np.array([1., 2, 3]), assume_a='pos', lower=True))
case("det", lambda: jsl.det(M), lambda: sl.det(M))
case("inv", lambda: jsl.inv(M), lambda: sl.inv(M))
case("pinv", lambda: jnp.linalg.pinv(M, rtol=0.4), lambda: np.linalg.pinv(M, rtol=0.4))
case("matrix_rank tol", lambda: jnp.linalg.matrix_rank(M, tol=2.5), lambda: np.linalg.matrix_rank(M, tol=2.5))
case("matrix_rank rtol", lambda: jnp.linalg.matrix_rank(M, rtol=0.5), lambda: np.linalg.matrix_rank(M, rtol=0.5))
case("matrix_rank hermitian", lambda: jnp.linalg.matrix_rank(M, hermitian=True), lambda: np.linalg.matrix_rank(M, hermitian=True))
case("matrix_rank 1d", lambda: jnp.linalg.matrix_rank(np.array([1., 0])), lambda: np.linalg.matrix_rank(np.array([1., 0])))
case("matrix_power neg", lambda: jnp.linalg.matrix_power(M, -2), lambda: np.linalg.matrix_power(M, -2))
case("norm fro axis kd", lambda: jnp.linalg.norm(np.arange(24.).reshape(2,3,4), ord='fro', axis=(0,2), keepdims=True), lambda: np.linalg.norm(np.arange(24.).reshape(2,3,4), ord='fro', axis=(0,2), keepdims=True))
for o in [None, 0, 1, -1, 2, -2, np.inf, -np.inf, 3, 0.5]:
    case(f"norm vec ord={o}", lambda o=o: jnp.linalg.norm(np.array([3., 0, -4]), ord=o), lambda o=o: np.linalg.norm(np.array([3., 0, -4]), ord=o))
for o in [None, 'fro', 'nuc', 1, -1, 2, -2, np.inf, -np.inf]:
    case(f"norm mat ord={o}", lambda o=o: jnp.linalg.norm(M, ord=o, axis=(1,0)), lambda o=o: np.linalg.norm(M, ord=o, axis=(1,0)))
case("norm int", lambda: jnp.linalg.norm(np.array([3, 4])), lambda: np.linalg.norm(np.array([3, 4])))
case("vector_norm", lambda: jnp.linalg.vector_norm(np.arange(24.).reshape(2,3,4), axis=(0,2), ord=3, keepdims=True), lambda: np.linalg.vector_norm(np.arange(24.).reshape(2,3,4), axis=(0,2), ord=3, keepdims=True))
case("vector_norm ord0 empty", lambda: jnp.linalg.vector_norm(np.zeros(0), ord=np.inf), lambda: np.linalg.vector_norm(np.zeros(0), ord=np.inf))
case("matrix_norm", lambda: jnp.linalg.matrix_norm(np.arange(24.).reshape(2,3,4), ord=2, keepdims=True), lambda: np.linalg.matrix_norm(np.arange(24.).reshape(2,3,4), ord=2, keepdims=True))
case("slogdet empty", lambda: jnp.linalg.slogdet(np.zeros((0,0))), lambda: tuple(np.linalg.slogdet(np.zeros((0,0)))))
case("det empty", lambda: jnp.linalg.det(np.zeros((0,0))), lambda: np.linalg.det(np.zeros((0,0))))
case("cond", lambda: jnp.linalg.cond(M, p='fro'), lambda: np.linalg.cond(M, p='fro'))
case("cond -2", lambda: jnp.linalg.cond(M, p=-2), lambda: np.linalg.cond(M, p=-2))
case("lstsq", lambda: jnp.linalg.lstsq(M[:, :2], np.array([1., 2, 3]), rcond=None), lambda: np.linalg.lstsq(M[:, :2], np.array([1., 2, 3]), rcond=None))
case("lstsq rank-def", lambda: jnp.linalg.lstsq(np.ones((3, 2)), np.array([1., 2, 3]), rcond=None)[1:], lambda: np.linalg.lstsq(np.ones((3, 2)), np.array([1., 2, 3]), rcond=None)[1:])
case("tensorsolve", lambda: jnp.linalg.tensorsolve(np.eye(6).reshape(6, 2, 3), np.ones(6)), lambda: np.linalg.tensorsolve(np.eye(6).reshape(6, 2, 3), np.ones(6)))
case("tensorinv", lambda: jnp.linalg.tensorinv(np.eye(6).reshape(2, 3, 6), ind=2), lambda: np.linalg.tensorinv(np.eye(6).reshape(2, 3, 6), ind=2))
case("multi_dot", lambda: jnp.linalg.multi_dot([np.ones(3), np.ones((3, 2)), np.ones(2)]), lambda: np.linalg.multi_dot([np.ones(3), np.ones((3, 2)), np.ones(2)]))
case("trace linalg", lambda: jnp.linalg.trace(np.arange(24.).reshape(2,3,4), offset=1), lambda: np.linalg.trace(np.arange(24.).reshape(2,3,4), offset=1))
case("diagonal linalg", lambda: jnp.linalg.diagonal(np.arange(24.).reshape(2,3,4), offset=-1), lambda: np.linalg.diagonal(np.arange(24.).reshape(2,3,4), offset=-1))
case("cross linalg", lambda: jnp.linalg.cross(np.array([1., 2, 3]), np.array([4., 5, 6])), lambda: np.linalg.cross(np.array([1., 2, 3]), np.array([4., 5, 6])))
case("outer linalg", lambda: jnp.linalg.outer(np.array([1., 2]), np.array([4., 5, 6])), lambda: np.linalg.outer(np.array([1., 2]), np.array([4., 5, 6])))
case("svdvals", lambda: jnp.linalg.svdvals(M), lambda: np.linalg.svdvals(M))
case("eigvalsh U", lambda: jnp.linalg.eigvalsh(np.triu(M), UPLO='U'), lambda: np.linalg.eigvalsh(np.triu(M), UPLO='U'))
case("qr r", lambda: np.abs(np.asarray(jnp.linalg.qr(M[:, :2], mode='r'))), lambda: np.abs(np.linalg.qr(M[:, :2], mode='r')))
case("sl.qr economic r", lambda: np.abs(np.asarray(jsl.qr(M[:, :2], mode='r')[0])), lambda: np.abs(sl.qr(M[:, :2], mode='r')[0]))
case("rsf2csf?", lambda: jsl.eigh_tridiagonal(np.array([1., 2, 3]), np.array([0.5, 0.5]), eigvals_only=True), lambda: sl.eigh_tridiagonal(np.array([1., 2, 3]), np.array([0.5, 0.5]), eigvals_only=True))
case("hessenberg?", lambda: jsl.funm(M, jnp.exp), lambda: sl.funm(M, np.exp))
case("schur", lambda: np.sort(np.diag(np.asarray(jsl.schur(M)[0]))), lambda: np.sort(np.diag(sl.schur(M)[0])))
case("expm_frechet", lambda: jsl.expm_frechet(M, np.eye(3)), lambda: sl.expm_frechet(M, np.eye(3)))
# ndimage
img = np.arange(20.).reshape(4, 5)
for mode in ["constant","nearest","wrap","mirror","reflect"]:
    for order in [0, 1]:
        crd = np.array([[-1.5, 0.3, 2.7, 4.2], [0.5, -2., 4.6, 6.]])
        case(f"map_coordinates {mode} o{order}", lambda mode=mode,order=order,crd=crd: jnd.map_coordinates(img, crd, order=order, mode=mode, cval=-3.), lambda mode=mode,order=order,crd=crd: nd.map_coordinates(img, crd, order=order, mode=mode, cval=-3.))
case("map_coordinates int0", lambda: jnd.map_coordinates(np.arange(5), [np.array([0.5, 1.5, 2.5])], order=0), lambda: nd.map_coordinates(np.arange(5), [np.array([0.5, 1.5, 2.5])], order=0))
# fft
v = np.array([1., 2, 0, -1, 3])
for t in [1, 2, 3, 4]:
    for nrm in [None, "ortho"]:
        case(f"dct t{t} {nrm}", lambda t=t,nrm=nrm: jfft.dct(v, type=t, norm=nrm), lambda t=t,nrm=nrm: sfft.dct(v, type=t, norm=nrm))
        case(f"idct t{t} {nrm}", lambda t=t,nrm=nrm: jfft.idct(v, type=t, norm=nrm), lambda t=t,nrm=nrm: sfft.idct(v, type=t, norm=nrm))
case("dct n", lambda: jfft.dct(v, n=8), lambda: sfft.dct(v, n=8))
case("dct n trunc", lambda: jfft.dct(v, n=3), lambda: sfft.dct(v, n=3))
case("dctn axes", lambda: jfft.dctn(img, axes=(1,), norm='ortho'), lambda: sfft.dctn(img, axes=(1,), norm='ortho'))
case("dctn s", lambda: jfft.dctn(img, s=(3, 6)), lambda: sfft.dctn(img, s=(3, 6)))
case("idctn s", lambda: jfft.idctn(img, s=(3, 6), axes=(0, 1)), lambda: sfft.idctn(img, s=(3, 6), axes=(0, 1)))
case("dct forward", lambda: jfft.dct(v, norm='forward'), lambda: sfft.dct(v, norm='forward'))
case("idct forward", lambda: jfft.idct(v, norm='forward'), lambda: sfft.idct(v, norm='forward'))
case("dct ortho orthogonalize?", lambda: jfft.dct(v, type=1, norm='ortho'), lambda: sfft.dct(v, type=1, norm='ortho'))
case("np.fft norm", lambda: jnp.fft.rfft(v, n=8, norm='forward'), lambda: np.fft.rfft(v, n=8, norm='forward'))
case("np.fft irfftn", lambda: jnp.fft.irfftn(np.fft.rfftn(img), s=(4, 5)), lambda: np.fft.irfftn(np.fft.rfftn(img), s=(4, 5)))
case("np.fft hfft", lambda: jnp.fft.hfft(np.array([1., 2j, 3]), n=6), lambda: np.fft.hfft(np.array([1., 2j, 3]), n=6))
case("np.fft ihfft", lambda: jnp.fft.ihfft(v, norm='ortho'), lambda: np.fft.ihfft(v, norm='ortho'))
case("np.fft fftfreq d", lambda: jnp.fft.fftfreq(5, d=0.1), lambda: np.fft.fftfreq(5, d=0.1))
case("np.fft rfftfreq", lambda: jnp.fft.rfftfreq(5, d=0.1), lambda: np.fft.rfftfreq(5, d=0.1))
case("np.fft fftshift axes", lambda: jnp.fft.fftshift(img, axes=1), lambda: np.fft.fftshift(img, axes=1))
case("np.fft fft2 s", lambda: jnp.fft.fft2(img, s=(2, 3), axes=(1, 0)), lambda: np.fft.fft2(img, s=(2, 3), axes=(1, 0)))
case("np.fft fftn axes None s", lambda: jnp.fft.fftn(img, s=(2, 7)), lambda: np.fft.fftn(img, s=(2, 7)))
# integrate
case("trapezoid sp", lambda: jsi.trapezoid(np.array([1., 2, 4]), x=np.array([0., 1, 3])), lambda: si.trapezoid(np.array([1., 2, 4]), x=np.array([0., 1, 3])))
# cluster / spatial
try:
    import jax.scipy.cluster.vq as jvq, scipy.cluster.vq as svq
    case("vq", lambda: jvq.vq(np.array([[0., 0], [3, 3], [1, 1]]), np.array([[0., 0], [2, 2]])), lambda: svq.vq(np.array([[0., 0], [3, 3], [1, 1]]), np.array([[0., 0], [2, 2]])))
except Exception as ex: print("vq import", ex)
from jax.scipy.spatial.transform import Rotation as JR
from scipy.spatial.transform import Rotation as SR
case("Rot euler", lambda: JR.from_euler('xyz', np.array([0.1, 0.2, 0.3])).as_matrix(), lambda: SR.from_euler('xyz', np.array([0.1, 0.2, 0.3])).as_matrix())
case("Rot euler deg intrinsic", lambda: JR.from_euler('ZYX', np.array([10., 20, 30]), degrees=True).as_quat(), lambda: SR.from_euler('ZYX', np.array([10., 20, 30]), degrees=True).as_quat())
case("Rot as_euler", lambda: JR.from_quat(np.array([0.1, 0.2, 0.3, 0.9])).as_euler('zxz'), lambda: SR.from_quat(np.array([0.1, 0.2, 0.3, 0.9])).as_euler('zxz'))
case("Rot rotvec deg", lambda: JR.from_rotvec(np.array([10., 0, 0]), degrees=True).as_rotvec(degrees=True), lambda: SR.from_rotvec(np.array([10., 0, 0]), degrees=True).as_rotvec(degrees=True))
case("Rot mrp", lambda: JR.from_mrp(np.array([0.1, 0.2, 0.3])).as_quat(), lambda: SR.from_mrp(np.array([0.1, 0.2, 0.3])).as_quat())
case("Rot magnitude", lambda: JR.from_quat(np.array([0.1, 0.2, 0.3, -0.9])).magnitude(), lambda: SR.from_quat(np.array([0.1, 0.2, 0.3, -0.9])).magnitude())
case("Rot inv apply", lambda: JR.from_euler('x', 0.5).apply(np.array([0., 1, 0]), inverse=True), lambda: SR.from_euler('x', 0.5).apply(np.array([0., 1, 0]), inverse=True))
case("Rot mean", lambda: JR.from_euler('x', np.array([0.1, 0.3])).mean().as_rotvec(), lambda: SR.from_euler('x', np.array([0.1, 0.3])).mean().as_rotvec())
case("Rot scalar_first", lambda: JR.from_quat(np.array([0.9, 0.1, 0.2, 0.3]), scalar_first=True).as_quat(), lambda: SR.from_quat(np.array([0.9, 0.1, 0.2, 0.3]), scalar_first=True).as_quat())
case("Rot canonical", lambda: JR.from_quat(np.array([0.1, 0.2, 0.3, -0.9])).as_quat(canonical=True), lambda: SR.from_quat(np.array([0.1, 0.2, 0.3, -0.9])).as_quat(canonical=True))
for name, jf, nf, kw in C:
    try:
        cmp(name, jf, nf, **kw)
    except Exception as ex:
        print("HARNESS", name, ex)
print("DONE scipy part", len(C))
