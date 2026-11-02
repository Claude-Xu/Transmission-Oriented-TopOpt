"""Parallel MMA, adapted from FEniTop under GPL-3.0.

Method of moving asymptotes: doi:10.1002/nme.1620240207.
"""

import numpy as np
from mpi4py import MPI
from scipy import sparse as sparse
from scipy.linalg import solve

def mma_optimizer(m, n, opt_iter, xval, xmin, xmax, xold1, xold2, df0dx, fval, dfdx, low, upp, a0=1, a=None, c=None, d=None, move=0.05, asyinit=0.5, asydecr=0.7, asyincr=1.2, low_bnd=0.002, up_bnd=1.0, albefa=0.1, feps=1e-06, comm=MPI.COMM_WORLD):
    if a is None:
        a = np.zeros(m)
    if c is None:
        c = np.full(m, 1000)
    if d is None:
        d = np.zeros(m)
    n_global = comm.allreduce(n, op=MPI.SUM)
    epsimin = np.sqrt(m + n_global) * 1e-09
    xrange = xmax - xmin
    if opt_iter < 2.5:
        low = xval - asyinit * xrange
        upp = xval + asyinit * xrange
    else:
        idx = (xval - xold1) * (xold1 - xold2)
        gamma = np.ones(n)
        gamma[idx > 0] = asyincr
        gamma[idx < 0] = asydecr
        low = xval - gamma * (xold1 - low)
        upp = xval + gamma * (upp - xold1)
        lowmin = xval - up_bnd * xrange
        lowmax = xval - low_bnd * xrange
        uppmin = xval + low_bnd * xrange
        uppmax = xval + up_bnd * xrange
        low = np.minimum(np.maximum(low, lowmin), lowmax)
        upp = np.maximum(np.minimum(upp, uppmax), uppmin)
    alpha = np.maximum.reduce([xmin, low + albefa * (xval - low), xval - move * xrange])
    beta = np.minimum.reduce([xmax, upp - albefa * (upp - xval), xval + move * xrange])
    idx = df0dx > 0
    (p0, q0) = (np.zeros(n), np.zeros(n))
    (p0[idx], p0[~idx]) = (1.001 * df0dx[idx], -0.001 * df0dx[~idx])
    (q0[idx], q0[~idx]) = (0.001 * df0dx[idx], -1.001 * df0dx[~idx])
    ul1 = upp - low
    (p0, q0) = ((p0 + feps / ul1) * (upp - xval) ** 2, (q0 + feps / ul1) * (xval - low) ** 2)
    (P_mat, Q_mat) = (np.zeros((m, n)), np.zeros((m, n)))
    dfdx = dfdx.reshape(m, n)
    idx = dfdx > 0
    (P_mat[idx], Q_mat[~idx]) = (dfdx[idx], -dfdx[~idx])
    (P_mat, Q_mat) = (P_mat * (upp - xval) ** 2, Q_mat * (xval - low) ** 2)
    b = P_mat @ (1.0 / (upp - xval)) + Q_mat @ (1.0 / (xval - low))
    b = comm.allreduce(b, op=MPI.SUM) - fval
    x_new = solve_subproblem(m, epsimin, low, upp, alpha, beta, p0, q0, P_mat, Q_mat, a0, a, b, c, d, comm=comm)
    change = comm.allreduce(np.max(np.abs(x_new - xval), initial=0), op=MPI.MAX)
    return (x_new, change, low, upp)

def solve_subproblem(m, epsimin, low, upp, alpha, beta, p0, q0, P_mat, Q_mat, a0, a, b, c, d, comm=MPI.COMM_WORLD):
    eps = 1.0
    x = 0.5 * (alpha + beta)
    xi = np.maximum(1.0 / (x - alpha), 1.0)
    eta = np.maximum(1.0 / (beta - x), 1.0)
    _lambda = np.ones(m)
    if comm.rank == 0:
        (y, z, s) = (np.ones(m), 1.0, np.ones(m))
        mu = np.maximum(0.5 * c, 1.0)
        zeta = 1.0
    while eps > epsimin:
        p_lambda = p0 + _lambda @ P_mat
        q_lambda = q0 + _lambda @ Q_mat
        g_vec = P_mat @ (1.0 / (upp - x)) + Q_mat @ (1.0 / (x - low))
        g_vec = comm.allreduce(g_vec, op=MPI.SUM)
        dpsi_dx = p_lambda / (upp - x) ** 2 - q_lambda / (x - low) ** 2
        res_x = comm.gather(dpsi_dx - xi + eta, root=0)
        res_xi = comm.gather(xi * (x - alpha) - eps, root=0)
        res_eta = comm.gather(eta * (beta - x) - eps, root=0)
        if comm.rank == 0:
            (res_x, res_xi, res_eta) = (np.hstack(res_x), np.hstack(res_xi), np.hstack(res_eta))
            res_y = c + d * y - mu - _lambda
            res_z = a0 - zeta - a @ _lambda
            res_lambda = g_vec - a * z - y + s - b
            res_mu = mu * y - eps
            res_zeta = zeta * z - eps
            res_s = _lambda * s - eps
            res = np.hstack([res_x, res_y, res_z, res_lambda, res_xi, res_eta, res_mu, res_zeta, res_s])
            res_norm = np.linalg.norm(res, ord=2)
            res_max = np.linalg.norm(res, ord=np.inf)
        else:
            (res_norm, res_max) = (None, None)
        res_norm = comm.bcast(res_norm, root=0)
        res_max = comm.bcast(res_max, root=0)
        newton_iter = 0
        while res_max > 0.9 * eps and newton_iter < 100:
            newton_iter += 1
            if newton_iter == 100:
                raise RuntimeError('MMA subproblem Newton iteration did not converge.')
            p_lambda = p0 + _lambda @ P_mat
            q_lambda = q0 + _lambda @ Q_mat
            G_mat = P_mat / (upp - x) ** 2 - Q_mat / (x - low) ** 2
            G_mat_global = comm.gather(G_mat, root=0)
            diag_x = 2 * (p_lambda / (upp - x) ** 3 + q_lambda / (x - low) ** 3) + xi / (x - alpha) + eta / (beta - x)
            diag_x_global = comm.gather(diag_x, root=0)
            if comm.rank == 0:
                G_mat_global = np.hstack(G_mat_global)
                diag_x_global = np.hstack(diag_x_global)
                diag_y = d + mu / y
                A_mat = sparse.spdiags(s / _lambda + 1.0 / diag_y, 0, m, m) + G_mat_global * (1.0 / diag_x_global) @ G_mat_global.T
                lhs = np.vstack([np.hstack([A_mat, a.reshape(-1, 1)]), np.hstack([a, -zeta / z])])
            dpsi_dx = p_lambda / (upp - x) ** 2 - q_lambda / (x - low) ** 2
            delta_x = dpsi_dx - eps / (x - alpha) + eps / (beta - x)
            delta_x_global = comm.gather(delta_x, root=0)
            g_vec = P_mat @ (1.0 / (upp - x)) + Q_mat @ (1.0 / (x - low))
            g_vec = comm.allreduce(g_vec, op=MPI.SUM)
            if comm.rank == 0:
                delta_x_global = np.hstack(delta_x_global)
                delta_y = c + d * y - _lambda - eps / y
                delta_z = a0 - a @ _lambda - eps / z
                delta_lambda = g_vec - a * z - y - b + eps / _lambda
                b_lambda = delta_lambda + delta_y / diag_y - G_mat_global @ (delta_x_global / diag_x_global)
                rhs = np.hstack([b_lambda, delta_z])
            solve_error = None
            if comm.rank == 0:
                try:
                    solution = solve(lhs, rhs)
                except (ValueError, np.linalg.LinAlgError) as error:
                    solve_error = str(error)
            solve_error = comm.bcast(solve_error, root=0)
            if solve_error is not None:
                raise RuntimeError(f'MMA subproblem solve failed: {solve_error}')
            if comm.rank == 0:
                (incr_lambda, incr_z) = (solution[:m], solution[m])
                incr_y = -delta_y / diag_y + incr_lambda / diag_y
                incr_mu = -mu + eps / y - mu * incr_y / y
                incr_zeta = -zeta + eps / z - zeta * incr_z / z
                incr_s = -s + eps / _lambda - s * incr_lambda / _lambda
            else:
                incr_lambda = None
            incr_lambda = comm.bcast(incr_lambda, root=0)
            incr_x = -delta_x / diag_x - incr_lambda @ G_mat / diag_x
            incr_xi = -xi + eps / (x - alpha) - xi * incr_x / (x - alpha)
            incr_eta = -eta + eps / (beta - x) + eta * incr_x / (beta - x)
            incr_xi_global = comm.gather(incr_xi, root=0)
            incr_eta_global = comm.gather(incr_eta, root=0)
            xi_global = comm.gather(xi, root=0)
            eta_global = comm.gather(eta, root=0)
            if comm.rank == 0:
                xi_global = np.hstack(xi_global)
                eta_global = np.hstack(eta_global)
                values = np.hstack([y, z, _lambda, xi_global, eta_global, mu, zeta, s])
                incr_xi_global = np.hstack(incr_xi_global)
                incr_eta_global = np.hstack(incr_eta_global)
                incr = np.hstack([incr_y, incr_z, incr_lambda, incr_xi_global, incr_eta_global, incr_mu, incr_zeta, incr_s])
            step_inv_alpha = np.max(-1.01 * incr_x / (x - alpha), initial=0)
            step_inv_beta = np.max(1.01 * incr_x / (beta - x), initial=0)
            step_inv_alpha = comm.allreduce(step_inv_alpha, op=MPI.MAX)
            step_inv_beta = comm.allreduce(step_inv_beta, op=MPI.MAX)
            if comm.rank == 0:
                step_inv = max(-1.01 * incr / values)
                step_inv = max(step_inv_alpha, step_inv_beta, step_inv, 1.0)
                step = 1.0 / step_inv
            else:
                step = None
            step = comm.bcast(step, root=0)
            (x_old, lambda_old) = (x.copy(), _lambda.copy())
            (xi_old, eta_old) = (xi.copy(), eta.copy())
            if comm.rank == 0:
                (y_old, z_old) = (y.copy(), z)
                (mu_old, zeta_old, s_old) = (mu.copy(), zeta, s.copy())
            search_iter = 0
            res_norm_new = 2 * res_norm
            while res_norm_new > res_norm and search_iter < 50:
                search_iter += 1
                x = x_old + step * incr_x
                _lambda = lambda_old + step * incr_lambda
                xi = xi_old + step * incr_xi
                eta = eta_old + step * incr_eta
                if comm.rank == 0:
                    y = y_old + step * incr_y
                    z = z_old + step * incr_z
                    mu = mu_old + step * incr_mu
                    zeta = zeta_old + step * incr_zeta
                    s = s_old + step * incr_s
                p_lambda = p0 + _lambda @ P_mat
                q_lambda = q0 + _lambda @ Q_mat
                g_vec = P_mat @ (1.0 / (upp - x)) + Q_mat @ (1.0 / (x - low))
                g_vec = comm.allreduce(g_vec, op=MPI.SUM)
                dpsi_dx = p_lambda / (upp - x) ** 2 - q_lambda / (x - low) ** 2
                res_x = comm.gather(dpsi_dx - xi + eta, root=0)
                res_xi = comm.gather(xi * (x - alpha) - eps, root=0)
                res_eta = comm.gather(eta * (beta - x) - eps, root=0)
                if comm.rank == 0:
                    (res_x, res_xi, res_eta) = (np.hstack(res_x), np.hstack(res_xi), np.hstack(res_eta))
                    res_y = c + d * y - mu - _lambda
                    res_z = a0 - zeta - a @ _lambda
                    res_lambda = g_vec - a * z - y + s - b
                    res_mu = mu * y - eps
                    res_zeta = zeta * z - eps
                    res_s = _lambda * s - eps
                    res = np.hstack([res_x, res_y, res_z, res_lambda, res_xi, res_eta, res_mu, res_zeta, res_s])
                    res_norm_new = np.linalg.norm(res, ord=2)
                else:
                    res_norm_new = None
                res_norm_new = comm.bcast(res_norm_new, root=0)
                step *= 0.5
            if not np.isfinite(res_norm_new) or res_norm_new > res_norm:
                raise RuntimeError('MMA subproblem line search did not converge.')
            res_norm = res_norm_new
            res_max = np.linalg.norm(res, ord=np.inf) if comm.rank == 0 else None
            res_max = comm.bcast(res_max, root=0)
        eps *= 0.1
    return x
