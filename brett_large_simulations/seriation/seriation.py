import os
import sys
import time
import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ot
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import scipy.stats
import umap
import pathos.multiprocessing as mp
import numexpr as ne
import geomloss
import torch
import pandas as pd
import wot
import umap_utils

# matplotlib.use("Agg")
matplotlib.use("TkAgg")


def flip_if_decreasing(x):
    return x if x[0] <= x[-1] else -x


def compute_curve_length(points_order, dist_mat):
    return dist_mat[tuple(np.argsort(points_order).repeat(2)[1:-1].reshape(-1, 2).T)].sum()


def compute_length_from_orders_dict(dist_mat, orders_dict):
    return {k: compute_curve_length(v, dist_mat) for k,v in orders_dict.items()}


# Edge distance between two orderings (integer-valued)
# https://www.cicirello.org/publications/cicirello-bict2019.pdf
def normalized_edge_distance(ord1, ord2):
    if len(ord1) != len(ord2):
        print("ERROR: ord1 and ord2 must have the same length")
        exit(-1)
    n = len(ord1)
    return 1/(n-1)*(n-1 - len(set.intersection({tuple(e) for e in ord1.repeat(2)[1:-1].reshape(-1, 2)},
                                               {tuple(e) for e in ord2.repeat(2)[1:-1].reshape(-1, 2)})))


# Kendall's tau distance between two orderings (integer-valued)
# https://en.wikipedia.org/wiki/Kendall_tau_distance
def kendall_tau_distance(ord1, ord2):
    if len(ord1) != len(ord2):
        print("ERROR: ord1 and ord2 must have the same length")
        exit(-1)
    n = len(ord1)
    return (1-scipy.stats.kendalltau(ord1, ord2).correlation)*n*(n-1)/4


# Set shared_support=True if all measures are defined on the same support, and only the weights are changing.
# In that case, X can be an |M,dim] array instead of the regular [T][N,dim], and weights is of shape [T,M].
def compute_distance_matrix(param_dict, X, weights=None, shared_support=False):

    print("Computing distance matrix")
    td0 = time.time()

    T = param_dict["T"]
    N = param_dict["N"]
    dim = param_dict["dim"]
    epsilon = param_dict["epsilon"]
    tau = param_dict["tau"]
    sink_iter = param_dict["sink_iter"]
    max_workers = param_dict["max_workers"]

    mmd_names_dict = {"MMD-e": "energy", "MMD-g": "gaussian", "MMD-l": "laplacian"}

    if weights is None:
        weights = np.tile(np.ones(N)/N,(T,1))

    Css = None
    if shared_support:
        if param_dict["reduce_dim"]:
            xi, yj, pca, mean = wot.ot.compute_pca(X, np.empty((0,X.shape[1])), n_components=dim)
            eigenvals = np.diag(pca.singular_values_)
            # In this function, C is normalized by its median
            Css = wot.ot.OTModel.compute_default_cost_matrix(xi, xi, eigenvals)
        else:
            xi = X[:, :, None]
            yjT = xi.T
            Css = ne.evaluate('sum((xi-yjT)**2, axis=1)')

    def worker(i):
        A = np.zeros(T)
        end = i+1 if param_dict["comp_diag"] else i
        for j in range(end):

            # Compute cost matrix
            xi = yj = eigenvals = None
            if shared_support:
                C = Css
            # if not shared_support:
            elif param_dict["reduce_dim"]:
                xi, yj, pca, mean = wot.ot.compute_pca(X[i], X[j], n_components=dim)
                eigenvals = np.diag(pca.singular_values_)
                # In this function, C is normalized by its median
                C = wot.ot.OTModel.compute_default_cost_matrix(xi, yj, eigenvals)
            else:
                xi = X[i][:, :, None]
                yj = X[j][:, :, None]
                yjT = yj.T
                C = ne.evaluate('sum((xi-yjT)**2, axis=1)')  # Compute pairwise ground distances

            # Compute distance
            if param_dict["marg_dist"] == "Sink":
                # The GPU version only does float32, so epsilon = 0.01 fails
                # A[j] = ot.gpu.sinkhorn(np.ones(N)/N, np.ones((N,1))/N, C, epsilon, numitermax=sink_iter)
                A[j] = ot.sinkhorn2(weights[i], weights[j], C, epsilon, numitermax=sink_iter)
            elif param_dict["marg_dist"] == "Unb-sink":
                A[j] = ot.sinkhorn_unbalanced2(weights[i], weights[j], C, epsilon, tau, numitermax=sink_iter)
            elif param_dict["marg_dist"] == "Sink-div":

                # Compute self-cost matrices
                if shared_support:
                    Cii = Cjj = C = Css
                elif param_dict["reduce_dim"]:
                    # In this function, C is normalized by its median
                    Cii = wot.ot.OTModel.compute_default_cost_matrix(xi, xi, eigenvals)
                    Cjj = wot.ot.OTModel.compute_default_cost_matrix(yj, yj, eigenvals)
                    C = wot.ot.OTModel.compute_default_cost_matrix(xi, yj, eigenvals)
                else:
                    xi = X[i][:, :, None]
                    yi = xi.T
                    Cii = ne.evaluate('sum((xi-yi)**2, axis=1)')
                    xj = X[j][:, :, None]
                    yj = xj.T
                    Cjj = ne.evaluate('sum((xj-yj)**2, axis=1)')

                dii = ot.sinkhorn2(weights[i], weights[i], Cii, epsilon, numitermax=sink_iter)
                djj = ot.sinkhorn2(weights[j], weights[j], Cjj, epsilon, numitermax=sink_iter)
                d = ot.sinkhorn2(weights[i], weights[j], C, epsilon, numitermax=sink_iter)
                A[j] = d - (dii + djj)/2
            elif param_dict["marg_dist"] == "EMD":
                A[j] = ot.emd2(weights[i], weights[j], np.ascontiguousarray(C))
            elif param_dict["marg_dist"].startswith("MMD"):
                loss = geomloss.SamplesLoss(mmd_names_dict[param_dict["marg_dist"]], blur=param_dict["MMD_blur"])
                A[j] = loss(torch.from_numpy(xi).squeeze(), torch.from_numpy(yj).squeeze())
            else:
                print("Unrecognized distance:",param_dict["marg_dist"])
                raise ValueError

        return A

    with mp.Pool(max_workers) as pool:
        result = pool.map(worker, np.arange(T))
    # result = [worker(i) for i in np.arange(T)]

    dist_mat = np.array(result)

    # Deal with negative values if any
    neg_vals = dist_mat[dist_mat < 0]
    if len(neg_vals) > 0:
        if not np.allclose(neg_vals,0,atol=1e-8):
            print("WARNING: Negative distance < -1e-8")
        dist_mat[dist_mat < 0] = 0

    # Fill the symmetric part of the matrix
    dist_mat += dist_mat.T

    t_dstmat = time.time() - td0
    print("Time dist matrix:",t_dstmat)

    return dist_mat, {"t_dstmat": "%0.2g"%t_dstmat}


# Do the seriation, and save and plot results
def seriation_save_and_plot(exp_dir, param_dict, dist_mat, img_result_dict):

    T = param_dict['T']

    # Save distance matrix
    np.save(os.path.join(exp_dir,"dist-matrix"), dist_mat)

    # Save image of distance matrix
    plt.figure()
    plt.imshow(dist_mat)
    plt.colorbar()
    plt.title("Distance matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir,img_result_dict["dist-matrix"]))
    plt.close()

    # Save distance histogram
    plt.figure()
    plt.hist(dist_mat[dist_mat > 0], 50)    # [dist_mat > 0] removes the diagonal
    plt.yscale('log', nonpositive='clip')
    plt.title("Distance histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, img_result_dict["dist-hist"]))
    plt.close()

    # Compute similarity matrix
    A, dist_mat_norm = compute_similarity_matrix(dist_mat, param_dict, similarity=param_dict["similarity"])

    # Compute orderings
    ordering_names = ['L', 'DA', 'DAD', 'UMAP', 'UKSD', 'UKRD']
    orders, times = compute_seriation(param_dict, A, dist_mat, ordering_methods=ordering_names, similarity=param_dict["similarity"])

    # Save similarity vs distance
    plt.figure()
    plt.plot(dist_mat_norm.flatten(), A.flatten(), '.', alpha=0.05)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel("Normalized distance")
    plt.ylabel("Similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, img_result_dict["simil-vs-dist"]))
    plt.close()

    # Not sure if useful
    # plt.figure()
    # [plt.plot(np.argsort(v), label=k) for k, v in orders.items()]
    # plt.legend()

    # Save similarity histogram
    plt.figure()
    plt.hist(A[A < 1], 50)  # [A < 1] removes the diagonal
    plt.yscale('log', nonpositive='clip')
    plt.title("Similarity histogram")
    plt.xlim([0,1])
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, img_result_dict["simil-hist"]))
    plt.close()

    # Save the orderings
    orderings = np.vstack(list(orders.values())).T
    df = pd.DataFrame(data=orderings, columns=orders.keys())
    df.to_csv(os.path.join(exp_dir, "seriations.csv"))

    # Compute lengths
    lengths = {k: compute_curve_length(v,dist_mat) for k,v in orders.items()}
    gt_length = compute_curve_length(np.arange(T), dist_mat)

    # Compute correlations and distances
    corr = {k: np.corrcoef(np.argsort(v), np.arange(T))[0, 1] for k, v in orders.items()}
    ktd = {k: kendall_tau_distance(np.argsort(v), np.arange(T)) for k, v in orders.items()}
    ed = {k: normalized_edge_distance(np.argsort(v), np.arange(T)) for k, v in orders.items()}

    # Plot lengths
    plt.figure()
    # Plot length of ground truth ordering
    plt.axhline(gt_length, c="orange")
    plt.text(5.3, gt_length, "GT", c='orange', verticalalignment='center')
    # Plot lengths of other orderings
    plt.plot(lengths.values(), '.-', markersize=20)
    plt.title("Lengths of the orderings")
    for i, (k, v) in enumerate(lengths.items()):
        plt.text(i + 0.1, v, k)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, img_result_dict["lengths2"]))
    plt.ylim(0.99*gt_length, gt_length + (dist_mat.mean()*T - gt_length)*0.75)
    plt.savefig(os.path.join(exp_dir, img_result_dict["lengths"]))
    plt.close()

    # Plot each ordering separately
    for k,v in orders.items():
        plt.figure()
        plt.plot(v)
        plt.title("Seriation - " + k)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, img_result_dict["seriation"+k]))
        plt.close()

    # Plot all orderings
    plt.figure(figsize=(15,8))
    # for i, name in enumerate(['L','UMAP','UKSD','DL','DLD','UKRD']):
    for i, name in enumerate(['L', 'UMAP', 'UKSD', 'DA', 'DAD', 'UKRD']):
        plt.subplot(2,3,i+1)
        plt.plot(orders[name])
        plt.title("Seriation - " + name)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir,img_result_dict["all"]))
    plt.close()

    # Embed points in 2D
    mapper2 = umap.UMAP(n_components=2,
                        n_neighbors=param_dict["neighb"],
                        min_dist=param_dict["mindist"],
                        n_epochs=param_dict["epochs"],
                        metric=param_dict["metric"],
                        random_state=param_dict["rand_seed"]
                        )
    coords2 = mapper2.fit_transform(dist_mat)

    # Plot the 2D UMAP
    plt.figure(figsize=(15, 10))
    plt.subplot(231)
    plt.title("Ground truth")
    plt.scatter(*coords2.T)
    plt.plot(*coords2.T)
    # for i, name in enumerate(['UMAP', 'UKSD', 'DL', 'DLD', 'UKRD']):
    for i, name in enumerate(['UMAP', 'UKSD', 'DA', 'DAD', 'UKRD']):
        plt.subplot(2,3,i+2)
        plt.title(name)
        plt.scatter(*coords2.T)
        plt.plot(*coords2[np.argsort(orders[name])].T)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, img_result_dict["umap2d"]))
    plt.close()

    # Scalar results
    ot_sampling_comp = param_dict["N"]**(-1/param_dict["dim"])
    # Convert to str
    times_str = {k: "%0.2g"%v for k, v in times.items()}
    lengths_str = {"length_"+k: "%0.5g"%v for k, v in lengths.items()}
    corr_str = {"corr_"+k: "%0.4g"%v for k, v in corr.items()}
    ktd_str = {"ktd_"+k: "%0.4g"%v for k, v in ktd.items()}
    ed_str = {"ed_"+k: "%d"%v for k, v in ed.items()}

    # Return additional scalar results as a dictionary, where keys will be the corresponding columns in the CSV.
    # Return the results either as their original dtype, or as how you want them to appear in the viewer:
    return {**times_str,
            **lengths_str,
            **corr_str, **ktd_str, **ed_str,
            "ot_sampling_comp": "%0.2g"%ot_sampling_comp,
            }


# Compute a similarity matrix from a distance matrix
# similarity can be "Exp-kernel" or "QOT".
# "Exp-kernel" will produce a dense similarity matrix
# "QOT" will produce a sparse similarity matrix, and can be returned as sparse or dense
def compute_similarity_matrix(dist_mat, param_dict, similarity="Exp-kernel", return_sparse_matrix=False):

    T = param_dict['T']
    dist_mat_norm = A = None
    if similarity == "Exp-kernel":
        # Normalize the distance matrix
        dist_mat_norm = dist_mat.copy()
        if param_dict["dist_mat_norm"] == 1:
            # This keeps the potential gap between 0 and the min non-zero distance.
            dist_mat_norm = dist_mat/dist_mat.max()
        if param_dict["dist_mat_norm"] == 2:
            # This removes the potential gap between 0 and the min non-zero distance.
            # It means that the min non-zero distance will become 0.
            dmin = np.min(dist_mat[dist_mat > 0])
            dmax = np.max(dist_mat[dist_mat > 0])
            dist_mat_norm[dist_mat > 0] = (dist_mat[dist_mat > 0] - dmin) / (dmax - dmin)
        if param_dict["dist_mat_norm"] == 3:
            # Same as above, except that min non-zero distance becomes alpha instead of 0
            dmin = np.min(dist_mat[dist_mat > 0])
            dmax = np.max(dist_mat[dist_mat > 0])
            alpha = 0.01
            dist_mat_norm[dist_mat > 0] = (dist_mat[dist_mat > 0] - dmin) / (dmax - dmin - alpha) + alpha

        # Build a similarity matrix
        A = np.exp(-dist_mat_norm**param_dict["dist_power"]/(param_dict["sim_sigma"]**2))
        A[A < param_dict['sim_cut']] = 0  # Apply threshold
    elif similarity == "QOT":
        C_diag0 = dist_mat**param_dict["dist_power"]
        Cmed = np.median(C_diag0)
        C = C_diag0 + np.diag(np.ones(C_diag0.shape[0]) + np.inf)
        A = ot.smooth.smooth_ot_dual(np.ones(T), np.ones(T), C, reg_type='l2', reg=param_dict["qot_reg"]*Cmed)
        dist_mat_norm = dist_mat
        if return_sparse_matrix:
            A = scipy.sparse.csr_matrix(A)
    else:
        print("ERROR: Unrecognized value for parameter 'similarity':",similarity)
        exit(-1)

    return A, dist_mat_norm


# Compute seriation from a similarity matrix A or distance matrix dist_mat, with different methods.
# ordering can be one of, or a list with any of: 'L','DA','DAD','DL','DLD','UMAP','UKSD','UKRD'. Default is "DAD".
# The matrix A is required for all ordering methods except "UMAP".
# The matrix dist_mat is required only for "UMAP" and "UKRD".
# It returns a dictionary where keys are the content of "ordering_method", and values are the corresponding ordering.
# similarity is the same parameter passed to `compute_similarity_matrix`, so values are "Exp-kernel" or "QOT"
def compute_seriation(param_dict, A=None, dist_mat=None, ordering_methods=None, similarity="Exp-kernel"):

    if ordering_methods is None:
        ordering_methods = ["DAD",]
    if not isinstance(ordering_methods,list):
        ordering_methods = list(ordering_methods)

    T = param_dict['T']

    # Pre-compute some matrices
    orders = {}
    Ddiag = L = Dm12 = None
    if len({"L", "DL", "DLD", "DA", "DAD"}.intersection(set(ordering_methods))) > 0:
        # Laplacian matrix
        Ddiag = np.sum(A, 1, keepdims=True)
    if len({"L","DL","DLD"}.intersection(set(ordering_methods))) > 0:
        # Laplacian matrix
        Dmat = np.diag(Ddiag.squeeze())
        L = Dmat - A
    if len({"DAD", "DLD"}.intersection(set(ordering_methods))) > 0:
        # Normalizing matrix D^{-1/2}
        Dm12 = np.diag(1/np.sqrt(Ddiag).squeeze())

    # Compute orderings
    times = {}
    for ordering_method in ordering_methods:

        if ordering_method == "L":
            t0 = time.time()
            # e_L, v_L = scipy.sparse.linalg.eigsh(L, k=2, which="SM")
            e_L, v_L = scipy.linalg.eigh(L)
            fvec_L = v_L[:, np.argsort(e_L)[1]]
            times['t_L'] = time.time() - t0
            orders["L"] = flip_if_decreasing(fvec_L)

        elif ordering_method == "DA":
            # Asymmetric normalized adjacency matrix
            DA = np.diag(1/Ddiag.squeeze()) @ A

            t0 = time.time()
            # e_DA, v_DA = scipy.sparse.linalg.eigs(DA, k=2, which="LM")
            e_DA, v_DA = scipy.linalg.eig(DA)
            fvec_DA = np.real(v_DA[:, np.argsort(e_DA)[-2]])
            times['t_DA'] = time.time()-t0
            orders["DA"] = flip_if_decreasing(fvec_DA)

        elif ordering_method == "DAD":
            # Symmetric normalized adjacency matrix
            DAD = Dm12 @ A @ Dm12

            t0 = time.time()
            # e_DAD, v_DAD = scipy.sparse.linalg.eigsh(DAD, k=2, which="LM")
            e_DAD, v_DAD = scipy.linalg.eigh(DAD)
            fvec_DAD = v_DAD[:, np.argsort(e_DAD)[-2]]
            times['t_DAD'] = time.time()-t0
            orders["DAD"] = flip_if_decreasing(fvec_DAD)

        elif ordering_method == "DL":
            # Asymmetric normalized laplacian
            DL = np.diag(1/Ddiag.squeeze()) @ L

            t0 = time.time()
            # e_DL, v_DL = scipy.sparse.linalg.eigs(DL, k=2, which="SM")  # DL is asymmetric
            e_DL, v_DL = scipy.linalg.eig(DL)
            fvec_DL = np.real(v_DL[:, np.argsort(e_DL)[1]])
            times['t_DL'] = time.time()-t0
            orders["DL"] = flip_if_decreasing(fvec_DL)

        elif ordering_method == "DLD":
            # Symmetric normalized laplacian
            DLD = Dm12 @ L @ Dm12

            t0 = time.time()
            # e_DLD, v_DLD = scipy.sparse.linalg.eigsh(DLD, k=2, which="SM")
            e_DLD, v_DLD = scipy.linalg.eigh(DLD)
            fvec_DLD = v_DLD[:, np.argsort(e_DLD)[1]]
            times['t_DLD'] = time.time()-t0
            orders["DLD"] = flip_if_decreasing(fvec_DLD)

        elif ordering_method == "UMAP":
            # Seriation with UMAP
            warnings.filterwarnings('ignore', '.*using precomputed metric.*', )
            t0 = time.time()
            mapper = umap.UMAP(n_components=1,
                               n_neighbors=param_dict["neighb"],
                               min_dist=param_dict["mindist"],
                               n_epochs=param_dict["epochs"],
                               metric=param_dict["metric"],
                               random_state=param_dict["rand_seed"]
                               )
            coords_UMAP = mapper.fit_transform(dist_mat)
            times['t_UMAP'] = time.time()-t0
            orders["UMAP"] = flip_if_decreasing(coords_UMAP.squeeze())

        elif ordering_method == "UKSD":
            # Seriation with UMAP and nearest neighbors from the similarity matrix
            # Transform neighbor graph into a UMAP compatible format
            n_neighbors = None if similarity == "QOT" else param_dict["neighb"]
            knn_simdist = umap_utils.nearest_neighbors_from_neighbor_graph(A, n_neighbors=n_neighbors)
            # Feed knn_simdist to UMAP
            if similarity == "QOT":
                # If similarity computed with QOT, take the max number of neighbors possible
                K_neighbors = knn_simdist[0].shape[1]
            else:
                # Else, take the amount necessary for UMAP.
                K_neighbors = param_dict["neighb"]
            t0 = time.time()
            knn_simdist_umap = umap.UMAP(n_components=1,
                                         n_neighbors=K_neighbors,
                                         min_dist=param_dict["mindist"],
                                         n_epochs=param_dict["epochs"],
                                         precomputed_knn=knn_simdist,
                                         random_state=param_dict["rand_seed"],
                                         force_approximation_algorithm=True)
            coords_UKSD = knn_simdist_umap.fit_transform(np.ones((T, 1)))
            times['t_UKSD'] = time.time()-t0
            orders["UKSD"] = flip_if_decreasing(coords_UKSD.squeeze())

        elif ordering_method == "UKRD":
            # Seriation with UMAP and nearest neighbors from the similarity matrix
            # Transform neighbor graph into a UMAP compatible format
            n_neighbors = None if similarity == "QOT" else param_dict["neighb"]
            knn_realdist = umap_utils.nearest_neighbors_from_neighbor_graph(A, n_neighbors=n_neighbors, D=dist_mat)
            # Feed knn_simdist to UMAP
            if similarity == "QOT":
                # If similarity computed with QOT, take the max number of neighbors possible
                K_neighbors = knn_realdist[0].shape[1]
            else:
                # Else, take the amount necessary for UMAP.
                K_neighbors = param_dict["neighb"]
            t0 = time.time()
            knn_realdist_umap = umap.UMAP(n_components=1,
                                          n_neighbors=K_neighbors,
                                          min_dist=param_dict["mindist"],
                                          n_epochs=param_dict["epochs"],
                                          precomputed_knn=knn_realdist,
                                          random_state=param_dict["rand_seed"],
                                          force_approximation_algorithm=True)
            coords_UKRD = knn_realdist_umap.fit_transform(np.ones((T, 1)))
            times['t_UKRD'] = time.time()-t0
            orders["UKRD"] = flip_if_decreasing(coords_UKRD.squeeze())

    return orders, times

