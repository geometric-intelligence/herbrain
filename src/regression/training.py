"""Functions for parameterized regression."""

import os

import numpy as np

os.environ["GEOMSTATS_BACKEND"] = "pytorch"  # noqa: E402

import geomstats.backend as gs
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

import H2_SurfaceMatch.utils.input_output as h2_io  # noqa: E402
from src.regression.geodesic_regression import GeodesicRegression


def save_regression_results(
    dataset_name,
    y,
    X,
    space,
    true_coef,
    regr_intercept,
    regr_coef,
    results_dir,
    config,
    estimator,
    model=None,
    y_hat=None,
    lr_score_array=None,
):
    """Save regression results to files.

    Parameters
    ----------
    dataset_name: string, either "synthetic_mesh" or "menstrual_mesh"
    y: input data given to regression (points on manifold)
    true_intercept: numpy array, the true intercept
    true_coef: numpy array, the true slope
    regr_intercept: numpy array, the intercept calculated via regression
    regr_coef: numpy array, the slope calculated via regression
    model: linear regression or geodesic regression
    estimator: string, the estimator used for regression
        GLS: Geodesic Least Squares applied to geodesic regression
        LLS: Linear Least Squares applied to geodesic regression
        Lin2015: Linear Least Squares applied to linear regression, then projected to the manifold.
        LR: Linear Regression
        PLS: maximum likelihood estimator of projected linear noise model
    results_directory: string, the directory in which to save the results
    y_hat: numpy array, the y values predicted by the regression model.
    """
    if model is None:
        suffix = f"{dataset_name}"
    else:
        suffix = f"{dataset_name}_model{model}_estimator{estimator}"
    true_intercept_path = os.path.join(results_dir, f"true_intercept_{suffix}")
    true_coef_path = os.path.join(results_dir, f"true_coef_{suffix}")
    regr_intercept_path = os.path.join(results_dir, f"regr_intercept_{suffix}")
    y_path = os.path.join(results_dir, f"y_{suffix}")
    X_path = os.path.join(results_dir, f"X_{suffix}")
    y_hat_path = os.path.join(results_dir, f"y_hat_{suffix}")

    if dataset_name == "synthetic_mesh" or dataset_name == "menstrual_mesh":
        faces = gs.array(space.faces).numpy()
        mesh_sequence_vertices = y
        h2_io.save_data(
            true_intercept_path,
            ".ply",
            gs.array(mesh_sequence_vertices[0]).numpy(),
            faces,
        )
        h2_io.save_data(
            regr_intercept_path,
            ".ply",
            gs.array(regr_intercept).numpy(),
            faces,
        )

        if not os.path.exists(y_path):
            os.makedirs(y_path)

        for i, mesh in enumerate(mesh_sequence_vertices):
            mesh_path = os.path.join(y_path, f"mesh_{i}")
            h2_io.save_data(
                mesh_path,
                ".ply",
                gs.array(mesh).numpy(),
                faces,
            )

        if y_hat is not None:
            if not os.path.exists(y_hat_path):
                os.makedirs(y_hat_path)

            for i, mesh in enumerate(y_hat):
                mesh_path = os.path.join(y_hat_path, f"mesh_{i}")
                h2_io.save_data(
                    mesh_path,
                    ".ply",
                    gs.array(mesh).numpy(),
                    faces,
                )

            if lr_score_array is not None:
                score_path = os.path.join(y_hat_path, f"R2_score_{suffix}")
                np.savetxt(score_path, lr_score_array)

    np.savetxt(true_coef_path, true_coef)
    np.savetxt(X_path, X)

    print("regr_coef.shape: ", regr_coef.shape)
    if len(regr_coef.shape) > 2:
        for i, coef in enumerate(regr_coef):
            regr_coef_path = os.path.join(results_dir, f"regr_coef_{suffix}_degree_{i}")
            np.savetxt(regr_coef_path, coef)


def fit_geodesic_regression(
    y,
    space,
    X,
    tol,
    intercept_hat_guess,
    coef_hat_guess,
    initialization="warm_start",
    estimator="GLS",
    compute_iterations=False,
    use_cuda=True,
    device_id=1,
):
    """Perform regression on parameterized meshes or benchmark data.

    Parameters
    ----------
    y:
        for meshes- list of vertices of meshes.
        for benchmark- list of points
    EACH MESH is a numpy array of shape (n, 3)
    space: space on which to perform regression
    X: list of X corresponding to y
    intercept_hat_guess: initial guess for intercept of regression fit
    coef_hat_guess: initial guess for slope of regression fit
    tol: tolerance for geodesic regression. If none logged, value 0.001.

    Returns
    -------
    intercept_hat: intercept of regression fit
    coef_hat: slope of regression fit
    """
    print("estimator: ", estimator)

    # maxiter was 100
    # method was riemannian
    gr = GeodesicRegression(
        space,
        center_X=False,
        method="extrinsic",
        compute_training_score=False,
        verbose=True,
        tol=tol,
        initialization=initialization,
        estimator=estimator,
        use_cuda=use_cuda,
        device_id=device_id,
        embedding_space_dim=3 * len(y[0]),
    )

    if intercept_hat_guess is None:
        intercept_hat_guess = gs.array(y[0])  # .to(device = device)
    elif intercept_hat_guess.shape != y[0].shape:
        raise ValueError("intercept_hat_guess must be None or have y[0].shape")

    if coef_hat_guess is None:
        coef_hat_guess = gs.array(y[1] - y[0])  # .to(device = device)

    gr.intercept_ = intercept_hat_guess
    gr.coef_ = coef_hat_guess

    gr.fit(gs.array(X), gs.array(y))

    intercept_hat, coef_hat = gr.intercept_, gr.coef_

    return intercept_hat, coef_hat, gr


def fit_linear_regression(y, X, return_p=False, X_df=None):  # , device = "cuda:0"):
    """Perform linear regression on parameterized meshes.

    Parameters
    ----------
    y:
        for meshes: vertices of mesh sequence to be fit
        for benchmark: points to be fit
    X: list of X corresponding to y

    Returns
    -------
    intercept_hat: intercept of regression fit
    coef_hat: slope of regression fit
    """
    original_point_shape = y[0].shape
    original_y = y
    original_X = X

    print("y.shape: ", y.shape)
    print("original_point_shape: ", original_point_shape)
    print("X.shape: ", X.shape)

    if return_p:
        y = gs.array(y.reshape((len(X), -1)))
        X = gs.array(X.reshape(len(X), -1))
        print("regression reshaped y.shape: ", y.shape)
        print("regression reshaped X.shape: ", X.shape)

        lr = LinearRegression()

        lr.fit(X, y)
        intercept_hat, coef_hat = lr.intercept_, lr.coef_

        percent_significant_p_values = calculate_p_values(original_X, original_y, lr)

    else:
        y = gs.array(y.reshape((len(X), -1)))
        X = gs.array(X.reshape(len(X), -1))
        print("regression reshaped y.shape: ", y.shape)

        lr = LinearRegression()
        lr.fit(X, y)

        intercept_hat, coef_hat = lr.intercept_, lr.coef_

    if X.shape[1] > 1:
        coef_hat = coef_hat.reshape(
            X.shape[1], original_point_shape[0], original_point_shape[1]
        )
    else:
        coef_hat = coef_hat.reshape(original_point_shape)

    print("coef_hat.shape: ", coef_hat.shape)

    intercept_hat = intercept_hat.reshape(original_point_shape)

    intercept_hat = gs.array(intercept_hat)
    coef_hat = gs.array(coef_hat)

    if return_p:
        return intercept_hat, coef_hat, lr, np.array(percent_significant_p_values)
    return intercept_hat, coef_hat, lr


# def stouffer_combination(p_value_matrix):
#     """Combine p-values using Stouffer's method.

#     Parameters
#     ----------
#     p_value_matrix: matrix of p-values for one vertex, for one coefficient

#     Returns
#     -------
#     combined_p_value: combined p-value for one coefficient
#     """
#     t_score_matrix = gs.array(norm.ppf(1 - np.array(p_value_matrix)))
#     combined_t_score = gs.sum(t_score_matrix) / gs.sqrt(np.array(t_score_matrix).size)
#     combined_p_value = 1 - norm.cdf(combined_t_score)
#     return combined_p_value


# def fisher_combination(p_value_matrix):
#     """Combine p-values using Fisher's method.

#     Note: i think there is an error in this method
#     """
#     t_score_matrix = gs.array(norm.ppf(1 - np.array(p_value_matrix)))
#     chi_square = gs.sum(t_score_matrix**2)
#     combined_p_value = 1 - norm.cdf(np.sqrt(chi_square))
#     return combined_p_value.pvalue


def percent_significant_p_values(p_values, alpha=0.05):
    """Calculate the percentage of vertices that are significant.

    Parameters
    ----------
    p_values: list of p-values for each coefficient
    alpha: significance level

    Returns
    -------
    percentage: percentage of p values that are significant
    """
    p_values_flat = p_values.flatten()
    num_p_values = len(p_values_flat)
    num_significant_values = 0
    for i in range(num_p_values):
        if p_values_flat[i] < alpha:
            num_significant_values += 1
    percentage = num_significant_values / num_p_values
    return percentage


def calculate_p_values(X, y, lr, method="stouffer", tails=2):
    """Calculate p-values for linear regression.

    Parameters
    ----------
    X: list of X corresponding to y
    y: vertices of mesh sequence to be fit
    lr: linear regression model
    tails: number of tails for t-distribution (1 or 2)
        1 tail means we are testing for significance in one direction
        2 tails means we are testing for significance in both directions
        (aka, 2 tails means we are just testing for any significance at all,
        in either direction from the null hypothesis)
    we choose to test for significance in one direction, because we are
    interested in whether the hormone levels have a positive or negative
    effect on the mesh.
    method: method for combining p-values. Options are "stouffer" and "fisher".

    Note: can't use sm.OLS(y, X).fit() because y is a 3D array,
    and sm.OLS() expects a 1D array. When we try to flatten y completely,
    we lose the information about which sample each vertex belongs to.

    Returns
    -------
    p_values: list of p-values for each coefficient
    """
    y_predictions = lr.predict(X)
    y_predictions = y_predictions.reshape(y.shape)
    residuals = gs.array(y - y_predictions)

    num_samples = X.shape[0]
    num_hormones = X.shape[1]
    num_vertices = y.shape[1]

    # add a constant to the X matrix
    X = gs.array(X)
    X = gs.concatenate((gs.ones((len(X), 1)), X), axis=1)

    variance = gs.linalg.norm(residuals) / (num_samples - num_hormones)

    print("varance.shape: ", variance.shape)
    xtx_inv = np.linalg.inv(np.dot(X.T, X))
    xtx_inv_diag = np.diag(xtx_inv)
    sqrt_xtx_inv_diag = gs.sqrt(xtx_inv_diag)
    sqrt_xtx_inv_diag_coef = sqrt_xtx_inv_diag[1:]

    print("lr.coef_: ", lr.coef_.shape)
    coef_hats = lr.coef_
    coef_hats = coef_hats.reshape(num_hormones, num_vertices, 3)
    print("coefs.shape: ", coef_hats.shape)

    percentage_significant_vertices_per_hormone = []
    percentage_significant_components_per_hormone = []
    for i in range(num_hormones):
        t_value_matrix = coef_hats[i] / (sqrt_xtx_inv_diag_coef[i] * variance)
        p_value_matrix = (
            tails
            * (1 - stats.t.cdf(np.abs(t_value_matrix), num_samples - num_hormones))
            * num_vertices
            * 3
        )

        num_significant_values = 0
        for i_vertex in range(num_vertices):
            vertex_p_value_matrix = p_value_matrix[i_vertex]
            p_value = stats.combine_pvalues(vertex_p_value_matrix, method=method).pvalue
            # if method is "stouffer":
            #     p_value = stouffer_combination(vertex_p_value_matrix)
            # elif method is "fisher":
            #     p_value = fisher_combination(vertex_p_value_matrix)
            # else:
            #     raise ValueError("method must be 'stouffer' or 'fisher'")
            if p_value < 0.05:
                num_significant_values += 1

        percentage_significant_components = percent_significant_p_values(p_value_matrix)
        percentage_significant_components_per_hormone.append(
            percentage_significant_components
        )

        percentage_significant_vertices = num_significant_values / num_vertices
        percentage_significant_vertices_per_hormone.append(
            percentage_significant_vertices
        )

    percentage_significant_vertices_per_hormone = gs.array(
        percentage_significant_vertices_per_hormone
    )
    percentage_significant_components_per_hormone = gs.array(
        percentage_significant_components_per_hormone
    )

    print(
        "percentage_significant_vertices_per_hormone: ",
        percentage_significant_vertices_per_hormone,
    )
    print(
        "percentage_significant_components_per_hormone: ",
        percentage_significant_components_per_hormone,
    )
    return percentage_significant_vertices_per_hormone

    # t_values = []
    # for i in range(num_hormones):
    #     t_values.append(coef_hats[i] / (sqrt_xtx_inv_diag_coef[i] * variance))
    # t_values = gs.array(t_values)

    # p_values = tails * (1 - stats.t.cdf(np.abs(t_values), num_samples - num_hormones))
    # p_value_medians = np.median(p_values, axis=1)
    # print("p_values: ", p_values)
    # print("median of vertex p-values: ", p_value_medians)

    # adjusted_p_values = []
    # for i in range(num_hormones):
    #     print(f"p-value for hormone {i}: {p_values[i]}")
    #     p_value = p_values[i]
    #     adjusted_p_value = multipletests(p_value, method='fdr_bh')[1]
    #     adjusted_p_values.append(adjusted_p_value)
    # adjusted_p_values = gs.array(adjusted_p_values)

    # adjusted_p_value_medians = np.median(adjusted_p_values, axis=1)
    # print("adjusted p-values: ", adjusted_p_values)
    # print("median of vertex adjusted p-values: ", adjusted_p_value_medians)

    # return p_values, p_value_medians, adjusted_p_values, adjusted_p_value_medians


# def calculate_p_values_via_f_statistic(X, y, lr, tails=2):
#     """Calculate p-values for linear regression.

#     Parameters
#     ----------
#     X: list of X corresponding to y
#     y: vertices of mesh sequence to be fit
#     lr: linear regression model

#     f statistic: F = (RSS1 - RSS2) / (p2 - p1) / (RSS2 / (n - p2))
#     where RSS1 is the residual sum of squares for the full model,
#     RSS2 is the residual sum of squares for the reduced model,
#     p1 is the number of parameters in the full model,
#     p2 is the number of parameters in the reduced model,
#     n is the number of samples.

#     We use the f statistic because beta hat is a vector of coefficients,
#     and we want to test whether the entire vector is significant.

#     Question: should the degrees of freedom include the number of vertices?
#     Answer: no, because the vertices are not parameters in the model.
#     Question: is this true even though coef_hat has the same shape as y?
#     Answer: yes, because the coefficients are not vertices.
#     Question: is this true even though there is one coefficient for each vertex?

#     """
#     y_predictions = lr.predict(X)
#     y_predictions = y_predictions.reshape(y.shape)
#     RSS_residuals_full = gs.array(y - y_predictions)

#     # N = numer of days
#     # p = number of hormones

#     num_samples = X.shape[0]
#     num_hormones = X.shape[1]
#     num_vertices = y.shape[1]
#     print("num_samples: ", num_samples)
#     print("num_hormones: ", num_hormones)
#     print("num_vertices: ", num_vertices)

#     num_features = num_hormones * num_vertices * 3

#     p_values = []
#     for i in range(num_hormones):
#         beta_hats.append(lr.coef_[i])

#         # Define degrees of freedom for the full model and the reduced model
#         df_full = num_samples * num_vertices * 3 * num_hormones - num_features
#         # df_reduced =

#         # Calculate the residual sum of squares for the reduced model
#         beta_hat_reduced = []
#         for j in range(num_hormones):
#             if i != j:
#                 beta_hat_reduced.append(lr.coef_[j])
#             else:
#                 beta_hat_reduced.append(gs.zeros(lr.coef_[j].shape))
#         beta_hat_reduced = gs.array(beta_hat_reduced)

#         lr_reduced = LinearRegression()
#         lr_reduced.coef_ = beta_hat_reduced
#         y_predictions_reduced = lr_reduced.predict(X)
#         y_predictions_reduced = y_predictions_reduced.reshape(y.shape)
#         RSS_residuals_reduced = gs.array(y - y_predictions_reduced)

#         # Calculate the F-statistic
#         RSS_full = gs.linalg.norm(RSS_residuals_full)
#         RSS_reduced = gs.linalg.norm(RSS_residuals_reduced)
#         f_statistic = ((RSS_reduced - RSS_full) / (df_full - df_reduced)) / (
#             RSS_full / (num_features - df_full)
#         )

#         # Calculate the p-value
#         p_value = f.sf(f_statistic, df_full, num_features - df_full)


def fit_polynomial_regression(y, X, degree=2):
    """Perform polynomial regression on parameterized meshes.

    Also used to perform multiple linear regression.

    Parameters
    ----------
    y: vertices of mesh sequence to be fit
    X: list of X corresponding to y

    Returns
    -------
    intercept_hat: intercept of regression fit
    coef_hat: slope of regression fit
    """
    original_point_shape = y[0].shape

    y = gs.array(y.reshape((len(X), -1)))
    X = gs.array(X.reshape(len(X), 1))

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)  # X_poly is a matrix of shape (len(X), degree + 1)
    # The extra row is filled with 1's, which is the "intercept" term.

    print("X_poly.shape: ", X_poly.shape)
    print("X_poly: ", X_poly)

    lr = LinearRegression()
    lr.fit(X_poly, y)

    intercept_hat, coef_hats = lr.intercept_, lr.coef_

    print("coef_hat.shape: ", coef_hats.shape)

    coef_hats = coef_hats.reshape(
        degree, original_point_shape[0], original_point_shape[1]
    )
    print("reshaped coef_hats.shape:", coef_hats.shape)

    # intercept_term = coef_hats[0] # note: this is essentially zero. ignore.
    # coef_hat_linear = coef_hats[0]
    # coef_hat_quadratic = coef_hats[1]

    # coef_hat_linear = coef_hat_linear.reshape(original_point_shape)
    # coef_hat_quadratic = coef_hat_quadratic.reshape(original_point_shape)
    intercept_hat = intercept_hat.reshape(original_point_shape)

    # coef_hat_linear = gs.array(coef_hat_linear)
    # coef_hat_quadratic = gs.array(coef_hat_quadratic)
    coef_hats = gs.array(coef_hats)
    intercept_hat = gs.array(intercept_hat)

    print("original_point_shape: ", original_point_shape)
    print("coef_hats.shape: ", coef_hats.shape)

    return intercept_hat, coef_hats, lr


def compute_R2(y, X, test_indices, train_indices):
    """Compute R2 score for linear regression.

    Parameters
    ----------
    X: list of X corresponding to y
    y: vertices of mesh sequence to be fit
        (flattened s.t. array dimension <= 2)
    lr: linear regression model

    Returns
    -------
    score_array: [adjusted R2 score, normal R2 score]
    """
    X_train = gs.array(X[train_indices])
    X_test = gs.array(X[test_indices])
    y_train = gs.array(y[train_indices])
    y_test = gs.array(y[test_indices])

    print("X_pred: ", X)
    print("X_train: ", X_train)
    print("X_test: ", X_test)

    # X_train = gs.array(X_train.reshape(len(X_train), len(X_train[0])))
    # X_test = gs.array(X_test.reshape(len(X_test), len(X_test[0])))
    X_train = gs.array(X_train.reshape(len(X_train), -1))
    X_test = gs.array(X_test.reshape(len(X_test), -1))
    y_train = gs.array(y_train.reshape((len(X_train), -1)))
    y_test = gs.array(y_test.reshape((len(X_test), -1)))

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred_for_lr = lr.predict(X_test)

    normal_r2_score = r2_score(y_test, y_pred_for_lr)

    train_sample_size = len(y_train)
    n_independent_variables = X_train.shape[1]
    print("train_sample_size (n): ", train_sample_size)
    print("n_independent_variables (p): ", n_independent_variables)

    Adj_r2 = 1 - (1 - normal_r2_score) * (train_sample_size - 1) / (
        train_sample_size - n_independent_variables - 1
    )

    print("Adjusted R2 score: ", Adj_r2)
    print("R2 score: ", normal_r2_score)
    score_array = np.array([Adj_r2, normal_r2_score])

    return score_array
