from collections import namedtuple

import numpy as np

from .vectorspace import VectorSpaceArrays, VectorSpaceHandles
from . import util
from . import parallel
from . py2to3 import range


def compute_BPOD_arrays(
    direct_vecs, adjoint_vecs, num_inputs=1, num_outputs=1,
    direct_mode_indices=None, adjoint_mode_indices=None,
    inner_product_weights=None, atol=1e-13, rtol=None):
    """Computes BPOD modes using data stored in arrays, using method of
    snapshots.

    Args:
        ``direct_vecs``: Array whose columns are direct data vectors
        (:math:`X`). They should be stacked so that if there are :math:`p`
        inputs, the first :math:`p` columns should all correspond to the same
        timestep.  For instance, these are often all initial conditions of
        :math:`p` different impulse responses.

        ``adjoint_vecs``: Array whose columns are adjoint data vectors
        (:math:`Y`). They should be stacked so that if there are :math:`q`
        outputs, the first :math:`q` columns should all correspond to the same
        timestep.  For instance, these are often all initial conditions of
        :math:`q` different adjoint impulse responses.

    Kwargs:
        ``num_inputs``: Number of inputs to the system.

        ``num_outputs``: Number of outputs of the system.

        ``direct_mode_indices``: List of indices describing which direct modes
        to compute.  Examples are ``range(10)`` or ``[3, 0, 6, 8]``.  If no
        mode indices are specified, then all modes will be computed.

        ``adjoint_mode_indices``: List of indices describing which adjoint
        modes to compute.  Examples are ``range(10)`` or ``[3, 0, 6, 8]``.  If
        no mode indices are specified, then all modes will be computed.

        ``inner_product_weights``: 1D or 2D array of inner product weights.
        Corresponds to :math:`W` in inner product :math:`v_1^* W v_2`.

        ``atol``: Level below which Hankel singular values are truncated.

        ``rtol``: Maximum relative difference between largest and smallest
        Hankel singular values.  Smaller ones are truncated.

    Returns:
        ``res``: Results of BPOD computation, stored in a namedtuple with
        the following attributes:

        * ``sing_vals``: 1D array of Hankel singular values (:math:`E`).

        * ``direct_modes``: Array whose columns are direct modes.

        * ``adjoint_modes``: Array whose columns are adjoint modes.

        * ``direct_proj_coeffs``: Array of projection coefficients for direct
          vector objects, expressed as a linear combination of direct BPOD
          modes.  Columns correspond to direct vector objects, rows correspond
          to direct BPOD modes.

        * ``adjoint_proj_coeffs``: Array of projection coefficients for adjoint
           vector objects, expressed as a linear combination of adjoint BPOD
           modes.  Columns correspond to adjoint vector objects, rows correspond
           to adjoint BPOD modes.

        * ``L_sing_vecs``: Array whose columns are left singular vectors of
          Hankel array (:math:`U`).

        * ``R_sing_vecs``: Array whose columns are right singular vectors of
          Hankel array (:math:`V`).

        * ``Hankel_array``: Hankel array (:math:`Y^* W X`).

        Attributes can be accessed using calls like ``res.direct_modes``.  To
        see all available attributes, use ``print(res)``.

    See also :py:class:`BPODHandles`.

    """
    if parallel.is_distributed():
        raise RuntimeError('Cannot run in parallel.')
    vec_space = VectorSpaceArrays(weights=inner_product_weights)

    # Compute first column (of chunks) of Hankel array
    all_adjoint_first_direct = vec_space.compute_inner_product_array(
        adjoint_vecs, direct_vecs[:, :num_inputs])
    all_adjoint_first_direct_list = [
        all_adjoint_first_direct[
            i * num_outputs:(i + 1) * num_outputs, :num_inputs]
        for i in range(all_adjoint_first_direct.shape[0] // num_outputs)]

    # Compute last row (of chunks) of Hankel array
    last_adjoint_all_direct = vec_space.compute_inner_product_array(
        adjoint_vecs[:, -num_outputs:], direct_vecs)
    last_adjoint_all_direct_list = [
        last_adjoint_all_direct[:, j * num_inputs:(j + 1) * num_inputs]
        for j in range(last_adjoint_all_direct.shape[1] // num_inputs)]

    # Compute Hankel array by computing unique chunks and restacking, rather
    # than computing all inner products.  This is more efficient, as it takes
    # advantage of the Hankel structure of the array to avoid duplicate
    # computations.
    Hankel_array = util.Hankel_chunks(
        all_adjoint_first_direct_list, last_adjoint_all_direct_list)

    # Compute BPOD modes
    L_sing_vecs, sing_vals, R_sing_vecs = util.svd(
        Hankel_array, atol=atol, rtol=rtol)
    sing_vals_sqrt_inv = np.diag(sing_vals ** -0.5)
    direct_build_coeffs = R_sing_vecs.dot(sing_vals_sqrt_inv)
    direct_modes = vec_space.lin_combine(
        direct_vecs, direct_build_coeffs,
        coeff_array_col_indices=direct_mode_indices)
    adjoint_build_coeffs = L_sing_vecs.dot(sing_vals_sqrt_inv)
    adjoint_modes = vec_space.lin_combine(
        adjoint_vecs, adjoint_build_coeffs,
        coeff_array_col_indices=adjoint_mode_indices)

    # Compute projection coefficients
    direct_proj_coeffs = np.diag(sing_vals ** 0.5).dot(R_sing_vecs.conj().T)
    adjoint_proj_coeffs = np.diag(sing_vals ** 0.5).dot(L_sing_vecs.conj().T)

    # Return a namedtuple
    BPOD_results = namedtuple(
        'BPOD_results', [
            'sing_vals', 'direct_modes', 'adjoint_modes',
            'direct_proj_coeffs', 'adjoint_proj_coeffs',
            'L_sing_vecs', 'R_sing_vecs', 'Hankel_array'])
    return BPOD_results(
        sing_vals=sing_vals,
        direct_modes=direct_modes, adjoint_modes=adjoint_modes,
        direct_proj_coeffs=direct_proj_coeffs,
        adjoint_proj_coeffs=adjoint_proj_coeffs,
        L_sing_vecs=L_sing_vecs, R_sing_vecs=R_sing_vecs,
        Hankel_array=Hankel_array)


class BPODHandles(object):
    """Balanced Proper Orthogonal Decomposition implemented for large datasets.

    Args:
        ``inner_product``: Function that computes inner product of two vector
        objects.

    Kwargs:
        ``put_array``: Function to put an array out of modred, e.g., write it to
        file.

      	``get_array``: Function to get an array into modred, e.g., load it from
        file.

        ``max_vecs_per_node``: Maximum number of vectors that can be stored in
        memory, per node.

        ``verbosity``: 1 prints progress and warnings, 0 prints almost nothing.

    Computes direct and adjoint BPOD modes from direct and adjoint vector
    objects (or handles).  Uses :py:class:`vectorspace.VectorSpaceHandles` for
    low level functions.

    Usage::

      myBPOD = BPODHandles(my_inner_product, max_vecs_per_node=500)
      myBPOD.compute_decomp(direct_vec_handles, adjoint_vec_handles)
      myBPOD.compute_direct_modes(range(50), direct_modes)
      myBPOD.compute_adjoint_modes(range(50), adjoint_modes)

    See also :py:func:`compute_BPOD_arrays` and :mod:`vectors`.
    """
    def __init__(
        self, inner_product, put_array=util.save_array_text,
        get_array=util.load_array_text,max_vecs_per_node=None, verbosity=1):
        """Constructor """
        self.get_array = get_array
        self.put_array = put_array
        self.verbosity = verbosity
        self.L_sing_vecs = None
        self.R_sing_vecs = None
        self.sing_vals = None
        self.Hankel_array = None
        # Class that contains all of the low-level vec operations
        self.vec_space = VectorSpaceHandles(
            inner_product=inner_product, max_vecs_per_node=max_vecs_per_node,
            verbosity=verbosity)
        self.direct_vec_handles = None
        self.adjoint_vec_handles = None


    def get_decomp(self, sing_vals_src, L_sing_vecs_src, R_sing_vecs_src):
        """Gets the decomposition arrays from sources (memory or file).

        Args:
            ``sing_vals_src``: Source from which to retrieve Hankel singular
            values.

            ``L_sing_vecs_src``: Source from which to retrieve left singular
            vectors of Hankel array.

            ``R_sing_vecs_src``: Source from which to retrieve right singular
            vectors of Hankel array.
        """
        self.sing_vals = np.squeeze(parallel.call_and_bcast(
            self.get_array, sing_vals_src))
        self.L_sing_vecs = parallel.call_and_bcast(
            self.get_array, L_sing_vecs_src)
        self.R_sing_vecs = parallel.call_and_bcast(
            self.get_array, R_sing_vecs_src)


    def get_Hankel_array(self, src):
        """Gets the Hankel array from source (memory or file).

        Args:
            ``src``: Source from which to retrieve Hankel singular values.
        """
        self.Hankel_array = parallel.call_and_bcast(self.get_array, src)


    def get_direct_proj_coeffs(self, src):
        """Gets the direct projection coefficients from source (memory or file).

        Args:
            ``src``: Source from which to retrieve direct projection
            coefficients.
        """
        self.direct_proj_coeffs = parallel.call_and_bcast(self.get_array, src)


    def get_adjoint_proj_coeffs(self, src):
        """Gets the adjoint projection coefficients from source (memory or
        file).

        Args:
            ``src``: Source from which to retrieve adjoint projection
            coefficients.
        """
        self.adjoint_proj_coeffs = parallel.call_and_bcast(self.get_array, src)


    def put_decomp(self, sing_vals_dest, L_sing_vecs_dest, R_sing_vecs_dest):
        """Puts the decomposition arrays in destinations (memory or file).

        Args:
            ``sing_vals_dest``: Destination in which to put Hankel singular
            values.

            ``L_sing_vecs_dest``: Destination in which to put left singular
            vectors of Hankel array.

            ``R_sing_vecs_dest``: Destination in which to put right singular
            vectors of Hankel array.
        """
        # Don't check if rank is zero because the following methods do.
        self.put_sing_vals(sing_vals_dest)
        self.put_L_sing_vecs(L_sing_vecs_dest)
        self.put_R_sing_vecs(R_sing_vecs_dest)


    def put_sing_vals(self, dest):
        """Puts Hankel singular values to ``dest``."""
        if parallel.is_rank_zero():
            self.put_array(self.sing_vals, dest)
        parallel.barrier()


    def put_L_sing_vecs(self, dest):
        """Puts left singular vectors of Hankel array to ``dest``."""
        if parallel.is_rank_zero():
            self.put_array(self.L_sing_vecs, dest)
        parallel.barrier()


    def put_R_sing_vecs(self, dest):
        """Puts right singular vectors of Hankel array to ``dest``."""
        if parallel.is_rank_zero():
            self.put_array(self.R_sing_vecs, dest)
        parallel.barrier()


    def put_Hankel_array(self, dest):
        """Puts Hankel array to ``dest``."""
        if parallel.is_rank_zero():
            self.put_array(self.Hankel_array, dest)
        parallel.barrier()


    def put_direct_proj_coeffs(self, dest):
        """Puts direct projection coefficients to ``dest``"""
        if parallel.is_rank_zero():
            self.put_array(self.direct_proj_coeffs, dest)
        parallel.barrier()


    def put_adjoint_proj_coeffs(self, dest):
        """Puts adjoint projection coefficients to ``dest``"""
        if parallel.is_rank_zero():
            self.put_array(self.adjoint_proj_coeffs, dest)
        parallel.barrier()


    def compute_SVD(self, atol=1e-13, rtol=None):
        """Computes singular value decomposition of the Hankel array.

       Kwargs:
            ``atol``: Level below which Hankel singular values are truncated.

            ``rtol``: Maximum relative difference between largest and smallest
            Hankel singular values.  Smaller ones are truncated.

        Useful if you already have the Hankel array and want to avoid
        recomputing it.

        Usage::

          my_BPOD.Hankel_array = pre_existing_Hankel_array
          my_BPOD.compute_SVD()
          my_BPOD.compute_direct_modes(
              range(10), mode_handles, direct_vec_handles=direct_vec_handles)
        """
        self.L_sing_vecs, self.sing_vals, self.R_sing_vecs =\
            parallel.call_and_bcast(
            util.svd, self.Hankel_array, atol=atol, rtol=rtol)


    def sanity_check(self, test_vec_handle):
        """Checks that user-supplied vector handle and vector satisfy
        requirements.

        Args:
            ``test_vec_handle``: A vector handle to test.

        See :py:meth:`vectorspace.VectorSpaceHandles.sanity_check`.
        """
        self.vec_space.sanity_check(test_vec_handle)


    def compute_decomp(
        self, direct_vec_handles, adjoint_vec_handles, num_inputs=1,
        num_outputs=1, atol=1e-13, rtol=None):
        """Computes Hankel array :math:`H=Y^*X` and its singular value
        decomposition :math:`UEV^*=H`.

        Args:
            ``direct_vec_handles``: List of handles for direct vector objects
            (:math:`X`).  They should be stacked so that if there are :math:`p`
            inputs, the first :math:`p` handles should all correspond to the
            same timestep.  For instance, these are often all initial conditions
            of :math:`p` different impulse responses.

            ``adjoint_vec_handles``: List of handles for adjoint vector objects
            (:math:`Y`).  They should be stacked so that if there are :math:`a`
            outputs, the first :math:`q` handles should all correspond to the
            same timestep.  For instance, these are often all initial conditions
            of :math:`p` different adjointimpulse responses.

        Kwargs:
            ``num_inputs``: Number of inputs to the system.

            ``num_outputs``: Number of outputs of the system.

            ``atol``: Level below which Hankel singular values are truncated.

            ``rtol``: Maximum relative difference between largest and smallest
            Hankel singular values.  Smaller ones are truncated.

        Returns:
            ``sing_vals``: 1D array of Hankel singular values (:math:`E`).

            ``L_sing_vecs``: Array of left singular vectors of Hankel array
            (:math:`U`).

            ``R_sing_vecs``: Array of right singular vectors of Hankel array
            (:math:`V`).
        """
        self.direct_vec_handles = direct_vec_handles
        self.adjoint_vec_handles = adjoint_vec_handles

        # Compute first column (of chunks) of Hankel array
        all_adjoint_first_direct = np.array(
            self.vec_space.compute_inner_product_array(
            self.adjoint_vec_handles, self.direct_vec_handles[:num_inputs]))

        # Compute last row (of chunks) of Hankel array
        last_adjoint_all_direct = np.array(
            self.vec_space.compute_inner_product_array(
            self.adjoint_vec_handles[-num_outputs:], self.direct_vec_handles))

        # Convert arrays of inner products into lists of array chunks
        all_adjoint_first_direct_list = [
            all_adjoint_first_direct[
                i * num_outputs:(i + 1) * num_outputs, :num_inputs]
            for i in range(all_adjoint_first_direct.shape[0] // num_outputs)]
        last_adjoint_all_direct_list = [
            last_adjoint_all_direct[:, j * num_inputs:(j + 1) * num_inputs]
            for j in range(last_adjoint_all_direct.shape[1] // num_inputs)]

        # Compute Hankel array by computing unique chunks and restacking.  This
        # is more efficient because it takes advantage of the Hankel structure
        # of the array to avoid duplicate computations.
        self.Hankel_array = parallel.call_and_bcast(
            util.Hankel_chunks,
            all_adjoint_first_direct_list, last_adjoint_all_direct_list)

        # Compute BPOD decomposition
        self.compute_SVD(atol=atol, rtol=rtol)

        # Return values
        return self.sing_vals, self.L_sing_vecs, self.R_sing_vecs


    def compute_direct_modes(
        self, mode_indices, mode_handles, direct_vec_handles=None):
        """Computes direct BPOD modes and calls ``put`` on them using mode
        handles.

        Args:
            ``mode_indices``: List of indices describing which direct modes to
            compute, e.g. ``range(10)`` or ``[3, 0, 5]``.

            ``mode_handles``: List of handles for direct modes to compute.

        Kwargs:
            ``direct_vec_handles``: List of handles for direct vector objects.
            Optional if given when calling :py:meth:`compute_decomp`.
        """
        if direct_vec_handles is not None:
            self.direct_vec_handles = util.make_iterable(direct_vec_handles)
        if self.direct_vec_handles is None:
            raise util.UndefinedError('direct_vec_handles undefined')
        build_coeffs = self.R_sing_vecs.dot(np.diag(self.sing_vals ** -0.5))
        self.vec_space.lin_combine(
            mode_handles, self.direct_vec_handles, build_coeffs,
            coeff_array_col_indices=mode_indices)


    def compute_adjoint_modes(
        self, mode_indices, mode_handles, adjoint_vec_handles=None):
        """Computes adjoint BPOD modes and calls ``put`` on them using mode
        handles.

        Args:
            ``mode_indices``: List of indices describing which adjoint modes to
            compute, e.g. ``range(10)`` or ``[3, 0, 5]``.

            ``mode_handles``: List of handles for adjoint modes to compute.

        Kwargs:
            ``adjoint_vec_handles``: List of handles for adjoint vector objects.
            Optional if given when calling :py:meth:`compute_decomp`.
        """
        if adjoint_vec_handles is not None:
            self.adjoint_vec_handles = util.make_iterable(adjoint_vec_handles)
        if self.adjoint_vec_handles is None:
            raise util.UndefinedError('adjoint_vec_handles undefined')
        build_coeffs = self.L_sing_vecs.dot(np.diag(self.sing_vals ** -0.5))
        self.vec_space.lin_combine(
            mode_handles, self.adjoint_vec_handles, build_coeffs,
            coeff_array_col_indices=mode_indices)


    def compute_direct_proj_coeffs(self):
        """Computes biorthogonal projection of direct vector objects onto
        direct BPOD modes, using adjoint BPOD modes.

        Returns:
            ``direct_proj_coeffs``: Array of projection coefficients for direct
            vector objects, expressed as a linear combination of direct BPOD
            modes.  Columns correspond to direct vector objects, rows
            correspond to direct BPOD modes.
        """
        self.direct_proj_coeffs = np.diag(self.sing_vals ** 0.5).dot(
            self.R_sing_vecs.conj().T)
        return self.direct_proj_coeffs


    def compute_adjoint_proj_coeffs(self):
        """Computes biorthogonal projection of adjoint vector objects onto
        adjoint BPOD modes, using direct BPOD modes.

        Returns:
            ``adjoint_proj_coeffs``: Array of projection coefficients for
            adjoint vector objects, expressed as a linear combination of
            adjoint BPOD modes.  Columns correspond to adjoint vector objects,
            rows correspond to adjoint BPOD modes.
        """
        self.adjoint_proj_coeffs = np.diag(self.sing_vals ** 0.5).dot(
            self.L_sing_vecs.conj().T)
        return self.adjoint_proj_coeffs
