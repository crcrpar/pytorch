import torch
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests, TEST_WITH_ROCM, TEST_WITH_SLOW
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes, skipCUDAIfRocm, skipMeta, ops)
from torch._six import inf, nan
from torch.testing._internal.common_methods_invocations import \
    (foreach_unary_op_db, foreach_binary_op_db, foreach_pointwise_op_db, foreach_min_max_op_db)

# Includes some values such that N * N won't be a multiple of 4,
# which should ensure we test the vectorized and non-vectorized
# kernel code paths.
N_values = [20, 23] if not TEST_WITH_SLOW else [23, 30, 300]

class TestForeach(TestCase):
    bin_ops = [
        (torch._foreach_add, torch._foreach_add_, torch.add),
        (torch._foreach_sub, torch._foreach_sub_, torch.sub),
        (torch._foreach_mul, torch._foreach_mul_, torch.mul),
        (torch._foreach_div, torch._foreach_div_, torch.div),
    ]

    def _get_test_data(self, device, dtype, N):
        if dtype in [torch.bfloat16, torch.bool, torch.float16]:
            tensors = [torch.randn(N, N, device=device).to(dtype) for _ in range(N)]
        elif dtype in torch.testing.get_all_int_dtypes():
            # Constrains the range between 1 and 10 for less stress on int8 tensors.
            tensors = [torch.randint(1, 10, (N, N), device=device, dtype=dtype) for _ in range(N)]
        else:
            tensors = [torch.randn(N, N, device=device, dtype=dtype) for _ in range(N)]

        return tensors

    def _test_binary_op(self, device, dtype, op, inputs1, inputs2, alpha=None):
        op_kwargs = {} if alpha is None else {'alpha': alpha}
        assert_kwargs = {}
        if self.device_type == 'cuda' and alpha is not None and dtype in (torch.bfloat16, torch.float16):
            # NOTE(mkozuki): the fastpath of tensorlists with alpha parameter is
            # less numerically stable than the slowpath.
            assert_kwargs = {
                'rtol': self.dtype_precisions[dtype][0],
                'atol': 0.05,
            }
        method = op.get_op()
        ref = op.get_ref()
        inplace = op.get_inplace()
        inplace_ref_name = op.get_inplace_ref_name()

        native_inputs2 = inputs2
        if isinstance(native_inputs2, (int, float, complex)):
            native_inputs2 = [native_inputs2 for _ in range(len(inputs1))]

        try:
            actual = method(inputs1, inputs2, **op_kwargs)
        except RuntimeError:
            with self.assertRaises(RuntimeError):
                [ref(i1, i2, **op_kwargs) for i1, i2 in zip(inputs1, native_inputs2)]
        else:
            expected = [ref(i1, i2, **op_kwargs) for i1, i2 in zip(inputs1, native_inputs2)]
            self.assertEqual(expected, actual, **assert_kwargs)

        if inplace is None:
            return
        copied_tensors = [t.clone().detach() for t in inputs1]
        try:
            inplace(copied_tensors, inputs2, **op_kwargs)
        except RuntimeError:
            with self.assertRaises(RuntimeError):
                [getattr(i1, inplace_ref_name)(i2, **op_kwargs) for i1, i2 in zip(inputs1, native_inputs2)]
        else:
            [getattr(i1, inplace_ref_name)(i2, **op_kwargs) for i1, i2 in zip(inputs1, native_inputs2)]
            self.assertEqual(inputs1, copied_tensors, **assert_kwargs)

    @ops(foreach_binary_op_db)
    def test_binary_ops_fastpath_tensorlists(self, device, dtype, op):
        if self.device_type != 'cuda':
            self.skipTest("fastpath tests is only for CUDA tensors")
        for N in N_values:
            inputs1 = op.sample_inputs(device, dtype, N, noncontiguous=False)
            inputs2 = op.sample_inputs(device, dtype, N, noncontiguous=False)
            self._test_binary_op(device, dtype, op, inputs1, inputs2)

    @ops(foreach_binary_op_db)
    def test_binary_ops_fastpath_scalarlist(self, device, dtype, op):
        if self.device_type != 'cuda':
            self.skipTest("fastpath tests is only for CUDA tensors")
        for N in N_values:
            for scalar in [1, 2.0, 3.0 + 0.3j, True]:
                inputs1 = op.sample_inputs(device, dtype, N, noncontiguous=False)
                inputs2 = [scalar for _ in range(N)]
                self._test_binary_op(device, dtype, op, inputs1, inputs2)

    @ops(foreach_binary_op_db)
    def test_binary_ops_fastpath_scalar(self, device, dtype, op):
        if self.device_type != 'cuda':
            self.skipTest("fastpath is only for CUDA tensors")
        for N in N_values:
            for scalar in [1, 2.0, 3.0 + 0.3j, True]:
                inputs1 = op.sample_inputs(device, dtype, N, noncontiguous=False)
                self._test_binary_op(device, dtype, op, inputs1, scalar)

    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_ops_slowpath_tensorlists(self, device, dtype, op):
        for N in N_values:
            inputs1 = op.sample_inputs(device, dtype, N, noncontiguous=True)
            inputs2 = op.sample_inputs(device, dtype, N, noncontiguous=True)
            self._test_binary_op(device, dtype, op, inputs1, inputs2)

    # TODO(mkozuki): Run the below test with `include_complex=True`.
    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes(include_complex=False))
    @ops(foreach_binary_op_db)
    def test_binary_ops_slowpath_scalarlist(self, device, dtype, op):
        for N in N_values:
            for scalar in [1, 2.0, 3.0 + 0.3j]:
                inputs1 = op.sample_inputs(device, dtype, N, noncontiguous=True)
                inputs2 = [scalar for _ in range(N)]
                self._test_binary_op(device, dtype, op, inputs1, inputs2)

    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_ops_slowpath_scalar(self, device, dtype, op):
        for N in N_values:
            for scalar in [1, 2.0, 3.0 + 0.3j]:
                inputs1 = op.sample_inputs(device, dtype, N, noncontiguous=True)
                self._test_binary_op(device, dtype, op, inputs1, scalar)

    @ops(foreach_binary_op_db)
    def test_binary_ops_tensorlists_with_alpha_fastpath(self, device, dtype, op):
        if self.device_type != 'cuda':
            self.skipTest("fastpath is only for CUDA tensors")
        if not op.supports_alpha_param:
            self.skipTest("{} does not support alpha parameter".format(op.name))
        for N in N_values:
            inputs1 = op.sample_inputs(device, dtype, N, noncontiguous=False)
            inputs2 = op.sample_inputs(device, dtype, N, noncontiguous=False)
            self._test_binary_op(device, dtype, op, inputs1, inputs2, alpha=10)

    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_ops_tensorlists_with_alpha_slowpath(self, device, dtype, op):
        for N in N_values:
            if not op.supports_alpha_param:
                self.skipTest("{} does not support alpha parameter".format(op.name))
            inputs1 = op.sample_inputs(device, dtype, N, noncontiguous=True)
            inputs2 = op.sample_inputs(device, dtype, N, noncontiguous=True)
            self._test_binary_op(device, dtype, op, inputs1, inputs2, alpha=10)

    # Unary Op Tests
    def _test_unary_op(self, device, dtype, op, tensors):
        method = op.get_op()
        ref = op.get_ref()
        inplace = op.get_inplace()
        inplace_ref_name = op.get_inplace_ref_name()

        try:
            actual = method(tensors)
        except RuntimeError as e_foreach:
            with self.assertRaises(RuntimeError):
                [ref(t) for t in tensors] 
        else:
            expected = [ref(t) for t in tensors]
            self.assertEqual(expected, actual)

        copied_tensors = [t.clone().detach() for t in tensors]
        try:
            inplace(copied_tensors)
        except RuntimeError:
            with self.assertRaises(RuntimeError):
                [getattr(t, inplace_ref_name)() for t in tensors]
        else:
            [getattr(t, inplace_ref_name)() for t in tensors]
            self.assertEqual(tensors, copied_tensors)

    @ops(foreach_unary_op_db)
    def test_unary_ops_fastpath(self, device, dtype, op):
        if self.device_type != 'cuda':
            self.skipTest("fastpath is only for CUDA tensors")
        for N in N_values:
            tensors = op.sample_inputs(device, dtype, N, noncontiguous=False)
            self._test_unary_op(device, dtype, op, tensors)

    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_unary_op_db)
    def test_unary_ops_slowpath(self, device, dtype, op):
        # TODO(mkozuki): Investigate why the skipped cases do not work.
        if op.name == "_foreach_sigmoid" and dtype in torch.testing.get_all_complex_dtypes():
            self.skipTest("_foreach_sigmoid on complex dtypes does not work")
        for N in N_values:
            tensors = op.sample_inputs(device, dtype, N, noncontiguous=True)
            self._test_unary_op(device, dtype, op, tensors)

    def _test_pointwise_op(self, device, dtype, op, tensors, inputs1, inputs2):
        assert_kwargs = {}
        if self.device_type == 'cuda' and dtype in (torch.bfloat16, torch.float16):
            assert_kwargs = {
                'rtol': self.dtype_precisions[dtype][0],
                'atol': 0.05,
            }
        method = op.get_op()
        ref = op.get_ref()
        inplace = op.get_inplace()
        inplace_ref_name = op.get_inplace_ref_name()

        try:
            actual = method(tensors, inputs1, inputs2)
        except RuntimeError:
            with self.assertRaises(RuntimeError):
                [ref(t, i1, i2) for t, i1, i2 in zip(tensors, inputs1, inputs2)]
        else:
            expected = [ref(t, i1, i2) for t, i1, i2 in zip(tensors, inputs1, inputs2)]
            self.assertEqual(expected, actual, **assert_kwargs)

        if inplace is None:
            return
        copied_tensors = [t.clone().detach() for t in tensors]
        if self.device_type == 'cpu' and dtype in (torch.bfloat16, torch.float16):
            with self.assertRaises(RuntimeError):
                inplace(copied_tensors, inputs1, inputs2)
            return
        try:
            inplace(copied_tensors, inputs1, inputs2)
        except RuntimeError:
            [getattr(t, inplace_ref_name)(i1, i2) for t, i1, i2 in zip(tensors, inputs1, inputs2)]
        else:
            [getattr(t, inplace_ref_name)(i1, i2) for t, i1, i2 in zip(tensors, inputs1, inputs2)]
            self.assertEqual(tensors, copied_tensors, **assert_kwargs)

    # FIXME(mkozuki): dtype of ForeachOpInfo is not correctly reflected to dtype configuration.
    @ops(foreach_pointwise_op_db)
    def test_pointwise_ops_fastpath(self, device, dtype, op):
        if self.device_type != 'cuda':
            self.skipTest('fastpath is only for cuda tensors')
        if dtype in torch.testing.get_all_int_dtypes() + [torch.bool, torch.bfloat16]:
            self.skipTest('integer, bool, and bfloat16 are not supported')
        for N in N_values:
            tensors = op.sample_inputs(device, dtype, N)
            inputs1 = op.sample_inputs(device, dtype, N)
            inputs2 = op.sample_inputs(device, dtype, N)
            self._test_pointwise_op(device, dtype, op, tensors, inputs1, inputs2)

    # FIXME(mkozuki): ForeachOpInfo is not correctly reflected to dtype configuration.
    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_pointwise_op_db)
    def test_pointwise_ops_slowpath(self, device, dtype, op):
        if dtype in torch.testing.get_all_int_dtypes() + [torch.bool]:
            self.skipTest("integer types and bool are not supported")
        for N in N_values:
            tensors = op.sample_inputs(device, dtype, N, noncontiguous=True)
            inputs1 = op.sample_inputs(device, dtype, N, noncontiguous=True)
            inputs2 = op.sample_inputs(device, dtype, N, noncontiguous=True)
            self._test_pointwise_op(device, dtype, op, tensors, inputs1, inputs2)

    @skipMeta
    @ops(foreach_min_max_op_db)
    def test_min_max(self, device, dtype, op):

        foreach_op = op.get_op()
        native_op = op.get_ref()

        for N in N_values:
            tensors1 = self._get_test_data(device, dtype, N)
            tensors2 = self._get_test_data(device, dtype, N)

            # Mimics cuda kernel dtype flow.  With fp16/bf16 input, runs in fp32 and casts output back to fp16/bf16.
            control_dtype = torch.float32 if (self.device_type == 'cuda' and
                                              dtype in (torch.float16, torch.bfloat16)) else dtype

            expected = [native_op(t1.to(dtype=control_dtype),
                                      t2.to(dtype=control_dtype)).to(dtype=dtype) for t1, t2 in zip(tensors1, tensors2)]
            actual = foreach_op(tensors1, tensors2)
            self.assertEqual(expected, actual)

    @skipMeta
    @ops(foreach_min_max_op_db)
    def test_min_max_inf_nan(self, device, dtype, op):

        if dtype in torch.testing.get_all_int_dtypes():
            raise self.skipTest('inf/nan cannot be converted to integer types')

        foreach_op = op.get_op()
        native_op = op.get_ref()

        a = [
            torch.tensor([inf], device=device, dtype=dtype),
            torch.tensor([-inf], device=device, dtype=dtype),
            torch.tensor([nan], device=device, dtype=dtype),
            torch.tensor([nan], device=device, dtype=dtype)
        ]

        b = [
            torch.tensor([-inf], device=device, dtype=dtype),
            torch.tensor([inf], device=device, dtype=dtype),
            torch.tensor([inf], device=device, dtype=dtype),
            torch.tensor([nan], device=device, dtype=dtype)
        ]

        expected = [native_op(a1, b1) for a1, b1 in zip(a, b)]
        actual = foreach_op(a, b)
        self.assertEqual(expected, actual)

        a = [
            torch.tensor([float('inf')], device=device, dtype=dtype),
            torch.tensor([-float('inf')], device=device, dtype=dtype),
            torch.tensor([float('nan')], device=device, dtype=dtype),
            torch.tensor([float('nan')], device=device, dtype=dtype)
        ]

        b = [
            torch.tensor([-float('inf')], device=device, dtype=dtype),
            torch.tensor([float('inf')], device=device, dtype=dtype),
            torch.tensor([float('inf')], device=device, dtype=dtype),
            torch.tensor([float('nan')], device=device, dtype=dtype)
        ]
        expected = [native_op(a1, b1) for a1, b1 in zip(a, b)]
        actual = foreach_op(a, b)
        self.assertEqual(expected, actual)

    #
    # Ops with scalar
    #
    @skipCUDAIfRocm
    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    def test_tensorlist_int_scalar_op(self, device, dtype):
        for N in N_values:
            for foreach_bin_op, foreach_bin_op_, torch_bin_op in self.bin_ops:
                tensors = self._get_test_data(device, dtype, N)
                scalar = 3

                if dtype == torch.bool:
                    if foreach_bin_op == torch._foreach_sub:
                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator,"):
                            res = foreach_bin_op(tensors, scalar)
                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator,"):
                            expected = [torch_bin_op(t, scalar) for t in tensors]

                        # Test In-place
                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator,"):
                            foreach_bin_op_(tensors, scalar)

                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator,"):
                            [t.sub_(scalar) for t in tensors]
                        continue

                    res = foreach_bin_op(tensors, scalar)
                    expected = [torch_bin_op(t, scalar) for t in tensors]
                    self.assertEqual(res, expected)

                    # Test In-place
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        foreach_bin_op_(tensors, scalar)

                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        [t.div_(scalar) for t in tensors]

                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        [t.mul_(scalar) for t in tensors]

                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        [t.add_(scalar) for t in tensors]

                    with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator, with a bool"):
                        [t.sub_(scalar) for t in tensors]
                    continue

                expected = [torch_bin_op(t, scalar) for t in tensors]
                res = foreach_bin_op(tensors, scalar)

                # In case of In-place division with integers, we can't change the dtype
                if foreach_bin_op_ == torch._foreach_div_ and dtype in torch.testing.integral_types():
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        [t.div_(scalar) for t in tensors]

                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        torch._foreach_div_(tensors, scalar)
                    continue

                self.assertEqual(res, expected)

                # In case of In-place op, we can't change the dtype
                foreach_bin_op_(tensors, scalar)
                self.assertEqual(tensors, expected)

    # TODO[Fix scalar list]:
    # We need to update codegen to correctly handle function overloads with float[] and int[].
    # As optimizers work with float tensors, the result will always be torch.float32 for now.
    # Current schema is using 'float[]' as scalar list type.
    @skipCUDAIfRocm
    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    def test_tensorlist_int_scalarlist_op(self, device, dtype):
        for N in N_values:
            for foreach_bin_op, foreach_bin_op_, torch_bin_op in self.bin_ops:
                tensors = self._get_test_data(device, dtype, N)
                scalars = [1 for _ in range(N)]
                expected = [torch_bin_op(t, s) for t, s in zip(tensors, scalars)]

                # we dont support bool and complex types on CUDA for now
                if dtype in torch.testing.get_all_complex_dtypes() and self.device_type == 'cuda':
                    with self.assertRaisesRegex(RuntimeError, "not implemented for"):
                        foreach_bin_op_(tensors, scalars)

                    with self.assertRaisesRegex(RuntimeError, "not implemented for"):
                        foreach_bin_op(tensors, scalars)
                    return

                res = foreach_bin_op(tensors, scalars)

                if dtype == torch.bool:
                    self.assertEqual(res, [torch_bin_op(t, s) for t, s in zip(tensors, scalars)])

                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        foreach_bin_op_(tensors, scalars)
                    return

                # test out of place
                self.assertEqual(res, expected)

                # test in-place
                if dtype in torch.testing.floating_types() and self.device_type == 'cpu':
                    foreach_bin_op_(tensors, scalars)
                    return
                else:
                    if foreach_bin_op_ == torch._foreach_div_ and dtype in torch.testing.integral_types():
                        with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                            foreach_bin_op_(tensors, scalars)
                    else:
                        foreach_bin_op_(tensors, scalars)
                        self.assertEqual(res, tensors)

    @skipCUDAIfRocm
    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    def test_tensorlist_float_scalar_op(self, device, dtype):
        for N in N_values:
            for foreach_bin_op, foreach_bin_op_, torch_bin_op in self.bin_ops:
                tensors = self._get_test_data(device, dtype, N)
                scalar = 3.3

                # Mimics cuda kernel dtype flow.  With fp16/bf16 input, runs in fp32 and casts output back to fp16/bf16.
                control_dtype = torch.float32 if (self.device_type == 'cuda' and
                                                  (dtype is torch.float16 or dtype is torch.bfloat16)) else dtype
                expected = [torch_bin_op(t.to(dtype=control_dtype),
                                         scalar) for t in tensors]
                if (dtype is torch.float16 or dtype is torch.bfloat16):
                    expected = [e.to(dtype=dtype) for e in expected]

                if dtype == torch.bool:
                    if foreach_bin_op == torch._foreach_sub:
                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator, with two bool"):
                            foreach_bin_op_(tensors, scalar)

                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator, with two bool"):
                            foreach_bin_op(tensors, scalar)
                    return

                res = foreach_bin_op(tensors, scalar)
                if (dtype is torch.float16 or dtype is torch.bfloat16) and TEST_WITH_ROCM:
                    self.assertEqual(res, expected, atol=1.e-3, rtol=self.dtype_precisions[dtype][0])
                else:
                    self.assertEqual(res, expected)

                if dtype in torch.testing.integral_types():
                    with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                        foreach_bin_op_(tensors, scalar)
                    return

                foreach_bin_op_(tensors, scalar)
                if (dtype is torch.float16 or dtype is torch.bfloat16) and TEST_WITH_ROCM:
                    self.assertEqual(tensors, expected, atol=1.e-3, rtol=self.dtype_precisions[dtype][0])
                else:
                    self.assertEqual(tensors, expected)

    @skipCUDAIfRocm
    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    def test_tensorlist_float_scalarlist_op(self, device, dtype):
        for N in N_values:
            for foreach_bin_op, foreach_bin_op_, torch_bin_op in self.bin_ops:
                tensors = self._get_test_data(device, dtype, N)
                scalars = [1.1 for _ in range(N)]

                # Bool case
                if dtype == torch.bool:
                    if foreach_bin_op == torch._foreach_sub:
                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator, with a bool tensor"):
                            expected = [torch_bin_op(t, s) for t, s in zip(tensors, scalars)]

                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator, with a bool tensor"):
                            res = foreach_bin_op(tensors, scalars)

                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator, with a bool tensor"):
                            [t.sub_(scalar) for t, scalar in zip(tensors, scalars)]

                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator, with a bool tensor"):
                            foreach_bin_op_(tensors, scalars)
                        continue

                    res = foreach_bin_op(tensors, scalars)
                    expected = [torch_bin_op(t, s) for t, s in zip(tensors, scalars)]
                    self.assertEqual(res, expected)

                    with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                        foreach_bin_op_(tensors, scalars)
                    continue

                # If incoming dtype is float16 or bfloat16, runs in float32 and casts output back to dtype.
                control_dtype = torch.float32 if (self.device_type == 'cuda' and
                                                  (dtype is torch.float16 or dtype is torch.bfloat16)) else dtype
                expected = [torch_bin_op(t.to(dtype=control_dtype), s) for t, s in zip(tensors, scalars)]
                if (dtype is torch.float16 or dtype is torch.bfloat16):
                    expected = [e.to(dtype=dtype) for e in expected]

                # we dont support bool and complex types on CUDA for now
                if (dtype in torch.testing.get_all_complex_dtypes() or dtype == torch.bool) and self.device_type == 'cuda':
                    with self.assertRaisesRegex(RuntimeError, "not implemented for"):
                        foreach_bin_op_(tensors, scalars)

                    with self.assertRaisesRegex(RuntimeError, "not implemented for"):
                        foreach_bin_op(tensors, scalars)
                    return

                res = foreach_bin_op(tensors, scalars)

                if dtype in torch.testing.integral_types() and self.device_type == 'cuda':
                    self.assertEqual(res, expected)
                    with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                        foreach_bin_op_(tensors, scalars)
                    continue
                else:
                    if (dtype is torch.float16 or dtype is torch.bfloat16) and TEST_WITH_ROCM:
                        self.assertEqual(res, expected, atol=1.e-3, rtol=self.dtype_precisions[dtype][0])
                    else:
                        self.assertEqual(res, expected)

                if dtype in torch.testing.integral_types() and self.device_type == "cpu":
                    with self.assertRaisesRegex(RuntimeError, "result type Float can't be cast to the desired output type"):
                        foreach_bin_op_(tensors, scalars)
                    continue

                foreach_bin_op_(tensors, scalars)
                if (dtype is torch.float16 or dtype is torch.bfloat16) and TEST_WITH_ROCM:
                    self.assertEqual(tensors, expected, atol=1.e-3, rtol=self.dtype_precisions[dtype][0])
                else:
                    self.assertEqual(tensors, expected)

    @skipCUDAIfRocm
    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    def test_tensorlist_complex_scalar_op(self, device, dtype):
        for N in N_values:
            for foreach_bin_op, foreach_bin_op_, torch_bin_op in self.bin_ops:
                tensors = self._get_test_data(device, dtype, N)
                scalar = 3 + 5j

                # Bool case
                if dtype == torch.bool:
                    if foreach_bin_op == torch._foreach_sub:
                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator,"):
                            foreach_bin_op_(tensors, scalar)

                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator,"):
                            foreach_bin_op(tensors, scalar)
                    continue

                res = foreach_bin_op(tensors, scalar)
                expected = [torch_bin_op(t, scalar) for t in tensors]
                self.assertEqual(res, expected)

                if dtype in torch.testing.get_all_fp_dtypes() and self.device_type == 'cuda':
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        foreach_bin_op_(tensors, scalar)
                    continue

                if dtype not in [torch.complex64, torch.complex128]:
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        foreach_bin_op_(tensors, scalar)
                else:
                    foreach_bin_op_(tensors, scalar)
                    self.assertEqual(res, tensors)

    @skipCUDAIfRocm
    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    def test_tensorlist_complex_scalarlist_op(self, device, dtype):
        for N in N_values:
            for foreach_bin_op, foreach_bin_op_, torch_bin_op in self.bin_ops:
                tensors = self._get_test_data(device, dtype, N)
                scalars = [3 + 5j for _ in range(N)]

                # Bool case
                if dtype == torch.bool:
                    if foreach_bin_op == torch._foreach_sub:
                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                            foreach_bin_op_(tensors, scalars)

                        with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                            foreach_bin_op(tensors, scalars)
                    continue

                expected = [torch_bin_op(t, s) for t, s in zip(tensors, scalars)]
                res = foreach_bin_op(tensors, scalars)
                self.assertEqual(res, expected)

                if dtype not in [torch.complex64, torch.complex128]:
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        foreach_bin_op_(tensors, scalars)
                else:
                    foreach_bin_op_(tensors, scalars)
                    self.assertEqual(res, tensors)

    @skipCUDAIfRocm
    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    def test_tensorlist_bool_scalar_op(self, device, dtype):
        for N in N_values:
            for foreach_bin_op, foreach_bin_op_, torch_bin_op in self.bin_ops:
                tensors = self._get_test_data(device, dtype, N)
                scalar = True

                if foreach_bin_op == torch._foreach_sub:
                    with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                        expected = [torch_bin_op(t, scalar) for t in tensors]

                    with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                        foreach_bin_op(tensors, scalar)

                    with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                        foreach_bin_op_(tensors, scalar)
                    continue

                expected = [torch_bin_op(t, scalar) for t in tensors]
                res = foreach_bin_op(tensors, scalar)
                self.assertEqual(expected, res)

                if dtype in torch.testing.integral_types_and(torch.bool) and foreach_bin_op == torch._foreach_div:
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output"):
                        foreach_bin_op_(tensors, scalar)
                else:
                    foreach_bin_op_(tensors, scalar)
                    self.assertEqual(tensors, res)

    @skipCUDAIfRocm
    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    def test_tensorlist_bool_scalarlist_op(self, device, dtype):
        for N in N_values:
            for foreach_bin_op, foreach_bin_op_, torch_bin_op in self.bin_ops:
                tensors = self._get_test_data(device, dtype, N)
                scalars = [True for _ in range(N)]

                # we dont support complex types on CUDA for now
                if (dtype in torch.testing.get_all_complex_dtypes()) and self.device_type == 'cuda':
                    # There are a two types of different errors that will be thrown.
                    # - Not implemented
                    # - Subtraction with a bool tensor
                    with self.assertRaises(RuntimeError):
                        foreach_bin_op_(tensors, scalars)

                    with self.assertRaises(RuntimeError):
                        foreach_bin_op(tensors, scalars)
                    continue

                if foreach_bin_op == torch._foreach_sub:
                    with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                        expected = [torch_bin_op(t, s) for t, s in zip(tensors, scalars)]

                    with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                        foreach_bin_op(tensors, scalars)

                    with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                        foreach_bin_op_(tensors, scalars)
                    continue

                expected = [torch_bin_op(t, s) for t, s in zip(tensors, scalars)]
                res = foreach_bin_op(tensors, scalars)
                self.assertEqual(expected, res)

                if dtype in torch.testing.integral_types_and(torch.bool) and foreach_bin_op == torch._foreach_div:
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output"):
                        foreach_bin_op_(tensors, scalars)
                else:
                    foreach_bin_op_(tensors, scalars)
                    self.assertEqual(tensors, res)

    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    def test_tensorlist_mixed_scalarlist_op(self, device, dtype):
        for N in N_values:
            for foreach_bin_op, foreach_bin_op_, torch_bin_op in self.bin_ops:
                tensors = self._get_test_data(device, dtype, N)
                scalars = [1, 1.1, 3 + 5j] + [True for _ in range(N - 3)]

                if foreach_bin_op == torch._foreach_sub:
                    with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                        expected = [torch_bin_op(t, s) for t, s in zip(tensors, scalars)]

                    with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                        foreach_bin_op(tensors, scalars)

                    # There are a two types of different errors that will be thrown.
                    # - Sub with bool is not allowed.
                    # - Result type can't be cast to the desired output type
                    with self.assertRaises(RuntimeError):
                        foreach_bin_op_(tensors, scalars)
                    continue

                expected = [torch_bin_op(t, s) for t, s in zip(tensors, scalars)]
                res = foreach_bin_op(tensors, scalars)
                self.assertEqual(expected, res)

                if dtype in torch.testing.get_all_complex_dtypes():
                    foreach_bin_op_(tensors, scalars)
                    self.assertEqual(expected, tensors)
                else:
                    with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
                        foreach_bin_op_(tensors, scalars)

    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_with_different_size_tensors(self, device, dtype):
        if dtype == torch.bool:
            return
        tensors = [torch.zeros(10 + n, 10 + n, device=device, dtype=dtype) for n in range(10)]
        expected = [torch.ones(10 + n, 10 + n, device=device, dtype=dtype) for n in range(10)]

        torch._foreach_add_(tensors, 1)
        self.assertEqual(expected, tensors)

    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar_with_empty_list_and_empty_tensor(self, device, dtype):
        # TODO: enable empty list case
        for tensors in [[torch.randn([0])]]:
            res = torch._foreach_add(tensors, 1)
            self.assertEqual(res, tensors)

            torch._foreach_add_(tensors, 1)
            self.assertEqual(res, tensors)

    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar_with_overlapping_tensors(self, device, dtype):
        tensors = [torch.ones(1, 1, device=device, dtype=dtype).expand(2, 1, 3)]
        expected = [torch.tensor([[[2, 2, 2]], [[2, 2, 2]]], dtype=dtype, device=device)]

        # bool tensor + 1 will result in int64 tensor
        if dtype == torch.bool:
            expected[0] = expected[0].to(torch.int64).add(1)

        res = torch._foreach_add(tensors, 1)
        self.assertEqual(res, expected)

    @skipMeta
    def test_bin_op_scalar_with_different_tensor_dtypes(self, device):
        tensors = [torch.tensor([1.1], dtype=torch.float, device=device),
                   torch.tensor([1], dtype=torch.long, device=device)]
        runtime_error = None
        try:
            torch._foreach_add(tensors, 1)
        except RuntimeError as e:
            runtime_error = e
        self.assertIsNone(runtime_error)

    #
    # Ops with list
    #
    @skipMeta
    def test_bin_op_list_error_cases(self, device):
        for bin_op, bin_op_, _ in self.bin_ops:
            tensors1 = []
            tensors2 = []

            # Empty lists
            with self.assertRaisesRegex(RuntimeError, "There were no tensor arguments to this function"):
                bin_op(tensors1, tensors2)
            with self.assertRaisesRegex(RuntimeError, "There were no tensor arguments to this function"):
                bin_op_(tensors1, tensors2)

            # One empty list
            tensors1.append(torch.tensor([1], device=device))
            with self.assertRaisesRegex(RuntimeError, "Tensor list must have same number of elements as scalar list."):
                bin_op(tensors1, tensors2)
            with self.assertRaisesRegex(RuntimeError, "Tensor list must have same number of elements as scalar list."):
                bin_op_(tensors1, tensors2)

            # Lists have different amount of tensors
            tensors2.append(torch.tensor([1], device=device))
            tensors2.append(torch.tensor([1], device=device))
            with self.assertRaisesRegex(RuntimeError, "Tensor lists must have the same number of tensors, got 1 and 2"):
                bin_op(tensors1, tensors2)
            with self.assertRaisesRegex(RuntimeError, "Tensor lists must have the same number of tensors, got 1 and 2"):
                bin_op_(tensors1, tensors2)

            # different devices
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                tensor1 = torch.zeros(10, 10, device="cuda:0")
                tensor2 = torch.ones(10, 10, device="cuda:1")
                with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                    bin_op([tensor1], [tensor2])
                with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                    bin_op_([tensor1], [tensor2])

            # Corresponding tensors with different sizes
            tensors1 = [torch.zeros(10, 10, device=device) for _ in range(10)]
            tensors2 = [torch.ones(11, 11, device=device) for _ in range(10)]
            with self.assertRaisesRegex(RuntimeError, "Corresponding tensors in lists must have the same size"):
                bin_op(tensors1, tensors2)
            with self.assertRaisesRegex(RuntimeError, r", got \[10, 10\] and \[11, 11\]"):
                bin_op_(tensors1, tensors2)

    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_list_different_sizes(self, device, dtype):
        tensors1 = [torch.zeros(10 + n, 10 + n, device=device, dtype=dtype) for n in range(10)]
        tensors2 = [torch.ones(10 + n, 10 + n, device=device, dtype=dtype) for n in range(10)]

        res = torch._foreach_add(tensors1, tensors2)
        torch._foreach_add_(tensors1, tensors2)
        self.assertEqual(res, tensors1)
        self.assertEqual(res, [torch.ones(10 + n, 10 + n, device=device, dtype=dtype) for n in range(10)])

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not found")
    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_list_slow_path(self, device, dtype):
        # 0-strides
        tensor1 = torch.rand(10, 10, device=device)
        tensor2 = torch.rand(1, device=device).expand_as(tensor1)
        res = torch._foreach_add([tensor1], [tensor2])
        torch._foreach_add_([tensor1], [tensor2])
        self.assertEqual(res, [tensor1])

        # different strides
        tensor1 = torch.zeros(10, 10, device=device, dtype=dtype)
        tensor2 = torch.ones(10, 10, device=device, dtype=dtype)
        res = torch._foreach_add([tensor1], [tensor2.t()])
        torch._foreach_add_([tensor1], [tensor2])
        self.assertEqual(res, [tensor1])

        # non contiguous
        tensor1 = torch.randn(5, 2, 1, 3, device=device)[:, 0]
        tensor2 = torch.randn(5, 2, 1, 3, device=device)[:, 0]
        self.assertFalse(tensor1.is_contiguous())
        self.assertFalse(tensor2.is_contiguous())
        res = torch._foreach_add([tensor1], [tensor2])
        torch._foreach_add_([tensor1], [tensor2])
        self.assertEqual(res, [tensor1])

        # sliced tensor
        tensor1 = torch.randn(5, 2, 1, 3, device=device).to(dtype)
        tensor2 = torch.randn(5, 2, 1, 3 * 7, device=device).to(dtype)[:, :, :, ::7]
        res = torch._foreach_add([tensor1], [tensor2])
        torch._foreach_add_([tensor1], [tensor2])
        self.assertEqual(res, [tensor1])

    # note: Below three tests (postfixed with `_tensors_on_different_devices`)
    # checks whether foreach works with lists of tensors on different devices
    # but tensors of the same index are on the same device, e.g., ['cuda', 'cpu].
    @ops(foreach_unary_op_db)
    def test_unary_op_tensors_on_different_devices(self, device, dtype, op):
        if self.device_type != 'cuda':
            self.skipTest('CUDA is necessary for tests with tensors on different devices')
        # tensors: ['cuda', 'cpu]
        tensors = op.sample_inputs(device, dtype, 2)
        tensors[1] = tensors[1].to('cpu')
        try:
            actual = op.get_method()(tensors)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), str(e)):
                [op.ref(t) for t in tensors]
        else:
            expected = [op.ref(t) for t in tensors]
            self.assertEqual(expected, actual)

        try:
            op.get_inplace()(tensors)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), str(e)):
                [getattr(t, op.ref.__name__ + '_')() for t in tensors]
        else:
            self.assertEqual(expected, tensors)

    @dtypes(*torch.testing.get_all_dtypes(include_bfloat16=True))
    def test_binary_op_tensors_on_different_devices(self, device, dtype):
        if self.device_type != 'cuda':
            self.skipTest('CUDA is necessary for tests with tensors on different devices')
        for foreach_op, foreach_op_, native_op in self.bin_ops:
            # `tensors1`: ['cuda', 'cpu']
            # `tensors2`: ['cuda', 'cpu']
            _cuda_tensors = self._get_test_data(device, dtype, 2)
            _cpu_tensors = self._get_test_data('cpu', dtype, 2)
            tensors1, tensors2 = [[_cuda_tensors[i], _cpu_tensors[i]] for i in range(2)]

            if dtype == torch.bool and native_op == torch.sub:
                with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                    foreach_op(tensors1, tensors2)
                with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                    foreach_op_(tensors1, tensors2)
                continue
            try:
                actual = foreach_op(tensors1, tensors2)
            except RuntimeError as e:
                with self.assertRaisesRegex(type(e), str(e)):
                    [native_op(t1, t2) for t1, t2 in zip(tensors1, tensors2)]
            else:
                expected = [native_op(t1, t2) for t1, t2 in zip(tensors1, tensors2)]
                self.assertEqual(expected, actual)
            try:
                foreach_op_(tensors1, tensors2)
            except RuntimeError as e:
                with self.assertRaisesRegex(type(e), str(e)):
                    [getattr(t1, native_op.__name__ + '_')(t2) for t1, t2 in zip(tensors1, tensors2)]
            else:
                self.assertEqual(actual, tensors1)

    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes(include_bfloat16=True))
    def test_pointwise_op_tensors_on_different_devices(self, device, dtype):
        if self.device_type != 'cuda':
            self.skipTest('CUDA is necessary for tests with tensors on different devices')

        pointwise_ops = [
            (torch._foreach_addcmul, torch._foreach_addcmul_, torch.addcmul),
            (torch._foreach_addcdiv, torch._foreach_addcdiv_, torch.addcdiv),
        ]
        for foreach_op, foreach_op_, native_op in pointwise_ops:
            # tensors1: ['cuda', 'cpu]
            # tensors2: ['cuda', 'cpu]
            # tensors3: ['cuda', 'cpu]
            _cuda_tensors = self._get_test_data(device, dtype, 3)
            _cpu_tensors = self._get_test_data('cpu', dtype, 3)
            tensors1, tensors2, tensors3 = [[_cuda_tensors[i], _cpu_tensors[i]] for i in range(3)]

            if native_op == torch.addcdiv:
                if dtype in torch.testing.get_all_int_dtypes() + [torch.bool]:
                    with self.assertRaisesRegex(RuntimeError, "Integer division with addcdiv"):
                        foreach_op(tensors1, tensors2, tensors3)
                    with self.assertRaisesRegex(RuntimeError, "Integer division with addcdiv"):
                        foreach_op_(tensors1, tensors2, tensors3)
                    continue
            try:
                actual = foreach_op(tensors1, tensors2, tensors3)
            except RuntimeError as e:
                with self.assertRaisesRegex(type(e), str(e)):
                    expected = [native_op(t1, t2, t3) for t1, t2, t3 in zip(tensors1, tensors2, tensors3)]
            else:
                expected = [native_op(t1, t2, t3) for t1, t2, t3 in zip(tensors1, tensors2, tensors3)]
                self.assertEqual(expected, actual)
            try:
                foreach_op_(tensors1, tensors2, tensors3)
            except RuntimeError as e:
                with self.assertRaisesRegex(type(e), str(e)):
                    [getattr(t1, native_op.__name__ + '_')(t2, t3) for t1, t2, t3 in zip(tensors1, tensors3, tensors3)]
            else:
                self.assertEqual(expected, tensors1)


instantiate_device_type_tests(TestForeach, globals())

if __name__ == '__main__':
    run_tests()
