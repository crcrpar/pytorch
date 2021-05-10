import torch
from torch.testing._internal.common_utils import TestCase, run_tests, TEST_WITH_SLOW
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes, skipMeta, ops)
from torch._six import inf, nan
from torch.testing._internal.common_methods_invocations import \
    (foreach_unary_op_db, foreach_binary_op_db, foreach_pointwise_op_db, foreach_min_max_op_db)

# Includes some values such that N * N won't be a multiple of 4,
# which should ensure we test the vectorized and non-vectorized
# kernel code paths.
N_values = [20, 23] if not TEST_WITH_SLOW else [23, 30, 300]

# List of values for `alpha` and `value` argument
# note: Complex numbers are excluded from Pointwise_scalars
# because they cannot be used as `value` of `torch.addcmul` and `torch.addcdiv`
Scalar_values = [1, 2.0, 3.0 + 0.3j, True]
Pointwise_scalars = [1, 2.0, True]

class TestForeach(TestCase):

    def _calc_control_dtype(self, dtype):
        # Mimics cuda kernel dtype flow.  With fp16/bf16 input, runs in fp32 and casts output back to fp16/bf16.
        if self.device_type == 'cuda' and dtype in (torch.bfloat16, torch.float16):
            return torch.float32
        else:
            return dtype

    def _test_binary_op(self, device, dtype, op, inputs1, inputs2, alpha=None, assert_kwargs=None, is_fastpath=False):
        op_kwargs = {} if alpha is None else {'alpha': alpha}
        if assert_kwargs is None:
            assert_kwargs = {}
        method = op.get_op()
        ref = op.get_ref()
        inplace = op.get_inplace()
        inplace_ref_name = op.get_inplace_ref_name()

        native_inputs2 = inputs2
        is_bool = dtype == torch.bool
        if not isinstance(native_inputs2, list):
            native_inputs2 = [native_inputs2 for _ in range(len(inputs1))]
        if not isinstance(native_inputs2[0], torch.Tensor) and not is_bool:
            for e in native_inputs2:
                if isinstance(e, bool):
                    is_bool = True
                    break

        if is_fastpath and not is_bool and not (method == torch._foreach_sub and dtype == torch.bool):
            self.assertEqual(
                [ref(i1, i2, **op_kwargs) for i1, i2 in zip(inputs1, native_inputs2)],
                method(inputs1, inputs2, **op_kwargs), **assert_kwargs)
        else:
            try:
                actual = method(inputs1, inputs2, **op_kwargs)
            except RuntimeError as e:
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
        except RuntimeError as e:
            with self.assertRaises(RuntimeError):
                [getattr(i1, inplace_ref_name)(i2, **op_kwargs) for i1, i2 in zip(inputs1, native_inputs2)]
        else:
            [getattr(i1, inplace_ref_name)(i2, **op_kwargs) for i1, i2 in zip(inputs1, native_inputs2)]
            self.assertEqual(inputs1, copied_tensors, **assert_kwargs)

    @skipMeta
    @ops(foreach_binary_op_db)
    def test_binary_ops_fastpath_tensorlists(self, device, dtype, op):
        for N in N_values:
            inputs1 = op.sample_inputs(device, dtype, N, noncontiguous=False)
            inputs2 = op.sample_inputs(device, dtype, N, noncontiguous=False)
            self._test_binary_op(device, dtype, op, inputs1, inputs2, is_fastpath=True)

    # FIXME: We need to update codegen to correctly handle function overloads with float[] and int[].
    # As optimizers work with float tensors, the result will always be torch.float32 for now.
    # Current schema is using 'float[]' as scalar list type.
    @skipMeta
    @ops(foreach_binary_op_db)
    def test_binary_ops_fastpath_scalarlist(self, device, dtype, op):
        for N in N_values:
            inputs1 = op.sample_inputs(device, dtype, N, noncontiguous=False)
            for scalar in Scalar_values:
                inputs2 = [scalar for _ in range(N)]
                self._test_binary_op(device, dtype, op, inputs1, inputs2, is_fastpath=True)
            else:
                # list of different types of scalars
                inputs2 = Scalar_values + [1.0 for _ in range(N - len(Scalar_values))]
                self._test_binary_op(device, dtype, op, inputs1, inputs2, is_fastpath=True)

    @skipMeta
    @ops(foreach_binary_op_db)
    def test_binary_ops_fastpath_scalar(self, device, dtype, op):
        for N in N_values:
            for scalar in Scalar_values:
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

    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_ops_slowpath_scalarlist(self, device, dtype, op):
        if dtype in torch.testing.get_all_complex_dtypes() and self.device_type == 'cuda':
            self.skipTest('FIXME')
        for N in N_values:
            for scalar in Scalar_values:
                inputs1 = op.sample_inputs(device, dtype, N, noncontiguous=True)
                inputs2 = [scalar for _ in range(N)]
                self._test_binary_op(device, dtype, op, inputs1, inputs2)

    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_ops_slowpath_scalar(self, device, dtype, op):
        for N in N_values:
            for scalar in Scalar_values:
                inputs1 = op.sample_inputs(device, dtype, N, noncontiguous=True)
                self._test_binary_op(device, dtype, op, inputs1, scalar)

    @skipMeta
    @ops(foreach_binary_op_db)
    def test_binary_ops_tensorlists_with_alpha_fastpath(self, device, dtype, op):
        if not op.supports_alpha_param:
            self.skipTest("{} does not support alpha parameter".format(op.name))
        # NOTE(mkozuki): the fastpath of tensorlists with alpha parameter behaves differently
        # from the native ones (torch.add & torch.sub) because
        # the former does math in FP32 for BFloat16/Float16 I/O while the latter not.
        assert_kwargs = None
        if dtype in (torch.bfloat16, torch.float16):
            assert_kwargs = {
                'rtol': self.dtype_precisions[dtype][0],
                'atol': 0.05,
            }
        for N in N_values:
            inputs1 = op.sample_inputs(device, dtype, N, noncontiguous=False)
            inputs2 = op.sample_inputs(device, dtype, N, noncontiguous=False)
            self._test_binary_op(device, dtype, op, inputs1, inputs2, alpha=10, assert_kwargs=assert_kwargs, is_fastpath=True)

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

    def _test_unary_op(self, device, dtype, op, tensors, is_fastpath):
        method = op.get_op()
        ref = op.get_ref()
        inplace = op.get_inplace()
        inplace_ref_name = op.get_inplace_ref_name()

        if is_fastpath:
            self.assertEqual([ref(t) for t in tensors], method(tensors))
        else:
            try:
                actual = method(tensors)
            except RuntimeError:
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

    @skipMeta
    @ops(foreach_unary_op_db)
    def test_unary_ops_fastpath(self, device, dtype, op):
        for N in N_values:
            tensors = op.sample_inputs(device, dtype, N, noncontiguous=False)
            self._test_unary_op(device, dtype, op, tensors, True)

    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_unary_op_db)
    def test_unary_ops_slowpath(self, device, dtype, op):
        for N in N_values:
            tensors = op.sample_inputs(device, dtype, N, noncontiguous=True)
            self._test_unary_op(device, dtype, op, tensors, False)

    def _test_pointwise_op(self, device, dtype, op, tensors, inputs1, inputs2, value=None, is_fastpath=False):
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

        if value is None:
            value = 1.0
        native_values = value
        if not isinstance(value, list):
            native_values = [value] * len(tensors)
        fe_kwargs = {'scalars': value} if isinstance(value, list) else {'value': value}

        if is_fastpath:
            self.assertEqual(
                [ref(t, i1, i2, value=v) for t, i1, i2, v in zip(tensors, inputs1, inputs2, native_values)],
                method(tensors, inputs1, inputs2, **fe_kwargs),
                **assert_kwargs
            )
        else:
            try:
                actual = method(tensors, inputs1, inputs2, **fe_kwargs)
            except RuntimeError:
                with self.assertRaises(RuntimeError):
                    [ref(t, i1, i2, value=v) for t, i1, i2, v in zip(tensors, inputs1, inputs2, native_values)]
            else:
                expected = [ref(t, i1, i2, value=v) for t, i1, i2, v in zip(tensors, inputs1, inputs2, native_values)]
                self.assertEqual(expected, actual, **assert_kwargs)

        if inplace is None:
            return
        copied_tensors = [t.clone().detach() for t in tensors]
        if self.device_type == 'cpu' and dtype in (torch.bfloat16, torch.float16):
            with self.assertRaises(RuntimeError):
                inplace(copied_tensors, inputs1, inputs2, **fe_kwargs)
            return
        if method == torch._foreach_addcdiv and dtype in torch.testing.get_all_int_dtypes() + [torch.bool]:
            with self.assertRaises(RuntimeError):
                inplace(copied_tensors, inputs1, inputs2, **fe_kwargs)
            return

        try:
            inplace(copied_tensors, inputs1, inputs2, **fe_kwargs)
        except RuntimeError as e:
            with self.assertRaisesRegex(RuntimeError, str(e)):
                [getattr(t, inplace_ref_name)(i1, i2, value=v) for t, i1, i2, v in zip(tensors, inputs1, inputs2, native_values)]
        else:
            [getattr(t, inplace_ref_name)(i1, i2, value=v) for t, i1, i2, v in zip(tensors, inputs1, inputs2, native_values)]
            self.assertEqual(tensors, copied_tensors, **assert_kwargs)

    @skipMeta
    @ops(foreach_pointwise_op_db)
    def test_pointwise_ops_fastpath(self, device, dtype, op):
        for N in N_values:
            tensors = op.sample_inputs(device, dtype, N)
            inputs1 = op.sample_inputs(device, dtype, N)
            inputs2 = op.sample_inputs(device, dtype, N)
            self._test_pointwise_op(device, dtype, op, tensors, inputs1, inputs2, is_fastpath=True)
            for scalar in Pointwise_scalars:
                self._test_pointwise_op(device, dtype, op, tensors, inputs1, inputs2, value=scalar, is_fastpath=True)
            else:
                values = Pointwise_scalars + [1.0] * (N - len(Pointwise_scalars))
                self._test_pointwise_op(device, dtype, op, tensors, inputs1, inputs2, value=values, is_fastpath=True)

    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_pointwise_op_db)
    def test_pointwise_ops_slowpath(self, device, dtype, op):
        for N in N_values:
            tensors = op.sample_inputs(device, dtype, N, noncontiguous=True)
            inputs1 = op.sample_inputs(device, dtype, N, noncontiguous=True)
            inputs2 = op.sample_inputs(device, dtype, N, noncontiguous=True)
            self._test_pointwise_op(device, dtype, op, tensors, inputs1, inputs2, is_fastpath=False)
            for scalar in Pointwise_scalars:
                self._test_pointwise_op(device, dtype, op, tensors, inputs1, inputs2, value=scalar, is_fastpath=False)
            else:
                values = Pointwise_scalars + [1.0] * (N - len(Pointwise_scalars))
                self._test_pointwise_op(device, dtype, op, tensors, inputs1, inputs2, value=values, is_fastpath=False)

    @skipMeta
    @ops(foreach_min_max_op_db)
    def test_min_max_fastpath(self, device, dtype, op):
        method, ref = op.get_op(), op.get_ref()
        for N in N_values:
            tensors1 = op.sample_inputs(device, dtype, N)
            tensors2 = op.sample_inputs(device, dtype, N)
            expected = [
                ref(
                    t1.to(dtype=self._calc_control_dtype(dtype)),
                    t2.to(dtype=self._calc_control_dtype(dtype)),
                ).to(dtype=dtype)
                for t1, t2 in zip(tensors1, tensors2)
            ]
            actual = method(tensors1, tensors2)
            self.assertEqual(expected, actual)

    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes(include_complex=False))
    @ops(foreach_min_max_op_db)
    def test_min_max_slowpath(self, device, dtype, op):
        method, ref = op.get_op(), op.get_ref()
        for N in N_values:
            tensors1 = op.sample_inputs(device, dtype, N, noncontiguous=True)
            tensors2 = op.sample_inputs(device, dtype, N, noncontiguous=True)
            expected = [
                ref(
                    t1.to(dtype=self._calc_control_dtype(dtype)),
                    t2.to(dtype=self._calc_control_dtype(dtype)),
                ).to(dtype=dtype) for t1, t2 in zip(tensors1, tensors2)
            ]
            actual = method(tensors1, tensors2)
            self.assertEqual(expected, actual)

    @skipMeta
    @ops(foreach_min_max_op_db)
    def test_min_max_inf_nan(self, device, dtype, op):
        if dtype in torch.testing.get_all_int_dtypes():
            self.skipTest('inf/nan cannot be converted to integer types')
        method, ref = op.get_op(), op.get_ref()

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

        expected = [ref(a1, b1) for a1, b1 in zip(a, b)]
        actual = method(a, b)
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
        expected = [ref(a1, b1) for a1, b1 in zip(a, b)]
        actual = method(a, b)
        self.assertEqual(expected, actual)

    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_ops_with_different_size_tensors(self, device, dtype, op):
        method = op.get_op()
        ref = op.get_ref()
        inplace = op.get_inplace()
        inplace_ref_name = op.get_inplace_ref_name()

        tensors1 = [op.sample_inputs(device, dtype, 10 + i)[0] for i in range(10)]
        tensors2 = [op.sample_inputs(device, dtype, 10 + i)[0] for i in range(10)]

        if method == torch._foreach_sub and dtype == torch.bool:
            with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                method(tensors1, tensors2)
            with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                inplace(tensors1, tensors2)
            return

        actual = method(tensors1, tensors2)
        expected = [ref(t1, t2) for t1, t2 in zip(tensors1, tensors2)]
        self.assertEqual(expected, actual)
        try:
            copied_tensors = [t1.clone().detach() for t1 in tensors1]
            inplace(tensors1, tensors2)
        except RuntimeError as e:
            with self.assertRaisesRegex(RuntimeError, str(e)):
                [getattr(t1, inplace_ref_name)(t2) for t1, t2 in zip(copied_tensors, tensors2)]
        else:
            self.assertEqual(expected, tensors1)

    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_ops_with_empty_tensor(self, device, dtype, op):
        tensors = [torch.ones(0, dtype=dtype, device=device)]
        scalar = 1
        method, ref, inplace, inplace_ref_name = op.get_op(), op.get_ref(), op.get_inplace(), op.get_inplace_ref_name()

        if method == torch._foreach_sub and dtype == torch.bool:
            with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                method(tensors, scalar)
            with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                inplace(tensors, scalar)
            return

        actual = method(tensors, scalar)
        expected = [ref(t, scalar) for t in tensors]
        self.assertEqual(expected, actual)
        try:
            inplace(tensors, scalar)
        except RuntimeError as e:
            with self.assertRaisesRegex(RuntimeError, str(e)):
                [getattr(t, inplace_ref_name)(scalar) for t in tensors]
        else:
            self.assertEqual(expected, tensors)

    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_ops_scalar_with_overlapping_tensors(self, device, dtype, op):
        method, ref = op.get_op(), op.get_ref()
        tensors = [torch.ones(1, 1, device=device, dtype=dtype).expand(2, 1, 3)]

        if dtype == torch.bool and method == torch._foreach_sub:
            with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                method(tensors, 1)
            with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                [ref(t, 1) for t in tensors]
            with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                op.get_inplace()(tensors, 1)
            return
        actual = method(tensors, 1)
        expected = [ref(t, 1) for t in tensors]
        self.assertEqual(expected, actual)

    @skipMeta
    @dtypes(torch.float32)
    @ops(foreach_binary_op_db)
    def test_binary_ops_scalar_with_different_tensor_dtypes(self, device, dtype, op):
        method, ref = op.get_op(), op.get_ref()
        tensors = [torch.tensor([1.1], dtype=torch.float, device=device),
                   torch.tensor([1], dtype=torch.long, device=device)]
        actual = method(tensors, 1)
        expected = [ref(t, 1) for t in tensors]
        self.assertEqual(expected, actual)

    @skipMeta
    @ops(foreach_binary_op_db)
    def test_binary_ops_list_error_cases(self, device, dtype, op):
        tensors1, tensors2 = [], []
        method, inplace = op.get_op(), op.get_inplace()

        # Empty lists
        with self.assertRaisesRegex(RuntimeError, "There were no tensor arguments to this function"):
            method(tensors1, tensors2)
        with self.assertRaisesRegex(RuntimeError, "There were no tensor arguments to this function"):
            inplace(tensors1, tensors2)

        # One empty list
        tensors1.append(torch.tensor([1], device=device))
        with self.assertRaisesRegex(RuntimeError, "Tensor list must have same number of elements as scalar list."):
            method(tensors1, tensors2)
        with self.assertRaisesRegex(RuntimeError, "Tensor list must have same number of elements as scalar list."):
            inplace(tensors1, tensors2)

        # Lists have different amount of tensors
        tensors2.append(torch.tensor([1], device=device))
        tensors2.append(torch.tensor([1], device=device))
        with self.assertRaisesRegex(RuntimeError, "Tensor lists must have the same number of tensors, got 1 and 2"):
            method(tensors1, tensors2)
        with self.assertRaisesRegex(RuntimeError, "Tensor lists must have the same number of tensors, got 1 and 2"):
            inplace(tensors1, tensors2)

        # different devices
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            tensor1 = torch.zeros(10, 10, device="cuda:0")
            tensor2 = torch.ones(10, 10, device="cuda:1")
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                method([tensor1], [tensor2])
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                inplace([tensor1], [tensor2])

        # Corresponding tensors with different sizes
        tensors1 = [torch.zeros(10, 10, device=device) for _ in range(10)]
        tensors2 = [torch.ones(11, 11, device=device) for _ in range(10)]
        with self.assertRaisesRegex(RuntimeError, "Corresponding tensors in lists must have the same size"):
            method(tensors1, tensors2)
        with self.assertRaisesRegex(RuntimeError, r", got \[10, 10\] and \[11, 11\]"):
            inplace(tensors1, tensors2)

    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_ops_different_sizes(self, device, dtype, op):
        tensors1 = [torch.zeros(10 + n, 10 + n, device=device, dtype=dtype) for n in range(10)]
        tensors2 = [torch.ones(10 + n, 10 + n, device=device, dtype=dtype) for n in range(10)]

        method = op.get_op()
        ref = op.get_ref()
        inplace = op.get_inplace()

        if dtype == torch.bool and op.get_op() == torch._foreach_sub:
            with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                method(tensors1, tensors2)
            with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                inplace(tensors1, tensors2)
            return
        actual = method(tensors1, tensors2)
        expected = [ref(t1, t2) for t1, t2 in zip(tensors1, tensors2)]
        self.assertEqual(expected, actual)
        try:
            inplace(tensors1, tensors2)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), str(e)):
                [getattr(t1, op.get_inplace_ref_name())(t2) for t1, t2 in zip(tensors1, tensors2)]
        else:
            self.assertEqual(actual, tensors1)

    @skipMeta
    @dtypes(torch.float32)
    @ops(foreach_binary_op_db)
    def test_binary_ops_noncontiguous_tensors(self, device, dtype, op):
        method, ref, inplace = op.get_op(), op.get_ref(), op.get_inplace()
        # 0-strides
        tensor1 = torch.rand(10, 10, device=device, dtype=dtype)
        tensor2 = torch.rand(1, device=device, dtype=dtype).expand_as(tensor1)
        expected = [ref(tensor1, tensor2)]
        actual = method([tensor1], [tensor2])
        inplace([tensor1], [tensor2])
        self.assertEqual(expected, actual)
        self.assertEqual(actual, [tensor1])

        # different strides
        tensor1 = torch.zeros(10, 10, device=device, dtype=dtype)
        tensor2 = torch.ones(10, 10, device=device, dtype=dtype)
        expected = [ref(tensor1, tensor2)]
        actual = method([tensor1], [tensor2])
        inplace([tensor1], [tensor2])
        self.assertEqual(expected, actual)
        self.assertEqual(expected, [tensor1])

        # non contiguous
        tensor1 = torch.randn(5, 2, 1, 3, device=device, dtype=dtype)[:, 0]
        tensor2 = torch.randn(5, 2, 1, 3, device=device, dtype=dtype)[:, 0]
        self.assertFalse(tensor1.is_contiguous())
        self.assertFalse(tensor2.is_contiguous())
        expected = [ref(tensor1, tensor2)]
        actual = method([tensor1], [tensor2])
        inplace([tensor1], [tensor2])
        self.assertEqual(expected, actual)
        self.assertEqual(expected, [tensor1])

        # sliced tensor
        tensor1 = torch.randn(5, 2, 1, 3, device=device, dtype=dtype).to(dtype)
        tensor2 = torch.randn(5, 2, 1, 3 * 7, device=device, dtype=dtype).to(dtype)[:, :, :, ::7]
        expected = [ref(tensor1, tensor2)]
        actual = method([tensor1], [tensor2])
        inplace([tensor1], [tensor2])
        self.assertEqual(expected, actual)
        self.assertEqual(expected, [tensor1])

    # note: Below three tests (postfixed with `_tensors_on_different_devices`)
    # checks whether foreach works with lists of tensors on different devices
    # but tensors of the same index are on the same device, e.g., ['cuda', 'cpu].
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_unary_op_db)
    def test_unary_ops_tensors_on_different_devices(self, device, dtype, op):
        if self.device_type != 'cuda':
            self.skipTest('CUDA is necessary for tests with tensors on different devices')
        # tensors: ['cuda', 'cpu]
        tensors = op.sample_inputs(device, dtype, 2)
        tensors[1] = tensors[1].to('cpu')
        method, ref, inplace, inplace_ref_name = op.get_op(), op.get_ref(), op.get_inplace(), op.get_inplace_ref_name()
        if method == torch._foreach_neg and dtype == torch.bool:
            with self.assertRaisesRegex(RuntimeError, "There is a bool tensor in the passed-in TensorList."):
                method(tensors)
            with self.assertRaisesRegex(RuntimeError, "There is a bool tensor in the passed-in TensorList."):
                inplace(tensors)
            return
        try:
            actual = method(tensors)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), str(e)):
                [ref(t) for t in tensors]
        else:
            expected = [ref(t) for t in tensors]
            self.assertEqual(expected, actual)

        try:
            inplace(tensors)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), str(e)):
                [getattr(t, inplace_ref_name)() for t in tensors]
        else:
            self.assertEqual(expected, tensors)

    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_ops_tensors_on_different_devices(self, device, dtype, op):
        if self.device_type != 'cuda':
            self.skipTest('CUDA is necessary for tests with tensors on different devices')
        # `tensors1`: ['cuda', 'cpu']
        # `tensors2`: ['cuda', 'cpu']
        _cuda_tensors = op.sample_inputs(device, dtype, 2)
        _cpu_tensors = op.sample_inputs('cpu', dtype, 2)
        tensors1, tensors2 = [[_cuda_tensors[i], _cpu_tensors[i]] for i in range(2)]
        method, ref, inplace, inplace_ref_name = op.get_op(), op.get_ref(), op.get_inplace(), op.get_inplace_ref_name()

        if dtype == torch.bool and method == torch._foreach_sub:
            with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                method(tensors1, tensors2)
            with self.assertRaisesRegex(RuntimeError, "Subtraction, the `-` operator"):
                inplace(tensors1, tensors2)
            return
        actual = method(tensors1, tensors2)
        expected = [ref(t1, t2) for t1, t2 in zip(tensors1, tensors2)]
        self.assertEqual(expected, actual)
        try:
            inplace(tensors1, tensors2)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), str(e)):
                [getattr(t1, inplace_ref_name)(t2) for t1, t2 in zip(tensors1, tensors2)]
        else:
            self.assertEqual(actual, tensors1)

    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_pointwise_op_db)
    def test_pointwise_ops_tensors_on_different_devices(self, device, dtype, op):
        if self.device_type != 'cuda':
            self.skipTest('CUDA is necessary for tests with tensors on different devices')
        # tensors1: ['cuda', 'cpu]
        # tensors2: ['cuda', 'cpu]
        # tensors3: ['cuda', 'cpu]
        _cuda_tensors = op.sample_inputs(device, dtype, 3)
        _cpu_tensors = op.sample_inputs('cpu', dtype, 3)
        tensors1, tensors2, tensors3 = [[_cuda_tensors[i], _cpu_tensors[i]] for i in range(3)]
        method, ref, inplace, inplace_ref_name = op.get_op(), op.get_ref(), op.get_inplace(), op.get_inplace_ref_name()

        if method == torch._foreach_addcdiv and dtype in [torch.bool] + torch.testing.get_all_int_dtypes():
            with self.assertRaisesRegex(RuntimeError, "Integer division with addcdiv"):
                method(tensors1, tensors2, tensors3)
            with self.assertRaisesRegex(RuntimeError, "Integer division with addcdiv"):
                inplace(tensors1, tensors2, tensors3)
            return
        try:
            actual = method(tensors1, tensors2, tensors3)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), str(e)):
                [ref(t1, t2, t3) for t1, t2, t3 in zip(tensors1, tensors2, tensors3)]
        else:
            expected = [ref(t1, t2, t3) for t1, t2, t3 in zip(tensors1, tensors2, tensors3)]
            self.assertEqual(expected, actual)
        try:
            inplace(tensors1, tensors2, tensors3)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), str(e)):
                [getattr(t1, inplace_ref_name)(t2, t3) for t1, t2, t3 in zip(tensors1, tensors3, tensors3)]
        else:
            self.assertEqual(expected, tensors1)


instantiate_device_type_tests(TestForeach, globals())

if __name__ == '__main__':
    run_tests()
