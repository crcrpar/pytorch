import re
import torch
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests, TEST_WITH_ROCM, TEST_WITH_SLOW
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes, skipCUDAIfRocm, skipMeta, ops)
from torch._six import inf, nan
from torch.testing._internal.common_methods_invocations import foreach_unary_op_db, foreach_binary_op_db

# Includes some values such that N * N won't be a multiple of 4,
# which should ensure we test the vectorized and non-vectorized
# kernel code paths.
N_values = [20, 23] if not TEST_WITH_SLOW else [23, 30, 300]
Alpha_values = [2, 1.23]

def _has_boolearn(inputs2):
    return any(isinstance(e, bool) for e in inputs2)

class RegularFuncWrapper:

    def __init__(self, func):
        self.func = func

    def __call__(self, inputs, **kwargs):
        return [self.func(*i, **kwargs) for i in zip(*inputs)]


class InplaceRefWrapper:

    def __init__(self, method_name):
        self.method_name = method_name

    def __call__(self, inputs, **kwargs):
        return [getattr(i1, self.method_name)(*rest, **kwargs) for i1, *rest in list(zip(*inputs))]

class ForeachFuncWrapper:

    def __init__(self, func, n_expected_cudaLaunchKernels):
        self.func = func
        self.n_expected_cudaLaunchKernels = n_expected_cudaLaunchKernels

    def __call__(self, inputs, is_cuda, is_fastpath, **kwargs):
        if is_cuda and torch.autograd.kineto_available():
            with torch.profiler.profile(activities=(torch.profiler.ProfilerActivity.CPU,)) as p:
                actual = self.func(*inputs, **kwargs)
            for e in p.key_averages():
                if e.key == 'cudaLaunchKernel':
                    if is_fastpath:
                        assert e.count == self.n_expected_cudaLaunchKernels
                    else:
                        assert e.count > self.n_expected_cudaLaunchKernels
            return actual
        else:
            return self.func(*inputs, **kwargs)

# TODO(mkozuki): Avoid `@skipMeta` as possible.
class TestForeach(TestCase):

    @property
    def is_cuda(self):
        return self.device_type == 'cuda'

    # note(mkozuki): It might be the case that the expected number of `cudaLaunchKernel`s
    # is greater than 1 once foreach functions internally separate their input `TensorList`s by
    # devices & dtypes into vectors of tensors.
    def _get_funcs(self, op, n_expected_cudaLaunchKernels=1):
        return (
            ForeachFuncWrapper(op.method_variant, n_expected_cudaLaunchKernels),
            RegularFuncWrapper(op.ref),
            ForeachFuncWrapper(op.inplace_variant, n_expected_cudaLaunchKernels),
            InplaceRefWrapper(op.ref_inplace),
        )

    # todo(mkozuki): remove this method once `TestForeach` is refactored with `@op` decorator.
    def _get_test_data(self, device, dtype, N):
        if dtype in [torch.bfloat16, torch.bool, torch.float16]:
            tensors = [torch.randn(N, N, device=device).to(dtype) for _ in range(N)]
        elif dtype in torch.testing.get_all_int_dtypes():
            # Constrains the range between 1 and 10 for less stress on int8 tensors.
            tensors = [torch.randint(1, 10, (N, N), device=device, dtype=dtype) for _ in range(N)]
        else:
            tensors = [torch.randn(N, N, device=device, dtype=dtype) for _ in range(N)]

        return tensors

    def _test_binary_op(self, device, dtype, op, inputs1, inputs2, is_fastpath, alpha=None, is_mixed_scalarlist=False):
        op_kwargs = {} if alpha is None else {'alpha': alpha}
        method, ref, inplace_method, inplace_ref = self._get_funcs(op)

        copied_inputs1 = [t.clone().detach() for t in inputs1]
        native_inputs2 = inputs2 if isinstance(inputs2, list) else [inputs2] * len(inputs1)

        is_subtraction = method.func == torch._foreach_sub
        use_fastpath_check = (
            is_fastpath and
            not _has_boolearn(native_inputs2) and
            not (is_subtraction and (dtype in (torch.bool,) or _has_boolearn(native_inputs2)))
        )

        if use_fastpath_check:
            actual = method((inputs1, inputs2), self.is_cuda, is_fastpath, **op_kwargs)
            expected = ref((inputs1, native_inputs2), **op_kwargs)
            # NOTE(mkozuki): Why using large tolerances when BFloat16 & Float16?
            # For BFloat16 and Float16 I/O, MTA implementation of addition and subtraction
            # does math in FP32 and then downcasts back the result into BFloat16/Float16
            # while `torch.add` & `torch.sub`, not.
            # When `alpha` or any scalar in inputs2 is not representable in 16-bits, the difference of results
            # gets way larger.
            rtol, atol = None, None
            if dtype in (torch.bfloat16, torch.float16):
                if TEST_WITH_ROCM:
                    rtol, atol = self.dtype_precisions[dtype]
                if (
                        isinstance(alpha, (int, float)) or
                        isinstance(inputs2, (int, float)) or
                        (isinstance(inputs2, list) and isinstance(inputs2[0], (int, float)))
                ):
                    rtol, atol = self.dtype_precisions[dtype]
                    atol = 0.5
            self.assertEqual(expected, actual, atol=atol, rtol=rtol)
            copied_inputs1 = [t.clone().detach() for t in inputs1]
            try:
                inplace_method((inputs1, inputs2), self.is_cuda, is_fastpath, **op_kwargs)
            except RuntimeError as e:
                with self.assertRaisesRegex(type(e), re.escape(str(e))):
                    inplace_ref((copied_inputs1, native_inputs2), **op_kwargs)
            else:
                inplace_ref((copied_inputs1, native_inputs2), **op_kwargs)
                self.assertEqual(copied_inputs1, inputs1, atol=atol, rtol=rtol)
            return
        else:
            try:
                actual = method((inputs1, inputs2), self.is_cuda, is_fastpath, **op_kwargs)
            except RuntimeError as e:
                with self.assertRaisesRegex(type(e), re.escape(str(e))):
                    ref((inputs1, native_inputs2), **op_kwargs)
            else:
                expected = ref((inputs1, native_inputs2), **op_kwargs)
                self.assertEqual(expected, actual)

            # note(mkozuki): Why special handling for inplace `sub` with mixed scalarlist?
            # Foreach one first does `sub_check(tensors[i], scalars[i])` and does math.
            # So, it will raise boolean subtraction error before any actual computation.
            # If there is no possibility of bool subtraction error, it's still possible that
            # result cannot be casted to the another dtype later.
            # Whilst, regular one calls `tensors[i].sub_(scalars[i])` in Python,
            # therefore, the order of two errors depends on the order of input scalarlist.
            pattern = "(Subtraction|result)"
            if is_subtraction and is_mixed_scalarlist:
                with self.assertRaisesRegex(RuntimeError, pattern):
                    inplace_method((inputs1, inputs2), self.is_cuda, is_fastpath, **op_kwargs)
                with self.assertRaisesRegex(RuntimeError, pattern):
                    inplace_ref((copied_inputs1, native_inputs2), **op_kwargs)
            else:
                try:
                    inplace_method((inputs1, inputs2), self.is_cuda, is_fastpath, **op_kwargs)
                except RuntimeError as e:
                    with self.assertRaisesRegex(type(e), re.escape(str(e))):
                        inplace_ref((copied_inputs1, native_inputs2), **op_kwargs)
                else:
                    inplace_ref((copied_inputs1, native_inputs2), **op_kwargs)
                    self.assertEqual(copied_inputs1, inputs1)

    def _test_binary_two_tensorlists(self, device, dtype, op, is_fastpath):
        for N in N_values:
            inputs1 = op.sample_inputs(device, dtype, N, noncontiguous=not is_fastpath)
            inputs2 = op.sample_inputs(device, dtype, N, noncontiguous=not is_fastpath)
            self._test_binary_op(device, dtype, op, inputs1, inputs2, is_fastpath)

            if dtype != torch.bool and op.supports_alpha_param:
                for alpha in Alpha_values:
                    if dtype in torch.testing.get_all_int_dtypes():
                        alpha = int(alpha)
                    self._test_binary_op(device, dtype, op, inputs1, inputs2, is_fastpath, alpha=alpha)

    @ops(foreach_binary_op_db)
    def test_binary_fastpath_two_tensorlists(self, device, dtype, op):
        self._test_binary_two_tensorlists(device, dtype, op, True)

    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_slowpath_two_tensorlists(self, device, dtype, op):
        self._test_binary_two_tensorlists(device, dtype, op, False)

    # If `scalar is None`, scalarlist consists of different types of scalars.
    def _test_binary_tensorlist_and_scalarlist(self, device, dtype, op, is_fastpath, scalar=None):
        for N in N_values:
            inputs1 = op.sample_inputs(device, dtype, N, noncontiguous=not is_fastpath)
            if scalar is None:
                Scalar_vlaues = [1, 2.2, 3.3 + 3.3j, True]
                inputs2 = Scalar_vlaues + [1.23] * (N - len(Scalar_vlaues))
            else:
                inputs2 = [scalar] * N

            self._test_binary_op(
                device, dtype, op, inputs1, inputs2, is_fastpath, alpha=None, is_mixed_scalarlist=scalar is None)

    # TODO[Fix scalar list]:
    # We need to update codegen to correctly handle function overloads with float[] and int[].
    # As optimizers work with float tensors, the result will always be torch.float32 for now.
    # Current schema is using 'float[]' as scalar list type.
    @skipCUDAIfRocm
    @ops(foreach_binary_op_db)
    def test_binary_fastpath_tensorlist_and_int_scalarlist(self, device, dtype, op):
        self._test_binary_tensorlist_and_scalarlist(device, dtype, op, True, scalar=1)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_slowpath_tensorlist_and_int_scalarlist(self, device, dtype, op):
        self._test_binary_tensorlist_and_scalarlist(device, dtype, op, False, scalar=1)

    @skipCUDAIfRocm
    @ops(foreach_binary_op_db)
    def test_binary_fastpath_tensorlist_and_float_scalarlist(self, device, dtype, op):
        self._test_binary_tensorlist_and_scalarlist(device, dtype, op, True, scalar=1.23)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_slowpath_tensorlist_and_float_scalarlist(self, device, dtype, op):
        self._test_binary_tensorlist_and_scalarlist(device, dtype, op, False, scalar=1.23)

    @skipCUDAIfRocm
    @ops(foreach_binary_op_db)
    def test_binary_fastpath_tensorlist_and_complex_scalarlist(self, device, dtype, op):
        self._test_binary_tensorlist_and_scalarlist(device, dtype, op, True, scalar=1.23 + 4.5j)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_slowpath_tensorlist_and_complex_scalarlist(self, device, dtype, op):
        self._test_binary_tensorlist_and_scalarlist(device, dtype, op, False, scalar=1.23 + 4.5j)

    @skipMeta
    @skipCUDAIfRocm
    @ops(foreach_binary_op_db)
    def test_binary_fastpath_tensorlist_and_bool_scalarlist(self, device, dtype, op):
        self._test_binary_tensorlist_and_scalarlist(device, dtype, op, True, scalar=True)

    @skipMeta
    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_slowpath_tensorlist_and_bool_scalarlist(self, device, dtype, op):
        self._test_binary_tensorlist_and_scalarlist(device, dtype, op, False, scalar=True)

    @skipMeta
    @ops(foreach_binary_op_db)
    def test_binary_fastpath_tensorlist_and_mixed_scalarlist(self, device, dtype, op):
        self._test_binary_tensorlist_and_scalarlist(device, dtype, op, True, scalar=None)

    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_slowpath_tensorlist_and_mixed_scalarlist(self, device, dtype, op):
        self._test_binary_tensorlist_and_scalarlist(device, dtype, op, False, scalar=None)

    def _test_binary_tensorlist_and_scalar(self, device, dtype, op, is_fastpath, scalar):
        for N in N_values:
            inputs1 = op.sample_inputs(device, dtype, N, noncontiguous=not is_fastpath)
            self._test_binary_op(device, dtype, op, inputs1, scalar, is_fastpath)

    @skipCUDAIfRocm
    @ops(foreach_binary_op_db)
    def test_binary_fastpath_tensorlist_and_int_scalar(self, device, dtype, op):
        self._test_binary_tensorlist_and_scalar(device, dtype, op, True, 1)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_slowpath_tensorlist_and_int_scalar(self, device, dtype, op):
        self._test_binary_tensorlist_and_scalar(device, dtype, op, False, 1)

    @skipCUDAIfRocm
    @ops(foreach_binary_op_db)
    def test_binary_fastpath_tensorlist_and_float_scalar(self, device, dtype, op):
        self._test_binary_tensorlist_and_scalar(device, dtype, op, True, 1.23)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_slow_tensorlist_and_float_scalar(self, device, dtype, op):
        self._test_binary_tensorlist_and_scalar(device, dtype, op, False, 1.23)

    @skipCUDAIfRocm
    @ops(foreach_binary_op_db)
    def test_binary_fastpath_tensorlist_and_complex_scalar(self, device, dtype, op):
        self._test_binary_tensorlist_and_scalar(device, dtype, op, True, 1.23 + 4.5j)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_slowpath_tensorlist_and_complex_scalar(self, device, dtype, op):
        self._test_binary_tensorlist_and_scalar(device, dtype, op, False, 1.23 + 4.5j)

    @skipMeta
    @skipCUDAIfRocm
    @ops(foreach_binary_op_db)
    def test_binary_fastpath_tensorlist_and_bool_scalar(self, device, dtype, op):
        self._test_binary_tensorlist_and_scalar(device, dtype, op, True, True)

    @skipMeta
    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_slowpath_tensorlist_and_bool_scalar(self, device, dtype, op):
        self._test_binary_tensorlist_and_scalar(device, dtype, op, False, True)

    def _test_bin_op_list(self, device, dtype, foreach_op, foreach_op_, torch_op):
        for N in N_values:
            tensors1 = self._get_test_data(device, dtype, N)
            tensors2 = self._get_test_data(device, dtype, N)

            # Mimics cuda kernel dtype flow.  With fp16/bf16 input, runs in fp32 and casts output back to fp16/bf16.
            control_dtype = torch.float32 if (self.device_type == 'cuda' and
                                              (dtype is torch.float16 or dtype is torch.bfloat16)) else dtype
            expected = [torch_op(tensors1[i].to(dtype=control_dtype),
                                 tensors2[i].to(dtype=control_dtype)).to(dtype=dtype) for i in range(N)]
            res = foreach_op(tensors1, tensors2)
            foreach_op_(tensors1, tensors2)
            self.assertEqual(res, tensors1)
            if (dtype is torch.float16 or dtype is torch.bfloat16) and TEST_WITH_ROCM:
                self.assertEqual(tensors1, expected, atol=1.e-3, rtol=self.dtype_precisions[dtype][0])
            else:
                self.assertEqual(tensors1, expected)

    def _test_pointwise_op(self, device, dtype, foreach_op, foreach_op_, torch_op):
        for N in N_values:
            # Constrains the range a bit for int8 tensors.
            values = [2 + (i % 5) for i in range(N)]
            for vals in [values[0], values]:
                tensors = self._get_test_data(device, dtype, N)
                tensors1 = self._get_test_data(device, dtype, N)
                tensors2 = self._get_test_data(device, dtype, N)

                # Mimics cuda kernel dtype flow.  With fp16/bf16 input, runs in fp32 and casts output back to fp16/bf16.
                control_dtype = torch.float32 if (self.device_type == 'cuda' and
                                                  (dtype is torch.float16 or dtype is torch.bfloat16)) else dtype

                if not isinstance(vals, list):
                    expected = [torch_op(tensors[i].to(dtype=control_dtype),
                                         tensors1[i].to(dtype=control_dtype),
                                         tensors2[i].to(dtype=control_dtype),
                                         value=values[0]).to(dtype=dtype) for i in range(N)]
                else:
                    expected = [torch_op(tensors[i].to(dtype=control_dtype),
                                         tensors1[i].to(dtype=control_dtype),
                                         tensors2[i].to(dtype=control_dtype),
                                         value=values[i]).to(dtype=dtype) for i in range(N)]

                res = foreach_op(tensors, tensors1, tensors2, vals)
                foreach_op_(tensors, tensors1, tensors2, vals)
                self.assertEqual(res, tensors)

                if (dtype is torch.float16 or dtype is torch.bfloat16) and TEST_WITH_ROCM:
                    self.assertEqual(tensors, expected, atol=3.e-3, rtol=self.dtype_precisions[dtype][0])
                else:
                    self.assertEqual(tensors, expected)

                # test error cases
                for op in [torch._foreach_addcmul, torch._foreach_addcmul_, torch._foreach_addcdiv, torch._foreach_addcdiv_]:
                    tensors = self._get_test_data(device, dtype, N)
                    tensors1 = self._get_test_data(device, dtype, N)
                    tensors2 = self._get_test_data(device, dtype, N)

                    with self.assertRaisesRegex(RuntimeError, "Tensor list must have same number of elements as scalar list."):
                        op(tensors, tensors1, tensors2, [2 for _ in range(N + 1)])

                    with self.assertRaisesRegex(RuntimeError, "Tensor list must have same number of elements as scalar list."):
                        op(tensors, tensors1, tensors2, [2 for _ in range(N - 1)])

                    msg = "Tensor lists must have the same number of tensors, got {} and {}".format(N + 1, N)

                    tensors = self._get_test_data(device, dtype, N + 1)
                    with self.assertRaisesRegex(RuntimeError, msg):
                        op(tensors, tensors1, tensors2, [2 for _ in range(N)])

                    tensors1 = self._get_test_data(device, dtype, N + 1)
                    with self.assertRaisesRegex(RuntimeError, msg):
                        op(tensors, tensors1, tensors2, [2 for _ in range(N)])

    # note(mkozuki): fastpath test uses dtypes which fastpath implementation supports.
    # To confirm the dtypes of `OpInfo` cover the dtypes that the function support,
    # this test does not use `try-except` for fastpath.
    def _regular_unary_test(self, dtype, method, ref, inputs, is_fastpath):
        if is_fastpath:
            self.assertEqual(ref(inputs), method(inputs, self.is_cuda, is_fastpath))
            return
        try:
            actual = method(inputs, self.is_cuda, is_fastpath)
        except RuntimeError as e:
            if method.func == torch._foreach_neg and dtype == torch.bool and self.is_cuda:
                # note(mkozuki): `foreach_neg` CUDA implementation throws a different message.
                with self.assertRaisesRegex(type(e), "Negation, the `-` operator, on a bool tensor is not supported."):
                    ref(inputs)
            else:
                with self.assertRaisesRegex(type(e), re.escape(str(e))):
                    ref(inputs)
        else:
            expected = ref(inputs)
            self.assertEqual(actual, expected)

    def _inplace_unary_test(self, dtype, inplace, inplace_ref, inputs, is_fastpath):
        copied_inputs = [[t.clone().detach() for t in tensors] for tensors in inputs]
        try:
            inplace(inputs, self.is_cuda, is_fastpath)
        except RuntimeError as e:
            if inplace.func == torch._foreach_neg_ and dtype == torch.bool and self.is_cuda:
                # note(mkozuki): `foreach_neg` CUDA implementation throws a different message.
                with self.assertRaisesRegex(type(e), "Negation, the `-` operator, on a bool tensor is not supported."):
                    inplace_ref(copied_inputs)
            else:
                with self.assertRaisesRegex(type(e), re.escape(str(e))):
                    inplace_ref(copied_inputs)
        else:
            inplace_ref(copied_inputs)
        self.assertEqual(copied_inputs, inputs)

    def _test_unary(self, device, dtype, op, N, is_fastpath):
        method, ref, inplace_method, inplace_ref = self._get_funcs(op)
        inputs = op.sample_inputs(device, dtype, N, noncontiguous=not is_fastpath),
        self._regular_unary_test(dtype, method, ref, inputs, is_fastpath)
        self._inplace_unary_test(dtype, inplace_method, inplace_ref, inputs, is_fastpath)

    @ops(foreach_unary_op_db)
    def test_unary_fastpath(self, device, dtype, op):
        for N in N_values:
            self._test_unary(device, dtype, op, N, is_fastpath=True)

    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_unary_op_db)
    def test_unary_slowpath(self, device, dtype, op):
        for N in N_values:
            self._test_unary(device, dtype, op, N, is_fastpath=False)

    #
    # Pointwise ops
    #
    @dtypes(*torch.testing.get_all_dtypes(include_bfloat16=False, include_bool=False, include_complex=False))
    def test_addcmul(self, device, dtype):
        if self.device_type == 'cpu':
            if dtype == torch.half:
                with self.assertRaisesRegex(RuntimeError, r"\"addcmul_cpu_out\" not implemented for \'Half\'"):
                    self._test_pointwise_op(device, dtype, torch._foreach_addcmul,
                                            torch._foreach_addcmul_, torch.addcmul)
                return

        self._test_pointwise_op(device, dtype, torch._foreach_addcmul, torch._foreach_addcmul_, torch.addcmul)

    @dtypes(*torch.testing.get_all_dtypes(include_bfloat16=False, include_bool=False, include_complex=False))
    def test_addcdiv(self, device, dtype):
        if dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
            with self.assertRaisesRegex(RuntimeError,
                                        "Integer division with addcdiv is no longer supported, and in a future"):
                self._test_pointwise_op(device, dtype, torch._foreach_addcdiv, torch._foreach_addcdiv_, torch.addcdiv)
            return

        if self.device_type == 'cpu':
            if dtype == torch.half:
                with self.assertRaisesRegex(RuntimeError, r"\"addcdiv_cpu_out\" not implemented for \'Half\'"):
                    self._test_pointwise_op(device, dtype, torch._foreach_addcdiv,
                                            torch._foreach_addcdiv_, torch.addcdiv)
                return
        self._test_pointwise_op(device, dtype, torch._foreach_addcdiv, torch._foreach_addcdiv_, torch.addcdiv)

    @dtypes(*torch.testing.get_all_dtypes(include_bfloat16=False, include_complex=False))
    def test_min_max(self, device, dtype):
        for N in N_values:
            tensors1 = self._get_test_data(device, dtype, N)
            tensors2 = self._get_test_data(device, dtype, N)

            # Mimics cuda kernel dtype flow.  With fp16/bf16 input, runs in fp32 and casts output back to fp16/bf16.
            control_dtype = torch.float32 if (self.device_type == 'cuda' and
                                              (dtype is torch.float16 or dtype is torch.bfloat16)) else dtype

            expected_max = [torch.max(tensors1[i].to(dtype=control_dtype),
                                      tensors2[i].to(dtype=control_dtype)).to(dtype=dtype) for i in range(N)]

            expected_min = [torch.min(tensors1[i].to(dtype=control_dtype),
                                      tensors2[i].to(dtype=control_dtype)).to(dtype=dtype) for i in range(N)]

            res_max = torch._foreach_maximum(tensors1, tensors2)
            self.assertEqual(res_max, expected_max)

            res_min = torch._foreach_minimum(tensors1, tensors2)
            self.assertEqual(res_min, expected_min)

    @dtypes(*(torch.testing.get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    def test_max_min_float_inf_nan(self, device, dtype):
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

        expected = [torch.max(a1, b1) for a1, b1 in zip(a, b)]
        res = torch._foreach_maximum(a, b)
        self.assertEqual(expected, res)

        expected = [torch.min(a1, b1) for a1, b1 in zip(a, b)]
        res = torch._foreach_minimum(a, b)
        self.assertEqual(expected, res)

    @dtypes(*(torch.testing.get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    def test_max_min_inf_nan(self, device, dtype):
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

        expected_max = [torch.max(a1, b1) for a1, b1 in zip(a, b)]
        res_max = torch._foreach_maximum(a, b)
        self.assertEqual(expected_max, res_max)

        expected_min = [torch.min(a1, b1) for a1, b1 in zip(a, b)]
        res_min = torch._foreach_minimum(a, b)
        self.assertEqual(expected_min, res_min)


    # This tests (implicitly) the behavior of slowpath
    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_with_different_size_tensors(self, device, dtype, op):
        method, ref, inplace_method, inplace_ref = self._get_funcs(op)
        tensors = [torch.zeros(10 + n, 10 + n, device=device, dtype=dtype) for n in range(10)]

        try:
            actual = method((tensors, 1), self.is_cuda, False)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                ref((tensors, [1] * 10))
        else:
            expected = ref((tensors, [1] * 10))
            self.assertEqual(expected, actual)

        copied_tensors = [t.clone().detach() for t in tensors]
        try:
            inplace_method((tensors, 1), self.is_cuda, False)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                inplace_ref((copied_tensors, [1] * 10))
        else:
            self.assertEqual(expected, tensors)

    # This tests (implicitly) the behavior of slowpath
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_scalar_with_empty_list_and_empty_tensor(self, device, dtype, op):
        method, ref, inplace_method, _ = self._get_funcs(op)
        # TODO: enable empty list case
        for tensors in [[torch.randn([0])]]:
            expected = ref((tensors, [1]))
            actual = method((tensors, 1), self.is_cuda, False)
            self.assertEqual(expected, actual)

            inplace_method((tensors, 1), self.is_cuda, False)
            self.assertEqual(expected, tensors)

    # This tests (implicitly) the behavior of slowpath
    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_scalar_with_overlapping_tensors(self, device, dtype, op):
        tensors = [torch.ones(1, 1, device=device, dtype=dtype).expand(2, 1, 3)]
        method, ref, _, _ = self._get_funcs(op)
        try:
            actual = method((tensors, 1), self.is_cuda, False)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                ref((tensors, [1] * len(tensors)))
        else:
            expected = ref((tensors, [1] * len(tensors)))
            self.assertEqual(expected, actual)

    # note(mkozuki): this test case fails with Meta at least in my local environment.
    # The message was
    # `AssertionError: NotImplementedError("Could not run 'aten::_foreach_add.Scalar' with arguments from the 'Meta' backend.`
    @dtypes(torch.float)
    @ops(foreach_binary_op_db)
    def test_binary_with_different_tensor_dtypes_and_scalar(self, device, dtype, op):
        tensors = [torch.tensor([1.1], dtype=torch.float, device=device),
                   torch.tensor([1], dtype=torch.long, device=device)]
        expected = [op.ref(t1, 1) for t1 in tensors]
        actual = op.method_variant(tensors, 1)
        self.assertEqual(expected, actual)

    #
    # Ops with list
    #
    @ops(foreach_binary_op_db)
    def test_binary_tensorlists_error_cases(self, device, dtype, op):
        method = op.method_variant
        inplace_method = op.inplace_variant
        tensors1 = []
        tensors2 = []

        # Empty lists
        with self.assertRaisesRegex(RuntimeError, "There were no tensor arguments to this function"):
            method(tensors1, tensors2)
        with self.assertRaisesRegex(RuntimeError, "There were no tensor arguments to this function"):
            inplace_method(tensors1, tensors2)

        # One empty list
        tensors1.append(torch.tensor([1], device=device, dtype=dtype))
        with self.assertRaisesRegex(RuntimeError, "Tensor list must have same number of elements as scalar list."):
            method(tensors1, tensors2)
        with self.assertRaisesRegex(RuntimeError, "Tensor list must have same number of elements as scalar list."):
            inplace_method(tensors1, tensors2)

        # Lists have different amount of tensors
        tensors2.append(torch.tensor([1], device=device))
        tensors2.append(torch.tensor([1], device=device))
        with self.assertRaisesRegex(RuntimeError, "Tensor lists must have the same number of tensors, got 1 and 2"):
            method(tensors1, tensors2)
        with self.assertRaisesRegex(RuntimeError, "Tensor lists must have the same number of tensors, got 1 and 2"):
            inplace_method(tensors1, tensors2)

        # Corresponding tensors with different sizes
        tensors1 = [torch.zeros(10, 10, dtype=dtype, device=device) for _ in range(10)]
        tensors2 = [torch.ones(11, 11, dtype=dtype, device=device) for _ in range(10)]
        with self.assertRaisesRegex(RuntimeError, "Corresponding tensors in lists must have the same size"):
            method(tensors1, tensors2)
        with self.assertRaisesRegex(RuntimeError, r", got \[10, 10\] and \[11, 11\]"):
            inplace_method(tensors1, tensors2)

    # This test goes through slowpath
    @skipMeta
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_with_list_of_different_size_tensors(self, device, dtype, op):
        method, ref, inplace_method, inplace_ref = self._get_funcs(op)
        tensors1 = [torch.zeros(10 + n, 10 + n, device=device, dtype=dtype) for n in range(10)]
        tensors2 = [torch.ones(10 + n, 10 + n, device=device, dtype=dtype) for n in range(10)]

        try:
            actual = method((tensors1, tensors2), self.is_cuda, False)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                ref((tensors1, tensors2))
        else:
            expected = ref((tensors1, tensors2))
            self.assertEqual(expected, actual)
            try:
                inplace_method((tensors1, tensors2), self.is_cuda, False)
            except RuntimeError as e:
                with self.assertRaisesRegex(type(e), re.escape(str(e))):
                    inplace_ref((tensors1, tensors2))
            else:
                self.assertEqual(expected, tensors1)

    @skipMeta
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not found")
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_tensorlists_nonstrided(self, device, dtype, op):
        method = op.method_variant
        inplace_method = op.inplace_variant
        ref = op.ref
        # Explicit broadcasting
        tensor1 = op.sample_inputs(device, dtype, 10)[0]
        tensor2 = op.sample_inputs(device, dtype, 1)[0].expand_as(tensor1)
        tensors1, tensors2 = [tensor1], [tensor2]
        try:
            actual = method(tensors1, tensors2)
        except RuntimeError:
            with self.assertRaises(RuntimeError):
                [op.ref(tensor1, tensor2)]
        else:
            expected = [op.ref(tensor1, tensor2)]
            self.assertEqual(expected, actual)
            try:
                inplace_method(tensors1, tensors2)
            except RuntimeError as e:
                pass
            else:
                self.assertEqual(expected, tensors1)

        # different strides
        tensor1 = torch.zeros(10, 10, device=device, dtype=dtype)
        tensor2 = torch.ones(10, 10, device=device, dtype=dtype)
        try:
            res = method([tensor1], [tensor2.t()])
        except RuntimeError:
            with self.assertRaises(RuntimeError):
                ref(tensor1, tensor2.t())
        else:
            expected = [ref(tensor1, tensor2.t())]
            self.assertEqual(expected, res)
            try:
                inplace_method([tensor1], [tensor2])
            except RuntimeError as e:
                with self.assertRaisesRegex(type(e), re.escape(str(e))):
                    getattr(tensor1, op.ref_inplace)(tensor2.t())
            else:
                self.assertEqual(expected, [tensor1])

        # sliced tensor
        tensor1 = torch.randn(5, 2, 1, 3, device=device).to(dtype)
        tensor2 = torch.randn(5, 2, 1, 3 * 7, device=device).to(dtype)[:, :, :, ::7]
        tensors1, tensors2 = [tensor1], [tensor2]
        try:
            actual = method(tensors1, tensors2)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                ref(tensor1, tensor2)
        else:
            expected = [op.ref(tensor1, tensor2)]
            self.assertEqual(expected, actual)
            try:
                inplace_method(tensors1, tensors2)
            except RuntimeError as e:
                with self.assertRaisesRegex(type(e), re.escape(str(e))):
                    getattr(tensor1, op.ref_inplace)(tensor2)
            else:
                self.assertEqual(expected, tensors1)

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
    @ops(foreach_binary_op_db)
    def test_binary_op_tensors_on_different_devices(self, device, dtype, op):
        if self.device_type != 'cuda':
            self.skipTest('CUDA is necessary for tests with tensors on different devices')
        # `tensors1`: ['cuda', 'cpu']
        # `tensors2`: ['cuda', 'cpu']
        _cuda_tensors = op.sample_inputs(device, dtype, 2, noncontiguous=False)
        _cpu_tensors = op.sample_inputs('cpu', dtype, 2, noncontiguous=False)
        tensors1, tensors2 = list(tensors for tensors in zip(_cuda_tensors, _cpu_tensors))
        method, ref, inplace_method, inplace_ref = self._get_funcs(op)

        try:
            actual = method((tensors1, tensors2), self.is_cuda, False)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                ref((tensors1, tensors2))
        else:
            expected = ref((tensors1, tensors2))
            self.assertEqual(expected, actual)
        try:
            inplace_method((tensors1, tensors2), self.is_cuda, False)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                inplace_ref((tensors1, tensors2))
        else:
            self.assertEqual(actual, tensors1)

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
            tensors1, tensors2, tensors3 = list(tensors for tensors in zip(_cuda_tensors, _cpu_tensors))

            try:
                actual = foreach_op(tensors1, tensors2, tensors3)
            except RuntimeError as e:
                with self.assertRaisesRegex(type(e), re.escape(str(e))):
                    expected = [native_op(t1, t2, t3) for t1, t2, t3 in zip(tensors1, tensors2, tensors3)]
            else:
                expected = [native_op(t1, t2, t3) for t1, t2, t3 in zip(tensors1, tensors2, tensors3)]
                self.assertEqual(expected, actual)
            try:
                foreach_op_(tensors1, tensors2, tensors3)
            except RuntimeError as e:
                with self.assertRaisesRegex(type(e), re.escape(str(e))):
                    [getattr(t1, native_op.__name__ + '_')(t2, t3) for t1, t2, t3 in zip(tensors1, tensors3, tensors3)]
            else:
                self.assertEqual(expected, tensors1)


instantiate_device_type_tests(TestForeach, globals())

if __name__ == '__main__':
    run_tests()
