#pragma once
#include <ATen/ATen.h>

#include <c10/util/irange.h>

#include <vector>
#include <unordered_map>
#include <utility>
#include <type_traits>
#include <string>

namespace at {
namespace native {
namespace {

using _device_key_t = std::string;
using _dtype_key_t = int8_t;
using _tensorvec_t = std::vector<at::Tensor>;
using _mta_input_t = std::vector<_tensorvec_t>;

// C++ version of https://github.com/pytorch/pytorch/blob/5211eeb22b17f706462bff4815f3c4bbf1af1c0c/torch/cuda/amp/grad_scaler.py#L194-L220
std::unordered_map<_device_key_t, std::unordered_map<_dtype_key_t, _tensorvec_t>>
_split_tensorlist_by_device_and_dtype(TensorList tensors) {
  std::unordered_map<_device_key_t, std::unordered_map<_dtype_key_t, _tensorvec_t>> hash;
  for (const auto & t : tensors) {
    const _device_key_t device_key = t.device().str();
    const _dtype_key_t dtype_key = static_cast<int8_t>(t.scalar_type());
    if (!hash.count(device_key)) {
      hash.emplace(device_key, std::unordered_map<_dtype_key_t, _tensorvec_t>{{dtype_key, _tensorvec_t{t}}});
    } else {
      if (!hash[device_key].count(dtype_key)) {
        hash[device_key].emplace(dtype_key, _tensorvec_t{t});
      } else {
        hash[device_key][dtype_key].emplace_back(t);
      }
    }
  }
  return hash;
}

// TODO(mkozuki): Write versions for binary&pointwise functions
/* Split TensorList by device & dtype.
 *
 * This function is used by in-place unary foreach functions.
 * Args:
 *  - TensorList tensors
 * Returns:
 *  - vector<vector<vector<Tensor>>: Caller can directly pass each element to MTA function.
 */
std::vector<_mta_input_t> split_tensorlist_for_inplace(TensorList tensors) {
  std::vector<_mta_input_t> grouped_tensors;
  for (auto & device_dtypetensors_pair : _split_tensorlist_by_device_and_dtype(tensors)) {
    for (auto & dtype_tensors_pair : device_dtypetensors_pair.second) {
      grouped_tensors.emplace_back(_mta_input_t{dtype_tensors_pair.second});
    }
  }
  return grouped_tensors;
}

// Check foreach API restrictions
// - Tensor lists must be non-empty.
// - All tensors in all lists must have the same dtype.
// - All TensorLists and ScalarLists must have the same number of elements.
// - Corresponding tensors must have the same size.
void check_foreach_api_restrictions(TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");
  auto expected_dtype = tensors[0].dtype();
  for (const auto& t : tensors) {
    TORCH_CHECK(t.dtype() == expected_dtype, "All tensors in the tensor list must have the same dtype.");
  }
}

void check_foreach_api_restrictions(const std::vector<at::Tensor>& tensors) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");
  auto expected_dtype = tensors[0].dtype();
  for (const auto& t : tensors) {
    TORCH_CHECK(t.dtype() == expected_dtype, "All tensors in the tensor list must have the same dtype.");
  }
}

void check_foreach_api_restrictions(TensorList tensors, ArrayRef<Scalar> scalars) {
  check_foreach_api_restrictions(tensors);
  TORCH_CHECK(tensors.size() == scalars.size(), "Tensor list must have same number of elements as scalar list.");
}

void check_foreach_api_restrictions(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors2.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() == tensors2.size(), "Tensor lists must have the same number of tensors, got ", tensors1.size(), " and ", tensors2.size());

  auto expected_dtype = tensors1[0].dtype();

  for (const auto i : c10::irange(tensors1.size())) {
    TORCH_CHECK(tensors1[i].dtype() == expected_dtype, "All tensors in the tensor list must have the same dtype.");
    TORCH_CHECK(tensors2[i].dtype() == expected_dtype, "All tensors in the tensor list must have the same dtype.");
    TORCH_CHECK(tensors1[i].sizes() == tensors2[i].sizes(), "Corresponding tensors in lists must have the same size, got ", tensors1[i].sizes(), " and ", tensors2[i].sizes());
  }
}

void check_foreach_api_restrictions(TensorList tensors1, TensorList tensors2, TensorList tensors3) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors2.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors3.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() == tensors2.size(), "Tensor lists must have the same number of tensors, got ", tensors1.size(), " and ", tensors2.size());
  TORCH_CHECK(tensors1.size() == tensors3.size(), "Tensor lists must have the same number of tensors, got ", tensors1.size(), " and ", tensors3.size());

  auto expected_dtype = tensors1[0].dtype();

  for (const auto i : c10::irange(tensors1.size())) {
    TORCH_CHECK(tensors1[i].dtype() == expected_dtype, "All tensors in the tensor list must have the same dtype.");
    TORCH_CHECK(tensors2[i].dtype() == expected_dtype, "All tensors in the tensor list must have the same dtype.");
    TORCH_CHECK(tensors1[i].sizes() == tensors2[i].sizes(), "Corresponding tensors in lists must have the same size, got ", tensors1[i].sizes(), " and ", tensors2[i].sizes());
    TORCH_CHECK(tensors1[i].sizes() == tensors3[i].sizes(), "Corresponding tensors in lists must have the same size, got ", tensors1[i].sizes(), " and ", tensors3[i].sizes());
  }
}

void check_foreach_api_restrictions(TensorList tensors1, TensorList tensors2, TensorList tensors3, ArrayRef<Scalar> scalars) {
  check_foreach_api_restrictions(tensors1, tensors2, tensors3);
  TORCH_CHECK(tensors1.size() == scalars.size(), "Tensor list must have same number of elements as scalar list, got ", tensors1.size(), " and ", scalars.size());
}

// To go via 'fast' path, several conditions must be satisfied
// - All tensors must be on the same device
// - All tensors must have strided layout
// - All tensors must be non-overlapping and dense
// - Resulting tensor must have the same dtype as the input one

bool will_promote_tensor(const Tensor& tensor, const Scalar& scalar, bool does_op_promote_integer_inputs_to_float = false) {
  // In case of division, integer inputs will result in float
  if (does_op_promote_integer_inputs_to_float) {
    if (at::isIntegralType(tensor.scalar_type(), /*includeBool*/ true)) {
      return true;
    }
  }
  auto result_dtype = at::result_type(tensor, scalar);
  return result_dtype != tensor.scalar_type();
}

// Please, make sure to call check_foreach_api_restrictions before calling this method.
// There is a set of preconditions that have to be satisfied.
template<typename T1, typename T2>
bool _check_fast_path_restrictions_impl(
  T1 tensorLists,
  T2 scalarList,
  const bool does_op_promote_integer_inputs_to_float) {
    // FIXME(mkozuki): ../aten/src/ATen/native/ForeachUtils.h:134:21: error: wrong number of template arguments (1, should be 2)  std::is_same< T1, ArrayRef<ArrayRef<at::Tensor>> >::value ||
    // TORCH_INTERNAL_ASSERT(
    //   std::is_same< T1, ArrayRef<ArrayRef<at::Tensor>> >::value ||
    //   std::is_same< T1, std::vector<std::vector<at::Tensor>> >::value,
    //   "`tensorLists` must be either `ArrayRef<TensorList>` or `std::vector<std::vector<at::Tensor>>`"
    // );
    auto expected_device = tensorLists[0][0].device();

    auto is_tensor_okay = [&](const Tensor& tensor) {
      return tensor.device() == expected_device &&
             tensor.layout() == at::kStrided &&
             tensor.is_non_overlapping_and_dense();
    };

    for (const auto& tensorList : tensorLists) {
      for (const auto& tensor : tensorList) {
        if (!is_tensor_okay(tensor)) {
          return false;
        }
      }
    }

    // Check if corresponding tensors in tensor lists have the same strides.
    for (int i=0; i < tensorLists.size(); i++) {
      for (int j=0; j < tensorLists[0].size(); j++) {
        if (tensorLists[0][j].strides() != tensorLists[i][j].strides()) {
          return false;
        }
      }
    }

    // For all j, tensorList[j][0] have the same shape and dtype. (this was a precondition
    // checked by `check_foreach_api_restrictions`). This means we only need to check if
    // {tensorList[0][0], tensorList[0][1], tensorList[0][2], ...} do type promotion with scalarLIst.
    for (int i=0; i < tensorLists[0].size(); i++) {
      if (does_op_promote_integer_inputs_to_float) {
        if (at::isIntegralType(tensorLists[0][i].scalar_type(), /*includeBool*/ true)) {
          return false;
        }
      }

      if (scalarList.size() == 1) {
        if (will_promote_tensor(tensorLists[0][i], scalarList[0])) {
          return false;
        }
      } else if (scalarList.size() > 1) {
        // Complex scalar list is not supported due to the limit for kernel launch argument (4KB)
        if (scalarList[i].isComplex()) {
          return false;
        }

        if (will_promote_tensor(tensorLists[0][i], scalarList[i])) {
          return false;
        }
      }
    }

    return true;
}

bool check_fast_path_restrictions(
  ArrayRef<TensorList> tensorLists,
  ArrayRef<Scalar> scalarList = {},
  bool does_op_promote_integer_inputs_to_float = false) {
    return _check_fast_path_restrictions_impl(tensorLists, scalarList, does_op_promote_integer_inputs_to_float);
}

bool check_fast_path_restrictions(
  const std::vector<std::vector<at::Tensor>>& tensorLists,
  ArrayRef<Scalar> scalarList = {},
  bool does_op_promote_integer_inputs_to_float = false) {
    return _check_fast_path_restrictions_impl(tensorLists, scalarList, does_op_promote_integer_inputs_to_float);
}

bool can_use_fast_route(ArrayRef<TensorList> tensorLists,
                        ArrayRef<Scalar> scalarList = {},
                        bool does_op_promote_integer_inputs_to_float = false) {
#ifdef __HIP_PLATFORM_HCC__
  return false;
#else
  return check_fast_path_restrictions(tensorLists, scalarList, does_op_promote_integer_inputs_to_float);
#endif
}

bool can_use_fast_route(const std::vector<std::vector<at::Tensor>>& tensorVec,
                        ArrayRef<Scalar> scalarList = {},
                        bool does_op_promote_integer_inputs_to_float = false) {
#ifdef __HIP_PLATFORM_HCC__
  return false;
#else
  return check_fast_path_restrictions(tensorVec, scalarList, does_op_promote_integer_inputs_to_float);
#endif
}

bool can_use_fast_route(TensorList tensors1, TensorList tensors2, bool does_op_promote_integer_inputs_to_float = false) {
#ifdef __HIP_PLATFORM_HCC__
  return false;
#else
  return can_use_fast_route({tensors1, tensors2}, {}, does_op_promote_integer_inputs_to_float);
#endif
}

}
}} // at::native
