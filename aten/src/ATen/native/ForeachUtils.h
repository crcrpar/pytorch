#pragma once
#include <ATen/ATen.h>

#include <c10/util/irange.h>

#include <vector>
#include <unordered_map>
#include <utility>
#include <type_traits>

namespace at {
namespace native {
namespace {

struct DeviceHashFunctor {
    size_t operator() (const at::Device & device) const {
        return std::hash<std::string>()(device.str());
    }
};

struct DtypeHashFunctor {
    size_t operator() (const at::ScalarType & scalar_type) const {
        return std::hash<int>()(static_cast<int>(scalar_type));
    }
};

using device_type_t = std::string;
using scalar_type_t = int;
using dtype_to_tensorvector_t = std::unordered_map< scalar_type_t, std::vector<at::Tensor> >;
using dev_dtype_tensors_t     = std::unordered_map< device_type_t, dtype_to_tensorvector_t>;
using dev_dtype_index_t       = std::unordered_map< device_type_t, std::unordered_map<scalar_type_t, std::vector<size_t>> >;

std::pair<dev_dtype_tensors_t, dev_dtype_index_t> _split_tensors_by_device_and_dtype(TensorList tensors) {
  dev_dtype_tensors_t hash;
  dev_dtype_index_t index_record;

  for (const auto & i : c10::irange(tensors.size())) {
    auto & t  = tensors[i];
    const std::string device_key = t.device().str();
    const int scalar_type_key = static_cast<int>(t.scalar_type());
    if (hash.count(device_key) > 0) {
      if (hash[device_key].count(scalar_type_key) > 0) {
        hash[device_key][scalar_type_key].emplace_back(t);
        index_record[device_key][scalar_type_key].emplace_back(i);
      } else {
        hash[device_key].emplace(scalar_type_key, std::vector<at::Tensor>{t});
        index_record[device_key].emplace(scalar_type_key, std::vector<size_t>{i});
      }
    } else {
      std::unordered_map<int, std::vector<at::Tensor>> tmp {{scalar_type_key, std::vector<at::Tensor>{t}}};
      hash.emplace(device_key, std::move(tmp));

      std::unordered_map<int, std::vector<size_t>> dd_idx {{scalar_type_key, std::vector<size_t>{i}}};
      index_record.emplace(device_key, dd_idx);
    }

  }
  // return std::make_pair<dev_dtype_tensors_t, dev_dtype_index_t>(hash, index_record);
  return {hash, index_record};
}

using triple_nested_tensor_vector_t = std::vector<std::vector<std::vector<at::Tensor>>>;
triple_nested_tensor_vector_t split_tensors_by_device_and_dtype(TensorList tensors) {
  auto hash = _split_tensors_by_device_and_dtype(tensors).first;
  std::vector<std::vector<std::vector<Tensor>>> grouped_tensors;
  for (auto & dev_pair : hash) {
    for (auto & dtype_pair : dev_pair.second) {
      auto _tensors = std::vector<std::vector<at::Tensor>>{dtype_pair.second};
      grouped_tensors.emplace_back(std::move(_tensors));
    }
  }
  return grouped_tensors;
}

using output_reorder_map_t = std::unordered_map<std::string, std::pair<size_t, std::vector<size_t>>>;
std::pair<triple_nested_tensor_vector_t, std::vector<size_t>>
split_tensors_by_device_and_dtype_with_order(TensorList tensors) {
  auto ret = _split_tensors_by_device_and_dtype(tensors);
  dev_dtype_tensors_t & hash = ret.first;
  dev_dtype_index_t & index_mapper = ret.second;

  std::vector<size_t> to_orig_order;
  to_orig_order.reserve(tensors.size());

  size_t i = 0;
  triple_nested_tensor_vector_t grouped_tensors;

  for (auto & dev_pair : hash) {
    const auto & device_key = dev_pair.first;
    for (auto & dtype_pair : dev_pair.second) {
      const auto & scalar_type_key = dtype_pair.first;
      auto _tensors = std::vector<std::vector<at::Tensor>>{dtype_pair.second};
      grouped_tensors.emplace_back(std::move(_tensors));

      for (const auto & idx : index_mapper[device_key][scalar_type_key]) {
        to_orig_order.emplace_back(idx);
      }
    }
  }

  return {grouped_tensors, to_orig_order};
}

void sort_results(std::vector<at::Tensor>& arr, std::vector<size_t>& index) {
  // Tensor old_target_e;
  for (size_t i = 0; i < arr.size(); i++) {
    while (index[i] != i) {
      int old_target_i = index[index[i]];
      Tensor old_target_e {std::move(arr[index[i]])};
      arr[index[i]] = arr[i];
      index[index[i]] = index[i];
      index[i] = old_target_i;
      arr[i] = old_target_e;
    }
  }
}

// void sort_output_tensors(std::vector<at::Tensor>& outputs, const std::vector<size_t>& to_orig_order) {
//   sort_vector_by_indices(outputs, output_reorder_map);
// }

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
