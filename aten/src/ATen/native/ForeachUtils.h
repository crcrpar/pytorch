#pragma once
#include <ATen/ATen.h>
#include <ATen/native/utils/ParamsHash.h>
#include <c10/util/irange.h>

namespace at {
namespace native {
namespace {
// Check if tensor list has either a boolean tensor or a integer tensor
bool has_integral_tensor(TensorList tensors, const bool includeBool) {
  return std::any_of(tensors.begin(), tensors.end(),
    [&includeBool](const auto & t) { return at::isIntegralType(t.scalar_type(), includeBool); });
}
// check if tensor list has bool tensors
bool has_bool_tensor(TensorList tensors) {
  return std::any_of(tensors.begin(), tensors.end(),
    [](const auto & t) -> bool { return t.scalar_type() == ScalarType::Bool; });
}

// Check foreach API restrictions
// - Tensor lists must be non-empty.
// - All TensorLists and ScalarLists must have the same number of elements.
// - Corresponding tensors must have the same size.
void check_foreach_api_restrictions(TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0, "Tensor list must have at least one tensor.");
}

void check_foreach_api_restrictions(TensorList tensors, ArrayRef<Scalar> scalars) {
  check_foreach_api_restrictions(tensors);
  TORCH_CHECK(tensors.size() == scalars.size(), "Tensor list must have same number of elements as scalar list.");
}

void check_foreach_api_restrictions(TensorList tensors1, TensorList tensors2) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors2.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() == tensors2.size(), "Tensor lists must have the same number of tensors, got ", tensors1.size(), " and ", tensors2.size());
}

void check_foreach_api_restrictions(TensorList tensors1, TensorList tensors2, TensorList tensors3) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors2.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors3.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() == tensors2.size(), "Tensor lists must have the same number of tensors, got ", tensors1.size(), " and ", tensors2.size());
  TORCH_CHECK(tensors1.size() == tensors3.size(), "Tensor lists must have the same number of tensors, got ", tensors1.size(), " and ", tensors3.size());
}

void check_foreach_api_restrictions(TensorList tensors1, TensorList tensors2, TensorList tensors3, ArrayRef<Scalar> scalars) {
  check_foreach_api_restrictions(tensors1, tensors2, tensors3);
  TORCH_CHECK(tensors1.size() == scalars.size(), "Tensor list must have same number of elements as scalar list, got ", tensors1.size(), " and ", scalars.size());
}

void check_foreach_api_restrictions(const std::vector<std::vector<at::Tensor>> & mta_input_tensors, ArrayRef<Scalar> scalars = {}) {
  switch (mta_input_tensors.size()) {
    case 1:
      check_foreach_api_restrictions(
        TensorList{mta_input_tensors[0]},
        scalars);
      break;
    case 2:
      TORCH_CHECK(scalars.size() == 0, "For Binary ops with two tensors, scalars must be 0");
      check_foreach_api_restrictions(
        TensorList{mta_input_tensors[0]},
        TensorList{mta_input_tensors[1]});
      break;
    case 3:
      if (scalars.size() == 0) {
        check_foreach_api_restrictions(
          TensorList{mta_input_tensors[0]},
          TensorList{mta_input_tensors[1]},
          TensorList{mta_input_tensors[2]});
      } else {
        check_foreach_api_restrictions(
          TensorList{mta_input_tensors[0]},
          TensorList{mta_input_tensors[1]},
          TensorList{mta_input_tensors[2]},
          scalars);
      }
      break;
    default:
      TORCH_CHECK(false, "Invalid number of input vectors of tensors.");
      break;
  }
}

// To go via 'fast' path, several conditions must be satisfied
// - All tensors in all lists must have the same dtype.
// - All tensors must be on the same device
// - All tensors must have strided layout
// - All tensors must be non-overlapping and dense
// - Resulting tensor must have the same dtype as the input one

// Please, make sure to call check_foreach_api_restrictions before calling this method.
// There is a set of preconditions that have to be satisfied.
bool check_fast_path_restrictions(
  ArrayRef<TensorList> tensorLists,
  ArrayRef<Scalar> scalarList = {},
  bool does_op_promote_integer_inputs_to_float = false) {
    const auto expected_dtype = tensorLists[0][0].dtype();
    const auto expected_device = tensorLists[0][0].device();

    auto is_tensor_okay = [&](const Tensor& tensor) {
      return tensor.dtype() == expected_dtype &&
             tensor.device() == expected_device &&
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

    // Check if corresponding tensors in tensor lists have the same sizes and strides.
    for (const auto& tensor_list : tensorLists) {
      for (const auto j : c10::irange(tensorLists[0].size())) {
        if (tensorLists[0][j].sizes() != tensor_list[j].sizes()) {
          return false;
        }
        if (tensorLists[0][j].strides() != tensor_list[j].strides()) {
          return false;
        }
      }
    }

    // This function has already checked that `tensorList[j][i]` for all j, i has the same dtype
    // using `is_tensor_okay` function above.
    // This means we only need to check if {tensorList[0][0], tensorList[0][1], tensorList[0][2], ...}
    // do type promotion with scalarLIst.
    for (const auto i : c10::irange(tensorLists[0].size())) {
      // For division, integer inputs will result in float.
      if (does_op_promote_integer_inputs_to_float) {
        if (at::isIntegralType(tensorLists[0][i].scalar_type(), /*includeBool*/ true)) {
          return false;
        }
      }
      if (scalarList.size() > 0) {
        const auto& scalar = scalarList.size() == 1 ? scalarList[0] : scalarList[i];
        const auto& tensor = tensorLists[0][i];
        // note(mkozuki): This check might be responsible for `_foreach_add(bool_tensors, bool_tensors)`
        // being pushed to slow path.
        if (tensor.scalar_type() != at::native::result_type(scalar, tensor)) {
          return false;
        }
      }
    }

    return true;
}

bool can_use_fast_route(ArrayRef<TensorList> tensorLists,
                        ArrayRef<Scalar> scalarList = {},
                        bool does_op_promote_integer_inputs_to_float = false) {
#if defined(USE_ROCM)
  return false;
#else
  return check_fast_path_restrictions(tensorLists, scalarList, does_op_promote_integer_inputs_to_float);
#endif
}

bool can_use_fast_route(TensorList tensors1, TensorList tensors2, bool does_op_promote_integer_inputs_to_float = false) {
#if defined(USE_ROCM)
  return false;
#else
  return can_use_fast_route({tensors1, tensors2}, {}, does_op_promote_integer_inputs_to_float);
#endif
}
// Tensor Grouping: Separate input Tensors of TensorList(s) by their device&dtype into `vector<vector<Tensor>>`.
#include <unordered_map>
#include <string>

using tensorvec_t = std::vector<at::Tensor>;
using mta_input_tensors_t = std::vector<tensorvec_t>;
using scalarvec_t = std::vector<Scalar>;
using indexvec_t = std::vector<size_t>;

class MTAInputTensorsAndScalars {
 public:
  mta_input_tensors_t mta_input_tensors;
  scalarvec_t mta_input_scalars;
  indexvec_t indices;

  MTAInputTensorsAndScalars() : mta_input_tensors{}, mta_input_scalars{}, indices{} {}

  // Unary & Binary with Scalar
  MTAInputTensorsAndScalars(const Tensor & t, const size_t index)
    : mta_input_tensors{ tensorvec_t{t} }, mta_input_scalars{}, indices{index} {}

  void append(const Tensor & t, const size_t index) {
    mta_input_tensors[0].emplace_back(t);
    indices.emplace_back(index);
  }

  // Binary with 2 TensorLists
  MTAInputTensorsAndScalars(const Tensor & t1, const Tensor & t2, const size_t index)
    : mta_input_tensors{ tensorvec_t{t1}, tensorvec_t{t2} }, mta_input_scalars{}, indices{index} {}

  void append(const Tensor & t1, const Tensor & t2, const size_t index) {
    mta_input_tensors[0].emplace_back(t1);
    mta_input_tensors[1].emplace_back(t2);
    indices.emplace_back(index);
  }

  // Binary with TensorList and ArrayRef<Scalar>
  MTAInputTensorsAndScalars(const Tensor & t, const Scalar & scalar, const size_t index)
    : mta_input_tensors{ tensorvec_t{t} }, mta_input_scalars{ scalar }, indices{index} {}

  void append(const Tensor & t, const Scalar & scalar, const size_t index) {
    mta_input_tensors[0].emplace_back(t);
    mta_input_scalars.emplace_back(scalar);
    indices.emplace_back(index);
  }

  // Pointwise
  MTAInputTensorsAndScalars(const Tensor & i, const Tensor & t1, const Tensor & t2, const size_t index)
    : mta_input_tensors{ tensorvec_t{i}, tensorvec_t{t1}, tensorvec_t{t2} }, mta_input_scalars{}, indices{index} {}

  void append(const Tensor & i, const Tensor & t1, const Tensor & t2, const size_t index) {
    mta_input_tensors[0].emplace_back(i);
    mta_input_tensors[1].emplace_back(t1);
    mta_input_tensors[2].emplace_back(t2);
    indices.emplace_back(index);
  }

  MTAInputTensorsAndScalars(const Tensor & i, const Tensor & t1, const Tensor & t2, const Scalar & scalar, const size_t index)
    : mta_input_tensors{ tensorvec_t{i}, tensorvec_t{t1}, tensorvec_t{t2} }, mta_input_scalars{ scalar }, indices{index} {}

  void append(const Tensor & i, const Tensor & t1, const Tensor & t2, const Scalar & scalar, const size_t index) {
    mta_input_tensors[0].emplace_back(i);
    mta_input_tensors[1].emplace_back(t1);
    mta_input_tensors[2].emplace_back(t2);
    mta_input_scalars.emplace_back(scalar);
    indices.emplace_back(index);
  }

  ArrayRef<Scalar> get_scalarList() const {
    if (mta_input_scalars.size() == 0) {
      return {};
    }
    return ArrayRef<Scalar>(mta_input_scalars);
  }

  TensorList get_tensorList(const size_t i) {
    return TensorList(mta_input_tensors[i]);
  }

  // Check vector<vector<Tensor>> and ArrayRef<Scalar> are compatible with MultiTensorApply or not.
  bool is_mta_compatible() const {
    return _is_mta_compatible({}, false);
  }

  bool is_mta_compatible(const Scalar & scalar) const {
    return _is_mta_compatible({scalar}, false);
  }

  bool is_mta_compatible(const bool does_op_promote_integer_inputs_to_float) const {
    return _is_mta_compatible(get_scalarList(), does_op_promote_integer_inputs_to_float);
  }

  bool is_mta_compatible(const Scalar & scalar, const bool does_op_promote_integer_inputs_to_float) const {
    return _is_mta_compatible({scalar}, does_op_promote_integer_inputs_to_float);
  }

 private:
  bool _is_mta_compatible(ArrayRef<Scalar> scalarList, const bool does_op_promote_integer_inputs_to_float) const {
    const auto & expected_dtype = mta_input_tensors[0][0].dtype();
    const auto & expected_device = mta_input_tensors[0][0].device();

    auto is_tensor_okay = [&](const Tensor& t) {
      return t.dtype() == expected_dtype &&
             t.device() == expected_device &&
             t.layout() == at::kStrided &&
             t.is_non_overlapping_and_dense();
    };

    for (const auto & tensors : mta_input_tensors) {
      for (const auto & t : tensors) {
        if (!is_tensor_okay(t)) {
          return false;
        }
      }
    }

    for (const auto & i : c10::irange(mta_input_tensors.size())) {
      for (const auto & j : c10::irange(mta_input_tensors[0].size())) {
        if (mta_input_tensors[0][j].strides() != mta_input_tensors[i][j].strides()) {
          return false;
        }
      }
    }

    for (const auto & i : c10::irange(mta_input_tensors[0].size())) {
      if (does_op_promote_integer_inputs_to_float) {
        if (at::isIntegralType(mta_input_tensors[0][i].scalar_type(), /* includeBool */ true)) {
          return false;
        }
      }
      if (scalarList.size() > 0) {
        const auto& scalar = scalarList.size() == 1 ? scalarList[0] : scalarList[i];
        if (scalar.isComplex()) {
          return false;
        }
        if (mta_input_tensors[0][i].scalar_type() != at::native::result_type(scalar, mta_input_tensors[0][i])) {
          return false;
        }
      }
    }
    return true;
  }
};

using devicekey_t = std::string;
template<int N_TENSORS> using dtypekey_t = std::array<uint8_t, N_TENSORS>;
template<int N_TENSORS> using DtypesHash = ParamsHash<std::array<uint8_t, N_TENSORS>>;
template<int N_TENSORS> using DtypesEqual = ParamsEqual<std::array<uint8_t, N_TENSORS>>;
template<int N_TENSORS> using dtype_map_t = std::unordered_map< dtypekey_t<N_TENSORS>, MTAInputTensorsAndScalars, DtypesHash<N_TENSORS>, DtypesEqual<N_TENSORS> >;
template<int N_TENSORS> using device_map_t = std::unordered_map< devicekey_t, dtype_map_t<N_TENSORS> >;

inline Scalar get_scalar(ArrayRef<Scalar> scalarList, const size_t i) {
  return scalarList.size() == 1 ? scalarList[0] : scalarList[i];
}

device_map_t<1> separate_tensors_by_device_and_dtype(TensorList tensors) {
  device_map_t<1> grouped_tensors_map;

  for (const auto & i : c10::irange(tensors.size())) {
    const at::Tensor & t = tensors[i];
    devicekey_t dev_key {t.device().str()};
    dtypekey_t<1> dtype_key {static_cast<uint8_t>(t.scalar_type())};

    if (!grouped_tensors_map.count(dev_key)) {
      grouped_tensors_map.emplace(dev_key, dtype_map_t<1>{{ dtype_key, MTAInputTensorsAndScalars(t, i) }});
    } else {
      if (!grouped_tensors_map[dev_key].count(dtype_key)) {
        grouped_tensors_map[dev_key].emplace(dtype_key, MTAInputTensorsAndScalars(t, i));
      } else {
        grouped_tensors_map[dev_key][dtype_key].append(t, i);
      }
    }
  }
  return grouped_tensors_map;
}

device_map_t<1> separate_tensors_by_device_and_dtype(TensorList tensors, ArrayRef<Scalar> scalarlist) {
  device_map_t<1> grouped_tensors_map;

  for (const auto & i : c10::irange(tensors.size())) {
    auto & t = tensors[i];
    devicekey_t dev_key = t.device().str();
    dtypekey_t<1> dtype_key { static_cast<uint8_t>(t.scalar_type()) };
    auto scalar = get_scalar(scalarlist, i);
    if (!grouped_tensors_map.count(dev_key)) {
      grouped_tensors_map.emplace(dev_key, dtype_map_t<1>{{ dtype_key, MTAInputTensorsAndScalars(t, scalar, i)  }});
    } else {
      if (!grouped_tensors_map[dev_key].count(dtype_key)) {
        grouped_tensors_map[dev_key].emplace(dtype_key, MTAInputTensorsAndScalars(t, scalar, i));
      } else {
        grouped_tensors_map[dev_key][dtype_key].append(t, scalar, i);
      }
    }
  }

  return grouped_tensors_map;
}

device_map_t<2> separate_tensors_by_device_and_dtype(TensorList tensors1, TensorList tensors2) {
  device_map_t<2> grouped_tensors_map;
  for (const auto & i : c10::irange(tensors1.size())) {
    auto t1 = tensors1[i];
    auto t2 = tensors2[i];
    TORCH_CHECK(t1.device() == t2.device(), "Operand tensors must be on the same device.");
    devicekey_t dev_key = t1.device().str();
    dtypekey_t<2> dtype_key { static_cast<uint8_t>(t1.scalar_type()), static_cast<uint8_t>(t2.scalar_type()) };
    if (!grouped_tensors_map.count(dev_key)) {
      grouped_tensors_map.emplace(dev_key, dtype_map_t<2>{{ dtype_key, MTAInputTensorsAndScalars(t1, t2, i) }});
    } else {
      if (!grouped_tensors_map[dev_key].count(dtype_key)) {
        grouped_tensors_map[dev_key].emplace(dtype_key, MTAInputTensorsAndScalars(t1, t2, i));
      } else {
        grouped_tensors_map[dev_key][dtype_key].append(t1, t2, i);
      }
    }
  }
  return grouped_tensors_map;
}

device_map_t<3> separate_tensors_by_device_and_dtype(TensorList input, TensorList tensors1, TensorList tensors2) {
  device_map_t<3> grouped_tensors_map;
  for (const auto & i : c10::irange(input.size())) {
    auto in_tensor = input[i];
    auto t1 = tensors1[i];
    auto t2 = tensors2[i];
    TORCH_CHECK(in_tensor.device() == t1.device() && t1.device() == t2.device(), "Operand tensors must be on the same device.");
    devicekey_t dev_key = in_tensor.device().str();
    dtypekey_t<3> dtype_key { static_cast<uint8_t>(in_tensor.scalar_type()), static_cast<uint8_t>(t1.scalar_type()), static_cast<uint8_t>(t2.scalar_type()) };
    if (!grouped_tensors_map.count(dev_key)) {
      grouped_tensors_map.emplace(dev_key, dtype_map_t<3>{{ dtype_key, MTAInputTensorsAndScalars(in_tensor, t1, t2, i) }});
    } else {
      if (!grouped_tensors_map[dev_key].count(dtype_key)) {
        grouped_tensors_map[dev_key].emplace(dtype_key, MTAInputTensorsAndScalars(in_tensor, t1, t2, i));
      } else {
        grouped_tensors_map[dev_key][dtype_key].append(in_tensor, t1, t2, i);
      }
    }
  }
  return grouped_tensors_map;
}

device_map_t<3> separate_tensors_by_device_and_dtype(TensorList input, TensorList tensors1, TensorList tensors2, ArrayRef<Scalar> scalarlist) {
  device_map_t<3> grouped_tensors_map;
  for (const auto & i : c10::irange(input.size())) {
    auto in_tensor = input[i];
    auto t1 = tensors1[i];
    auto t2 = tensors2[i];
    TORCH_CHECK(in_tensor.device() == t1.device() && t1.device() == t2.device(), "Operand tensors must be on the same device.");
    devicekey_t dev_key = in_tensor.device().str();
    dtypekey_t<3> dtype_key { static_cast<uint8_t>(in_tensor.scalar_type()), static_cast<uint8_t>(t1.scalar_type()), static_cast<uint8_t>(t2.scalar_type()) };
    auto scalar = get_scalar(scalarlist, i);
    if (!grouped_tensors_map.count(dev_key)) {
      grouped_tensors_map.emplace(dev_key, dtype_map_t<3>{{ dtype_key, MTAInputTensorsAndScalars(in_tensor, t1, t2, scalar, i) }});
    } else {
      if (!grouped_tensors_map[dev_key].count(dtype_key)) {
        grouped_tensors_map[dev_key].emplace(dtype_key, MTAInputTensorsAndScalars(in_tensor, t1, t2, scalar, i));
      } else {
        grouped_tensors_map[dev_key][dtype_key].append(in_tensor, t1, t2, scalar, i);
      }
    }
  }
  return grouped_tensors_map;
}

}
}} // at::native
