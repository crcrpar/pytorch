#pragma once
#include <ATen/ATen.h>

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

  for (const auto i : c10::irange(tensors1.size())) {
    TORCH_CHECK(tensors1[i].sizes() == tensors2[i].sizes(), "Corresponding tensors in lists must have the same size, got ", tensors1[i].sizes(), " and ", tensors2[i].sizes());
  }
}

void check_foreach_api_restrictions(TensorList tensors1, TensorList tensors2, TensorList tensors3) {
  TORCH_CHECK(tensors1.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors2.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors3.size() > 0, "Tensor list must have at least one tensor.");
  TORCH_CHECK(tensors1.size() == tensors2.size(), "Tensor lists must have the same number of tensors, got ", tensors1.size(), " and ", tensors2.size());
  TORCH_CHECK(tensors1.size() == tensors3.size(), "Tensor lists must have the same number of tensors, got ", tensors1.size(), " and ", tensors3.size());

  for (const auto i : c10::irange(tensors1.size())) {
    TORCH_CHECK(tensors1[i].sizes() == tensors2[i].sizes(), "Corresponding tensors in lists must have the same size, got ", tensors1[i].sizes(), " and ", tensors2[i].sizes());
    TORCH_CHECK(tensors1[i].sizes() == tensors3[i].sizes(), "Corresponding tensors in lists must have the same size, got ", tensors1[i].sizes(), " and ", tensors3[i].sizes());
  }
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

// TODO(mkozuki): Consider whether we really need this function or not.
// Note that, there is a possibility that foreach fastpath supports type promotion in the future,
// which might complicate the functionality this function should provides.
// However, as of now, the check of division op with integer inputs is duplicated.
// `check_fast_path_restrictions` does the same thing in it before calling this function.
bool will_promote_tensor(const Tensor& tensor, const Scalar& scalar, bool does_op_promote_integer_inputs_to_float = false) {
  // In case of division, integer inputs will result in float
  if (does_op_promote_integer_inputs_to_float &&
      at::isIntegralType(tensor.scalar_type(), /* includeBool */ true)) {
    return true;
  }
  return tensor.scalar_type() != at::native::result_type(scalar, tensor);
}

// TODO(mkozuki): Remove device-dtype check once tensor grouping is completed.
// Please, make sure to call check_foreach_api_restrictions before calling this method.
// There is a set of preconditions that have to be satisfied.
bool check_fast_path_restrictions(
  ArrayRef<TensorList> tensorLists,
  ArrayRef<Scalar> scalarList = {},
  bool does_op_promote_integer_inputs_to_float = false) {
    const auto expected_dtype = tensorLists[0][0].dtype();
    const auto expected_device = tensorLists[0][0].device();
    if (!expected_device.is_cuda()) {
      return false;
    }

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

    // Check if corresponding tensors in tensor lists have the same strides.
    for (int i=0; i < tensorLists.size(); i++) {
      for (int j=0; j < tensorLists[0].size(); j++) {
        if (tensorLists[0][j].strides() != tensorLists[i][j].strides()) {
          return false;
        }
      }
    }

    // This function has already checked that `tensorList[j][i]` for all j, i has the same dtype
    // using `is_tensor_okay` function above.
    // checked by `check_foreach_api_restrictions`).
    // This means we only need to check if {tensorList[0][0], tensorList[0][1], tensorList[0][2], ...}
    // do type promotion with scalarLIst.
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
        // FIXME(mkozuki): Consider specializing `TensorListScalarListMetadata` for complex dtypes
        // to access the following comment.
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

bool can_use_fast_route(ArrayRef<TensorList> tensorLists,
                        ArrayRef<Scalar> scalarList = {},
                        bool does_op_promote_integer_inputs_to_float = false) {
#ifdef __HIP_PLATFORM_HCC__
  return false;
#else
  return check_fast_path_restrictions(tensorLists, scalarList, does_op_promote_integer_inputs_to_float);
#endif
}

bool can_use_fast_route(TensorList tensors1, TensorList tensors2, bool does_op_promote_integer_inputs_to_float = false) {
#ifdef __HIP_PLATFORM_HCC__
  return false;
#else
  return can_use_fast_route({tensors1, tensors2}, {}, does_op_promote_integer_inputs_to_float);
#endif
}

// note(mkozuki): should write a dedicated check_fast_path_restrictions for this?
bool can_use_fast_route(
    const std::vector<std::vector<Tensor>>& mta_input_tensors,
    ArrayRef<Scalar> scalarList = {},
    bool does_op_promote_integer_inputs_to_float = false) {
#ifdef __HIP_PLATFORM_HCC_
  return false;
#else
  std::vector<TensorList> tmp;
  for (const auto & tensorvec : mta_input_tensors) {
    tmp.emplace_back(TensorList{tensorvec});
  }
  return check_fast_path_restrictions(ArrayRef<TensorList>{tmp}, scalarList, does_op_promote_integer_inputs_to_float);
#endif
}
// Tensor Grouping: Separate input Tensors of TensorList(s) by their device&dtype into `vector<vector<Tensor>>`.
#include <unordered_map>
#include <string>

using tensorvec_t = std::vector<at::Tensor>;
using mta_input_tensors_t = std::vector<tensorvec_t>;
using scalarvec_t = std::vector<Scalar>;
using indexvec_t = std::vector<size_t>;

struct MTAInputTensorsAndScalars {
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
  MTAInputTensorsAndScalars(Tensor & t1, Tensor & t2, const size_t index)
    : mta_input_tensors{ tensorvec_t{t1}, tensorvec_t{t2} }, mta_input_scalars{}, indices{index} {}

  void append(Tensor & t1, Tensor & t2, const size_t index) {
    mta_input_tensors[0].emplace_back(t1);
    mta_input_tensors[1].emplace_back(t2);
    indices.emplace_back(index);
  }

  // Binary with TensorList and ArrayRef<Scalar>
  MTAInputTensorsAndScalars(const Tensor & t, Scalar & scalar, const size_t index)
    : mta_input_tensors{ tensorvec_t{t} }, mta_input_scalars{ scalar }, indices{index} {}

  void append(const Tensor & t, Scalar & scalar, const size_t index) {
    mta_input_tensors[0].emplace_back(t);
    mta_input_scalars.emplace_back(scalar);
    indices.emplace_back(index);
  }

  // Pointwise
  MTAInputTensorsAndScalars(Tensor & i, Tensor & t1, Tensor & t2, Scalar & scalar, const size_t index)
    : mta_input_tensors{ tensorvec_t{i}, tensorvec_t{t1}, tensorvec_t{t2} }, mta_input_scalars{ scalar }, indices{index} {}

  void append(Tensor & i, Tensor & t1, Tensor & t2, Scalar & scalar, const size_t index) {
    mta_input_tensors[0].emplace_back(i);
    mta_input_tensors[1].emplace_back(t1);
    mta_input_tensors[2].emplace_back(t2);
    mta_input_scalars.emplace_back(scalar);
    indices.emplace_back(index);
  }

  ArrayRef<Scalar> scalars() const {
    TORCH_CHECK(mta_input_scalars.size() > 0, "No scalar values stored.");
    return ArrayRef<Scalar>(mta_input_scalars);
  }
};

using devicekey_t = std::string;
template<int N_TENSORS> using dtypekey_t = std::array<uint8_t, N_TENSORS>;

template<int N_TENSORS>
struct DtypeHashFunctor {
  size_t operator() (const dtypekey_t<N_TENSORS> dtypes) const {
    size_t seed = 0;
    for (const auto & v : dtypes) {
      seed ^= std::hash<uint8_t>()(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    }
    return seed;
  }
};

template<int N_TENSORS> using dtype_map_t = std::unordered_map< dtypekey_t<N_TENSORS>, MTAInputTensorsAndScalars, DtypeHashFunctor<N_TENSORS> >;
template<int N_TENSORS> using device_map_t = std::unordered_map< devicekey_t, dtype_map_t<N_TENSORS> >;

Scalar get_scalar(ArrayRef<Scalar> scalarlist, const size_t i) {
  switch (scalarlist.size()) {
    case 0:
      return 0;
      break;
    case 1:
      return scalarlist[0];
      break;
    default:
      return scalarlist[i];
      break;
  }
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

device_map_t<3> separate_tensors_by_device_and_dtype(TensorList input, TensorList tensors1, TensorList tensors2, ArrayRef<Scalar> scalarlist = {}) {
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
