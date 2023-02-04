#include <ATen/native/cuda/fused_adam_utils.cuh>

#include <ATen/Device.h>
#include <ATen/ScalarType.h>
#include <unordered_map>

namespace at { namespace native {
grouped_nested_tensorlists_t group_tensorlists(ArrayRef<TensorList> nested_tensorlists, const bool has_step_list) {
  TORCH_CHECK_GT(nested_tensorlists.size(), 0);
  TORCH_CHECK(std::all_of(
        nested_tensorlists.cbegin(), nested_tensorlists.cend(),
        [&](const auto & tensorlist) { return nested_tensorlists[0].size() == tensorlist.size(); }));

  const auto num_lists = nested_tensorlists.size();
  const auto num_tensors = nested_tensorlists[0].size();
  grouped_nested_tensorlists_t grouped_nested_tensorlists;

  for (const auto & tensor_idx : c10::irange(num_tensors)) {
    const auto device = nested_tensorlists[0][tensor_idx].device();
    const auto dtype = nested_tensorlists[0][tensor_idx].scalar_type();
    TORCH_CHECK([&]() {
      if (!has_step_list) {
        return std::all_of(
            nested_tensorlists.cbegin(), nested_tensorlists.cend(),
            [&](const auto & tensorlist) {
              return tensorlist[tensor_idx].device() == device && tensorlist[tensor_idx].scalar_type() == dtype;
            });
      } else {
        for (const auto & i : c10::irange(num_lists - 1)) {
          if (!(nested_tensorlists[i][tensor_idx].device() == device && nested_tensorlists[i][tensor_idx].scalar_type() == dtype)) {
            return false;
          }
        }
        return nested_tensorlists[num_lists - 1][tensor_idx].device() == device;
      }
    }());

    if (grouped_nested_tensorlists.count(device)) {
      if (grouped_nested_tensorlists.at(device).count(dtype)) {
        std::vector<std::vector<Tensor>> values;
        for (const auto & i : c10::irange(num_lists)) {
          // Assuming the tensors are hosted on one device and have one dtype.
          std::vector<Tensor> init;
          init.reserve(num_tensors);
          values.emplace_back(init);
        }
        grouped_nested_tensorlists[device].emplace(dtype, values);
      } else {
        for (const auto & i : c10::irange(num_lists)) {
          grouped_nested_tensorlists[device][dtype][i].emplace_back(nested_tensorlists[i][tensor_idx]);
        }
      }
    } else {
      std::vector<std::vector<Tensor>> values;
      for (const auto & i : c10::irange(num_lists)) {
        values.emplace_back(std::vector<Tensor>{nested_tensorlists[i][tensor_idx]});
      }
      dtype_nested_tensorlists_t init_value{{dtype, values}};
      grouped_nested_tensorlists.emplace(device, init_value);
    }
  }
  return grouped_nested_tensorlists;
}
}} // namespace at::native
