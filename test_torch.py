from torch.utils.cpp_extension import load_inline
import torch

cpp_source = """
#include <torch/extension.h>

torch::Tensor dummy_add(torch::Tensor x) {
    return x + 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dummy_add", &dummy_add, "Add 1 to tensor");
}
"""

ext = load_inline(
    name="dummy_add_extension",
    cpp_sources=[cpp_source],
    functions="dummy_add",
    extra_cflags=["-O3"],
    extra_ldflags=["-lcurand", "-L/usr/local/cuda-12.8/targets/x86_64-linux/lib"],
    extra_include_paths=["/usr/local/cuda-12.8/include"],
    verbose=True
)

x = torch.tensor([1.0, 2.0, 3.0])
print("Output from dummy_add:", ext.dummy_add(x))
