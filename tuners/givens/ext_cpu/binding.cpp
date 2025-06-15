#include <torch/extension.h>
#include <vector>

// 声明
torch::Tensor goft_rot_cpu(torch::Tensor, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> goft_rot_backward_cpu(torch::Tensor, torch::Tensor,
                                                 torch::Tensor, torch::Tensor);

/* ---- 普通 C++ dispatch ---- */
torch::Tensor goft_rot(torch::Tensor x,
                       torch::Tensor c,
                       torch::Tensor s) {
  return goft_rot_cpu(x, c, s);
}

std::vector<torch::Tensor> goft_rot_backward(torch::Tensor g,
                                             torch::Tensor x,
                                             torch::Tensor c,
                                             torch::Tensor s) {
  return goft_rot_backward_cpu(g, x, c, s);
}

/* ---- Autograd glue ---- */
class GOFTRotFn : public torch::autograd::Function<GOFTRotFn> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor x,
                               torch::Tensor c,
                               torch::Tensor s) {
    ctx->save_for_backward({x, c, s});
    return goft_rot(x, c, s);
  }
  static std::vector<torch::Tensor> backward(torch::autograd::AutogradContext *ctx,
                                             std::vector<torch::Tensor> grad_outs) {
    auto saved = ctx->get_saved_variables();
    auto x = saved[0], c = saved[1], s = saved[2];
    auto grads = goft_rot_backward(grad_outs[0], x, c, s);
    return {grads[0], grads[1], grads[2]};
  }
};

/* ---- PyBind ---- */
TORCH_LIBRARY(goft, m) {
  m.def("rot", &goft_rot);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rot_autograd", [](torch::Tensor x, torch::Tensor c, torch::Tensor s) {
    return GOFTRotFn::apply(x, c, s);
  });
}
