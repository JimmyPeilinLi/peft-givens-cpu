#include <torch/extension.h>
#include <vector>

// 声明
torch::Tensor goft_rot_cpu(torch::Tensor, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> goft_rot_backward_cpu(torch::Tensor, torch::Tensor,
                                                 torch::Tensor, torch::Tensor);

torch::Tensor goft_rot_stack_cpu(torch::Tensor, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> goft_rot_stack_backward_cpu(torch::Tensor, torch::Tensor,
                                                       torch::Tensor, torch::Tensor);

/* ---- 单旋转 C++ dispatch ---- */
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

/* ---- Stack C++ dispatch ---- */
torch::Tensor goft_rot_stack(torch::Tensor x,
                       torch::Tensor c,
                       torch::Tensor s) {
  return goft_rot_stack_cpu(x, c, s);
}

std::vector<torch::Tensor> goft_rot_stack_backward(torch::Tensor g,
                                             torch::Tensor x,
                                             torch::Tensor c,
                                             torch::Tensor s) {
  return goft_rot_stack_backward_cpu(g, x, c, s);
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

class GOFTRotStackFn : public torch::autograd::Function<GOFTRotStackFn>{
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::Tensor x,
                               torch::Tensor cos_stack,
                               torch::Tensor sin_stack){
    ctx->saved_data["orig_sizes"] = x.sizes();
    ctx->save_for_backward({x, cos_stack, sin_stack});
    return goft_rot_stack(x, cos_stack, sin_stack);
  }
  static std::vector<torch::Tensor> backward(torch::autograd::AutogradContext* ctx,
                                             std::vector<torch::Tensor> g_o){
    auto saved = ctx->get_saved_variables();
    auto x = saved[0], cos = saved[1], sin = saved[2];
    
    // auto grads = goft_rot_stack_backward(g_o[0], x, cos, sin);
    auto x2d = x.reshape({-1, x.size(-1)});
    auto grads = goft_rot_stack_backward(g_o[0], x2d, cos, sin);

    // 将 grad_x 还原回原 shape
    auto orig_sizes = ctx->saved_data["orig_sizes"].toIntVector();
    grads[0] = grads[0].reshape(orig_sizes);

    return {grads[0], grads[1], grads[2]};
  }
};

/* ---- PyBind ---- */
TORCH_LIBRARY(goft, m) {
  m.def("rot", &goft_rot);
  m.def("rot_stack", &goft_rot_stack);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rot_autograd", [](torch::Tensor x, torch::Tensor c, torch::Tensor s) {
    return GOFTRotFn::apply(x, c, s);
  });
  m.def("rot_stack_autograd",
        [](torch::Tensor x, torch::Tensor c, torch::Tensor s){
            return GOFTRotStackFn::apply(x, c, s);
        });
}
