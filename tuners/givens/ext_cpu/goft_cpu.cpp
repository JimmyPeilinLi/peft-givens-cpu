#include <torch/extension.h>
#include <ATen/Parallel.h>

template <typename scalar_t>
static void rot_forward_kernel(const scalar_t* x, const scalar_t* c,
                               const scalar_t* s, scalar_t* y,
                               int64_t B, int64_t D) {
  at::parallel_for(0, B, 0, [&](int64_t beg, int64_t end) {
    for (int64_t b = beg; b < end; ++b) {
      auto x_ptr = x + b * D;
      auto y_ptr = y + b * D;
      for (int64_t i = 0; i < D; i += 2) {
        scalar_t xi = x_ptr[i];
        scalar_t xj = (i + 1 < D) ? x_ptr[i + 1] : scalar_t(0);
        scalar_t cv = c[i];
        scalar_t sv = s[i];
        y_ptr[i]         =  cv * xi - sv * xj;
        if (i + 1 < D)
          y_ptr[i + 1] =  sv * xi + cv * xj;
      }
    }
  });
}

torch::Tensor goft_rot_cpu(torch::Tensor x,
                           torch::Tensor cos_v,
                           torch::Tensor sin_v) {
  TORCH_CHECK(x.dim() == 2, "x must be [B, D]");
  TORCH_CHECK(x.device().is_cpu(), "CPU tensor expected");

  auto y = torch::empty_like(x);
  int64_t B = x.size(0);
  int64_t D = x.size(1);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "goft_rot_cpu_forward", [&] {
    rot_forward_kernel<scalar_t>(x.data_ptr<scalar_t>(),
                                 cos_v.data_ptr<scalar_t>(),
                                 sin_v.data_ptr<scalar_t>(),
                                 y.data_ptr<scalar_t>(),
                                 B, D);
  });
  return y;
}

/* ---------- 反向 ---------- */

template <typename scalar_t>
static void rot_backward_kernel(const scalar_t* go, const scalar_t* x,
                                const scalar_t* c, const scalar_t* s,
                                scalar_t* gx, scalar_t* gc, scalar_t* gs,
                                int64_t B, int64_t D) {
  at::parallel_for(0, B, 0, [&](int64_t beg, int64_t end) {
    for (int64_t b = beg; b < end; ++b) {
      auto go_ptr = go + b * D;
      auto x_ptr  = x  + b * D;
      auto gx_ptr = gx + b * D;
      for (int64_t i = 0; i < D; i += 2) {
        scalar_t x1 = x_ptr[i];
        scalar_t x2 = (i + 1 < D) ? x_ptr[i + 1] : scalar_t(0);
        scalar_t cv = c[i];
        scalar_t sv = s[i];
        scalar_t go1 = go_ptr[i];
        scalar_t go2 = (i + 1 < D) ? go_ptr[i + 1] : scalar_t(0);

        /* grad wrt input */
        gx_ptr[i]         =  cv * go1 + sv * go2;
        if (i + 1 < D)
          gx_ptr[i + 1] = -sv * go1 + cv * go2;

        /* atomic accumulate grad wrt cos/sin */
        // scalar_t gcos_local = x1 * go1 - x2 * go2;
        // scalar_t gsin_local = -x2 * go1 - x1 * go2;
        // at::native::cpu_atomic_add(gc + i, gcos_local);
        // at::native::cpu_atomic_add(gs + i, gsin_local);

        // grad wrt cos & sin —— 用 OpenMP atomic
        #pragma omp atomic
        gc[i] += x1 * go1 - x2 * go2;

        #pragma omp atomic
        gs[i] += -x2 * go1 - x1 * go2;
      }
    }
  });
}

std::vector<torch::Tensor> goft_rot_backward_cpu(torch::Tensor grad_out,
                                                 torch::Tensor x,
                                                 torch::Tensor cos_v,
                                                 torch::Tensor sin_v) {
  auto grad_x   = torch::empty_like(x);
  auto grad_cos = torch::zeros_like(cos_v);
  auto grad_sin = torch::zeros_like(sin_v);

  int64_t B = x.size(0), D = x.size(1);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "goft_rot_cpu_backward", [&] {
    rot_backward_kernel<scalar_t>(grad_out.data_ptr<scalar_t>(),
                                  x.data_ptr<scalar_t>(),
                                  cos_v.data_ptr<scalar_t>(),
                                  sin_v.data_ptr<scalar_t>(),
                                  grad_x.data_ptr<scalar_t>(),
                                  grad_cos.data_ptr<scalar_t>(),
                                  grad_sin.data_ptr<scalar_t>(),
                                  B, D);
  });
  return {grad_x, grad_cos, grad_sin};
}
