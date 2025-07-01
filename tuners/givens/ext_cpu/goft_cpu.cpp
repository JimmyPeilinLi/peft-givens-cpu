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

// 一次执行 K 个 Givens 旋转
torch::Tensor goft_rot_stack_cpu(torch::Tensor x,
                                 torch::Tensor cos_stack,   // [K, D]
                                 torch::Tensor sin_stack){  // [K, D]
  TORCH_CHECK(x.dim() >= 2, "x must be [B,S,D] or [B,D]");
  TORCH_CHECK(x.device().is_cpu(), "CPU tensor expected");

  auto orig_shape = x.sizes();
  const int64_t D = x.size(-1);
  const int64_t B = x.numel() / D;
  const int64_t K = cos_stack.size(0);

  x = x.reshape({B, D});                 // [B, D]
  auto y = x.clone();                    // out

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "rot_stack_forward", [&]{
    auto yptr = y.data_ptr<scalar_t>();
    for (int64_t k=0; k<K; ++k){
      const scalar_t* c = cos_stack[k].data_ptr<scalar_t>();
      const scalar_t* s = sin_stack[k].data_ptr<scalar_t>();

      at::parallel_for(0, B, 0, [&](int64_t beg, int64_t end){
        for (int64_t b=beg; b<end; ++b){
          auto row = yptr + b*D;
          for (int64_t i=0;i<D;i+=2){
            scalar_t xi=row[i], xj=(i+1<D?row[i+1]:scalar_t(0));
            scalar_t cv=c[i],   sv=s[i];
            row[i]         =  cv*xi - sv*xj;
            if (i+1<D) row[i+1] =  sv*xi + cv*xj;
          }
        }
      });
    }
  });

  return y.reshape(orig_shape);
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

std::vector<torch::Tensor> goft_rot_stack_backward_cpu(
        torch::Tensor grad_out,          // [*, D]
        torch::Tensor x,
        torch::Tensor cos_stack,
        torch::Tensor sin_stack){

  const int64_t K = cos_stack.size(0);
  auto gx = torch::zeros_like(x);
  auto gcos = torch::zeros_like(cos_stack);
  auto gsin = torch::zeros_like(sin_stack);

  /* --- 反向迭代：最后一层先求梯度 --- */
  auto y_cur = x.clone();                         // y0 = input

  // 正向先全部旋转，得到最终输出 & 中间缓存
  std::vector<torch::Tensor> y_cache{y_cur};
  for(int64_t k=0;k<K;++k){
      y_cur = goft_rot_cpu(y_cur, cos_stack[k], sin_stack[k]);
      y_cache.push_back(y_cur);
  }
  // 现在 grad = grad_out
  auto grad_cur = grad_out.reshape_as(y_cur).contiguous();

  for (int64_t k=K-1; k>=0; --k){
      // y_(k) 是旋转前
      auto y_prev = y_cache[k];
      auto grads  = goft_rot_backward_cpu(grad_cur, y_prev,
                                          cos_stack[k], sin_stack[k]);
      grad_cur = grads[0];                // 传递给下一层
      gcos[k]  = grads[1];
      gsin[k]  = grads[2];
  }
  gx = grad_cur;                          // w.r.t. input
  return {gx, gcos, gsin};
}