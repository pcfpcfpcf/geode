#ifndef GEODE_CUDA_H
#define GEODE_CUDA_H

#include <stddef.h>

/* A CUDA driver context plus the JIT'd kernels this executor runs. The driver
   API is dlopen'd and the kernels JIT from embedded PTX, so any compute
   capability works without a build-time toolchain -- the same trick the probe
   uses, kept independent of it because the probe is an offline tool with its
   own lifetime. */
typedef struct CudaDevice CudaDevice;

int cuda_start(CudaDevice **out, char *err, size_t errsz);
void cuda_stop(CudaDevice *dev);

/* Free vram, for the caller that has to fit more than one model on the gpu. */
unsigned long long cuda_vram_free(CudaDevice *dev);

unsigned long long cuda_alloc(CudaDevice *dev, size_t bytes, char *err,
                              size_t errsz);
void cuda_free(CudaDevice *dev, unsigned long long ptr);
int cuda_copy_to(CudaDevice *dev, unsigned long long dst, const void *src,
                 size_t bytes);
int cuda_copy_from(CudaDevice *dev, void *dst, unsigned long long src,
                   size_t bytes);

/* Pins a host buffer so the small per-layer transfers skip the driver's
   pageable staging round trip, which costs ~1 ms a copy on this driver.
   Best effort: an unpinned buffer still works, just slower. */
int cuda_host_pin(CudaDevice *dev, void *ptr, size_t bytes);
void cuda_host_unpin(CudaDevice *dev, void *ptr);

/* Drains the queue; a deferred kernel fault surfaces here. */
int cuda_sync(CudaDevice *dev);

/* Weight row encodings cuda_gemm reads. Q4_K streams at quant size; F16 is
   what every other encoding is uploaded as, dequantized once on the host --
   half the bytes of f32, which matters on every token of decode. F32 stays
   for callers that need exact host values. */
enum { CUDA_W_F32, CUDA_W_Q4K, CUDA_W_F16 };

/* out[tokens][rows] = rows · x[tokens], where rows is a stack of n_head
   matrices of head_out rows each and x holds one n_in-wide segment per head
   per token, segments x_head_stride elements apart:
     out[r_tok * out_row_stride + row]
       = dot(W_row, x[r_tok * x_row_stride + head * x_head_stride ..])
   A plain matmul is the n_head == 1 case. `type` says how W rows are packed.

   The stack is walked as one row array -- row = head * head_out + r_in -- so
   the kernel needs head alignment only to stage x: head_out must be a
   multiple of 8 (one warp block's row count) whenever n_head > 1, and a Q4_K
   row must be a whole number of 256-wide blocks. cuda_start checks both
   shapes with a self-test kernel launch, so a bad call fails loudly there
   rather than silently here. */
void cuda_gemm(CudaDevice *dev, unsigned long long w, unsigned long long x,
               unsigned long long out, int n_in, int head_out, int n_head,
               int x_row_stride, int x_head_stride, int out_row_stride,
               int n_tokens, int type);

/* Attention geometry, mirrored in the kernels' parameter block. Field order
   is load-bearing: the PTX reads it field by field. */
typedef struct {
    int kind;        /* model.h's AttentionKind: ATTN_MLA or ATTN_GQA */
    int n_head;
    int n_head_kv;   /* GQA only */
    int head_dim_k;
    int rank;        /* MLA only: kv_lora_rank */
    int rope_dim;    /* MLA only: the rotary slice of a query head */
    int n_segments;  /* quantize scales per slot */
    int cache_width; /* int8 per slot */
    int slot_bytes;  /* cache_width + n_segments * 4, scales interleaved */
    int q_stride;    /* n_head * head_dim_k floats per token */
    int latent_stride; /* n_head * rank, MLA only */
    int out_stride;  /* MLA: n_head * rank; GQA: n_head * head_dim_v */
    int fold_width;  /* MLA: rank; GQA: head_dim_v */
    float kq_scale;
    int n_ctx;
    float rms_eps; /* rmsnorm epsilon for the prep kernels; the PTX reads
                      fields by offset, so this one goes last */
} CudaGeometry;

/* The two halves of the attention pipeline the old design round-tripped
   through the cpu: query norm+rope, and kv norm+rope+quantize straight into
   the layer's cache slot at (position, position+n_tokens). One thread per
   head (q) or per cache segment (kv) per token; n_head and n_segments must
   fit a warp, which check_geometry already guarantees. The query is roped
   in place -- GQA norms each head first against `weight`, MLA leaves the
   nope slice alone and rotates only the tail. The kv kernel reads the kv
   projection, applies the same normalization and rotation the cpu tier
   does, and writes the slot's bytes and scales so the cache never crosses
   the bus. Geometry (including rms_eps) comes from the device block. */
void cuda_q_prep(CudaDevice *dev, unsigned long long query,
                 unsigned long long cos_sin, unsigned long long weight,
                 unsigned long long geometry_on_device, int n_tokens);
void cuda_kv_prep(CudaDevice *dev, unsigned long long kv_projected,
                  unsigned long long cos_sin, unsigned long long norm,
                  unsigned long long slot_base,
                  unsigned long long geometry_on_device, int n_tokens);

/* activated[i] = silu(gate[i]) * up[i] over the first `total` f32 elements.
   The feed-forward's activation on the gpu, between the gate/up gemm and the
   down gemm. */
void cuda_swiglu(CudaDevice *dev, unsigned long long gate,
                 unsigned long long up, unsigned long long out, size_t total);

/* Scores every cached position of one layer for every head and token,
   softmaxes each row, and folds the weighted cache into the attention output:
   MLA into the latent buffer that v_b reads, GQA straight into the head-value
   buffer that attn_output reads. The query must already be normed and rotated
   (rope for MLA covers only the rotary slice, as on the cpu).

   `cache` points at the layer's first slot -- slot p is cache[p *
   slot_bytes], its scales at cache[p * slot_bytes + cache_width].
   `scores` is n_head * n_tokens * n_ctx floats, scored positions only.
   `query_latent` is MLA's k_b output; GQA passes `query` again. `geometry`
   is the same block uploaded to the device. */
void cuda_attention(CudaDevice *dev, unsigned long long cache,
                    unsigned long long query, unsigned long long query_latent,
                    unsigned long long scores, unsigned long long out,
                    int position, int n_tokens, const CudaGeometry *geometry,
                    unsigned long long geometry_on_device);

/* The first error a kernel or copy hit, or NULL. Held here rather than
   threaded through every launch because a fault after the start self-test
   means the driver is dying and the process cannot meaningfully continue. */
const char *cuda_fault(const CudaDevice *dev);

#endif