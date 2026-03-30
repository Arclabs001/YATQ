"""
WHT-based TurboQuant following llama.cpp implementation exactly.

Key insight from llama.cpp:
- Keys are stored in WHT domain (quantized)
- Query is transformed with WHT on-the-fly during attention
- Values are stored in WHT domain, IWHT applied after attention
- Scale factor: WHT(query) is scaled by scale/D

This matches the fused kernel approach in fattn-vec.cuh and fattn-common.cuh.
"""

import torch
import math
from typing import Tuple, Optional


# Sign patterns from llama.cpp cpy-utils.cuh (256 bits for 256 elements)
TBQ_SIGNS = bytes([
    0xa7, 0x3b, 0x91, 0xf4, 0x6d, 0xc2, 0x58, 0x0e,
    0xb3, 0x7f, 0x24, 0xd6, 0x89, 0x45, 0xea, 0x1c,
    0x63, 0xaf, 0xd8, 0x52, 0x97, 0x0b, 0xe1, 0x3d,
    0x76, 0xc4, 0x19, 0xfe, 0x4a, 0x85, 0x2c, 0xdb,
])

# Independent sign pattern for QJL SRHT (from llama.cpp)
QJL_SIGNS = bytes([
    0xd3, 0x4e, 0xa8, 0x17, 0x9c, 0x5b, 0xe6, 0x31,
    0x72, 0xb9, 0x0d, 0xf5, 0x43, 0x8a, 0x6e, 0xc7,
    0x58, 0x2f, 0x94, 0xe1, 0xb6, 0x3d, 0x0a, 0x7c,
    0xc5, 0x61, 0xd8, 0x4f, 0xa3, 0x97, 0x1e, 0x85,
])


def get_sign(signs: bytes, idx: int) -> float:
    """Get sign (-1.0 or +1.0) at index idx from sign bytes."""
    return -1.0 if ((signs[idx >> 3] >> (idx & 7)) & 1) else 1.0


def serial_wht(x: torch.Tensor) -> torch.Tensor:
    """
    Serial Walsh-Hadamard Transform (in-place butterfly algorithm).

    For d=256, this is equivalent to H @ x where H is the Hadamard matrix.
    Note: WHT is orthogonal up to scaling: H @ H^T = d * I
    So inverse is H @ x / d

    Args:
        x: tensor of shape (..., d) where d must be a power of 2

    Returns:
        transformed tensor of same shape (not normalized)
    """
    shape = x.shape
    d = shape[-1]

    # Flatten to process last dimension
    x_flat = x.reshape(-1, d).clone()

    # Butterfly algorithm: for len = 1, 2, 4, 8, ..., d/2
    length = 1
    while length < d:
        for i in range(0, d, 2 * length):
            for j in range(length):
                u = x_flat[:, i + j].clone()
                v = x_flat[:, i + j + length].clone()
                x_flat[:, i + j] = u + v
                x_flat[:, i + j + length] = u - v
        length *= 2

    return x_flat.reshape(shape)


def inverse_wht(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse Walsh-Hadamard Transform.
    Since H @ H = d * I, the inverse is H @ x / d.
    """
    return serial_wht(x) / x.shape[-1]


# Lloyd-Max boundaries and centroids for N(0, 1) distribution
# Computed via Lloyd-Max algorithm (alternating optimization)

LM_BOUNDARIES = {
    # 2-bit: 3 boundaries for 4 levels
    2: torch.tensor([-0.9816, 0.0, 0.9816]),
    # 3-bit: 7 boundaries for 8 levels
    3: torch.tensor([-1.7479, -1.05, -0.5005, 0.0, 0.5005, 1.05, 1.7479]),
    # 4-bit: 15 boundaries for 16 levels
    4: torch.tensor([-2.4008, -1.8435, -1.4371, -1.0993, -0.7996, -0.5224, -0.2582,
                     0.0, 0.2582, 0.5224, 0.7996, 1.0993, 1.4371, 1.8435, 2.4008]),
    # 6-bit: 63 boundaries for 64 levels
    6: torch.tensor([-3.5169, -3.1058, -2.8233, -2.6015, -2.4159, -2.2544, -2.1103, -1.9792,
                     -1.8584, -1.7457, -1.6397, -1.5392, -1.4433, -1.3515, -1.2630, -1.1774,
                     -1.0943, -1.0134, -0.9344, -0.8571, -0.7812, -0.7066, -0.6331, -0.5606,
                     -0.4889, -0.4178, -0.3473, -0.2773, -0.2077, -0.1383, -0.0691, 0.0,
                     0.0691, 0.1383, 0.2077, 0.2773, 0.3473, 0.4178, 0.4889, 0.5606,
                     0.6331, 0.7066, 0.7812, 0.8571, 0.9344, 1.0134, 1.0943, 1.1774,
                     1.2630, 1.3515, 1.4433, 1.5392, 1.6397, 1.7457, 1.8584, 1.9792,
                     2.1103, 2.2544, 2.4159, 2.6015, 2.8233, 3.1058, 3.5169]),
    # 8-bit: 255 boundaries for 256 levels
    8: torch.tensor([-4.5020, -4.1704, -3.9485, -3.7782, -3.6388, -3.5200, -3.4161, -3.3236,
                     -3.2400, -3.1636, -3.0933, -3.0280, -2.9670, -2.9098, -2.8558, -2.8047,
                     -2.7562, -2.7101, -2.6659, -2.6237, -2.5831, -2.5441, -2.5065, -2.4701,
                     -2.4350, -2.4009, -2.3679, -2.3357, -2.3044, -2.2738, -2.2440, -2.2148,
                     -2.1862, -2.1582, -2.1307, -2.1037, -2.0771, -2.0508, -2.0250, -1.9995,
                     -1.9742, -1.9493, -1.9246, -1.9002, -1.8759, -1.8519, -1.8280, -1.8042,
                     -1.7807, -1.7572, -1.7338, -1.7106, -1.6875, -1.6644, -1.6414, -1.6185,
                     -1.5956, -1.5728, -1.5500, -1.5273, -1.5046, -1.4820, -1.4593, -1.4367,
                     -1.4142, -1.3916, -1.3691, -1.3465, -1.3240, -1.3015, -1.2790, -1.2565,
                     -1.2341, -1.2116, -1.1891, -1.1667, -1.1442, -1.1218, -1.0993, -1.0769,
                     -1.0544, -1.0320, -1.0095, -0.9871, -0.9646, -0.9422, -0.9198, -0.8973,
                     -0.8749, -0.8525, -0.8300, -0.8076, -0.7852, -0.7627, -0.7403, -0.7179,
                     -0.6954, -0.6730, -0.6506, -0.6281, -0.6057, -0.5833, -0.5608, -0.5384,
                     -0.5160, -0.4935, -0.4711, -0.4487, -0.4262, -0.4038, -0.3814, -0.3589,
                     -0.3365, -0.3141, -0.2916, -0.2692, -0.2468, -0.2243, -0.2019, -0.1795,
                     -0.1570, -0.1346, -0.1122, -0.0897, -0.0673, -0.0449, -0.0224, 0.0,
                     0.0224, 0.0449, 0.0673, 0.0897, 0.1122, 0.1346, 0.1570, 0.1795,
                     0.2019, 0.2243, 0.2468, 0.2692, 0.2916, 0.3141, 0.3365, 0.3589,
                     0.3814, 0.4038, 0.4262, 0.4487, 0.4711, 0.4935, 0.5160, 0.5384,
                     0.5608, 0.5833, 0.6057, 0.6281, 0.6506, 0.6730, 0.6954, 0.7179,
                     0.7403, 0.7627, 0.7852, 0.8076, 0.8300, 0.8525, 0.8749, 0.8973,
                     0.9198, 0.9422, 0.9646, 0.9871, 1.0095, 1.0320, 1.0544, 1.0769,
                     1.0993, 1.1218, 1.1442, 1.1667, 1.1891, 1.2116, 1.2341, 1.2565,
                     1.2790, 1.3015, 1.3240, 1.3465, 1.3691, 1.3916, 1.4142, 1.4367,
                     1.4593, 1.4820, 1.5046, 1.5273, 1.5500, 1.5728, 1.5956, 1.6185,
                     1.6414, 1.6644, 1.6875, 1.7106, 1.7338, 1.7572, 1.7807, 1.8042,
                     1.8280, 1.8519, 1.8759, 1.9002, 1.9246, 1.9493, 1.9742, 1.9995,
                     2.0250, 2.0508, 2.0771, 2.1037, 2.1307, 2.1582, 2.1862, 2.2148,
                     2.2440, 2.2738, 2.3044, 2.3357, 2.3679, 2.4009, 2.4350, 2.4701,
                     2.5065, 2.5441, 2.5831, 2.6237, 2.6659, 2.7101, 2.7562, 2.8047,
                     2.8558, 2.9098, 2.9670, 3.0280, 3.0933, 3.1636, 3.2400, 3.3236,
                     3.4161, 3.5200, 3.6388, 3.7782, 3.9485, 4.1704, 4.5020]),
}

LM_CENTROIDS = {
    # 2-bit: 4 centroids
    2: torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104]),
    # 3-bit: 8 centroids
    3: torch.tensor([-2.1519, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1519]),
    # 4-bit: 16 centroids
    4: torch.tensor([-2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3880, -0.1284,
                     0.1284, 0.3880, 0.6568, 0.9423, 1.2562, 1.6180, 2.0690, 2.7326]),
    # 6-bit: 64 centroids
    6: torch.tensor([-3.7674, -3.2664, -2.9452, -2.7015, -2.5016, -2.3302, -2.1786, -2.0419,
                     -1.9165, -1.8002, -1.6912, -1.5882, -1.4902, -1.3965, -1.3064, -1.2195,
                     -1.1352, -1.0533, -0.9735, -0.8954, -0.8188, -0.7437, -0.6696, -0.5967,
                     -0.5245, -0.4532, -0.3824, -0.3122, -0.2424, -0.1729, -0.1037, -0.0345,
                     0.0345, 0.1037, 0.1729, 0.2424, 0.3122, 0.3824, 0.4532, 0.5245,
                     0.5967, 0.6696, 0.7437, 0.8188, 0.8954, 0.9735, 1.0533, 1.1352,
                     1.2195, 1.3064, 1.3965, 1.4902, 1.5882, 1.6912, 1.8002, 1.9165,
                     2.0419, 2.1786, 2.3302, 2.5016, 2.7015, 2.9452, 3.2664, 3.7674]),
    # 8-bit: 256 centroids
    8: torch.tensor([-4.7062, -4.2978, -4.0430, -3.8540, -3.7025, -3.5751, -3.4649, -3.3674,
                     -3.2798, -3.2002, -3.1271, -3.0595, -2.9965, -2.9375, -2.8820, -2.8296,
                     -2.7799, -2.7326, -2.6875, -2.6444, -2.6030, -2.5632, -2.5249, -2.4880,
                     -2.4523, -2.4177, -2.3842, -2.3516, -2.3198, -2.2889, -2.2587, -2.2292,
                     -2.2004, -2.1721, -2.1443, -2.1171, -2.0903, -2.0639, -2.0378, -2.0121,
                     -1.9868, -1.9617, -1.9369, -1.9123, -1.8880, -1.8638, -1.8399, -1.8161,
                     -1.7924, -1.7689, -1.7455, -1.7222, -1.6990, -1.6759, -1.6529, -1.6299,
                     -1.6070, -1.5842, -1.5614, -1.5387, -1.5160, -1.4933, -1.4706, -1.4480,
                     -1.4254, -1.4029, -1.3803, -1.3578, -1.3353, -1.3128, -1.2903, -1.2678,
                     -1.2453, -1.2228, -1.2004, -1.1779, -1.1554, -1.1330, -1.1105, -1.0881,
                     -1.0656, -1.0432, -1.0207, -0.9983, -0.9759, -0.9534, -0.9310, -0.9086,
                     -0.8861, -0.8637, -0.8412, -0.8188, -0.7964, -0.7739, -0.7515, -0.7291,
                     -0.7066, -0.6842, -0.6618, -0.6393, -0.6169, -0.5945, -0.5720, -0.5496,
                     -0.5272, -0.5047, -0.4823, -0.4599, -0.4374, -0.4150, -0.3926, -0.3701,
                     -0.3477, -0.3253, -0.3028, -0.2804, -0.2580, -0.2355, -0.2131, -0.1907,
                     -0.1682, -0.1458, -0.1234, -0.1009, -0.0785, -0.0561, -0.0337, -0.0112,
                     0.0112, 0.0337, 0.0561, 0.0785, 0.1009, 0.1234, 0.1458, 0.1682,
                     0.1907, 0.2131, 0.2355, 0.2580, 0.2804, 0.3028, 0.3253, 0.3477,
                     0.3701, 0.3926, 0.4150, 0.4374, 0.4599, 0.4823, 0.5047, 0.5272,
                     0.5496, 0.5720, 0.5945, 0.6169, 0.6393, 0.6618, 0.6842, 0.7066,
                     0.7291, 0.7515, 0.7739, 0.7964, 0.8188, 0.8412, 0.8637, 0.8861,
                     0.9086, 0.9310, 0.9534, 0.9759, 0.9983, 1.0207, 1.0432, 1.0656,
                     1.0881, 1.1105, 1.1330, 1.1554, 1.1779, 1.2004, 1.2228, 1.2453,
                     1.2678, 1.2903, 1.3128, 1.3353, 1.3578, 1.3803, 1.4029, 1.4254,
                     1.4480, 1.4706, 1.4933, 1.5160, 1.5387, 1.5614, 1.5842, 1.6070,
                     1.6299, 1.6529, 1.6759, 1.6990, 1.7222, 1.7455, 1.7689, 1.7924,
                     1.8161, 1.8399, 1.8638, 1.8880, 1.9123, 1.9369, 1.9617, 1.9868,
                     2.0121, 2.0378, 2.0639, 2.0903, 2.1171, 2.1443, 2.1721, 2.2004,
                     2.2292, 2.2587, 2.2889, 2.3198, 2.3516, 2.3842, 2.4177, 2.4523,
                     2.4880, 2.5249, 2.5632, 2.6030, 2.6444, 2.6875, 2.7326, 2.7799,
                     2.8296, 2.8820, 2.9375, 2.9965, 3.0595, 3.1271, 3.2002, 3.2798,
                     3.3674, 3.4649, 3.5751, 3.7025, 3.8540, 4.0430, 4.2978, 4.7062]),
}


def lloyd_max_quantize(x: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Lloyd-Max quantization for N(0, 1) distribution.

    Args:
        x: tensor of shape (..., d), assumed to follow N(0, 1) after WHT
        bits: number of bits (2, 3, 4, 6, or 8)

    Returns:
        indices: quantized indices
        centroids: reconstructed values (same shape as x)
    """
    n_levels = 2 ** bits
    boundaries = LM_BOUNDARIES[bits].to(x.device)
    centroids = LM_CENTROIDS[bits].to(x.device)

    # Ensure centroids has correct length
    if len(centroids) != n_levels:
        # Generate uniform quantization as fallback
        centroids = torch.linspace(-3.0, 3.0, n_levels, device=x.device)

    # Ensure boundaries has correct length
    if len(boundaries) != n_levels - 1:
        # Generate uniform boundaries as fallback
        boundaries = torch.linspace(-3.0, 3.0, n_levels - 1, device=x.device)

    # Find index for each element
    indices = torch.zeros_like(x, dtype=torch.long)
    for i, b in enumerate(boundaries):
        indices = indices + (x >= b).long()

    # Clamp to valid range
    indices = indices.clamp(0, n_levels - 1)

    # Get centroids
    reconstructed = centroids[indices]

    return indices, reconstructed


class TurboQuantWHT:
    """
    WHT-based TurboQuant matching llama.cpp exactly.

    The key difference from random rotation:
    - Uses deterministic WHT (Walsh-Hadamard Transform)
    - Uses fixed sign patterns (TBQ_SIGNS for MSE, QJL_SIGNS for QJL)
    - Block size must be 256 (WHT requires power of 2)
    - Stores keys in WHT domain, transforms query on-the-fly
    """

    def __init__(self, dim: int, bits: int, block_size: Optional[int] = None):
        """
        Args:
            dim: dimension of vectors
            bits: bits per element (2, 3, 4, 6, or 8)
            block_size: WHT block size (default: next power of 2 >= dim)
        """
        assert bits in [2, 3, 4, 6, 8], f"bits must be 2, 3, 4, 6, or 8"

        self.dim = dim
        self.bits = bits

        # Determine block size (must be power of 2)
        if block_size is None:
            block_size = 1
            while block_size < dim:
                block_size *= 2

        self.block_size = block_size
        self.padded = block_size != dim

        # Precompute sign tensors (for block_size)
        self.tbq_signs = torch.tensor(
            [get_sign(TBQ_SIGNS, i % 256) for i in range(block_size)],
            dtype=torch.float32
        )
        self.qjl_signs = torch.tensor(
            [get_sign(QJL_SIGNS, i % 256) for i in range(block_size)],
            dtype=torch.float32
        )

    def quantize_key(self, key: torch.Tensor, use_qjl: bool = False) -> dict:
        """
        Quantize key vectors for KV cache storage.

        Following llama.cpp's quantize_f32_tbq*_block:
        1. Compute L2 norm
        2. Normalize
        3. Apply TBQ signs
        4. Apply WHT
        5. Lloyd-Max quantize

        Args:
            key: tensor of shape (..., dim)
            use_qjl: if True, also compute QJL residual data

        Returns:
            dict with quantized data for attention computation
        """
        original_shape = key.shape
        key_float = key.float()
        d = self.dim
        bs = self.block_size

        # Reshape to (n_vectors, dim)
        key_flat = key_float.reshape(-1, d)
        n_vectors = key_flat.shape[0]

        # Pad if necessary
        if self.padded:
            key_flat = F.pad(key_flat, (0, bs - d))

        # Step 1: Compute L2 norm per vector (on original dim)
        vec_norms = torch.norm(key_flat[:, :d], dim=-1)  # (n_vectors,)

        # Step 2: Normalize
        key_norm = key_flat / (vec_norms.unsqueeze(-1) + 1e-10)

        # Step 3: Apply TBQ signs
        signs = self.tbq_signs.to(key.device)
        key_signed = key_norm * signs.unsqueeze(0)

        # Step 4: Apply WHT
        key_wht = serial_wht(key_signed)

        # Step 5: Lloyd-Max quantization
        indices, centroids = lloyd_max_quantize(key_wht, self.bits)

        # Store: indices, norm
        result = {
            'vec_norm': vec_norms,  # (n_vectors,)
            'indices': indices,  # (n_vectors, block_size)
            'centroids_wht': centroids,  # (n_vectors, block_size) - centroids in WHT domain
            'key_wht_norm': key_wht,  # For debugging
        }

        if use_qjl:
            # QJL: compute residual in WHT domain
            residual = key_wht - centroids  # (n_vectors, block_size)

            # Residual norm per vector
            residual_norm = torch.norm(residual, dim=-1)  # (n_vectors,)

            # Apply QJL signs and WHT (SRHT = D2 @ H @ D1)
            # Matching llama.cpp: sign(WHT(qjl_signs * residual))
            qjl_s = self.qjl_signs.to(key.device)
            residual_qjl_signed = residual * qjl_s.unsqueeze(0)
            residual_qjl_wht = serial_wht(residual_qjl_signed)

            # Take sign (1 for >= 0, -1 for < 0)
            qjl_sign_bits = (residual_qjl_wht >= 0).float() * 2 - 1

            # d_qjl = gamma * norm (matching llama.cpp line 325)
            d_qjl = residual_norm * vec_norms

            result['qjl_signs'] = qjl_sign_bits  # (n_vectors, block_size)
            result['residual_norm'] = residual_norm  # (n_vectors,)
            result['d_qjl'] = d_qjl  # (n_vectors,)

        return result

    def quantize_value(self, value: torch.Tensor) -> dict:
        """
        Quantize value vectors for KV cache storage.

        Same as key quantization but simpler (no QJL needed).
        Values are stored in WHT domain for efficiency.
        """
        original_shape = value.shape
        value_float = value.float()
        d = self.dim
        bs = self.block_size

        value_flat = value_float.reshape(-1, d)
        n_vectors = value_flat.shape[0]

        # Pad if necessary
        if self.padded:
            value_flat = F.pad(value_flat, (0, bs - d))

        # L2 norm (on original dim)
        vec_norms = torch.norm(value_flat[:, :d], dim=-1)

        # Normalize + signs + WHT
        value_norm = value_flat / (vec_norms.unsqueeze(-1) + 1e-10)
        signs = self.tbq_signs.to(value.device)
        value_signed = value_norm * signs.unsqueeze(0)
        value_wht = serial_wht(value_signed)

        # Lloyd-Max quantize
        indices, centroids = lloyd_max_quantize(value_wht, self.bits)

        return {
            'vec_norm': vec_norms,
            'indices': indices,
            'centroids_wht': centroids,
        }

    def compute_attention_scores(
        self,
        query: torch.Tensor,
        key_data: dict,
        use_qjl: bool = False,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute attention scores <query, keys> using WHT-based method.

        Following llama.cpp's vec_dot_fattn_vec_KQ_tbq*:
        1. Apply TBQ signs to query
        2. Apply WHT to query
        3. Compute inner product with centroids in WHT domain
        4. Multiply by key norm

        Args:
            query: (batch, dim) query vectors
            key_data: dict from quantize_key()
            use_qjl: if True, add QJL correction
            scale: attention scale (default 1/sqrt(dim))

        Returns:
            scores: (batch, n_keys) attention scores
        """
        if scale is None:
            scale = 1.0 / math.sqrt(self.dim)

        d = self.dim
        bs = self.block_size
        query_float = query.float()
        batch = query_float.shape[0]
        n_keys = key_data['vec_norm'].shape[0]

        # Pad query if necessary
        if self.padded:
            query_float = F.pad(query_float, (0, bs - d))

        # Step 1-2: Apply signs + WHT to query
        signs = self.tbq_signs.to(query.device)
        query_signed = query_float * signs.unsqueeze(0)
        query_wht = serial_wht(query_signed)

        # Step 3: Inner product with centroids in WHT domain
        centroids_wht = key_data['centroids_wht']  # (n_keys, block_size)
        key_norms = key_data['vec_norm']  # (n_keys,)

        # Query scaled by scale/D (from llama.cpp line 271-272)
        # Note: D here is block_size for WHT scaling
        query_scaled = query_wht * scale / bs

        # Inner product (only on original dim columns)
        scores = torch.matmul(query_scaled[:, :d], centroids_wht[:, :d].T)  # (batch, n_keys)

        # Multiply by key norms
        scores = scores * key_norms.unsqueeze(0)

        if not use_qjl:
            return scores

        # QJL correction (matching llama.cpp fattn-common.cuh lines 689-691)
        # Formula: d_qjl * (Q_qjl · qjl_signs)
        # Where Q_qjl = WHT(Q_wht * qjl_signs) * scale * sqrt(pi/2) / D^2

        # Apply QJL signs to query_wht and do second WHT
        qjl_s = self.qjl_signs.to(query.device)
        query_qjl_signed = query_wht * qjl_s.unsqueeze(0)  # Apply signs
        query_qjl_wht = serial_wht(query_qjl_signed)  # Second WHT

        # QJL inner product with stored signs
        qjl_signs = key_data['qjl_signs']  # (n_keys, block_size)
        d_qjl = key_data['d_qjl']  # (n_keys,)

        # QJL correction: d_qjl * (query_qjl_wht · qjl_signs)
        # Scaling: scale * sqrt(pi/2) / D^2 (from llama.cpp line 318)
        qjl_factor = math.sqrt(math.pi / 2)
        qjl_scale = scale * qjl_factor / (bs * bs)

        # Only on original dim columns
        qjl_ip = torch.matmul(query_qjl_wht[:, :d], qjl_signs[:, :d].T)  # (batch, n_keys)

        # Add correction
        scores = scores + d_qjl.unsqueeze(0) * qjl_ip * qjl_scale

        return scores

    def reconstruct_key(self, key_data: dict) -> torch.Tensor:
        """
        Reconstruct key vectors from quantized data (for debugging/testing).

        Args:
            key_data: dict from quantize_key()

        Returns:
            reconstructed keys: (n_keys, dim)
        """
        centroids_wht = key_data['centroids_wht']  # (n_keys, block_size)
        vec_norms = key_data['vec_norm']
        bs = self.block_size
        d = self.dim

        # Inverse WHT to get back to original domain
        signs = self.tbq_signs.to(centroids_wht.device)
        centroids_orig = inverse_wht(centroids_wht) * signs.unsqueeze(0)

        # Rescale by norm and take only original dim
        reconstructed = centroids_orig[:, :d] * vec_norms.unsqueeze(-1)

        return reconstructed


# Test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from turboquant import TurboQuantMSE, TurboQuantProd

    print("=" * 70)
    print("WHT-based TurboQuant Test")
    print("=" * 70)

    dim = 256
    bits = 3

    torch.manual_seed(42)
    keys = torch.randn(100, dim)
    query = torch.randn(10, dim)

    # True inner products
    true_ip = torch.matmul(query, keys.T)

    # WHT version
    wht_quant = TurboQuantWHT(dim, bits)

    # Test MSE-only (scale=1 for raw inner product comparison)
    key_data_mse = wht_quant.quantize_key(keys, use_qjl=False)
    scores_mse = wht_quant.compute_attention_scores(query, key_data_mse, use_qjl=False, scale=1.0)

    # Test with QJL
    key_data_qjl = wht_quant.quantize_key(keys, use_qjl=True)
    scores_qjl = wht_quant.compute_attention_scores(query, key_data_qjl, use_qjl=True, scale=1.0)

    # Random rotation version for comparison
    mse_quant = TurboQuantMSE(dim, bits, seed=42)
    prod_quant = TurboQuantProd(dim, bits, seed=42)

    mse_recon, _, _ = mse_quant.quantize(keys, return_indices=True)
    prod_result = prod_quant.quantize(keys)

    mse_ip = torch.matmul(query, mse_recon.T)
    prod_ip = prod_quant.inner_product(query, prod_result)

    def metrics(est, true):
        mse = torch.mean((est - true) ** 2).item()
        bias = torch.mean(est - true).item() / torch.mean(torch.abs(true)).item() * 100
        var = torch.var(est - true).item()
        return mse, bias, var

    print(f"\n{'Method':<30} {'MSE':>12} {'Bias%':>10} {'Variance':>12}")
    print("-" * 65)

    print(f"{'Random MSE (3b)':<30} {metrics(mse_ip, true_ip)[0]:>12.4f} {metrics(mse_ip, true_ip)[1]:>+10.2f} {metrics(mse_ip, true_ip)[2]:>12.2f}")
    print(f"{'Random QJL (2b+1b)':<30} {metrics(prod_ip, true_ip)[0]:>12.4f} {metrics(prod_ip, true_ip)[1]:>+10.2f} {metrics(prod_ip, true_ip)[2]:>12.2f}")
    print(f"{'WHT MSE (3b)':<30} {metrics(scores_mse, true_ip)[0]:>12.4f} {metrics(scores_mse, true_ip)[1]:>+10.2f} {metrics(scores_mse, true_ip)[2]:>12.2f}")
    print(f"{'WHT QJL (2b+1b)':<30} {metrics(scores_qjl, true_ip)[0]:>12.4f} {metrics(scores_qjl, true_ip)[1]:>+10.2f} {metrics(scores_qjl, true_ip)[2]:>12.2f}")

    # Test reconstruction
    print("\n" + "=" * 70)
    print("Key Reconstruction Test")
    print("=" * 70)

    key_recon = wht_quant.reconstruct_key(key_data_mse)
    recon_mse = torch.mean((key_recon - keys) ** 2).item()
    print(f"Reconstruction MSE: {recon_mse:.6f}")
    print(f"Original key norm (avg): {torch.mean(torch.norm(keys, dim=-1)).item():.4f}")
    print(f"Reconstructed key norm (avg): {torch.mean(torch.norm(key_recon, dim=-1)).item():.4f}")

    print("\n" + "=" * 70)