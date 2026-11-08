"""Chapter 8 numerical walkthrough; pure Python, no CUDA/GPU required.

Run from the repository root:
    python3 kernels/interview/book/notes/examples/ch08_rope_transpose_walkthrough.py

The transpose simulation follows base.cuh's 16x16 thread block and 64x17
shared tile. Bank profiles describe one scalar shared-memory instruction,
not measured GPU latency or throughput.
"""

from collections import defaultdict
import math


def rope(vector, position, base=10000.0):
    result = []
    dimension = len(vector)
    assert dimension % 2 == 0
    for pair in range(dimension // 2):
        angle = position * base ** (-2.0 * pair / dimension)
        cosine, sine = math.cos(angle), math.sin(angle)
        a, b = vector[2 * pair:2 * pair + 2]
        result.extend((a * cosine - b * sine, a * sine + b * cosine))
    return result


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(values):
    exponents = [math.exp(x - max(values)) for x in values]
    denominator = sum(exponents)
    return [x / denominator for x in exponents]


def print_matrix(name, matrix):
    print(name)
    for row in matrix:
        print("  [" + ", ".join(f"{x:.9f}" for x in row) + "]")


def bank_profile(word_indices):
    # Identical-word reads are broadcasts, not distinct-address conflicts.
    words_per_bank = defaultdict(set)
    for word in word_indices:
        words_per_bank[word % 32].add(word)
    return len(words_per_bank), max(map(len, words_per_bank.values()))


def run():
    q = [1.0, 2.0, 3.0, 4.0]
    k = [2.0, 1.0, 1.0, 0.0]
    values = [[10.0, 0.0, 1.0, 0.0],
              [0.0, 20.0, 0.0, 1.0],
              [5.0, 5.0, 1.0, 1.0]]
    queries = [rope(q, position) for position in range(3)]
    keys = [rope(k, position) for position in range(3)]
    gram = [[dot(query, key) for key in keys] for query in queries]
    scores = [[x / 2.0 for x in row] for row in gram]
    probabilities = [softmax(scores[m][:m + 1]) + [0.0] * (2 - m)
                     for m in range(3)]
    outputs = [[sum(probabilities[m][n] * values[n][d] for n in range(3))
                for d in range(4)] for m in range(3)]
    for vector in queries:
        assert math.isclose(dot(vector, vector), 30.0, abs_tol=1e-12)
    for vector in keys:
        assert math.isclose(dot(vector, vector), 6.0, abs_tol=1e-12)
    assert math.isclose(gram[1][0], gram[2][1], abs_tol=1e-12)
    for row in probabilities:
        assert math.isclose(sum(row), 1.0, abs_tol=1e-12)
    for name, matrix in [("Q_rot, positions 0/1/2", queries),
                         ("K_rot, positions 0/1/2", keys),
                         ("Q_rot @ K_rot.T (before causal mask)", gram),
                         ("scores = Gram / sqrt(4)", scores),
                         ("causal probabilities", probabilities),
                         ("attention outputs", outputs)]:
        print_matrix(name, matrix)
    print_matrix("prefill K_rot.T, shape 4x2", list(zip(*keys[:2])))

    # X is a completely specified 128x32 FP32-exact integer matrix.
    rows, columns = 128, 32
    x = [[float(1000 * r + c) for c in range(columns)] for r in range(rows)]
    y = [[None] * rows for _ in range(columns)]
    write_count = [[0] * rows for _ in range(columns)]
    for block_y in range(rows // 64):
        for block_x in range(columns // 16):
            tile = [[None] * 17 for _ in range(64)]
            # All producers finish before any consumer (the CUDA barrier).
            for ty in range(16):
                for tx in range(16):
                    for i in range(4):
                        r = block_y * 64 + ty * 4 + i
                        c = block_x * 16 + tx
                        tile[ty * 4 + i][tx] = x[r][c]
            for ty in range(16):
                for tx in range(16):
                    for j in range(4):
                        r = block_x * 16 + ty
                        c = block_y * 64 + tx * 4 + j
                        value = tile[tx * 4 + j][ty]
                        assert value is not None
                        y[r][c] = value
                        write_count[r][c] += 1
            assert all(tile[r][16] is None for r in range(64))
    assert all(y[c][r] == x[r][c] and write_count[c][r] == 1
               for r in range(rows) for c in range(columns))
    assert y[5][8:12] == [8005.0, 9005.0, 10005.0, 11005.0]
    print("transpose 128x32 -> 32x128: all 4096 outputs checked")
    print("producer (tx=5,ty=2), consumer (tx=2,ty=5):", y[5][8:12])

    for width in (16, 17):
        for warp in range(8):
            for component in range(4):
                write_words, read_words = [], []
                for lane in range(32):
                    linear = warp * 32 + lane
                    tx, ty = linear % 16, linear // 16
                    write_words.append((ty * 4 + component) * width + tx)
                    read_words.append((tx * 4 + component) * width + ty)
                assert bank_profile(write_words)[1] == 2
                assert bank_profile(read_words)[1] == (16 if width == 16 else 2)
        first_read = [(4 * (lane % 16)) * width + lane // 16
                      for lane in range(32)]
        first_write = [(4 * (lane // 16)) * width + lane % 16
                       for lane in range(32)]
        print(f"tile[64][{width}], warp 0, component 0:")
        print("  shared write banks:", [word % 32 for word in first_write])
        print("  shared read banks: ", [word % 32 for word in first_read])
        print("  (active banks, max distinct words/bank):",
              bank_profile(first_write), bank_profile(first_read))
    assert bank_profile([0] * 32) == (1, 1)
    assert bank_profile([33 * lane for lane in range(32)]) == (32, 1)

    input_words = [(lane // 16) * columns + lane % 16 for lane in range(32)]
    naive_output_words = [(lane % 16) * rows + lane // 16 for lane in range(32)]
    packed_output_words = [(lane // 16) * rows + 4 * (lane % 16) + j
                           for lane in range(32) for j in range(4)]
    sectors = lambda words: len({word * 4 // 32 for word in words})
    assert sectors(input_words) == 4
    assert sectors(naive_output_words) == 16
    assert sectors(packed_output_words) == 16
    assert 64 * 17 * 4 == 4352
    assert rows * columns * 4 * 2 == 32768
    print("32-byte sectors, naive warp read/write:",
          sectors(input_words), sectors(naive_output_words))
    print("32-byte sectors, padded float4 warp write:", sectors(packed_output_words))
    print("PASS: RoPE, causal attention, relative-position identity, full transpose,")
    print("      producer/consumer mapping, all warp/component bank profiles and sectors.")


if __name__ == "__main__":
    run()
