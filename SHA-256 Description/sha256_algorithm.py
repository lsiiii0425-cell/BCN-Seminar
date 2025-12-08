import math

# --- SHA-256 Constants and Initial Hash Values (H) ---
# Initial hash values (h0 to h7), computed from the fractional parts 
# of the square roots of the first 8 primes (2..19)
H = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
]

# Сonstants (k0 to k63), computed from the fractional parts 
# of the cube roots of the first 64 primes (2..311)
K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]

# --- Bitwise Logic and Rotation Functions (32-bit operations) ---

def right_rotate(n, d, bits=32):
    # (n >> d) | (n << (bits - d)) 
    return ((n >> d) | (n << (bits - d))) & 0xFFFFFFFF

def right_shift(n, d):
    # n >> d
    return (n >> d) & 0xFFFFFFFF

def Ch(e, f, g):
    # Choice function: (e AND f) XOR ((NOT e) AND g) 
    return (e & f) ^ (~e & g)

def Maj(a, b, c):
    # Majority function: (a AND b) XOR (a AND c) XOR (b AND c) 
    return (a & b) ^ (a & c) ^ (b & c)

def Sigma0(a):
    # (a ROTR 2) XOR (a ROTR 13) XOR (a ROTR 22) 
    return right_rotate(a, 2) ^ right_rotate(a, 13) ^ right_rotate(a, 22)

def Sigma1(e):
    # (e ROTR 6) XOR (e ROTR 11) XOR (e ROTR 25) 
    return right_rotate(e, 6) ^ right_rotate(e, 11) ^ right_rotate(e, 25)

def sigma0(w):
    # (w ROTR 7) XOR (w ROTR 18) XOR (w SHR 3) 
    return right_rotate(w, 7) ^ right_rotate(w, 18) ^ right_shift(w, 3)

def sigma1(w):
    # (w ROTR 17) XOR (w ROTR 19) XOR (w SHR 10) 
    return right_rotate(w, 17) ^ right_rotate(w, 19) ^ right_shift(w, 10)

# --- Pre-processing (Encoding and Padding) ---

def preprocess_message(input_string):
    # Step 1: Encode input into binary (UTF-8 standard)
    # For 'Blockchain', this results in an 80-bit binary string.
    
    # Python's default bytes encoding for strings is often UTF-8, which covers ASCII.
    # We convert to binary string representation of the bytes.
    binary_message = ''.join(format(byte, '08b') for byte in input_string.encode('utf-8'))
    
    # Step 2: Padding 
    original_length = len(binary_message)
    
    # 2a: Append a single '1' bit
    binary_message += '1'
    
    # 2b: Append k zero bits
    # The message must be a multiple of 512 bits.
    # Pad until length is 448 mod 512 (which is 64 bits short of 512).
    # Since we added '1', length is now original_length + 1.
    k = (448 - (original_length + 1) % 512) % 512
    binary_message += '0' * k
    
    # 2c: Append the 64-bit original length 
    length_bits = format(original_length, '064b')
    binary_message += length_bits
    
    # Step 2: Break padded message into 512-bit blocks
    # For 'Blockchain', N=1 block
    blocks = []
    for i in range(0, len(binary_message), 512):
        blocks.append(binary_message[i:i + 512])
        
    return blocks

# --- Hashing (Main SHA-256 Loop) ---

def sha256_hash(blocks):
    # Initialize working hash values h0 to h7 to the initial constants H 
    h = list(H) 

    # Process each 512-bit block [cite: 678]
    for block in blocks:
        # 1. Prepare the Message Schedule W (W0 to W63) 
        W = []
        # W0 to W15 (sixteen 32-bit "words") 
        for i in range(0, 512, 32):
            W.append(int(block[i:i + 32], 2))

        # W16 to W63 iteratively calculated 
        for t in range(16, 64):
            # Wt = sigma1(W[t-2]) + W[t-7] + sigma0(W[t-15]) + W[t-16] (mod 2^32)
            W.append((sigma1(W[t-2]) + W[t-7] + sigma0(W[t-15]) + W[t-16]) & 0xFFFFFFFF)

        # 2. Initialize working variables a, b, c, d, e, f, g, h 
        a, b, c, d, e, f, g, hh = h 

        # 3. Compression loop (64 rounds) 
        for t in range(64):
            # T1 = h + Sigma1(e) + Ch(e, f, g) + Kt + Wt (mod 2^32) 
            T1 = (hh + Sigma1(e) + Ch(e, f, g) + K[t] + W[t]) & 0xFFFFFFFF
            
            # T2 = Sigma0(a) + Maj(a, b, c) (mod 2^32) 
            T2 = (Sigma0(a) + Maj(a, b, c)) & 0xFFFFFFFF

            # Update working variables (shift operation) 
            hh = g                                  # h = g
            g = f                                   # g = f
            f = e                                   # f = e
            e = (d + T1) & 0xFFFFFFFF               # e = d + T1
            d = c                                   # d = c
            c = b                                   # c = b
            b = a                                   # b = a
            a = (T1 + T2) & 0xFFFFFFFF              # a = T1 + T2

        # 4. Add final values of working variables to hash values (h0 to h7) [cite: 831]
        h[0] = (h[0] + a) & 0xFFFFFFFF 
        h[1] = (h[1] + b) & 0xFFFFFFFF
        h[2] = (h[2] + c) & 0xFFFFFFFF
        h[3] = (h[3] + d) & 0xFFFFFFFF
        h[4] = (h[4] + e) & 0xFFFFFFFF
        h[5] = (h[5] + f) & 0xFFFFFFFF
        h[6] = (h[6] + g) & 0xFFFFFFFF
        h[7] = (h[7] + hh) & 0xFFFFFFFF
        
    return h

# --- Final Digest Generation ---

def generate_digest(final_hash_values):
    # Step 3: Append hash values h0..h7 to get final 256-bit digest 
    # Return digest in hexadecimal format 
    digest_hex = ''
    for h_val in final_hash_values:
        # Format each 32-bit word as 8 hexadecimal characters
        digest_hex += format(h_val, '08x')
        
    return digest_hex

# --- Main Execution ---

def sha256(message):
    blocks = preprocess_message(message)
    final_hash_values = sha256_hash(blocks)
    return generate_digest(final_hash_values)

# --- Run for input "Blockchain" ---
input = "Blockchain"
digest = sha256(input)

print(f"Input: {input}")
print(f"SHA-256 Digest: {digest}")