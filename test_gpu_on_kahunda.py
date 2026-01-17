import torch

# Check if the handshake is successful
print(f"Driver/Hardware link: {torch.cuda.is_available()}")

# Run a small GPU operation
try:
    a = torch.ones(10).cuda()
    b = a * 2
    print("GPU Test: Success!")
except Exception as e:
    print(f"GPU Test: Failed with error {e}")