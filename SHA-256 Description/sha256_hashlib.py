from hashlib import sha256
input = 'Blockchain'
output = sha256(input.encode()).hexdigest()
print(output)