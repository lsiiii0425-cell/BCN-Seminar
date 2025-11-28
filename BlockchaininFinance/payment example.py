# Minimal payment example (account-based ledger) for presentation/demo
# - Simple in-memory ledger with accounts and balances
# - Transactions are created, 'signed' (mock), and applied after verification
# - Use this to illustrate on-chain payments, double-spend checks, and mempool processing
#
# Run: python3 payment_example.py

import hashlib
import time
import json
from collections import defaultdict

def mock_sign(private_key, message):
    # Mock signature: hash of private_key + message (for demo only)
    return hashlib.sha256(f"{private_key}:{message}".encode()).hexdigest()

def mock_verify(public_key, message, signature):
    # In this demo public_key is same as private_key; verify by recomputing
    return mock_sign(public_key, message) == signature

class Transaction:
    def __init__(self, from_acct, to_acct, amount, nonce, signature=None):
        self.from_acct = from_acct
        self.to_acct = to_acct
        self.amount = amount
        self.nonce = nonce
        self.signature = signature
        self.timestamp = time.time()

    def payload(self):
        return json.dumps({
            "from": self.from_acct,
            "to": self.to_acct,
            "amount": self.amount,
            "nonce": self.nonce,
            "timestamp": self.timestamp
        }, sort_keys=True)

    def sign(self, private_key):
        self.signature = mock_sign(private_key, self.payload())

class Ledger:
    def __init__(self):
        self.balances = defaultdict(int)
        self.nonces = defaultdict(int)
        self.mempool = []

    def submit_tx(self, tx):
        # Basic verification: signature, nonce, balance
        if not mock_verify(tx.from_acct, tx.payload(), tx.signature):
            print("Invalid signature — reject tx")
            return False
        if tx.nonce != self.nonces[tx.from_acct] + 1:
            print("Invalid nonce — possible replay or missing txs")
            return False
        if self.balances[tx.from_acct] < tx.amount:
            print("Insufficient funds — reject tx")
            return False
        # Accept to mempool
        self.mempool.append(tx)
        print("Tx accepted to mempool")
        return True

    def mine_block(self):
        # Very simple block application: apply all mempool txs in order
        print("Mining block with", len(self.mempool), "tx(s)")
        for tx in self.mempool:
            self.balances[tx.from_acct] -= tx.amount
            self.balances[tx.to_acct] += tx.amount
            self.nonces[tx.from_acct] += 1
        self.mempool = []
        print("Block applied")

if __name__ == "__main__":
    # Demo run
    ledger = Ledger()
    # Setup accounts (for demo, public_key == private_key)
    ledger.balances["Alice"] = 100
    ledger.balances["Bob"] = 20

    print("Initial balances:", dict(ledger.balances))

    # Alice sends 30 to Bob
    tx1 = Transaction("Alice", "Bob", 30, nonce=1)
    tx1.sign("Alice")  # mock sign with private key "Alice"
    ledger.submit_tx(tx1)

    # Attempt double spend: Alice tries to reuse same nonce
    tx2 = Transaction("Alice", "Carol", 80, nonce=1)
    tx2.sign("Alice")
    ledger.submit_tx(tx2)  # should fail nonce or insufficient funds if nonce allowed

    ledger.mine_block()
    print("Balances after block:", dict(ledger.balances))

    # Now Alice can send again with nonce=2
    tx3 = Transaction("Alice", "Carol", 50, nonce=2)
    tx3.sign("Alice")
    ledger.submit_tx(tx3)
    ledger.mine_block()
    print("Final balances:", dict(ledger.balances))
