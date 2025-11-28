# Minimal consensus mechanism examples for presentation/demo
# - ProofOfWork: simple mining by finding a nonce meeting a difficulty prefix
# - ProofOfAuthority: rotating set of authorized validators (no heavy crypto)
# - PBFT: tiny simulation of pre-prepare / prepare / commit phases
# - ProofOfStake: proposer chosen by stake-weighted random selection
#
# Run: python3 consensus_examples.py

import hashlib
import time
import random
from collections import defaultdict, Counter

# -------------------
# Proof of Work (PoW)
# -------------------
class ProofOfWork:
    def __init__(self, difficulty=3):
        self.difficulty = difficulty  # number of leading hex zeros required

    def mine(self, data, max_tries=1000000):
        prefix = '0' * self.difficulty
        for nonce in range(max_tries):
            h = hashlib.sha256(f"{data}{nonce}".encode()).hexdigest()
            if h.startswith(prefix):
                return nonce, h
        return None, None

# -------------------
# Proof of Authority (PoA) - simple
# -------------------
class ProofOfAuthority:
    def __init__(self, validators):
        # validators: list of validator IDs (strings)
        self.validators = list(validators)

    def proposer_for_round(self, round_number):
        # deterministic round-robin proposer selection
        return self.validators[round_number % len(self.validators)]

    def create_block(self, round_number, payload):
        proposer = self.proposer_for_round(round_number)
        block = {
            "proposer": proposer,
            "round": round_number,
            "payload": payload,
            "signature": f"sig({proposer})"  # placeholder signature
        }
        return block

# -------------------
# PBFT (simplified educational simulation)
# -------------------
class PBFTNode:
    def __init__(self, node_id, nodes):
        self.id = node_id
        self.nodes = nodes  # list of node ids
        self.prepared = set()
        self.committed = set()

def run_pbft(nodes_count=4):
    # PBFT tolerates f faulty nodes where n >= 3f+1
    nodes = [PBFTNode(i, list(range(nodes_count))) for i in range(nodes_count)]
    primary = 0
    request = "transfer $10"
    print(f"PBFT demo: n={nodes_count}, primary={primary}, request='{request}'")

    # Pre-prepare (primary sends)
    preprepare = {"view": 0, "seq": 1, "request": request, "primary": primary}
    print(" primary -> pre-prepare:", preprepare)

    # Prepare: every node (including primary) broadcasts prepare
    prepares = defaultdict(set)
    for node in nodes:
        prepares[(preprepare["view"], preprepare["seq"])].add(node.id)
    print(" prepares received from nodes:", sorted(list(prepares[(0,1)])))

    # If 2f+1 prepares reached, mark prepared
    f = (nodes_count - 1) // 3
    prepared_quorum = 2 * f + 1
    if len(prepares[(0,1)]) >= prepared_quorum:
        print(f" prepared quorum reached ({len(prepares[(0,1)])} >= {prepared_quorum})")

    # Commit: nodes broadcast commit; when 2f+1 commits, the request is executed
    commits = prepares[(0,1)]  # in this small sim assume same nodes commit
    if len(commits) >= prepared_quorum:
        print(f" commit quorum reached ({len(commits)}). Request EXECUTED: {request}")
    else:
        print(" commit quorum NOT reached in this demo")

# -------------------
# Proof of Stake (PoS) - simple weighted proposer selection
# -------------------
class ProofOfStake:
    def __init__(self, stakes):
        # stakes: dict validator_id -> stake_amount (numeric)
        self.stakes = dict(stakes)

    def choose_proposer(self):
        total = sum(self.stakes.values())
        if total == 0:
            return random.choice(list(self.stakes.keys()))
        r = random.random() * total
        upto = 0
        for v, s in self.stakes.items():
            upto += s
            if r <= upto:
                return v
        # fallback
        return list(self.stakes.keys())[-1]

# -------------------
# Demo runner
# -------------------
if __name__ == "__main__":
    print("==== Proof of Work demo ====")
    pow_demo = ProofOfWork(difficulty=3)  # adjust difficulty for demo speed
    start = time.time()
    nonce, h = pow_demo.mine("demo-block")
    elapsed = time.time() - start
    print(" mined nonce:", nonce, "hash:", h, f"(took {elapsed:.3f}s)")

    print("\n==== Proof of Authority demo ====")
    poa = ProofOfAuthority(validators=["A", "B", "C"])
    for r in range(3):
        blk = poa.create_block(r, payload=f"tx{r}")
        print(" round", r, "proposer:", blk["proposer"], "signature:", blk["signature"])

    print("\n==== PBFT demo ====")
    run_pbft(nodes_count=4)

    print("\n==== Proof of Stake demo ====")
    stakes = {"Alice": 50, "Bob": 30, "Carol": 20}
    pos = ProofOfStake(stakes)
    picks = Counter(pos.choose_proposer() for _ in range(1000))
    print(" proposer distribution over 1000 picks (approx proportional to stake):", picks)
