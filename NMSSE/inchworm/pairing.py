from typing import List, Tuple

Pair = Tuple[float, float]
Pairing = List[Pair]


def generate_pairings(times: List[float]) -> List[Pairing]:
    if not times:
        return [[]]
    first = times[0]
    out: List[Pairing] = []
    for k in range(1, len(times)):
        second = times[k]
        pair = (first, second) if first <= second else (second, first)
        rest = times[1:k] + times[k+1:]
        for sub in generate_pairings(rest):
            out.append([pair] + sub)
    return out


def is_linked_two_Pairs(a: Pair, b: Pair) -> bool:
    a1, a2 = a if a[0] <= a[1] else (a[1], a[0])
    b1, b2 = b if b[0] <= b[1] else (b[1], b[0])
    
    return (a1 <= b1 <= a2 <= b2) or (b1 <= a1 <= b2 <= a2)


def linked_components_Pairs(pairs: Pairing) -> List[Pairing]:
    n = len(pairs)
    if n == 0:
        return []
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if is_linked_two_Pairs(pairs[i], pairs[j]):
                adj[i].append(j)
                adj[j].append(i)

    comps: List[Pairing] = []
    seen = [False]*n
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        idxs = []
        while stack:
            v = stack.pop()
            idxs.append(v)
            for u in adj[v]:
                if not seen[u]:
                    seen[u] = True
                    stack.append(u)
        comps.append([pairs[i] for i in idxs])
    return comps


def generate_linked_Pairs(times: List[float]) -> List[Pairing]:
    all_pairings = generate_pairings(times)
    return [p for p in all_pairings if len(linked_components_Pairs(p)) == 1]


def is_inchworm_proper_Pairing(pairing: Pairing, s_star: float) -> bool:
    # Each linked component has at least one endpoint >= s_star
    for comp in linked_components_Pairs(pairing):
        endpoints = []
        for a, b in comp:
            endpoints.append(a)
            endpoints.append(b)
        if not any(t >= s_star for t in endpoints):
            return False
    return True


def generate_inchworm_proper_pairings(times: List[float], s_star: float) -> List[Pairing]:
    if len(times) % 2 != 0:
        raise ValueError("time length must be even.")
    all_pairings = generate_pairings(times)
    return [p for p in all_pairings if is_inchworm_proper_Pairing(p, s_star)]