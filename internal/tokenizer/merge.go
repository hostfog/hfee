package tokenizer

import "container/heap"

// merge repeatedly joins the adjacent pair with the lowest priority value
// (ties go to the leftmost pair) until priority knows no remaining pair.
func merge(syms []string, priority func(left, right string) (float64, bool)) []string {
	if len(syms) < 2 {
		return syms
	}

	type node struct {
		text       string
		prev, next int
	}
	nodes := make([]node, len(syms))
	for i, s := range syms {
		nodes[i] = node{text: s, prev: i - 1, next: i + 1}
	}
	nodes[len(nodes)-1].next = -1

	q := &pairQueue{}
	push := func(l, r int) {
		if l < 0 || r < 0 {
			return
		}
		if p, ok := priority(nodes[l].text, nodes[r].text); ok {
			heap.Push(q, pair{left: l, right: r, prio: p, size: len(nodes[l].text) + len(nodes[r].text)})
		}
	}
	for i := 0; i+1 < len(nodes); i++ {
		push(i, i+1)
	}

	for q.Len() > 0 {
		p := heap.Pop(q).(pair)
		l, r := &nodes[p.left], &nodes[p.right]
		// skip stale entries: either side was merged away or changed since the push
		if l.text == "" || r.text == "" || l.next != p.right || len(l.text)+len(r.text) != p.size {
			continue
		}

		l.text += r.text
		r.text = ""
		l.next = r.next
		if r.next >= 0 {
			nodes[r.next].prev = p.left
		}
		push(l.prev, p.left)
		push(p.left, l.next)
	}

	out := syms[:0]
	for i := 0; i >= 0; i = nodes[i].next {
		out = append(out, nodes[i].text)
	}
	return out
}

type pair struct {
	left, right int
	prio        float64
	size        int
}

type pairQueue []pair

func (q pairQueue) Len() int { return len(q) }
func (q pairQueue) Less(i, j int) bool {
	if q[i].prio != q[j].prio {
		return q[i].prio < q[j].prio
	}
	return q[i].left < q[j].left
}
func (q pairQueue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q *pairQueue) Push(x any)   { *q = append(*q, x.(pair)) }
func (q *pairQueue) Pop() any {
	old := *q
	p := old[len(old)-1]
	*q = old[:len(old)-1]
	return p
}
