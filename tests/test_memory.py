from aetheragents import InMemoryVectorStore, MemoryManager


def test_short_term_window():
    m = MemoryManager("a")
    for i in range(5):
        m.add_message("user", f"msg {i}")
    recent = m.get_recent_context(3)
    assert len(recent) == 3
    assert recent[-1]["content"] == "msg 4"


def test_vector_search_ranks_relevant_first():
    m = MemoryManager("a")
    m.add_message("user", "I love hiking in the mountains")
    m.add_message("user", "Python is a great programming language")
    m.add_message("user", "The weather is sunny today")
    hits = m.search("tips about a programming language", n_results=2)
    assert hits
    assert "Python" in hits[0]["document"]


def test_inmemory_store_query():
    s = InMemoryVectorStore()
    s.add("1", "alpha beta gamma")
    s.add("2", "delta epsilon zeta")
    assert s.count() == 2
    hits = s.query("beta", k=1)
    assert hits[0]["id"] == "1"


def test_recall_alias():
    m = MemoryManager("a")
    m.remember("the launch code is 1234")
    hits = m.recall("launch code")
    assert hits and "launch" in hits[0]["document"]
