"""Debug keyword extraction for failing multi-hop queries."""
import os, sys
os.chdir("/Users/zachary/Workspace/LongtermMemory/mosaic")
sys.path.insert(0, ".")

_stop = {
    "the", "and", "for", "are", "was", "were", "has", "have", "had",
    "that", "this", "with", "from", "what", "which", "who", "how",
    "does", "did", "not", "been", "but", "they", "them", "than",
    "can", "her", "his", "she", "you", "all", "any", "some", "will",
}

queries = [
    "How many times did Audrey and Andew plan to hike together?",
    "What outdoor activities has Andrew done other than hiking in nature?",
    "What is something that Audrey often dresses up her dogs with?",
    "What are the breeds of Audrey's dogs?",
    "What is the biggest stressor in Andrew's life besides not being able to hike frequently?",
]

for q in queries:
    q_lower = q.lower()
    q_words = set()
    for w in q_lower.split():
        w = w.strip(".,!?()[]\"':;")
        if len(w) > 2:
            q_words.add(w)
    q_words -= _stop
    print(f"Q: {q[:60]}")
    print(f"  Keywords: {sorted(q_words)}")
    print()
