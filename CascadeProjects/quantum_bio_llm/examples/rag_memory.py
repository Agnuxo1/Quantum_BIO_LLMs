import numpy as np
from collections import defaultdict

class WordNode:
    def __init__(self, word):
        self.word = word
        self.count = 1
        self.next = defaultdict(lambda: {'word': '', 'count': 0})
        self.color = {
            'hue': np.random.rand() * 360,
            'saturation': 1.0,
            'brightness': 1.0,
            'alpha': 0.5
        }
        self.documents = set()
        self.lastAccessed = 0
        self.strength = 1.0

class RAGMemory:
    def __init__(self):
        self.nodes = defaultdict(WordNode)
        self.totalDocuments = 0
        self.lastConsolidation = 0

    def add_document(self, text):
        words = text.split()
        for word in words:
            if word not in self.nodes:
                self.nodes[word] = WordNode(word)  # Properly initialize the WordNode
            node = self.nodes[word]
            node.count += 1
            node.lastAccessed = np.datetime64('now')
            node.documents.add(text)
            self.update_next_words(node, words)

        self.totalDocuments += 1

    def update_next_words(self, node, words):
        for i in range(len(words) - 1):
            if words[i] == node.word:
                next_word = words[i + 1]
                node.next[next_word]['word'] = next_word
                node.next[next_word]['count'] += 1

    def retrieve_response(self, query):
        words = query.split()
        mixed_color = self.mix_colors([self.nodes[word].color for word in words if word in self.nodes])
        return mixed_color

    def mix_colors(self, colors):
        if not colors:
            return {'hue': 0, 'saturation': 0, 'brightness': 0, 'alpha': 0}
        total_hue = sum(color['hue'] for color in colors)
        avg_hue = total_hue / len(colors)
        return {
            'hue': avg_hue,
            'saturation': 1.0,
            'brightness': 1.0,
            'alpha': 0.5
        }

# Example usage
if __name__ == '__main__':
    rag_memory = RAGMemory()
    rag_memory.add_document('Two and Three are important numbers.')
    rag_memory.add_document('Three is often used with Four.')
    response = rag_memory.retrieve_response('Two plus Three')
    print(f'Response Color: {response}')